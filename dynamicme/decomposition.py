#============================================================
# File decomposition.py
#
# class  
#
# Decomposition methods
#
# Laurence Yang, SBRG, UCSD
#
# 21 Feb 2018:  first version
#============================================================

from six import iteritems
from builtins import range
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from cobra.core import Solution
from cobra import Reaction, Metabolite, Model
from cobra import DictList
from cobra.solvers import gurobi_solver
from gurobipy import *
from qminos.quadLP import QMINOS
from cobra.solvers.gurobi_solver import _float, variable_kind_dict
from cobra.solvers.gurobi_solver import parameter_defaults, parameter_mappings
from cobra.solvers.gurobi_solver import sense_dict, status_dict, objective_senses, METHODS

import gurobipy as grb
import numpy as np
import cobra
import warnings

class Decomposer(object):
    def __init__(self, cobra_model, objective_sense, quadratic_component=None, solver='gurobi'):
        self.objective_sense = objective_sense
        milp = self.to_solver_model(cobra_model, quadratic_component, solver)
        self.milp = milp
        self._INF = 1e3 # Not too big if multiplying with binary var.
        self._master = None
        self._sub = None
        self._A = None
        self._Acsc = None
        self._B = None
        self._Bcsc = None
        self._d = None
        self._csenses = None
        self._xs = None
        self._ys = None
        self._wa = None
        self._wl = None
        self._wu = None
        self._xl = None
        self._xu = None
        self._cx  = None
        self._fy  = None

    def to_solver_model(self, cobra_model, quadratic_component=None, solver='gurobi'):
        """
        Create solver-specific model, while keeping track of variable and constraint mapping.
        Easiest solution: keep the same rxn(var) / met(constraint) names.
        """
        from cobra.solvers.gurobi_solver import _float, variable_kind_dict
        from cobra.solvers.gurobi_solver import parameter_defaults, parameter_mappings
        from cobra.solvers.gurobi_solver import sense_dict, status_dict, objective_senses, METHODS


        if solver != 'gurobi':
            raise ValueError("Only solver=gurobi supported, currently")

        lp = grb.Model(cobra_model.id)
        params = parameter_defaults
        for k,v in iteritems(params):
            gurobi_solver.set_parameter(lp, k, v)

        # Add and track variables
        variable_list = [lp.addVar(_float(x.lower_bound),
                                   _float(x.upper_bound),
                                   float(x.objective_coefficient),
                                   variable_kind_dict[x.variable_kind],
                                   str(x.id))
                         for i, x in enumerate(cobra_model.reactions)]
        reaction_to_variable = dict(zip(cobra_model.reactions,
                                        variable_list))
        # Integrate new variables
        lp.update()

        # Add and track constraints
        for i, the_metabolite in enumerate(cobra_model.metabolites):
            constraint_coefficients = []
            constraint_variables = []
            for the_reaction in the_metabolite._reaction:
                constraint_coefficients.append(_float(the_reaction._metabolites[the_metabolite]))
                constraint_variables.append(reaction_to_variable[the_reaction])
            #Add the metabolite to the problem
            lp.addConstr(LinExpr(constraint_coefficients, constraint_variables),
                         sense_dict[the_metabolite._constraint_sense.upper()],
                         the_metabolite._bound,
                         str(the_metabolite.id))

        # Set objective to quadratic program
        if quadratic_component is not None:
            gurobi_solver.set_quadratic_objective(lp, quadratic_component)

        lp.update()

        return lp

    def convert_solution(self):
        """
        Format solution back to cobra from master and subproblem(s).
        master : master problem
        submodels : list of submodels
        cobra_mdl : original cobra model used to create MILP
        """


    def benders_decomp(self):
        """
        master, sub = benders_decomp()

        Decomposes self.milp into master and sub problems

        Returns
        master, sub

        TODO: for multiple subproblems.
        """
        self._split_constraints()
        master = self.make_master()
        sub = self.make_sub()
        self._master = master
        self._sub = sub

        return master, sub

    def _split_constraints(self):
        A,B,d,csenses,xs,ys,C,b,csenses_mp = split_constraints(self.milp)
        self._A = A
        self._Acsc = A.tocsc()
        self._B = B
        self._Bcsc = B.tocsc()
        self._d = d
        self._csenses = csenses
        self._C = C
        self._b = b
        self._csenses_mp = csenses_mp
        self._x0 = xs
        self._y0 = ys

    def make_master(self):
        """
        min     z
        y,z
        s.t.    Cy [<=>] b
                z >= f'y + (dk-By)'wA_i,k + lk*wl_i,k - uk*wu_i,k,  i \in OptimalityCuts
                (dk-By)'wA_i,k + lk*wl_i,k - uk*wu_i,k <= 0,        i \in FeasibilityCuts
        """
        LB = -self._INF
        UB = self._INF
        if self._y0 is None:
            self._split_constraints()

        ys0 = self._y0
        ny = len(ys0)
        C = self._C
        b = self._b
        csenses_mp = self._csenses_mp

        B = self._B
        fy = np.array([yj.Obj for yj in ys0])
        master = Model('master')
        z = master.addVar(LB, UB, 0., GRB.CONTINUOUS, 'z')
        ys = [master.addVar(y.LB, y.UB, y.Obj, y.VType, y.VarName) for y in ys0]
        # Add Cy [<=>] b
        for i in range(C.shape[0]):
            csense= csenses_mp[i]
            bi    = b[i]
            cinds = C.indices[C.indptr[i]:C.indptr[i+1]]
            coefs = C.data[C.indptr[i]:C.indptr[i+1]]
            expr  = LinExpr(coefs, [ys[j] for j in cinds])
            cons  = master.addConstr(expr, csense, bi, name='MP_%s'%i)

        # Add z >= f'y initially
        rhs = LinExpr(fy, ys)
        master.addConstr(z, GRB.GREATER_EQUAL, rhs)
        master.setObjective(z, GRB.MINIMIZE)

        self._z = z
        self._ys = ys
        self._fy = fy

        master._decomposer = self   # Need this reference to access methods inside callback
        master._verbosity = 0
        master._gaptol = 1e-6
        master._precision_sub = 'double'

        master.Params.Presolve = 0          # To be safe, turn off
        master.Params.LazyConstraints = 1   # Required to use cbLazy
        master.Params.IntFeasTol = 1e-9

        master.update()

        return master

    def make_sub_primal(self, yopt):
        """
        Make primal subproblem
        min(max) c'x
        s.t.    Ax [<=>] d-B*y
                l <= x <= u
        """
        obj_sense = self.objective_sense
        xs0 = self._x0
        A = self._A
        d = self._d
        B = self._B
        csenses = self._csenses
        m = len(d)
        n = len(xs0)
        nx = n
        xl = np.array([x.LB for x in xs0])
        xu = np.array([x.UB for x in xs0])
        cx  = [x.Obj for x in xs0]

        sub = Model('sub_primal')
        xs  = [sub.addVar(x.LB, x.UB, x.Obj, x.VType, x.VarName) for x in xs0]
        By  = B*yopt
        for i in range(m):
            cinds = A.indices[A.indptr[i]:A.indptr[i+1]]
            coefs = A.data[A.indptr[i]:A.indptr[i+1]]
            ax    = LinExpr(coefs, [xs[j] for j in cinds])
            rhs   = d[i] - By[i]
            cons  = sub.addConstr(ax, csenses[i], rhs, name=str(i))

        sub.update()
        sub_wrapper = DecompModel(sub)

        return sub_wrapper

    def make_sub(self):
        """
        Constraint doesn't change.
        Objective changes with RMP solution, so yopt is optional when initiating.
        Allow dual to become unbounded to detect infeasible primal (i.e., no
        artificial box constraints)
        """
        INF = GRB.INFINITY
        ZERO = 1e-15
        if self._x0 is None:
            self._split_constraints()

        xs0 = self._x0
        A = self._A
        d = self._d
        csenses = self._csenses

        m = len(d)
        n = len(xs0)
        nx = n
        sub = Model('sub')
        sub.Params.InfUnbdInfo = 1
        lb_dict = {GRB.EQUAL: -INF, GRB.GREATER_EQUAL: 0., GRB.LESS_EQUAL: -INF}
        ub_dict = {GRB.EQUAL: INF, GRB.GREATER_EQUAL: INF, GRB.LESS_EQUAL: 0.}
        wa = [sub.addVar(lb_dict[sense], ub_dict[sense], 0., GRB.CONTINUOUS, 'wa[%d]'%i) \
                for i,sense in enumerate(csenses)]
        wl = [sub.addVar(0., INF, 0., GRB.CONTINUOUS, 'wl[%d]'%i) for i in range(n)]
        wu = [sub.addVar(0., INF, 0., GRB.CONTINUOUS, 'wu[%d]'%i) for i in range(n)]
        xl = np.array([x.LB for x in xs0])
        xu = np.array([x.UB for x in xs0])
        cx  = [x.Obj for x in xs0]

        self._xl = xl
        self._xu = xu
        self._cx  = cx
        self._wa = wa
        self._wl = wl
        self._wu = wu

        # This dual constraint never changes
        Acsc = self._Acsc
        dual_cons = []
        for j,x0 in enumerate(xs0):
            rinds = Acsc.indices[Acsc.indptr[j]:Acsc.indptr[j+1]]
            coefs = Acsc.data[Acsc.indptr[j]:Acsc.indptr[j+1]]
            expr  = LinExpr(coefs, [wa[i] for i in rinds])
            #expr.addTerms([1., -1.], [wl[j], wu[j]])
            #cons  = sub.addConstr(expr+wl[j]-wu[j], GRB.EQUAL, cx[j], name=x0.VarName)
            if x0.LB>=0 and x0.UB>=0:
                csense = GRB.LESS_EQUAL
            elif x0.LB<-ZERO and x0.UB>ZERO:
                csense = GRB.EQUAL
            elif x0.LB<-ZERO and x0.UB<=ZERO:
                csense = GRB.GREATER_EQUAL
            else:
                print('Did not account for lb=%g and ub=%g'%(x0.LB,x0.UB))
                raise ValueError

            cons  = sub.addConstr(expr+wl[j]-wu[j], csense, cx[j], name=x0.VarName)
            dual_cons.append(cons)

        sub.update()

        sub_wrapper = DecompModel(sub)

        return sub_wrapper

    def update_subobj(self, yopt):
        sub = self._sub
        if sub is None:
            sub = self.make_sub(yopt)

        d = self._d
        B = self._B
        wa = self._wa
        wl = self._wl
        wu = self._wu
        xl = self._xl
        xu = self._xu
        m,ny = B.shape
        n = len(xl)

        #sub.setObjective(
        #    sum([d[i]*wa[i]-sum([B[i,j]*yopt[j]*wa[i] for j in range(ny)]) \
        #        for i in range(m)]) + \
        #        sum([xl[j]*wl[j] for j in range(n)]) - \
        #        sum([xu[j]*wu[j] for j in range(n)]), GRB.MAXIMIZE)

        dBy = d - B*yopt
        cinds = dBy.nonzero()[0]
        try:
            dBywa = LinExpr([dBy[j] for j in cinds], [wa[j] for j in cinds])
            obj = dBywa + LinExpr(xl,wl) - LinExpr(xu,wu)
            sub.setObjective(obj, GRB.MAXIMIZE)
            sub.update()
        except GurobiError as e:
            print('Caught GurobiError (%s) in update_subobj(yopt=%s)'%(repr(e),yopt))

    def make_optcut(self):
        z = self._z
        fy = self._fy
        ys = self._ys
        d = self._d
        B = self._B
        xl = self._xl
        xu = self._xu
        m = len(d)
        n = len(xl)
        ny = len(ys)
        #wa = np.array([w.X for w in self._wa])
        #wl = np.array([w.X for w in self._wl])
        #wu = np.array([w.X for w in self._wu])
        wa = np.array([self._sub.x_dict[w.VarName] for w in self._wa])
        wl = np.array([self._sub.x_dict[w.VarName] for w in self._wl])
        wu = np.array([self._sub.x_dict[w.VarName] for w in self._wu])

        Bcsc = self._Bcsc

        # cut = z >= quicksum([fy[j]*ys[j] for j in range(ny)]) + \
        #         quicksum([d[i]*wa[i]-quicksum([B[i,j]*ys[j]*wa[i] for j in range(ny)]) \
        #         for i in range(m)]) + \
        #         quicksum([xl[j]*wl[j] for j in range(n)]) -\
        #         quicksum([xu[j]*wu[j] for j in range(n)])

        waB  = wa*Bcsc
        cinds = waB.nonzero()[0]
        try:
            waBy = LinExpr([waB[j] for j in cinds], [ys[j] for j in cinds])
            cut = z >= LinExpr(fy,ys) + sum(d*wa) - waBy + sum(xl*wl) - sum(xu*wu)
        except GurobiError as e:
            print('Caught GurobiError (%s) in make_optcut()'%repr(e))

        return cut

    def make_feascut(self):
        """
        Using unbounded ray information (Model.UnbdRay)
        """
        # wa = [w.X for w in self._wa]
        # wl = [w.X for w in self._wl]
        # wu = [w.X for w in self._wu]
        wa = np.array([self._sub.x_dict[w.VarName] for w in self._wa])
        wl = np.array([self._sub.x_dict[w.VarName] for w in self._wl])
        wu = np.array([self._sub.x_dict[w.VarName] for w in self._wu])
        ys = self._ys
        d = self._d
        Bcsc = self._Bcsc
        xl = self._xl
        xu = self._xu
        m = len(d)
        n = len(xl)

        waB  = wa*Bcsc
        cinds = waB.nonzero()[0]
        try:
            waBy = LinExpr([waB[j] for j in cinds], [ys[j] for j in cinds])
            cut = sum(d*wa) - waBy + sum(xl*wl) - sum(xu*wu) <= 0
        except GurobiError as e:
            print('Caught GurobiError (%s) in make_feascut'%repr(e))

        return cut

    def calc_sub_objval(self, yopt):
        sub = self._sub
        fy = self._fy
        objval = sub.ObjVal + sum(fy*yopt)
        #objval = sub.ObjVal + sum([fy[j]*yopt[j] for j in range(len(yopt))])

        return objval

    def get_sub(self):
        if self._sub is None:
            self._sub = self.make_sub()

        return self._sub

    def get_master(self):
        if self._master is None:
            self._master = self.make_master()
        return self._master


class StackedDecomposer(Decomposer):
    """
    Multi-cut version of decomposer. Works with stacked models and (sub)model generators.
    """
    def __init__(self, model_dict, objective_sense, quadratic_component=None, solver='gurobi'):
        self.model_dict = model_dict
        self.objective_sense = objective_sense
        #milp = self.to_solver_model(cobra_model, quadratic_component, solver)
        #self.milp = milp
        self._INF = 1e3 # Not too big if multiplying with binary var.
        self._master = None
        self._sub = None
        self._A = None
        self._Acsc = None
        self._B = None
        self._Bcsc = None
        self._d = None
        self._csenses = None
        self._xs = None
        self._ys = None
        self._wa = None
        self._wl = None
        self._wu = None
        self._xl = None
        self._xu = None
        self._cx  = None
        self._fy  = None

    def update_subobj(self, yopt):
        """
        Update submodel objectives
        """
        # Return generator of submodels
        submodels = self.get_submodels()

        for sub in submodels:
            d = sub._d
            B = sub._B
            wa = sub._wa
            wl = sub._wl
            wu = sub._wu
            xl = sub._xl
            xu = sub._xu
            m,ny = B.shape
            n = len(xl)

            dBy = d - B*yopt
            cinds = dBy.nonzero()[0]
            try:
                dBywa = LinExpr([dBy[j] for j in cinds], [wa[j] for j in cinds])
                obj = dBywa + LinExpr(xl,wl) - LinExpr(xu,wu)
                sub.setObjective(obj, GRB.MAXIMIZE)
                sub.update()
            except GurobiError as e:
                print('Caught GurobiError (%s) in update_subobj(yopt=%s)'%(repr(e),yopt))

    def make_feascut(self, yopt, zmaster, sub):
        wa = np.array([sub.x_dict[w.VarName] for w in sub._wa])
        wl = np.array([sub.x_dict[w.VarName] for w in sub._wl])
        wu = np.array([sub.x_dict[w.VarName] for w in sub._wu])
        ys = sub._ys    # ys same across all subproblems
        d = sub._d
        Bcsc = sub._Bcsc
        xl = sub._xl
        xu = sub._xu
        m = len(d)
        n = len(xl)

        waB  = wa*Bcsc
        cinds = waB.nonzero()[0]
        try:
            waBy = LinExpr([waB[j] for j in cinds], [ys[j] for j in cinds])
            cut = sum(d*wa) - waBy + sum(xl*wl) - sum(xu*wu) <= 0
        except GurobiError as e:
            print('Caught GurobiError (%s) in make_feascut'%repr(e))

        return cut

    def make_optcut(self, sub):
        z = sub._z     # same across all subproblems 
        fy = sub._fy    # same across all subproblems
        ys = sub._ys
        d = sub._d
        B = sub._B
        xl = sub._xl
        xu = sub._xu
        m = len(d)
        n = len(xl)
        ny = len(ys)
        wa = np.array([sub.x_dict[w.VarName] for w in sub._wa])
        wl = np.array([sub.x_dict[w.VarName] for w in sub._wl])
        wu = np.array([sub.x_dict[w.VarName] for w in sub._wu])

        Bcsc = sub._Bcsc

        waB  = wa*Bcsc
        cinds = waB.nonzero()[0]
        try:
            waBy = LinExpr([waB[j] for j in cinds], [ys[j] for j in cinds])
            cut = z >= LinExpr(fy,ys) + sum(d*wa) - waBy + sum(xl*wl) - sum(xu*wu)
        except GurobiError as e:
            print('Caught GurobiError (%s) in make_optcut()'%repr(e))

        return cut


    def get_submodels(self):
        pass


    def make_sub_base(self):
        """
        Create base submodel.
        """
        INF = GRB.INFINITY
        ZERO = 1e-15
        if self._x0 is None:
            self._split_constraints()

        xs0 = self._x0
        A = self._A
        d = self._d
        csenses = self._csenses

        m = len(d)
        n = len(xs0)
        nx = n
        sub = Model('sub')
        sub.Params.InfUnbdInfo = 1
        lb_dict = {GRB.EQUAL: -INF, GRB.GREATER_EQUAL: 0., GRB.LESS_EQUAL: -INF}
        ub_dict = {GRB.EQUAL: INF, GRB.GREATER_EQUAL: INF, GRB.LESS_EQUAL: 0.}
        wa = [sub.addVar(lb_dict[sense], ub_dict[sense], 0., GRB.CONTINUOUS, 'wa[%d]'%i) \
                for i,sense in enumerate(csenses)]
        wl = [sub.addVar(0., INF, 0., GRB.CONTINUOUS, 'wl[%d]'%i) for i in range(n)]
        wu = [sub.addVar(0., INF, 0., GRB.CONTINUOUS, 'wu[%d]'%i) for i in range(n)]
        xl = np.array([x.LB for x in xs0])
        xu = np.array([x.UB for x in xs0])
        cx  = [x.Obj for x in xs0]

        self._xl = xl
        self._xu = xu
        self._cx  = cx
        self._wa = wa
        self._wl = wl
        self._wu = wu

        # This dual constraint never changes
        Acsc = self._Acsc
        dual_cons = []
        for j,x0 in enumerate(xs0):
            rinds = Acsc.indices[Acsc.indptr[j]:Acsc.indptr[j+1]]
            coefs = Acsc.data[Acsc.indptr[j]:Acsc.indptr[j+1]]
            expr  = LinExpr(coefs, [wa[i] for i in rinds])
            #expr.addTerms([1., -1.], [wl[j], wu[j]])
            #cons  = sub.addConstr(expr+wl[j]-wu[j], GRB.EQUAL, cx[j], name=x0.VarName)
            if x0.LB>=0 and x0.UB>=0:
                csense = GRB.LESS_EQUAL
            elif x0.LB<-ZERO and x0.UB>ZERO:
                csense = GRB.EQUAL
            elif x0.LB<-ZERO and x0.UB<=ZERO:
                csense = GRB.GREATER_EQUAL
            else:
                print('Did not account for lb=%g and ub=%g'%(x0.LB,x0.UB))
                raise ValueError

            cons  = sub.addConstr(expr+wl[j]-wu[j], csense, cx[j], name=x0.VarName)
            dual_cons.append(cons)

        sub.update()

        sub_wrapper = DecompModel(sub)

        return sub_wrapper








class DecompModel(object):
    """
    Class for Benders decomposition
    """
    def __init__(self, model):
        self.model = model
        qsolver = QMINOS()
        qsolver.set_realopts('lp', {'Feasibility tol':1e-15, 'Optimality tol':1e-15})
        qsolver.set_realopts('lp_q2', {'Feasibility tol':1e-15, 'Optimality tol':1e-15})
        self.qsolver = qsolver

        self.lp_basis = None
        self.qp_basis = None
        self.A = None
        self.B = None
        self.d = None
        self.xs = None
        self.ys = None
        self.xl = None
        self.xu = None
        self.csense = None
        self._ObjVal = None
        self.x_dict = None

    @property
    def ObjVal(self):
        return self._ObjVal

    @ObjVal.setter
    def ObjVal(self, value):
        self._ObjVal = value

    def __getattr__(self, attr):
        return getattr(self.model, attr)

    # def __setattr__(self, name, value):
    #     super(DecompModel, self).__setattr__(name, value)

    def optimize(self, precision='gurobi'):
        model = self.model
        if precision=='gurobi':
            try:
                model.optimize()
                if model.Status==GRB.OPTIMAL:
                    self.xopt = np.array([x.X for x in model.getVars()])
                    self.x_dict = {x.VarName:x.X for x in model.getVars()}
                    self.ObjVal = model.ObjVal
                elif model.Status == GRB.UNBOUNDED:
                    ray = model.UnbdRay
                    self.xopt   = ray
                    self.x_dict = {x.VarName:ray[j] for j,x in enumerate(model.getVars())}
                    self.ObJVal = np.nan #model.ObjVal

            except GurobiError as e:
                print('Caught GurobiError in DecompModel.optimize(): %s'%repr(e))
        else:
            self.qminos_solve(precision)

    def qminos_solve(self, precision):
        csense_dict = {'=':'E', '<':'L', '>':'G'}
        model = self.model
        qsolver = self.qsolver
        obj_sense = model.ModelSense
        if obj_sense==GRB.MAXIMIZE:
            obj_sense_str = 'Maximize'
        else:
            obj_sense_str = 'Minimize'
        for alg in ['lp','lp_d','lp_q2']:
            qsolver.opt_strlist[alg][0] = obj_sense_str

        #if self.A is None:
        # Need to update every time for now
        A,B,d,csenses0,xs,ys,C,b,csenses_mp = split_constraints(model)
        xl = np.array([x.LB for x in xs])
        xu = np.array([x.UB for x in xs])
        csenses = [csense_dict[sense] for sense in csenses0]
        cx = np.array([x.OBJ for x in xs])
        self.cx = cx
        self.xl = xl
        self.xu = xu

        if len(ys)>0:
            yopt = [y.X for y in ys]
            b = d - B*yopt
        else:
            b = d

        basis = self.lp_basis

        xall,stat,hs = qsolver.solvelp(A,b,cx,xl,xu,csenses,precision,basis=basis, verbosity=0)
        nx = len(cx)
        xopt = xall[0:nx]

        self.xopt = xopt
        self.stat = stat
        self.lp_basis = hs

        self.ObjVal = sum(cx*xopt)
        self.x_dict = {xj.VarName:xopt[j] for j,xj in enumerate(xs)}

        # Translate qminos solution to original solver format

def split_cobra(model):
    """
    INPUTS
    model : cobra model

    Splits constraints into continuous and integer parts.
    Complicating constraints:       Ax + By [<=>] d
    Integer-only constraints:            Cy [<=>] b
    Continuous-only constraints:    Dx      [<=>] e
    """
    intvar_types = ['integer','binary']
    xs = [x for x in model.reactions if x.variable_kind == 'continuous']
    ys = [x for x in model.reactions if x.variable_kind in intvar_types]
    csenses = [x._constraint_sense for x in model.metabolites]
    d = [x._bound for x in model.metabolites]
    xdata = []
    xrow_inds = []
    xcol_inds = []

    ydata = []
    yrow_inds = []
    ycol_inds = []
    for row_idx, met in enumerate(model.metabolites):
        for rxn in met.reactions:
            coeff = rxn.metabolites[met]
            if rxn.variable_kind == 'continuous':
                col_idx = xs.index(rxn)
                xdata.append(coeff)
                xcol_inds.append(col_idx)
                xrow_inds.append(row_idx)
            elif rxn.variable_kind in intvar_types:
                col_idx = ys.index(rxn)
                ydata.append(coeff)
                ycol_inds.append(col_idx)
                yrow_inds.append(row_idx)
            else:
                print("rxn.variable_kind must be in ['continuous','integer','binary']")
                raise Exception("rxn.variable_kind must be in ['continuous','integer','binary']")

    M  = len(model.metabolites)
    nx = len(xs)
    ny = len(ys)
    A = coo_matrix((xdata, (xrow_inds, xcol_inds)), shape=(M,nx)).tocsr()
    B = coo_matrix((ydata, (yrow_inds, ycol_inds)), shape=(M,ny)).tocsr()
    # Find y-only rows
    boolb = abs(A).sum(axis=1) == 0
    b_rows = boolb.nonzero()[0]
    C = B[b_rows,:]
    d0 = np.array(d)
    b = d0[b_rows]
    csenses0 = np.array(csenses)
    bsenses = csenses0[b_rows]

    boold = ~boolb
    d_rows = boold.nonzero()[0]
    Amix = A[d_rows,:]
    Bmix = B[d_rows,:]
    dmix = d0[d_rows]
    csenses_mix = csenses0[d_rows]

    return Amix, Bmix, dmix, csenses_mix, xs, ys, C, b, bsenses



def split_constraints(model):
    """
    Splits constraints into continuous and integer parts:
    Complicating constraints:       Ax + By [<=>] d
    Integer-only constraints:            Cy [<=>] b
    Continuous-only constraints:    Dx      [<=>] e
    """
    constrs = model.getConstrs()
    xs = [x for x in model.getVars() if x.VType == GRB.CONTINUOUS]
    x_inds = {x:i for i,x in enumerate(xs)}
    ys = [x for x in model.getVars() if x.VType in (GRB.INTEGER, GRB.BINARY)]
    y_inds = {x:i for i,x in enumerate(ys)}
    # Need to make index dictionary since __eq__ method of grb var 
    # prevents use of list(vars).index(...)    
    csenses = []
    d = []

    xdata = []
    xrow_inds = []
    xcol_inds = []

    ydata = []
    yrow_inds = []
    ycol_inds = []

    for row_idx, constr in enumerate(constrs):
        csenses.append(constr.Sense)
        d.append(constr.RHS)
        row = model.getRow(constr)
        for j in range(row.size()):
            coeff = row.getCoeff(j)
            vj = row.getVar(j)
            if vj in x_inds:
                col_idx = x_inds[vj]
                xdata.append(coeff)
                xcol_inds.append(col_idx)
                xrow_inds.append(row_idx)
            elif vj in y_inds:
                col_idx = y_inds[vj]
                ydata.append(coeff)
                ycol_inds.append(col_idx)
                yrow_inds.append(row_idx)
            else:
                print('vj should be in either x_inds or y_inds')
                raise Exception('vj should be in either x_inds or y_inds')

    M  = len(constrs)
    nx = len(xs)
    ny = len(ys)
    A = coo_matrix((xdata, (xrow_inds, xcol_inds)), shape=(M,nx)).tocsr()
    B = coo_matrix((ydata, (yrow_inds, ycol_inds)), shape=(M,ny)).tocsr()
    #********************************************************
    # DEBUG
    #********************************************************
    # Find y-only rows
    boolb = abs(A).sum(axis=1) == 0
    b_rows = boolb.nonzero()[0]
    C = B[b_rows,:]
    d0 = np.array(d)
    b = d0[b_rows]
    csenses0 = np.array(csenses)
    bsenses = csenses0[b_rows]

    boold = ~boolb
    d_rows = boold.nonzero()[0]
    Amix = A[d_rows,:]
    Bmix = B[d_rows,:]
    dmix = d0[d_rows]
    csenses_mix = csenses0[d_rows]
    #********************************************************

    return Amix, Bmix, dmix, csenses_mix, xs, ys, C, b, bsenses




class BendersSubmodel(object):
    """
    Primal (Benders) decomposition submodel.
    """
    def __init__(self, cobra_model, solver='gurobi'):
        A,B,d,csenses,xs,ys,C,b,csenses_mp = split_cobra(cobra_model)
        self._A = A
        self._Acsc = A.tocsc()
        self._B = B
        self._Bcsc = B.tocsc()
        self._d = d
        self._csenses = csenses
        self._C = C
        self._b = b
        self._csenses_mp = csenses_mp
        self._x0 = xs
        self._y0 = ys

        solver_csenses = [sense_dict[i] for i in csenses]
        self.init_model(xs, A, d, solver_csenses)

    def init_model(self, xs0, A, d, csenses):
        INF = GRB.INFINITY
        ZERO = 1e-15
        m = len(d)
        n = len(xs0)
        nx = n
        model = grb.Model('sub')
        model.Params.InfUnbdInfo = 1
        lb_dict = {GRB.EQUAL: -INF, GRB.GREATER_EQUAL: 0., GRB.LESS_EQUAL: -INF}
        ub_dict = {GRB.EQUAL: INF, GRB.GREATER_EQUAL: INF, GRB.LESS_EQUAL: 0.}
        wa = [model.addVar(lb_dict[sense], ub_dict[sense], 0., GRB.CONTINUOUS, 'wa[%d]'%i) \
                for i,sense in enumerate(csenses)]
        wl = [model.addVar(0., INF, 0., GRB.CONTINUOUS, 'wl[%d]'%i) for i in range(n)]
        wu = [model.addVar(0., INF, 0., GRB.CONTINUOUS, 'wu[%d]'%i) for i in range(n)]
        xl = np.array([x.lower_bound for x in xs0])
        xu = np.array([x.upper_bound for x in xs0])
        cx  = [x.objective_coefficient for x in xs0]

        self._xl = xl
        self._xu = xu
        self._cx  = cx
        self._wa = wa
        self._wl = wl
        self._wu = wu

        # This dual constraint never changes
        Acsc = self._Acsc
        dual_cons = []
        for j,x0 in enumerate(xs0):
            rinds = Acsc.indices[Acsc.indptr[j]:Acsc.indptr[j+1]]
            coefs = Acsc.data[Acsc.indptr[j]:Acsc.indptr[j+1]]
            expr  = LinExpr(coefs, [wa[i] for i in rinds])
            if x0.lower_bound>=0 and x0.upper_bound>=0:
                csense = GRB.LESS_EQUAL
            elif x0.lower_bound<-ZERO and x0.upper_bound>ZERO:
                csense = GRB.EQUAL
            elif x0.lower_bound<-ZERO and x0.upper_bound<=ZERO:
                csense = GRB.GREATER_EQUAL
            else:
                print('Did not account for lb=%g and ub=%g'%(x0.lower_bound,x0.upper_bound))
                raise ValueError

            cons  = model.addConstr(expr+wl[j]-wu[j], csense, cx[j], name=x0.id)
            dual_cons.append(cons)

        model.update()
        self.model = model

    def update_obj(self, yopt):
        model = self.model
        d = self._d
        B = self._B
        wa = self._wa
        wl = self._wl
        wu = self._wu
        xl = self._xl
        xu = self._xu
        m,ny = B.shape
        n = len(xl)

        dBy = d - B*yopt
        cinds = dBy.nonzero()[0]
        try:
            dBywa = LinExpr([dBy[j] for j in cinds], [wa[j] for j in cinds])
            obj = dBywa + LinExpr(xl,wl) - LinExpr(xu,wu)
            model.setObjective(obj, GRB.MAXIMIZE)
            model.update()
        except GurobiError as e:
            print('Caught GurobiError (%s) in update_subobj(yopt=%s)'%(repr(e),yopt))

    def make_optcut(self):
        pass

    def make_feascut(self):
        pass

