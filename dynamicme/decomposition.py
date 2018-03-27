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

from __future__ import division
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

from dynamicme.callback_gurobi import cb_benders_multi

import gurobipy as grb
import numpy as np
import cobra
import warnings
import time

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
        master._gaptol = 1e-4   # relative gaptol
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
                    #self.xopt = np.array([x.X for x in model.getVars()])
                    self.x_dict = {x.VarName:x.X for x in model.getVars()}
                    self.ObjVal = model.ObjVal
                elif model.Status == GRB.UNBOUNDED:
                    ray = model.UnbdRay
                    #self.xopt   = ray
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

        #self.xopt = xopt
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


class BendersMaster(object):
    """
    Restricted Master Problem for Primal (Benders) decomposition.

    min     z
    y,z
    s.t.    Cy [<=>] b
            z >= f'y + sum_k tki,                           i \in OptimalityCuts
            tki >= (dk-By)'wA_i,k + lk*wl_i,k - uk*wu_i,k,  i \in OptimalityCuts
            (dk-By)'wA_i,k + lk*wl_i,k - uk*wu_i,k <= 0,    i \in FeasibilityCuts
            tki >= zDk* - u*Hk*y    (cross-decomposition from Lagrangean subproblems)
    """
    def __init__(self, cobra_model, solver='gurobi'):
        self.cobra_model = cobra_model
        self.sub_dict = {}
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
        self._INF = 1e6
        self.optcuts = set()      # Need to keep all cuts in case they are dropped at a node
        self.feascuts = set()
        self.UB = 1e15
        self.LB = -1e15
        self.verbosity = 0
        self.gaptol = 1e-4    # relative gap tolerance
        self.precision_sub = 'gurobi'
        self.print_iter = 20
        self.max_iter = 1000.

        self.model = self.init_model(ys, C, b, B, csenses_mp)

    def init_model(self, ys0, C, b, B, csenses):
        LB = -self._INF
        UB = self._INF

        ys0 = self._y0
        ny = len(ys0)

        fy = np.array([yj.objective_coefficient for yj in ys0])
        model = grb.Model('master')
        z = model.addVar(LB, UB, 0., GRB.CONTINUOUS, 'z')
        ys = [model.addVar(y.lower_bound, y.upper_bound, y.objective_coefficient,
            variable_kind_dict[y.variable_kind], y.id) for y in ys0]
        # Add Cy [<=>] b
        for i in range(C.shape[0]):
            csense= sense_dict[csenses[i]]
            bi    = b[i]
            cinds = C.indices[C.indptr[i]:C.indptr[i+1]]
            coefs = C.data[C.indptr[i]:C.indptr[i+1]]
            expr  = LinExpr(coefs, [ys[j] for j in cinds])
            cons  = model.addConstr(expr, csense, bi, name='MP_%s'%i)

        # Add z >= f'y + sum_k tk initially
        # Add z = f'y + sum_k wk*tk initially
        # and tk >= ...
        rhs = LinExpr(fy, ys)
        model.addConstr(z, GRB.EQUAL, rhs, 'z_cut')

        # Set objective: min z
        model.setObjective(z, GRB.MINIMIZE)

        self._z = z
        self._ys = ys
        self._fy = fy

        model._master = self  # Need this reference to access methods inside callback

        # model.Params.Presolve = 0          # To be safe, turn off
        model.Params.LazyConstraints = 1   # Required to use cbLazy
        model.Params.IntFeasTol = 1e-9

        model.update()
        self.model = model

        return model

    def add_submodels(self, sub_dict, weight_dict=None):
        """
        Add submodels.
        Inputs
        sub_dict : dict of BendersSubmodel objects
        """
        INF = self._INF

        zcut = self.model.getConstrByName('z_cut')
        for k,v in iteritems(sub_dict):
            self.sub_dict[k] = v
            # Also update/add tk variable and constraints
            tk_id = 'tk_%s'%k
            tk = self.model.getVarByName(tk_id)
            if tk is None:
                tk = self.model.addVar(-INF,INF,0.,GRB.CONTINUOUS,tk_id)
            # Update z cut: z = f'y + sum_k wk*tk
            # z - f'y - sum_k wk*tk = 0
            wk = v._weight
            self.model.chgCoeff(zcut, tk, -wk)

        self.model.update()


    def make_optcut(self, sub, cut_type='multi'):
        """
        z == f'y + sum_k tki,                           i \in OptimalityCuts
        (Above remains the same. Just append to tk)
        tki >= (dk-By)'wA_i,k + lk*wl_i,k - uk*wu_i,k,  i \in OptimalityCuts
        """
        INF = GRB.INFINITY
        # Can't add variable during callback--malloc
        #tk = self.model.addVar(-INF,INF,0.,GRB.CONTINUOUS,'tk')

        z = self._z
        fy = self._fy
        ys = self._ys
        d = sub._d
        B = sub._B
        xl = sub._xl
        xu = sub._xu
        m = len(d)
        n = len(xl)
        ny = len(ys)
        wa = np.array([sub.model.x_dict[w.VarName] for w in sub._wa])
        wl = np.array([sub.model.x_dict[w.VarName] for w in sub._wl])
        wu = np.array([sub.model.x_dict[w.VarName] for w in sub._wu])

        Bcsc = sub._Bcsc
        waB  = wa*Bcsc
        cinds = waB.nonzero()[0]
        tk = self.model.getVarByName('tk_%s'%sub._id)
        try:
            waBy = LinExpr([waB[j] for j in cinds], [ys[j] for j in cinds])
            optcut = tk >= sum(d*wa) - waBy + sum(xl*wl) - sum(xu*wu)
        except GurobiError as e:
            print('Caught GurobiError (%s) in make_optcut()'%repr(e))

        return optcut

    def make_feascut(self, sub):
        """
        Using unbounded ray information (Model.UnbdRay)
        """
        wa = np.array([sub.model.x_dict[w.VarName] for w in sub._wa])
        wl = np.array([sub.model.x_dict[w.VarName] for w in sub._wl])
        wu = np.array([sub.model.x_dict[w.VarName] for w in sub._wu])
        ys = self._ys
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

    def optimize(self, two_phase=False, cut_strategy='default', single_tree=True):
        """
        Optimize, possibly using various improvements.

        Inputs
        two_phase : phase 1 (LP relaxation), phase 2 (original, keeping P1 cuts)
        cut_strategy :
            None (default, one cut per subproblem),
            "mw" (Magnanti-Wong non-dominated),
            "maximal" (Sherali-Lunday cuts)
        single_tree : single search tree using solver callbacks (lazy constraints)
        """
        model = self.model
        if two_phase:
            self.solve_relaxed(cut_strategy)

        if single_tree:
            model.optimize(cb_benders_multi)

    def solve_relaxed(self, cut_strategy='default'):
        """
        Solve LP relaxation.
        """
        model = self.model
        OutputFlag = model.Params.OutputFlag
        ytypes = [y.VType for y in self._ys]
        model.Params.OutputFlag = 0
        for y in self._ys:
            y.VType = GRB.CONTINUOUS
        self.solve_loop(cut_strategy)

        # Make integer/binary again
        for y,yt in zip(self._ys,ytypes):
            y.VType = yt
        model.Params.OutputFlag = OutputFlag

    def solve_loop(self, cut_strategy='default'):
        """
        Solve without callbacks.
        Useful for solving LP relaxation.
        """
        model = self.model
        max_iter = self.max_iter
        print_iter = self.print_iter
        precision_sub = self.precision_sub
        cut_strategy = cut_strategy.lower()
        UB = 1e15
        LB = -1e15
        gap = UB-LB
        relgap = gap/(1e-10+abs(UB))
        bestUB = UB
        bestLB = LB
        gaptol = self.gaptol

        fy = self._fy
        ys = self._ys
        z = self._z
        yopt = np.zeros(len(ys))
        y0 = np.array([min(1.,y.UB) for y in ys])

        sub_dict = self.sub_dict

        tic = time.time()
        print("%12.10s%12.8s%12.8s%12.8s%12.8s%12.8s%12.10s%12.8s" % (
            'Iter','UB','LB','Best UB','Best LB','gap','relgap(%)','time(s)'))

        for _iter in range(max_iter):
            if np.mod(_iter, print_iter)==0:
                toc = time.time()-tic
                print("%12.10s%12.8s%12.8s%12.8s%12.8s%12.8s%12.10s%12.8s" % (
                    _iter,UB,LB,bestUB,bestLB,gap,relgap*100,toc))

            model.optimize()
            yprev = yopt
            yopt = np.array([y.X for y in ys])
            sub_objs = []
            opt_sub_inds = []

            # Update core point
            if cut_strategy=='mw':
                y0 = 0.5*(y0+yopt)
                y0[y0<model.Params.IntFeasTol] = 0.

            for sub_ind,sub in iteritems(sub_dict):
                sub.update_obj(yopt)
                sub.model.optimize(precision=precision_sub)
                if sub.model.Status==GRB.Status.UNBOUNDED:
                    feascut = self.make_feascut(sub)
                    model.addConstr(feascut)
                else:
                    sub_obj = sub._weight*sub.model.ObjVal
                    sub_objs.append(sub_obj)
                    opt_sub_inds.append(sub_ind)
                    if cut_strategy=='mw':
                        make_mw_cut(sub, y0)
                    elif cut_strategy=='maximal':
                        warnings.warn("Maximal cut not supported yet")

            LB = z.X
            UB = sum(fy*yopt) + sum(sub_objs)
            LB = LB if abs(LB)>1e-10 else 0.
            UB = UB if abs(UB)>1e-10 else 0.

            bestUB = min(bestUB,UB)
            bestLB = max(bestLB,LB)

            gap = UB-LB
            relgap = gap/(1e-10+abs(UB))

            if relgap < gaptol:
                toc = time.time()-tic
                print("%12.10s%12.8s%12.8s%12.8s%12.8s%12.8s%12.10s%12.8s" % (
                    _iter,UB,LB,bestUB,bestLB,gap,relgap*100,toc))
                print("relgap (%g) < gaptol (%g). Done!"%(relgap, gaptol))
                break
            else:
                for k,sub_ind in enumerate(opt_sub_inds):
                    tk = model.getVarByName('tk_%s'%sub_ind)
                    sub = sub_dict[sub_ind]
                    if tk.X < sub_objs[k]:
                        cut = self.make_optcut(sub)
                        model.addConstr(cut)

    def make_mw_cut(self, sub, y0):
        pass

    def make_maximal_cut(self, sub):
        pass





class BendersSubmodel(object):
    """
    Primal (Benders) decomposition submodel.
    """
    def __init__(self, cobra_model, _id, solver='gurobi', weight=1.):
        self._id = _id
        self.cobra_model = cobra_model
        self._weight = weight
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
        self._x0 = DictList(xs)
        self._y0 = DictList(ys)

        self.model = self.init_model(xs, A, d, csenses)

    def init_model(self, xs0, A, d, csenses):
        INF = GRB.INFINITY
        ZERO = 1e-15
        m = len(d)
        n = len(xs0)
        nx = n
        model = grb.Model('sub')
        model.Params.InfUnbdInfo = 1
        model.Params.OutputFlag = 0
        csenses = [sense_dict[c] for c in csenses]
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
        model = DecompModel(model)

        return model

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


class LagrangeSubmodel(object):
    """
    Lagrangean (dual) subproblem

    min  f'yk + c'xk + uk'*Hk*yk
    xk,yk
    s.t. Axk + Byk [<=>] d
               Cyk [<=>] b
         Dxk       [<=>] e

    where uk are Lagrange multipliers updated by Lagrangean master problem,
    and Hk constrain yk to be the same across all subproblems.
    """
    def __init__(self, cobra_model, _id, solver='gurobi', weight=1.):
        self._id = _id
        self.cobra_model = cobra_model
        self._weight = weight
        A,B,d,dsenses,xs,ys,C,b,bsenses = split_cobra(cobra_model)
        self._A = A
        self._Acsc = A.tocsc()
        self._B = B
        self._Bcsc = B.tocsc()
        self._d = d
        self._dsenses = dsenses
        self._C = C
        self._b = b
        self._bsenses = bsenses
        self._x0 = DictList(xs)
        self._y0 = DictList(ys)
        self._H = None

        # self.model = self.init_model(xs, A, d, csenses)
        self.model = self.init_model()

    def init_model(self):
        model = grb.Model('sub')
        model.Params.OutputFlag = 0
        # These constraints don't change
        A = self._A
        Acsc = self._Acsc
        B = self._B
        Bcsc = self._Bcsc
        d = self._d
        dsenses = self._dsenses
        C = self._C
        b = self._b
        bsenses = self._bsenses
        xs0 = self._x0
        ys0 = self._y0

        fy = np.array([yj.objective_coefficient for yj in ys0])
        ys = [model.addVar(y.lower_bound, y.upper_bound, y.objective_coefficient,
            variable_kind_dict[y.variable_kind], y.id) for y in ys0]
        cx = np.array([xj.objective_coefficient for xj in xs0])
        xs = [model.addVar(x.lower_bound, x.upper_bound, x.objective_coefficient,
            variable_kind_dict[x.variable_kind], x.id) for x in xs0]

        self._xs = xs
        self._ys = ys
        self._fy = fy
        self._cx = cx

        # Add Cy [<=>] b
        for i in range(C.shape[0]):
            bsense= bsenses[i]
            bi    = b[i]
            cinds = C.indices[C.indptr[i]:C.indptr[i+1]]
            coefs = C.data[C.indptr[i]:C.indptr[i+1]]
            expr  = LinExpr(coefs, [ys[j] for j in cinds])
            cons  = model.addConstr(expr, bsense, bi, name='Cy_%s'%i)
        # Add Axk + Byk [<=>] dk
        for i in range(A.shape[0]):
            # A*x
            cinds = A.indices[A.indptr[i]:A.indptr[i+1]]
            coefs = A.data[A.indptr[i]:A.indptr[i+1]]
            exprA = LinExpr(coefs, [xs[j] for j in cinds])
            # B*y
            cinds = B.indices[B.indptr[i]:B.indptr[i+1]]
            coefs = B.data[B.indptr[i]:B.indptr[i+1]]
            exprB = LinExpr(coefs, [ys[j] for j in cinds])
            # Ax + By [<=>] d
            di    = d[i]
            dsense= dsenses[i]
            cons = model.addConstr(exprA+exprB, dsense, di, name="AxBy_%s"%i)

        model.update()

        model.Params.IntFeasTol = 1e-9

        return model

    def update_obj(self, uk):
        """
        min  f'yk + c'xk + uk'*Hk*yk
        """
        ys = self._ys
        xs = self._xs
        fy = self._fy
        cx = self._cx
        Hk = self._H
        uH = uk*Hk

        model = self.model

        obj_fun = LinExpr(fy,ys) + LinExpr(cx,xs) + LinExpr(uH,ys)
        model.setObjective(obj_fun, GRB.MINIMIZE)
        model.update()

    def optimize(self, xk, yk):
        """
        Solve with warm-start
        """
        model = self.model
        xs = self._xs
        ys = self._ys

        for x,xopt in zip(xs,xk):
            x.Start = xopt

        for y,yopt in zip(ys,yk):
            y.Start = yopt

        model.optimize()


class LagrangeMaster(object):
    """
    Lagrangean (dual) master problem:

    max  z + delta/2 ||u - u*||2
    u,z,tk
    s.t  z  <= sum_k wk*tk
         tk <= f'yk + c'xk + u*Hk*yk,    forall k
         tk <= zpk* + u*H*y,            forall k

    Cross-decomposition:
    zpk* is the Benders subproblem objective
    """
    def __init__(self, cobra_model, solver='gurobi'):
        self.cobra_model = cobra_model
        self.sub_dict = {}
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
        self._INF = 1e3

        self.model = self.init_model(ys)

    def init_model(self, ys0):
        INF = self._INF

        model = grb.Model('master')
        z = model.addVar(-INF,INF,0.,GRB.CONTINUOUS,'z')
        self._z = z
        # Lagrange multiplier for each constraint making y same across conditions
        # z  <= sum_k wk*sk
        model.addConstr(z, GRB.LESS_EQUAL, 0., 'z_cut')
        self.model = model
        model.update()

        return model

    def update_obj(self, delta=1, u0=None):
        """
        #----------------------------------------------------
        # max  z - delta/2 ||u - u0||2
        #----------------------------------------------------
        """
        model = self.model
        z = self._z
        us = self._us
        if u0 is None:
            u0 = np.zeros(len(us))
        # Quadratic part: delta/2 * u^2
        unorm_quad = grb.QuadExpr()
        for u in us:
            unorm_quad.addTerms(delta/2., u, u)
        # Linear part(s): -delta * u0*u
        unorm_lin = LinExpr(-delta*u0, us)
        # Constant part: u0^2
        unorm_const = delta/2.*sum(u0*u0)
        obj_fun = z - unorm_lin - unorm_quad - unorm_const
        model.setObjective(obj_fun, GRB.MAXIMIZE)
        model.update()

        return obj_fun

    def add_submodels(self, sub_dict):
        """
        Add submodels.
        Inputs
        sub_dict : dict of LagrangeSubmodel objects
        """
        INF = self._INF
        zcut = self.model.getConstrByName('z_cut')
        for sub_ind,sub in iteritems(sub_dict):
            self.sub_dict[sub_ind] = sub
            # Also update tk variables and constraints
            tk_id = 'tk_%s'%sub_ind
            tk = self.model.getVarByName(tk_id)
            if tk is None:
                tk = self.model.addVar(-INF,INF,0.,GRB.CONTINUOUS,tk_id)
            # Update cut
            # z <= sum_k wk*tk
            # z - sum_k wk*tk <= 0
            wk = sub._weight
            self.model.chgCoeff(zcut, tk, -wk)

        # Update non-anticipativity constraints using ALL submodels
        nsub = len(self.sub_dict.keys())
        ny = len(self._y0)
        # Update us
        INF = self._INF
        us = []
        for j in range(ny):
            for k in range(nsub-1):
                vid = 'u_%d%d'%(j,k)
                ujk = self.model.getVarByName(vid)
                if ujk is None:
                    # Multiplier for an equality constraint
                    ujk = self.model.addVar(-INF,INF,0.,GRB.CONTINUOUS,vid)
                us.append(ujk)

        self._us = us

        # Each submodel's Hk: [ny*(K-1) x ny] matrix
        Hm = ny*(nsub-1)
        Hn = ny
        for k,(sub_ind,sub) in enumerate(iteritems(sub_dict)):
            if k < nsub-1:
                # Last set doesn't need self
                data_self = np.ones(ny).tolist()
                rows_self = np.arange(k*ny,(k+1)*ny).tolist() #range(ny) + k*ny
                cols_self = list(range(ny))
            else:
                data_self = []
                rows_self = []
                cols_self = []
            if k > 0:
                data_other= (-np.ones(ny)).tolist()
                rows_other= np.arange((k-1)*ny,k*ny).tolist() #range(ny) + (k-1)*ny
                cols_other= list(range(ny))
            else:
                data_other= []
                rows_other= []
                cols_other= []

            data = data_self + data_other
            rows = rows_self + rows_other
            cols = cols_self + cols_other

            Hk = coo_matrix((data, (rows, cols)), shape=(Hm,Hn)).tocsr()
            sub._H = Hk

        self.update_obj()

        self.model.update()

    def make_optcut(self, sub, xk, yk):
        """
        tk <= f'yk + c'xk + u*Hk*yk,    forall k
        Hk = [ny*(K-1) x ny] matrix
        """
        tk  = self.model.getVarByName('tk_%s'%sub._id)
        fy  = sub._fy
        cx  = sub._cx
        Hk  = sub._H
        yH  = Hk*yk
        us  = self._us
        try:
            optcut = tk <= sum(fy*yk) + sum(cx*xk) + LinExpr(yH,us)
        except GurobiError as e:
            print('Caught GurobiError (%s) in make_optcut()'%repr(e))

        return optcut

    def make_benderscut(self, sub, zpk, yopt):
        """
        tk <= zpk* + u*H*y,            forall k
        """
        tk  = self.model.getVarByName('tk_%s'%sub._id)
        Hk  = sub._H
        yH  = Hk*yopt
        us  = self._us
        try:
            cut = tk <= zpk + LinExpr(yH,us)
        except GurobiError as e:
            print('Caught GurobiError (%s) in make_benderscut()'%repr(e))

        return cut
