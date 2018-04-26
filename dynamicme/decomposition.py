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
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, identity
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
        master._absgaptol = 1e-6   # absolute gaptol
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
        #****************************************************
        # For now, update in case constraints/vars changed
        model.update()
        #****************************************************
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
            yopt = np.array([y.X for y in ys])
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

def split_cobra(model, first_stage_vars=None):
    """
    INPUTS
    model : cobra model

    Splits constraints into continuous and integer parts.
    Complicating constraints:       Ax + By [<=>] d
    First-stage-only constraints:        Cy [<=>] b
    Second-stage-only constraints:  Dx      [<=>] e
    """
    intvar_types = ['integer','binary']
    if first_stage_vars is None:
        xs = [x for x in model.reactions if x.variable_kind == 'continuous']
        ys = [x for x in model.reactions if x.variable_kind in intvar_types]
    else:
        ys = first_stage_vars
        xs = [x for x in model.reactions if x not in first_stage_vars]

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
            if rxn in ys:
                col_idx = ys.index(rxn)
                ydata.append(coeff)
                ycol_inds.append(col_idx)
                yrow_inds.append(row_idx)
            else:
                #if rxn.variable_kind == 'continuous':
                col_idx = xs.index(rxn)
                xdata.append(coeff)
                xcol_inds.append(col_idx)
                xrow_inds.append(row_idx)
            #elif rxn.variable_kind in intvar_types:
            # else:
            #     print("rxn.variable_kind must be in ['continuous','integer','binary']")
            #     raise Exception("rxn.variable_kind must be in ['continuous','integer','binary']")

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

    mets_b = [model.metabolites[i] for i in b_rows]
    mets_d = [model.metabolites[i] for i in d_rows]

    return Amix, Bmix, dmix, csenses_mix, xs, ys, C, b, bsenses, mets_d, mets_b



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
        A,B,d,csenses,xs,ys,C,b,csenses_mp,mets_d,mets_b = split_cobra(cobra_model)
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
        self._mets_d = mets_d
        self._mets_b = mets_b
        self._INF = 1e6
        self.optcuts = set()      # Need to keep all cuts in case they are dropped at a node
        self.feascuts = set()
        self.UB = 1e100
        self.LB = -1e100
        self.verbosity = 0
        self.gaptol = 1e-4    # relative gap tolerance
        self.absgaptol = 1e-6    # relative gap tolerance
        self.precision_sub = 'gurobi'
        self.print_iter = 20
        self.max_iter = 1000.
        self.y0 = None
        self.cut_strategy = 'default'
        self.int_sols = []
        self.nsol_keep = 5
        self.x_dict = {}
        self._iter = 0.

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

        model.Params.Presolve = 0          # To be safe, turn off
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
            v.master = self     # also register as its master
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

    def make_lagrange_cut(self, sub, uH, zLR):
        """
        Make Lagrange dual cut:
        f'y + tk + u'Hk*yk >= zLRk
        where tk >= submodel k objective
        """
        z = self._z
        fy = self._fy
        ys = self._ys
        tk = self.model.getVarByName('tk_%s'%sub._id)
        try:
            cut = LinExpr(fy,ys) + tk + LinExpr(uH,ys) >= zLR
        except GurobiError as e:
            print('Caught GurobiError (%s) in make_optcut()'%repr(e))

        return cut



    def make_feascut_from_primal(self, sub):
        """
        Make feascut using FarkasDual of primal.
        """
        cut = None

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
        self.cut_strategy = cut_strategy
        if two_phase:
            self.solve_relaxed(cut_strategy)
            # Reset umaxs
            for sub in self.sub_dict.values():
                sub.maximal_dict['u']=None

        if single_tree:
            model.optimize(cb_benders_multi)
        else:
            self.solve_loop(cut_strategy)

        # Sometimes heuristic finds incumbent s.t. previous constraints
        # that is within gap without chance to exclude that incumbent
        # with cuts based on integer solution

        yopt = np.array([y.X for y in self._ys])
        return yopt

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
        OutputFlag = model.Params.OutputFlag
        model.Params.OutputFlag = 0
        max_iter = self.max_iter
        print_iter = self.print_iter
        precision_sub = self.precision_sub
        cut_strategy = cut_strategy.lower()
        UB = 1e100
        LB = -1e100
        gap = UB-LB
        relgap = gap/(1e-10+abs(UB))
        bestUB = UB
        bestLB = LB
        gaptol = self.gaptol
        absgaptol = self.absgaptol
        verbosity = self.verbosity

        fy = self._fy
        ys = self._ys
        z = self._z
        yopt = None #np.zeros(len(ys))
        ybest = yopt
        # y0 = np.array([min(1.,y.UB) for y in ys])
        y0 = self.y0

        sub_dict = self.sub_dict

        tic = time.time()
        if cut_strategy=='maximal':
            print("%12.10s%12.10s%12.10s%12.10s%12.10s%12.10s%12.10s%12.8s" % (
                'Iter','UB','LB','Best UB','umax','gap','relgap(%)','time(s)'))
        else:
            print("%12.10s%12.10s%12.10s%12.10s%12.10s%12.10s%12.10s%12.8s" % (
                'Iter','UB','LB','Best UB','Best LB','gap','relgap(%)','time(s)'))

        for _iter in range(max_iter):
            if np.mod(_iter, print_iter)==0:
                toc = time.time()-tic
                if cut_strategy=='maximal':
                    umax = max([sub.maximal_dict['u'] for sub in sub_dict.values()])
                    umaxf = umax if umax is not None else np.inf
                    print("%12.10s%12.6g%12.6g%12.6g%12.6g%12.6g%12.6g%12.8s" % (
                        _iter,UB,LB,bestUB,umaxf,gap,relgap*100,toc))
                else:
                    print("%12.10s%12.6g%12.6g%12.6g%12.6g%12.6g%12.6g%12.8s" % (
                        _iter,UB,LB,bestUB,bestLB,gap,relgap*100,toc))

            model.optimize()
            yprev = yopt
            yopt = np.array([y.X for y in ys])
            self.int_sols.append(yopt)
            if len(self.int_sols)>self.nsol_keep:
                self.int_sols.pop(0)
            #************************************************
            # DEBUG TODO: hmmm....
            # No need to round. Subgradient of LP relaxation is still
            # a subgradient of integer problem.
            # yopt = np.array([np.round(y.X) for y in ys])
            #************************************************
            sub_objs = []
            opt_sub_inds = []

            # Update core point
            if cut_strategy in ['mw','maximal']:
                if y0 is None:
                    if yprev is None:
                        pass
                    else:
                        # Wait till we have at least two int sols
                        if sum(abs(yprev-yopt))>1e-9:
                            y0 = 0.5*(yprev+yopt)
                            y0[y0<model.Params.IntFeasTol] = 0.
                else:
                    y0 = self.update_corepoint(yopt, y0)
                    y0[y0<model.Params.IntFeasTol] = 0.

            for sub_ind,sub in iteritems(sub_dict):
                if cut_strategy=='proximal':
                    sub.update_proximal_obj(yopt)
                if cut_strategy=='maximal' and y0 is not None:
                    sub.update_maximal_obj(yopt,y0,_iter,absgaptol)
                else:
                    sub.update_obj(yopt)
                sub.model.optimize(precision=precision_sub)
                #********************************************
                # Elaborate procedure to repair MW cuts
                if sub.model.Status == GRB.Status.INFEASIBLE:
                    if verbosity>0:
                        print("Submodel %s is infeasible! Attempting fix."%sub_ind)
                    if cut_strategy=='mw':
                        # Try to get a normal optimality cut instead.
                        # Alternatively, might have been nearly unbounded.
                        cons = sub.model.getConstrByName('fixobjval')
                        if verbosity>1:
                            print("fixobjval: %s %s %s"%(
                                sub.model.getRow(cons), cons.Sense, cons.RHS))
                        if cons is not None:
                            # Drastic:
                            sub.model.model.remove(cons)
                            sub.model.model.update()
                            sub.model.optimize(precision=precision_sub)
                            if verbosity>0:
                                if sub.model.Status==GRB.Status.INFEASIBLE:
                                    print("****************************")
                                    print("Could not make submodel %s feasible"%sub_ind)
                                else:
                                    print("Submodel without mw constraint: %s (%s)"%(
                                        status_dict[sub.model.Status],sub.model.Status))
                #********************************************
                if sub.model.Status==GRB.Status.UNBOUNDED:
                    feascut = self.make_feascut(sub)
                    model.addConstr(feascut)
                elif sub.model.Status==GRB.Status.OPTIMAL:
                    sub_obj = sub._weight*sub.model.ObjVal
                    sub_objs.append(sub_obj)
                    opt_sub_inds.append(sub_ind)
                    if cut_strategy=='mw':
                        if y0 is not None:
                            self.solve_mw_cut(sub, y0)

            LB = z.X
            UB = sum(fy*yopt) + sum(sub_objs)
            # LB = LB if abs(LB)>1e-10 else 0.
            # UB = UB if abs(UB)>1e-10 else 0.

            if UB < bestUB:
                bestUB = UB
                ybest = yopt
                self.x_dict = {v.VarName:v.X for v in model.getVars()}
                # for y,yb in zip(ys,ybest):
                #     y.Start = yb

            bestLB = LB
            gap = bestUB-LB
            relgap = gap/(1e-10+abs(bestUB))

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

        # Reset outputflag
        model.Params.OutputFlag = OutputFlag

        return ybest

    def update_corepoint(self, yopt, y0):
        """
        Update core point
        """
        if y0 is not None:
            y0 = 0.5*(y0 + yopt)

        return y0

    def solve_mw_cut(self, sub, y0):
        """
        Add or update constraint to fix objective:
        max  u'(d-B*yc)
        s.t. u'(d-B*yopt) = ObjVal(yopt)
        """
        cons = sub.model.model.getConstrByName('fixobjval')
        if cons is None:
            expr = sub.model.model.getObjective() == sub.model.ObjVal
            cons = sub.model.model.addConstr(expr, name='fixobjval')
        else:
            cons.RHS = sub.model.ObjVal
            cons.Sense = GRB.EQUAL
            # Reset all coeffs
            row = sub.model.getRow(cons)
            for j in range(row.size()):
                v = row.getVar(j)
                sub.model.chgCoeff(cons, v, 0.)
            # Update actual coefficients
            obj = sub.model.getObjective()
            for j in range(obj.size()):
                v = obj.getVar(j)
                sub.model.chgCoeff(cons, v, obj.getCoeff(j))
        # Change objective function to core point
        sub.update_obj(y0)
        sub.model.update()
        sub.model.optimize()
        if sub.model.Status==GRB.Status.INFEASIBLE:
            print("Submodel %s Infeasible after adding mw constraint"%sub._id)
            raise Exception("Submodel %s Infeasible after adding mw constraint"%sub._id)
        # Relax the constraint for next iteration
        cons.Sense = GRB.GREATER_EQUAL
        cons.RHS   = -GRB.INFINITY
        sub.model.update()



class BendersSubmodel(object):
    """
    Primal (Benders) decomposition submodel.
    """
    def __init__(self, cobra_model, _id, solver='gurobi', weight=1.):
        self._id = _id
        self.cobra_model = cobra_model
        self._weight = weight
        self.master = None
        A,B,d,csenses,xs,ys,C,b,csenses_mp,mets_d,mets_b = split_cobra(cobra_model)
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
        self._mets_d = mets_d
        self._mets_b = mets_b
        self.maximal_dict = {'u':None, 'L':2., 'M':1.}

        self.model = self.init_model(xs, A, d, csenses)
        self.primal = None
        # Proximal bundle
        self.D_prox = None
        self.wa0 = None
        self.wl0 = None
        self.wu0 = None
        self.delta = 1.
        self.delta_min = 1e-20
        self.delta_mult = 0.5
        self.bundle_mult = 0.9
        self.bundle_tol = 1e-6

    def init_model(self, xs0, A, d, csenses):
        INF = GRB.INFINITY
        ZERO = 1e-15
        m = len(d)
        n = len(xs0)
        nx = n
        model = grb.Model('dual')
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

        dBy = d - B*yopt
        cinds = dBy.nonzero()[0]
        try:
            dBywa = LinExpr([dBy[j] for j in cinds], [wa[j] for j in cinds])
            obj = dBywa + LinExpr(xl,wl) - LinExpr(xu,wu)
            model.setObjective(obj, GRB.MAXIMIZE)
            model.update()
        except GurobiError as e:
            print('Caught GurobiError (%s) in update_subobj(yopt=%s)'%(repr(e),yopt))

    def solve_proximal(self,wa0,wl0,wu0):
        """
        Just solution to last proximal point.
        """
        zbest = self.master.LB

        return zbest

    def update_proximal_point(self):
        """
        Determine if proximal point changes based on predicted ascent.
        """
        model = self.model
        # Model solved?
        if model.ObjVal is not None:
            wa0 = self.wa0
            wl0 = self.wl0
            wu0 = self.wu0

            if self.wa0 is None:
                wa0 = np.zeros(len(self._wa))
                self.wa0 = wa0
            if self.wl0 is None:
                wl0 = np.zeros(len(self._wl))
                self.wl0 = wl0
            if self.wu0 is None:
                wu0 = np.zeros(len(self._wu))
                self.wu0 = wu0

            if self.D_prox is None:
                self.D_prox = self.solve_proximal(wa0,wl0,wu0)

            D_prox = self.D_prox

            z = self.master._z
            if hasattr(z,'X'):
                LB = z.X
            else:
                LB = -1e100

            D_new = LB
            UB = model.ObjVal

            pred_ascent = UB - D_prox
            ascent_tol = self.bundle_tol
            if abs(pred_ascent) < ascent_tol:
                converged = True
            else:
                converged = False
                if D_new >= (D_prox + self.bundle_mult*pred_ascent):
                    wak = np.array([w.X for w in self._wa])
                    wlk = np.array([w.X for w in self._wl])
                    wuk = np.array([w.X for w in self._wu])

                    wa0 = wak
                    wl0 = wlk
                    wu0 = wuk
                    # Compute D_prox at updated prox point
                    D_prox = self.solve_proximal(wa0,wl0,wu0)
                else:
                    # Null step
                    pass

            self.D_prox = D_prox
            delta = self.delta
            delta = max(self.delta_min, self.delta_mult*delta)
            self.delta = delta

        else:
            wa0 = np.zeros(len(self._wa))
            wl0 = np.zeros(len(self._wl))
            wu0 = np.zeros(len(self._wu))
            self.wa0 = wa0
            self.wl0 = wl0
            self.wu0 = wu0
            delta = self.delta


        return wa0,wl0,wu0,delta


    def update_proximal_obj(self, yopt):
        """
        Proximal bundle
        max  w'(b-B*yopt) + u*w'(b-B*ycore) + xl'wl - xu'wu - delta/2 ||w-w0||^2

        """
        model = self.model
        d = self._d
        B = self._B
        wa = self._wa
        wl = self._wl
        wu = self._wu
        xl = self._xl
        xu = self._xu

        wa0, wl0, wu0, delta = self.update_proximal_point()

        dBy = d - B*yopt
        cinds = dBy.nonzero()[0]
        try:
            # Quadratic part: delta/2 * w^2
            wa_quad = grb.QuadExpr()
            wl_quad = grb.QuadExpr()
            wu_quad = grb.QuadExpr()
            for w in wa:
                wa_quad.addTerms(delta/2.,w,w)
            for w in wl:
                wl_quad.addTerms(delta/2.,w,w)
            for w in wu:
                wu_quad.addTerms(delta/2.,w,w)
            # Linear part(s): -delta * w0*u
            wa_lin = LinExpr(-delta*wa0, wa)
            wl_lin = LinExpr(-delta*wl0, wl)
            wu_lin = LinExpr(-delta*wu0, wu)
            # Constant part: w0^2
            wa_const = delta/2.*sum(wa0*wa0)
            wl_const = delta/2.*sum(wl0*wl0)
            wu_const = delta/2.*sum(wu0*wu0)

            dBywa = LinExpr([dBy[j] for j in cinds], [wa[j] for j in cinds])
            obj = dBywa + LinExpr(xl,wl) - LinExpr(xu,wu) - \
                    wa_lin - wl_lin - wu_lin - \
                    wa_quad - wl_quad - wu_quad - \
                    wa_const - wl_const - wu_const
            model.setObjective(obj, GRB.MAXIMIZE)
            model.update()
        except GurobiError as e:
            print('Caught GurobiError (%s) in update_subobj(yopt=%s)'%(repr(e),yopt))


    def update_maximal_obj(self, yopt, ycore, _iter, absgaptol=1e-6):
        """
        Maximal cuts by Sherali and Lunday (2013) Ann Oper Res, 210:57-72.
        Weight update scheme by Oliveira et al. (2014) Comput Oper Res, 49:47-58.

        For MIP:
        min  c'x + f'y
        s.t. Ax + By = b
             Cy = d
             xl <= x <= xu
             y integer

        Solve the dual subproblem:
        #----------------------------------------------------
        max  w'(b-B*yopt) + u*w'(b-B*ycore) + xl'wl - xu'wu
        s.t. w'A = c
        #----------------------------------------------------

        Critically, u must be small enough that
        UB - LB <= absgaptol at optimality
        i.e.,
        w'(b-B*yopt) + l'wl - u'wu  + u*w'(b-B*ycore) - z <= absgaptol

        Additionally,
        update weights (to guarantee convergence):
        sum_k=1,...,inf uk --> inf and
        uk --> 0 as k --> inf
        E.g., u{k+1} = alpha{k+1}*u{k}
        where alpha{k+1} = l/k,
        where l=2, k is the iteration number,
        and u{0} = absgaptol/(M*theta),
        where theta=absgaptol + max{0,max{boi}} - min{0,min{boi}},
        with bo = b-B*yopt,
        and M is the penalty for infeasibility

        Note dual of modified dual subproblem is
        min  c'x
        s.t. Ax = b-B*yopt + u*(b-B*ycore)
        """
        model = self.model
        d = self._d
        B = self._B
        wa = self._wa
        wl = self._wl
        wu = self._wu
        xl = self._xl
        xu = self._xu
        maximal_dict = self.maximal_dict
        umax = maximal_dict['u']
        L = maximal_dict['L']
        M = maximal_dict['M']

        dBy = d - B*yopt
        dByc = d - B*ycore
        #----------------------------------------------------
        # Update weight
        if _iter<=0 or umax is None:
            maxboi = max(dByc)
            minboi = min(dByc)
            theta = absgaptol + max(0,maxboi)-min(0,minboi)
            #theta = max(0,maxboi)-min(0,minboi)
            umax = absgaptol/theta
            #umax = min(umax, absgaptol)
        else:
            # alpha = L/max(1.,_iter)
            alpha = 1/np.log10(max(10.,_iter))
            umax = alpha*umax

        self.maximal_dict['u'] = umax
        #----------------------------------------------------
        u_dByc = umax*dByc
        cinds = dBy.nonzero()[0]
        cinds_c = u_dByc.nonzero()[0]
        try:
            dBywa = LinExpr([dBy[j] for j in cinds], [wa[j] for j in cinds])
            dByc_wa = LinExpr([u_dByc[j] for j in cinds_c], [wa[j] for j in cinds_c])
            obj = dBywa + dByc_wa + LinExpr(xl,wl) - LinExpr(xu,wu)

            model.setObjective(obj, GRB.MAXIMIZE)
            model.update()
        except GurobiError as e:
            print('Caught GurobiError (%s) in update_subobj(yopt=%s)'%(repr(e),yopt))


    def make_update_primal(self, yopt, cx=None, obj_sense='minimize'):
        """
        Make or update primal problem.
        min  c'x + f'y
        s.t. Ax = d-By
             l<=x<=u
        """
        csenses = self._csenses
        A = self._A
        B = self._B
        d = self._d
        mets_d = self._mets_d
        xs0 = self._x0  # DictList of cobra vars
        ys0 = self._y0

        if self.primal is None:
            xl= np.array([x.lower_bound for x in xs0])
            xu= np.array([x.upper_bound for x in xs0])
            if cx is None:
                cx= np.array([x.objective_coefficient for x in xs0])

            model = grb.Model('primal')
            model.Params.InfUnbdInfo=1
            model.Params.OutputFlag=0

            csenses = [sense_dict[c] for c in csenses]
            xs = [model.addVar(x.lower_bound, x.upper_bound, x.objective_coefficient,
                variable_kind_dict[x.variable_kind], x.id) for x in xs0]

            # Ax = d-By
            By = B*yopt
            Am = A.shape[0]
            for i in range(Am):
                met    = mets_d[i]
                csense = csenses[i]
                di     = d[i]
                #------------------------------------------------
                cinds  = A.indices[A.indptr[i]:A.indptr[i+1]]
                coefs  = A.data[A.indptr[i]:A.indptr[i+1]]
                lhs    = LinExpr(coefs, [xs[j] for j in cinds])
                rhs    = di-By[i]
                #------------------------------------------------
                cons   = model.addConstr(lhs, csense, rhs, name=met.id)

            # Objective: min(max) c'x
            obj = LinExpr(cx, xs)
            model.setObjective(obj, objective_senses[obj_sense])
            model.update()

            self.primal = model
        else:
            model = self.primal
            # Update: Ax = d-B*yopt
            By = B*yopt
            for i,cons in enumerate(model.getConstrs()):
                cons.RHS = d[i]-By[i]
            # Updated objective provided for some reason?
            if cx is not None:
                xs  = model.getVars()
                obj = LinExpr(cx, xs)
                model.setObjective(obj, objective_senses[obj_sense])

            model.update()

        return model


    def repair_infeas(self, master, yopt):
        """
        Happens if primal is unbounded.
        But, at minimum we have box constraints on primal.
        Therefore, this case only arises when solver mistakes unbounded
        for infeasible.
        If this happens, attempt to fix the situation by
        solving the sub primal and using FarkasDual to get the unbounded ray of dual.
        """
        primal = self.make_update_primal(yopt)
        feascut = master.make_feascut_from_primal(self)

        return feascut


class LagrangeSubmodel(object):
    """
    Lagrangean (dual) subproblem

    min  f'yk + c'xk + u'*Hk*yk
    xk,yk
    s.t. Axk + Byk [<=>] d
               Cyk [<=>] b
         Dxk       [<=>] e

    where uk are Lagrange multipliers updated by Lagrangean master problem,
    and Hk constrain yk to be the same across all subproblems.
    """
    def __init__(self, cobra_model, _id, first_stage_vars=None, solver='gurobi', weight=1.,
            Q=None, bisection=False):
        """
        """
        self._id = _id
        self.cobra_model = cobra_model
        self.solver = solver
        self._weight = weight
        A,B,d,dsenses,xs,ys,C,b,bsenses,mets_d,mets_b = split_cobra(cobra_model, first_stage_vars)
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
        self._mets_d = mets_d
        self._mets_b = mets_b
        self._H = None
        self._Q  = Q
        self.yopt = None
        self.x_dict={}
        self.ObjVal = None
        self.max_alt = None
        self.bisection = bisection
        self.bisect_lb = None
        self.bisect_ub = None
        self.bisect_abs_precision = 0.01
        self.bisect_rel_precision = None
        self.bisect_max_iter = 1e3
        self.bisect_verbosity = 0
        self.subs_bounds = None
        self.subs_constraints = None

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
        min  f'yk + c'xk + uk'*Hk*yk + 1/2 x'*Q*x
        """
        ys = self._ys
        xs = self._xs
        fy = self._fy
        cx = self._cx
        Hk = self._H
        Q = self._Q

        uH = uk*Hk

        model = self.model
        obj_lin = LinExpr(fy,ys) + LinExpr(cx,xs) + LinExpr(uH,ys)
        if Q is None:
            obj_fun = obj_lin
        else:
            var_dict = {i:v for i,v in enumerate(model.getVars())}
            obj_quad = grb.QuadExpr()
            for (ind0,ind1),val in Q.todok().items():
                x0 = var_dict[ind0]
                x1 = var_dict[ind1]
                obj_quad.addTerms(val/2., x0, x1)
            obj_fun = obj_lin + obj_quad

        model.setObjective(obj_fun, GRB.MINIMIZE)
        model.update()

    def optimize(self, xk=None, yk=None, bisection=None):
        """
        Solve with warm-start.
        Some solvers do this automatically.
        """
        if bisection is None:
            bisection = self.bisection
        solver = self.solver
        model = self.model
        xs = self._xs
        ys = self._ys

        if xk is not None:
            # Start only used for integer/binary variables no?
            for x,xopt in zip(xs,xk):
                x.Start = xopt

        if yk is not None:
            for y,yopt in zip(ys,yk):
                y.Start = yopt

        if bisection:
            objval = self.solve_bisection(
                    self.bisect_abs_precision, self.bisect_lb, self.bisect_ub,
                    rel_precision=self.bisect_rel_precision,
                    max_iter=self.bisect_max_iter,
                    verbosity=self.bisect_verbosity)
        else:
            model.optimize()
            if model.SolCount>0:
                objval = model.ObjVal
                self.yopt = np.array([y.X for y in ys])
                self.x_dict = {v.VarName:v.X for v in model.getVars()}
                self.ObjVal = objval
            else:
                objval = np.nan

        return objval

    def solve_bisection(self, abs_precision, a, b, minimize=True, max_iter=1e3,
            rel_precision=None, verbosity=0):
        """
        Solve using bisection

        Inputs
        bisect_var : which variable to bisect.
        abs_precision : final precision = b-a
        rel_precision : final relative precision = |b-a|/(1e-10 + |b|)
        a : starting lower bound
        b : starting upper bound
        minimize : bisect to minimize, if False, maximize.
        """
        ybest = None
        xd_best = None
        best_obj = np.nan

        mid = (b-a)/2.
        abs_prec = b-a
        rel_prec = abs(b-a) / (1e-10 + abs(b))
        if rel_precision is not None:
            converged_rel = rel_prec <= rel_precision
        else:
            converged_rel = False
        if abs_precision is not None:
            converged_abs = abs_prec <= abs_precision
        else:
            converged_abs = False
        converged = converged_abs or converged_rel

        _iter = 0
        tic = time.time()
        #----------------------------------------------------
        if verbosity>0:
            print("%8s%12s%12s%12s%12s%12s%12s%12s" % (
                'Iter','point','a_new','b_new','gap','relgap(%)','objval','Time(s)'))
            print("%8s%12s%12s%12s%12s%12s%12s%12s" % (
                '-'*7,'-'*11,'-'*11,'-'*11,'-'*11,'-'*11,'-'*11,'-'*11))

        while not converged and _iter < max_iter:
            _iter += 1
            mid = (b+a)/2.
            self.substitute_mu(mid)
            objval = self.optimize(bisection=False)

            if np.isnan(objval):
                if minimize:
                    a = mid
                else:
                    b = mid
            else:
                # Save best sol
                best_obj = objval
                ybest = self.yopt
                xd_best = self.x_dict

                if minimize:
                    b = mid
                else:
                    a = mid
            # Update interval
            rel_prec = abs(b-a) / (1e-10 + abs(b))
            abs_prec = b-a
            # Check convergence
            if rel_precision is not None:
                converged_rel = rel_prec <= rel_precision
            else:
                converged_rel = False
            if abs_precision is not None:
                converged_abs = abs_prec <= abs_precision
            else:
                converged_abs = False
            converged = converged_abs or converged_rel
            toc = time.time()-tic
            #------------------------------------------------
            if verbosity>0:
                print(
                "%8.6s%12.4g%12.4g%12.4g%12.4g%12.4g%12.4g%10.8s" % (
                _iter,mid,a,b,abs_prec,rel_prec,objval,toc))


        self.yopt = ybest
        self.x_dict = xd_best
        self.ObjVal = best_obj

        return best_obj

    def substitute_mu(self, mu_val):
        """
        Substitute value in for symbolic variable
        subs_bounds : list of (rxn,lb_expr,ub_expr)
            - lb_expr is any function where lb_expr(mu_k) = lb_k. Same for ub_expr.
        subs_constraints : list of (met,rxn,expr)
        """
        subs_bounds = self.subs_bounds
        if subs_bounds is not None:
            for (rxn,lb,ub) in subs_bounds:
                if lb is not None:
                    self.update_bound(rxn, lb=lb(mu_val))
                if ub is not None:
                    self.update_bound(rxn, ub=ub(mu_val))

        subs_constraints = self.subs_constraints
        if subs_constraints is not None:
            for (met,rxn,expr) in subs_constraints:
                self.update_coeff(met,rxn,expr(mu_val))

    def update_bound(self, rxn_id, lb=None, ub=None):
        """
        Update bounds in problem
        """
        model = self.model
        v = model.getVarByName(rxn_id)
        if v is None:
            raise KeyError("variable %s not found!"%rxn_id)
        if lb is not None:
            v.LB = lb
        if ub is not None:
            v.UB = ub


    def compile_substitution(self, subs_bounds=None, subs_constraints=None):
        """
        Compile substitutable bounds and/or constraint coefficients.
        """



class LagrangeMaster(object):
    """
    Given problem

    min  sum_k fk'y + sum_k ck'xk
    xk,y
    s.t. Ak*xk + Bk*y [<=>] dk
         lk <= xk <= uk
         y integer

    Reformulate for k=1,...,K
    min  sum_k fk'yk + sum_k ck'xk
    xk,yk
    s.t. Ak*xk + Bk*yk [<=>] dk
         yk - y1 = 0,    k=2,..,K   (asymmetric non-anticipativity constraints)
         lk <= xk <= uk
         yk integer

    Lagrange relaxation
    min  sum_k fk'yk + sum_k ck'xk + sum_k u'*Hk*yk
    xk,yk
    s.t. Ak*xk + Bk*yk [<=>] dk
         lk <= xk <= uk
         sum_k uk = 0
         yk integer

    If yk all binary, special Hk available.

    Lagrangean (dual) master problem:

    max  z - delta/2 ||u - u*||2
    u,z,tk
    s.t  z  <= sum_k wk*tk
         tk <= fk'yk + c'xk + u*Hk*yk,    forall k=1,..,K   (supergradient cut)
         sum_k uk = 0

         If Cross-decomposition add cut:
         tk <= zpk* + u*Hk*y,            forall k
         zpk* is the Benders subproblem objective

    In general, the solution to Lagrangean dual must be converted to a
    primal feasible solution. Heuristics are commonly used.
    Can also use cross-decomposition.

    If candidate feasible solution is infeasible, add integer cut to
    exclude it from all subproblems.

    The feasible solution gives a valid upper bound.

    #--------------------------------------------------------
    # Derivation of supergradient cuts
    L(x,y,u) = f'y  + c'x  + u'H*y
    D(u)  = inf L(x,y,u)
          = inf f'y + c'x + u'H*y
    D(u)  = L(x^,y^,u)
         <= L(x0,y0,u)             # Since D(u) is infimum, L at any other u0 greater or equal
          = f'y0 + c'x0 + u'H*y0
          = f'y0 + c'x0 + u0'H*y0 + u'H*y0 - u0'H*y0
          = D(u0) + (H*y0)'(u-u0)
    Thus, D(u) - D(u0) <= (H*y0)'(u-u0).
    Therefore, H*y0 is a supergradient of D(.) at u0.
    """
    def __init__(self, cobra_model, first_stage_vars=None, solver='gurobi', method='ld'):
        """
        Input
        method : 'lr' Lagrangean relaxation, 'ld' Lagrangean decomposition
        """
        self.cobra_model = cobra_model
        self.sub_dict = {}
        self.primal_dict={}
        A,B,d,csenses,xs,ys,C,b,csenses_mp,mets_d,mets_b = split_cobra(
                cobra_model,first_stage_vars)
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
        self._mets_d = mets_d
        self._mets_b = mets_b
        self._us = None # Initialized with add_submodels
        self._z = None
        self._INF = 1e3
        self.verbosity = 1
        self.gaptol = 1e-4    # relative gap tolerance
        self.absgaptol = 1e-6    # relative gap tolerance
        self.feastol = 1e-6
        self.penaltytol = 1e-6
        self.precision_sub = 'gurobi'
        self.print_iter = 10
        self.time_limit = 1e30  # seconds
        self.max_iter = 1e6
        self.delta_min = 1e-10
        self.delta_mult = 0.5
        self.bundle_mult = 0.5
        self.max_max_alt = 200
        self.best_dict = {}
        self.uk = None
        self.x_dict = {}
        self.yopt = None

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
        model.Params.OutputFlag = 0
        model.update()

        return model

    def update_obj(self, delta=1., u0=None):
        """
        #----------------------------------------------------
        # max  z - delta/2 ||u - u0||2
        # max  z - delta/2*(u^2 - 2*u*u0 + u0^2)
        # max  z - delta/2*u^2 + delta*u*u0 - delta/2*u0^2
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

    def add_submodels(self, sub_dict, antic_method='asymmetric'):
        """
        Add submodels.
        Inputs
        sub_dict : dict of LagrangeSubmodel objects
        antic_method : nonanticipativity constraint method.
                       Ignored if determined that all y are binary--uses tighter formulation.
            'extra': xk-x = 0
            'stagger': x1=x2, ... xk-1=xk
            'asymmetric':x1=x2, x1=x3, ..., x1=xk   # Reported to be the best (Oliveira, 2013)
        """
        INF = self._INF
        model = self.model
        zcut = model.getConstrByName('z_cut')
        for sub_ind,sub in iteritems(sub_dict):
            self.sub_dict[sub_ind] = sub
            # Also update tk variables and constraints
            tk_id = 'tk_%s'%sub_ind
            tk = model.getVarByName(tk_id)
            if tk is None:
                tk = model.addVar(-INF,INF,0.,GRB.CONTINUOUS,tk_id)
            # Update cut
            # z <= sum_k wk*tk
            # z - sum_k wk*tk <= 0
            wk = sub._weight
            model.chgCoeff(zcut, tk, -wk)

        # Update non-anticipativity constraints using ALL submodels
        nsub = len(self.sub_dict.keys())
        ny = len(self._y0)
        # Check if all binary
        are_binary = [y.upper_bound<=1 for y in self._y0]
        is_binary = np.all(are_binary)

        # Update us
        INF = self._INF

        if is_binary:
            us = [model.addVar(-INF,INF,0.,GRB.CONTINUOUS,'u_0%d'%j) for j in range(ny)]
        else:
            us = []
            nsub_max = nsub
            if antic_method in ['stagger','asymmetric']:
                nsub_max=nsub-1
            for k in range(nsub_max):
                for j in range(ny):
                    vid = 'u_%d%d'%(k,j)
                    ukj = model.getVarByName(vid)
                    if ukj is None:
                        # Multiplier for an equality constraint
                        ukj = model.addVar(-INF,INF,0.,GRB.CONTINUOUS,vid)
                    us.append(ukj)

        self._us = us

        # if y all binary, H1 = 1-p1*eye(n). H_{k>1}=-pk*eye(n)
        # else each submodel's Hk: [ny*(K-1) x ny] matrix
        weights = [sub._weight for sub in sub_dict.values()]
        w_tot = sum(weights)

        if is_binary:
            for k,(sub_ind,sub) in enumerate(iteritems(sub_dict)):
                p = sub._weight / w_tot
                if k==0:
                    Hk = (1-p)*identity(ny)
                else:
                    Hk = -p*identity(ny)
                sub._H = Hk
        else:
            Hm = ny*nsub_max
            Hn = ny
            if antic_method=='extra':
                # Just identity matrix
                for k,(sub_ind,sub) in enumerate(iteritems(sub_dict)):
                    data = np.ones(ny)
                    rows = np.arange(k*ny,(k+1)*ny)
                    cols = range(ny)
                    Hk = coo_matrix((data,(rows,cols)), shape=(Hm,Hn)).tocsr()
                    sub._H = Hk

            elif antic_method=='stagger':
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

            elif antic_method=='asymmetric':
                for k,(sub_ind,sub) in enumerate(iteritems(sub_dict)):
                    if k==0:
                        data = np.ones(ny*nsub_max)
                        rows = range(ny*nsub_max)
                        cols = [ii for i in [range(ny) for kk in range(nsub_max)] for ii in i]
                    else:
                        data= -np.ones(ny)
                        rows= np.arange((k-1)*ny,k*ny)
                        cols= range(ny)

                    Hk = coo_matrix((data, (rows, cols)), shape=(Hm,Hn)).tocsr()
                    sub._H = Hk


        self.update_obj()

        self.model.update()

    def make_supercut(self, sub_dict):
        """
        Make supergradient cut for subproblem k:

        z <= sum_k fk'yk + sum_k ck'xk + u*sum_k Hk*yk

        Use stored solution for best LB
        """
        z = self._z
        us = self._us
        try:
            fys = [sum(sub._fy*np.array([sub.x_dict[v.VarName] for v in sub._ys]))
                    for sub in sub_dict.values()]
            cxs = [sum(sub._cx*np.array([sub.x_dict[v.VarName] for v in sub._xs]))
                    for sub in sub_dict.values()]
            yHs = [sub._H*np.array([sub.x_dict[v.VarName] for v in sub._ys])
                    for sub in sub_dict.values()]
            cut = z <= sum(fys) + sum(cxs) + LinExpr(sum(yHs),us)
        except GurobiError as e:
            print('Caught GurobiError (%s) in make_supercut()'%repr(e))

        return cut

    def make_multicut(self, sub):
        """
        Make supergradient cut for subproblem k:

        tk <= f'yk + c'xk + 1/2 x'*Q*x + u*Hk*yk
        Hk = [ny*(K-1) x ny] matrix
        """
        tk  = self.model.getVarByName('tk_%s'%sub._id)
        fy  = sub._fy
        cx  = sub._cx
        Hk  = sub._H
        Q   = sub._Q
        xk = np.array([sub.x_dict[v.VarName] for v in sub._xs])
        yk = np.array([sub.x_dict[v.VarName] for v in sub._ys])

        yH  = Hk*yk
        us  = self._us
        if Q is None:
            quadTerm = 0.
        else:
            vk = np.array([sub.x_dict[v.VarName] for v in sub.model.getVars()])
            quadTerm = 0.5*np.dot((Q*vk), vk)
        try:
            cut = tk <= sum(fy*yk) + sum(cx*xk) + LinExpr(yH,us) + quadTerm
        except GurobiError as e:
            print('Caught GurobiError (%s) in make_supercut()'%repr(e))

        return cut

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

    def solve_lagrangian(self, uk):
        """
        Solve Lagrangian given Lagrange multipliers
        """
        sub_dict = self.sub_dict
        sub_objs = []
        for sub_ind,sub in iteritems(sub_dict):
            sub.update_obj(uk)
            #********************************************
            obj = sub.optimize(yk=sub.yopt) * sub._weight
            #********************************************
            sub_objs.append(obj)
            if np.isnan(obj):
                # if sub.model.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
                if verbosity>0:
                    #print("Relaxed subproblem %s is Infeasible or Unbounded. Status=%s (%s)."%(
                    #    sub_ind, status_dict[sub.model.Status], sub.model.Status))
                    print("Relaxed subproblem %s has objval=%s."%(sub_ind, obj))
                return None

        objval = sum(sub_objs)
        return objval


    def optimize(self, feasible_methods=['best_rounding','enumerate'], bundle=False,
            multicut=True, max_alt=10, alt_method='pool', nogood_cuts=False,
            early_heuristics=[], strictUB=False):
        """
        x_dict = optimize()

        Solution procedure
        Init Lagrange multipliers, u
        1) Update u in Lagrange relaxed subproblems
        2) Solve relaxed subproblems: LB
        3) Construct feasible solution for complicating variables
        4) Solve primal subproblems using feasible solution: UB
        5) if gap = UB-LB < gaptol stop, else continue
        6) Solve Lagrange dual problem: u(k+1). Goto 1)

        Inputs
        feasible_methods :methods to construct feasible solution
                          for complicating (integer) variables.
                          - best_rounding : binary (or golden section) search rounding
                          - average : simple rounding
                          - enumerate : enumerate alternate optima
                          - benders : Benders' decomposition, making overall
                            procedure a cross-decomposition.
        bundle : if True, only update uk-->wk+1 if
                            D(wk+1) >= D(uk) + a*(D(wk+1)-D(uk))
                            where a\in (0,1) is an Armijo-like parameter.
                            Else, wk+1 = uk
        max_alt : maximum alternate optima to pool or search for when
                  duality gap meets threshold
        alt_method : Use solver's pool feature (pool) or with int cuts (intcuts)

        Outputs
        x_dict : {x.VarName:x.X}
        """
        model = self.model
        sub_dict = self.sub_dict
        max_iter = self.max_iter
        print_iter = self.print_iter
        time_limit = self.time_limit
        gaptol = self.gaptol
        absgaptol = self.absgaptol
        feastol = self.feastol
        penaltytol = self.penaltytol
        verbosity = self.verbosity
        delta_min = self.delta_min
        delta_mult= self.delta_mult
        delta = 1.
        bundle_mult = self.bundle_mult
        D_prox = None    # proximal point objval

        if feasible_methods is None:
            feasible_methods = []

        z = self._z
        us = self._us
        nu = len(us)

        feasUB = 1e100
        dualUB = 1e100  # Overestimator from master, with supergradient-based polyhedral approx
                        # Should decrease monotonically
        UB = 1e100      # UB from feasible solution. Not necessarily monotonic, depending
                        # method used to construct feasible solution.
        LB = -1e100     # Underestimator from dual relaxed subproblems
        bestLB = LB
        bestUB = UB     # Given by best feasible solution, not the master problem
        if strictUB:
            gap = feasUB-bestLB
        else:
            gap = bestUB-bestLB
        if strictUB:
            relgap = gap/(1e-10+abs(feasUB))
        else:
            relgap = gap/(1e-10+abs(bestUB))
        uk = np.zeros(nu)
        self.uk = uk
        u0 = np.zeros(nu)   # Trust region center
        ubest = uk
        x_dict = {}
        self.log_rows = []

        status_dict[GRB.SUBOPTIMAL] = 'suboptimal'
        status_dict[GRB.NUMERIC] = 'numeric'

        tic = time.time()
        print("%8.6s%22s%22s%10.8s%10.9s%10.8s%30s" % (
            'Iter','UB','LB','gap','relgap(%)','penalty','Time(s)'))
        print("%8.6s%22s%22s%10.8s%10.9s%10.8s%30s" % (
            '-'*7,'-'*19,'-'*19,'-'*9,'-'*9,'-'*9,'-'*29))
        print("%8.6s%11.8s%11.8s%11.8s%11.8s%10.8s%10.8s%10.8s%10.8s%10.8s%10.8s" % (
            '','Best','Feasible','Sub','Best','','','','total','master','sub'))

        # Don't pool solutions until gap below tolerance
        for sub in sub_dict.values():
            sub.model.Params.PoolSearchMode=0
            sub.model.Params.PoolSolutions =1

        for _iter in range(max_iter):
            #----------------------------------------------------
            # Solve Master
            #----------------------------------------------------
            tic_master = time.time()
            uk = self.solve_master(delta, u0)
            self.uk = uk
            toc_master = time.time()-tic_master
            if model.Status != GRB.OPTIMAL:
                warnings.warn("Solution available but Master solver status=%s (%s)."%(
                    status_dict[model.Status], model.Status))
                if verbosity>1:
                    print("Master solver status=%s (%s)."%(
                        status_dict[model.Status], model.Status))
            if uk is None:
                raise Exception("Master solver status=%s (%s). Aborting."%(
                    status_dict[model.Status], model.Status))

            # self.update_obj(delta, u0)
            # model.optimize()
            # toc_master = time.time()-tic_master
            # #if model.Status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
            # if model.SolCount > 0:
            #     uk = np.array([u.X for u in self._us])
            #     if model.Status != GRB.OPTIMAL:
            #         warnings.warn("Solution available but Master solver status=%s (%s)."%(
            #             status_dict[model.Status], model.Status))
            #         if verbosity>1:
            #             print("Master solver status=%s (%s)."%(
            #                 status_dict[model.Status], model.Status))
            # else:
            #     raise Exception("Master solver status=%s (%s). Aborting."%(
            #         status_dict[model.Status], model.Status))

            #----------------------------------------------------
            # Solve Subproblems
            #----------------------------------------------------
            tic_sub = time.time()
            sub_results = self.solve_subproblems(uk)
            sub_objs = []
            for sub_result in sub_results:
                sub_ind = sub_result['id']
                obj = sub_result['obj']
                if np.isnan(obj):
                    if verbosity>0:
                        print("Relaxed subproblem %s has objval=%s."%(sub_ind, obj))
                    return None
                else:
                    cut = sub_result['cut']
                    model.addConstr(cut)
                    sub_objs.append(obj)

#             sub_objs = []
#             for sub_ind,sub in iteritems(sub_dict):
#                 sub.update_obj(uk)
#                 #********************************************
#                 obj = sub.optimize(yk=sub.yopt) * sub._weight
#                 #********************************************
#                 sub_objs.append(obj)
#                 if not np.isnan(obj):
#                     # sub.model.Status == GRB.OPTIMAL:
#                     if multicut:
#                         cut = self.make_multicut(sub)
#                         model.addConstr(cut)
#                 elif np.isnan(obj):
#                     #elif sub.model.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
#                     # If RELAXED subproblem is INFEASIBLE, ORIGINAL problem was INFEASIBLE.
#                     # raise Exception(
#                     #         "Relaxed subproblem is Infeasible or Unbounded. Status=%s (%s)."%(
#                     #     status_dict[sub.model.Status], sub.model.Status))
#                     if verbosity>0:
#                         #print("Relaxed subproblem %s is Infeasible or Unbounded. Status=%s (%s)."%(
#                         #    sub_ind, status_dict[sub.model.Status], sub.model.Status))
#                         print("Relaxed subproblem %s has objval=%s."%(
#                             sub_ind, obj))
#                     return None

            toc_sub = time.time()-tic_sub

            if not multicut:
                cut = self.make_supercut(sub_dict)
                model.addConstr(cut)

            #----------------------------------------------------
            # Compute UB
            UB = z.X        # Lagrange dual UB
            dualUB = UB
            # if UB < bestUB:
            #     bestUB = UB
            bestUB = UB     # Since proximal penalty might bias it
            #----------------------------------------------------
            # Update bounds and check convergence
            # Lagrange multipliers were chosen to find the max lb
            LB = sum(sub_objs)
            if LB > bestLB:
                bestLB = LB
                ubest  = uk     # Record best multipliers that improved LB
                self.uopt = ubest
                self.x_dict = {v.VarName:v.X for v in model.getVars()}
            #----------------------------------------------------
            # Can use Heuristics before desired gap reached
            if early_heuristics:
                for heuristic in early_heuristics:
                    yfeas, feasUBk, is_optimal = self.feasible_heuristics(heuristic)
                    if yfeas is not None:
                        if feasUBk < bestUB:
                            bestUB = feasUBk
                        if feasUBk < feasUB:
                            feasUB = feasUBk
                            ybest = yfeas
                            self.yopt = ybest
                            for j,yj in enumerate(ybest):
                                x_dict[sub._ys[j].VarName] = yj
                            if verbosity>1:
                                print("Best Heuristic solution has objval=%s"%(feasUB))
                            if strictUB:
                                gap = abs(feasUB-bestLB)
                            else:
                                gap = abs(bestUB-bestLB)
                            if abs(gap)<absgaptol:
                                gap = 0.
                            if strictUB:
                                relgap = gap/(1e-10+abs(feasUB))
                            else:
                                relgap = gap/(1e-10+abs(bestUB))
                            if relgap <= gaptol:
                                #--------------------------------------------
                                # Final QC that sum_k Hk*yk = 0
                                Hys = []
                                for sub_ind,sub in iteritems(sub_dict):
                                    yk = np.array([sub.x_dict[x.VarName] for x in sub._ys])
                                    Hyk = sub._H*yk
                                    Hys.append(Hyk)
                                if verbosity>2:
                                    print("sum_k Hk*yk: %s"%(sum(Hys)))
                                    if sum([abs(x) for x in sum(Hys)]) > feastol:
                                        print("WARNING: Non-anticipativity constraints not satisfied.")
                                #--------------------------------------------
                                # Final Log
                                toc = time.time()-tic
                                res_u = delta/2.*sum((uk-u0)**2)    # total penalty term
                                self.log_rows.append({'iter':_iter,'bestUB':bestUB,'feasUB':feasUB,
                                    'LB':LB,'bestLB':bestLB,'gap':gap,'relgap':relgap*100,'delta':delta,
                                    'res_u':res_u,'t_total':toc,'t_master':toc_master,'t_sub':toc_sub})
                                break


            # Quadratic stabilization term can cause "best" LB and UB to change
            # non monotonically, not guaranteeing bestUB > bestLB
            if strictUB:
                gap = abs(feasUB-bestLB)
            else:
                gap = abs(bestUB-bestLB)
            if abs(gap)<absgaptol:
                gap = 0.
            if strictUB:
                relgap = gap/(1e-10+abs(feasUB))
            else:
                relgap = gap/(1e-10+abs(bestUB))
            res_u = delta/2.*sum((uk-u0)**2)    # total penalty term
            #------------------------------------------------
            # Log progress
            toc = time.time()-tic
            self.log_rows.append({'iter':_iter,'bestUB':bestUB,'feasUB':feasUB,
                'LB':LB,'bestLB':bestLB,'gap':gap,'relgap':relgap*100,'delta':delta,
                'res_u':res_u,'t_total':toc,'t_master':toc_master,'t_sub':toc_sub})
            #------------------------------------------------

            #------------------------------------------------
            # Save best sol so far
            self.best_dict = {'feasUB':feasUB, 'bestUB':bestUB, 'bestLB':bestLB,
                    'gap':gap, 'relgap':relgap}

            #------------------------------------------------
            # Check convergence
            if relgap <= gaptol and abs(res_u)<penaltytol:
                #--------------------------------------------
                toc = time.time()-tic
                if verbosity>0:
                    print(
                    "%8.6s%11.4g%11.4g%11.4g%11.4g%10.4g%10.4g%10.3g%10.8s%10.8s%10.8s" % (
                    _iter,bestUB,feasUB,LB,bestLB,gap,relgap*100,res_u,toc,toc_master,toc_sub))
                #--------------------------------------------
                # Now, look for a feasible solution
                # Spend user-defined portion of total time on different strategies,
                # including heuristics, B&B, and enumeration.
                #--------------------------------------------
                if feasible_methods:
                    for method in feasible_methods:
                        if method=='enumerate':
                            yfeas, feasUBk, is_optimal = self.enum_alt_opt(first_feasible=True,
                                    max_alt=max_alt, method=alt_method, nogood_cuts=nogood_cuts)
                        else:
                            yfeas, feasUBk, is_optimal = self.feasible_heuristics(method)
                        if yfeas is not None:
                            if feasUBk < feasUB:
                                feasUB = feasUBk
                                ybest = yfeas
                                self.yopt = ybest
                                for j,yj in enumerate(ybest):
                                    x_dict[sub._ys[j].VarName] = yj
                                if verbosity>1:
                                    print("Best Heuristic solution has objval=%s"%(feasUB))

                            if is_optimal:
                                #--------------------------------------------
                                # Final QC that sum_k Hk*yk = 0
                                Hys = []
                                for sub_ind,sub in iteritems(sub_dict):
                                    yk = np.array([sub.x_dict[x.VarName] for x in sub._ys])
                                    Hyk = sub._H*yk
                                    Hys.append(Hyk)
                                if verbosity>1:
                                    print("sum_k Hk*yk: %s"%(sum(Hys)))
                                    if sum([abs(x) for x in sum(Hys)]) > feastol:
                                        print("WARNING: Non-anticipativity constraints not satisfied.")
                                #--------------------------------------------
                                # Final Log
                                toc = time.time()-tic
                                self.log_rows.append({'iter':_iter,'bestUB':bestUB,'feasUB':feasUB,
                                    'LB':LB,'bestLB':bestLB,'gap':gap,'relgap':relgap*100,'delta':delta,
                                    'res_u':res_u,'t_total':toc,'t_master':toc_master,'t_sub':toc_sub})
                                break
                        else:
                            if verbosity>0:
                                print("Feasible solution not among alt opt. Trying next iteration.")
                    #--------------------------------------------
                    # END: Recover primal feasible solution.
                    #--------------------------------------------
                else:
                    break

            D_new  = LB
            u0, delta, D_prox, pred_ascent = self.update_proximal(uk, u0, delta, D_new, D_prox)

            if verbosity>2:
                print("pred_ascent=%s. Updating D_prox from %s to %s."%(
                    pred_ascent,D_prox,D_new))

            ascent_tol = self.feastol
            if abs(pred_ascent) < ascent_tol:
                print("Master ascent of %s < %s  below threshold. Stopping."%(
                    pred_ascent, ascent_tol))
                toc = time.time()-tic
                # Final Log
                self.log_rows.append({'iter':_iter,'bestUB':bestUB,'feasUB':feasUB,
                    'LB':LB,'bestLB':bestLB,'gap':gap,'relgap':relgap*100,'delta':delta,
                    'res_u':res_u,'t_total':toc,'t_master':toc_master,'t_sub':toc_sub})
                if verbosity>0:
                    print(
                    "%8.6s%11.4g%11.4g%11.4g%11.4g%10.4g%10.4g%10.3g%10.8s%10.8s%10.8s" % (
                    _iter,bestUB,feasUB,LB,bestLB,gap,relgap*100,res_u,toc,toc_master,toc_sub))
                break
            #------------------------------------------------

            if np.mod(_iter, print_iter)==0:
                toc = time.time()-tic
                if verbosity>0:
                    print(
                    "%8.6s%11.4g%11.4g%11.4g%11.4g%10.4g%10.4g%10.3g%10.8s%10.8s%10.8s" % (
                    _iter,bestUB,feasUB,LB,bestLB,gap,relgap*100,res_u,toc,toc_master,toc_sub))
            # Check timelimit
            toc = time.time()-tic
            if toc > time_limit:
                if verbosity>0:
                    print("Stopping due to time limit of %g seconds."%time_limit)
                break
            else:
                # Update time limits on all (sub)models
                time_left = time_limit-toc
                model.Params.TimeLimit = time_left
                for sub in sub_dict.values():
                    sub.model.Params.TimeLimit = time_left

        return x_dict

    def restore_feasible(self):
        """
        Restore primal feasible solution.
        """

    def solve_relaxed(self,*args,**kwargs):
        """
        Solve LP relaxation.
        """
        # Should not try to make integer feasible since LP relaxation
        if not kwargs.has_key('feasible_methods'):
            kwargs['feasible_methods'] = None

        model = self.model
        sub_dict = self.sub_dict
        xtype_dict = {sub_ind:[x.VType for x in sub._xs] for sub_ind,sub in iteritems(sub_dict)}
        ytype_dict = {sub_ind:[y.VType for y in sub._ys] for sub_ind,sub in iteritems(sub_dict)}
        for sub in sub_dict.values():
            for y in sub._ys:
                y.VType = GRB.CONTINUOUS
            for x in sub._xs:
                x.VType = GRB.CONTINUOUS
        x_dict = self.optimize(*args,**kwargs)

        # Make integer/binary again
        for sub_ind,sub in iteritems(sub_dict):
            for y,yt in zip(sub._ys,ytype_dict[sub_ind]):
                y.VType = yt
            for x,xt in zip(sub._xs,xtype_dict[sub_ind]):
                x.VType = xt

        return x_dict

    def add_int_cut(self, sub, yk):
        """
        Add int cut to submodel.
        For binary variables:
        sum_{yk=1} (1-y) + sum_{yk=0} yk >= 1
        -sum_{yk=1} yk + sum_{yk=0} yk >= 1 - n1

        For general integers?:
        sum_{yk>0} (yk - y) + sum_{yk==0} yk >= 1
        """
        model = sub.model
        n1 = 0.
        ys = sub._ys
        coeffs = np.zeros(len(ys))
        for j,y in enumerate(yk):
            if y >= 0.5:
                coeffs[j] = -1.
                n1 += y
            else:
                coeffs[j] = 1.
        lhs = LinExpr(coeffs, ys)
        rhs = 1-n1
        cut = model.addConstr(lhs, GRB.GREATER_EQUAL, rhs, name='intcut')

        return cut

    def enum_alt_opt(self, first_feasible=True, max_alt=10, method='pool', nogood_cuts=False):
        """
        Enumerate solutions by adding integer cuts while keeping objective fixed.

        Inputs
        first_feasible : stop as soon as feasible solution found
        """
        sub_dict = self.sub_dict
        yfeas = None
        is_optimal = False
        feasUBk = None
        stat_dict = {}
        opt_obj = sum([sub.ObjVal for sub in sub_dict.values()])
        feas_objs = []
        feas_sols = []

        if method=='pool':
            # Use solver feature to pool n best sols
            for sub_ind,sub in iteritems(sub_dict):
                if sub.max_alt is None:
                    sub.max_alt = max_alt
                sub.model.Params.PoolSearchMode=2   # Keep n best sols
                sub.model.Params.PoolSolutions = max(1,sub.max_alt)
                sub.optimize()
                bestobj = sub.ObjVal

                if sub.model.SolCount>0 and hasattr(sub.model,'PoolObjVal'):
                    alt_sols = []
                    for k in range(sub.model.SolCount):
                        sub.model.Params.SolutionNumber=k
                        objk = sub.model.PoolObjVal
                        # Only keep if alt optimal for this subproblem
                        if abs(objk-bestobj)/(1e-10+abs(bestobj))<=self.gaptol:
                            yk = np.array([y.Xn for y in sub._ys])
                            alt_sols.append(yk)
                        else:
                            # Ordered by worsening obj so can break at first subopt obj
                            break
                    #----------------------------------------
                    # Run this after collecting all solutions since it modifies the problem
                    alt_objs = []
                    for k,yk in enumerate(alt_sols):
                        obj_dict, feas_dict, sub_stats = self.check_feasible(yk)
                        are_feas = feas_dict.values()
                        tot_obj = sum(obj_dict.values())
                        alt_objs.append(tot_obj)
                        gap = abs(tot_obj-opt_obj)
                        relgap = abs(tot_obj-opt_obj)/(1e-10+abs(opt_obj))

                        is_optimal = (gap <= self.absgaptol) or (relgap <= self.gaptol)
                        stats = np.array(sub_stats.values())
                        stat_dict[sub_ind] = stats

                        if np.all(are_feas):
                            feas_objs.append(tot_obj)
                            feas_sols.append(yk)
                            if is_optimal:
                                feasUBk = tot_obj
                                yfeas   = yk
                                # Want to stop enumerating immediately
                                if self.verbosity > 1:
                                    print("Best obj=%g. Tot obj=%g. obj=%g for %s"%(
                                        opt_obj, tot_obj, sub.ObjVal,sub_ind))
                                    print("Alt optimum %d in subprob %s is feasible! Done."%(
                                        k, sub_ind))
                                return yfeas, feasUBk, is_optimal

                    # Integer (no-good) cuts exclude alt opt infeasible points
                    if nogood_cuts:
                        for yk in alt_sols:
                            self.add_int_cut(sub, yk)
                        if self.verbosity > 1:
                            print("Excluded %d infeasible alt optima from sub %s"%(
                                len(alt_sols), sub_ind))

                    # Adpatively expand or shrink number of sols to keep for this sub
                    if len(alt_sols)==sub.max_alt:
                        sub.max_alt = min(2*sub.max_alt, self.max_max_alt)
                    else:
                        sub.max_alt = max(1,len(alt_sols))

                else:
                    if self.verbosity>1:
                        print("No pool solution available for %s"%sub_ind)
                    break
            #----------------------------------------
            # Save best feasible so far
            if len(feas_objs)>0:
                feasUBk = min(feas_objs)
                yfeas   = feas_sols[feas_objs.index(feasUBk)]
                if self.verbosity > 1:
                    print("Best feasible solution among %d alt feasible has objval=%s"%(
                        len(feas_sols), feasUBk))
            else:
                if self.verbosity>1:
                    print("No feasible solution enumerated.")
        else:
            fixobj_id = 'fixobjval'
            if self.verbosity>1:
                print("Finding feasible solution with optimal obj=%s"%opt_obj)

            for sub_ind,sub in iteritems(sub_dict):
                #------------------------------------------------
                # Check if feasible and still optimal
                y0 = sub.yopt
                obj_dict, feas_dict, sub_stats = self.check_feasible(y0)
                are_feas = feas_dict.values()
                tot_obj = sum(obj_dict.values())
                is_optimal = abs(tot_obj-opt_obj)/(1e-10+abs(opt_obj)) <= self.gaptol
                stats = np.array(sub_stats.values())
                stat_dict[sub_ind] = stats

                #if is_optimal and np.all(stats==GRB.OPTIMAL):
                if is_optimal and np.all(are_feas):
                    yfeas = y0
                    if self.verbosity > 1:
                        print("Tot obj=%g. obj=%g for %s"%(
                            tot_obj, sub.ObjVal,sub_ind))
                        print("Found feasible alt opt before int cuts. Stopping.")
                    break
                else:
                    # Constrain this submodel's objective
                    model = sub.model
                    cons = model.getConstrByName(fixobj_id)
                    if cons is None:
                        expr = model.getObjective() == sub.ObjVal
                        cons = model.addConstr(expr, name=fixobj_id)
                    else:
                        cons.RHS = sub.ObjVal
                        cons.Sense = GRB.EQUAL
                        # Reset all coeffs
                        row = model.getRow(cons)
                        for j in range(row.size()):
                            v = row.getVar(j)
                            model.chgCoeff(cons, v, 0.)
                        # Update to actual coeffs
                        obj = model.getObjective()
                        for j in range(obj.size()):
                            v = obj.getVar(j)
                            model.chgCoeff(cons, v, obj.getCoeff(j))

                    #------------------------------------------------
                    # Enumerate alt optima
                    n_alt = 0
                    intcuts = []
                    for icut in range(sub.max_alt):
                        # Add int cut
                        intcut = self.add_int_cut(sub, [v.X for v in sub._ys])
                        intcuts.append(intcut)
                        model.update()
                        # Solve and check if feasible
                        sub.optimize()
                        is_feasible = sub.model.SolCount>0
                        if not is_feasible:
                            break
                        else:
                            n_alt += 1
                            y0 = sub.yopt
                            obj_dict, feas_dict, sub_stats = self.check_feasible(y0)
                            tot_obj = sum(obj_dict.values())
                            is_optimal = abs(tot_obj-opt_obj)/(1e-10+abs(opt_obj)) <= self.gaptol
                            #if is_optimal and np.all(stats==GRB.OPTIMAL):
                            are_feas = feas_dict.values()
                            if is_optimal and np.all(are_feas):
                                yfeas = y0
                                if self.verbosity > 1:
                                    print("Alt optima %d is feasible! Stopping."%n_alt)
                                    # print("Found feasible alt opt. Stopping.")
                                break
                    if n_alt == sub.max_alt:
                        sub.max_alt = sub.max_alt*2
                    else:
                        sub.max_alt = n_alt
                    #------------------------------------------------
                    # Relax all the constraints for next outer iter
                    cons = model.getConstrByName(fixobj_id)
                    cons.Sense = GRB.LESS_EQUAL
                    cons.RHS = GRB.INFINITY
                    # model.remove(cons)
                    if self.verbosity>1:
                        print("Tot obj=%g. Found %d alt optima with obj=%g for %s"%(
                            tot_obj, n_alt, sub.ObjVal,sub_ind))
                    for intcut in intcuts:
                        model.remove(intcut)
                    model.update()

        return yfeas, feasUBk, is_optimal


    def feasible_heuristics(self, heuristic='average'):
        """
        Lagrangian heuristics for feasible primal recovery.
        """
        yfeas = None
        feasUB = None
        is_optimal = False

        sub_dict = self.sub_dict
        opt_obj = sum([sub.ObjVal for sub in sub_dict.values()])

        if heuristic=='average':
            ymat = np.array([[sub._weight*sub.x_dict[v.VarName] for v in sub._ys] for 
                sub in sub_dict.values()])
            #------------------------------------------------
            # Only need to round if not continuous
            # y0 = ymat.mean(axis=0).round()
            ym = ymat.mean(axis=0)
            y0 = np.array(
                    [ymi if y.VType==GRB.CONTINUOUS else ymi.round() for y,ymi in zip(sub._ys,ym)])
            #------------------------------------------------
            obj_dict, feas_dict, sub_stats = self.check_feasible(y0)

            are_feas = feas_dict.values()
            tot_obj = sum(obj_dict.values())
            gap = abs(tot_obj-opt_obj)
            relgap = abs(tot_obj-opt_obj)/(1e-10+abs(opt_obj))
            is_optimal = gap <= self.absgaptol or relgap <= self.gaptol

            if np.all(are_feas):
                yfeas = y0
                feasUB = tot_obj

        elif heuristic=='best_rounding':
            """
            Binary search to find the best rounding threshold
            """
            ymat = np.array([[sub._weight*sub.x_dict[v.VarName] for v in sub._ys] for 
                sub in sub_dict.values()])
            ym = ymat.mean(axis=0)
            #------------------------------------------------
            # Binary search to find best rounding threshold
            a = 0.
            b = 1.
            precision = 0.1
            dist  = b-a
            tot_obj = 1e100
            max_iter = 1e3

            for _iter in range(max_iter):
                mid = (a+b)/2.
                x1  = (a+mid)/2.
                x2  = (mid+b)/2.
                # Only round if not continuous
                y1 = np.array(
                    [ymi if y.VType==GRB.CONTINUOUS else binary_thresh(ymi,x1)
                        for y,ymi in zip(sub._ys,ym)])
                y2 = np.array(
                    [ymi if y.VType==GRB.CONTINUOUS else binary_thresh(ymi,x2)
                        for y,ymi in zip(sub._ys,ym)])
                # Fix the set covering constraint if necessary...
                obj_dict1, feas_dict1, sub_stats1 = self.check_feasible(y1)
                obj_dict2, feas_dict2, sub_stats2 = self.check_feasible(y2)
                obj1 = sum(obj_dict1.values())
                obj2 = sum(obj_dict2.values())
                # Ensure feasible
                if not all(feas_dict1.values()) or not all(feas_dict2.values()):
                    # Then need to set more binaries to 1
                    b = mid
                else:
                    if obj1 < obj2:
                        b = mid
                        yfeas = y1
                        feasUB = obj1
                        tot_obj = obj1
                    else:
                        a = mid
                        yfeas = y2
                        feasUB = obj2
                        tot_obj = obj2

                dist = b-a
                if dist < precision:
                    break

            gap = abs(tot_obj-opt_obj)
            relgap = abs(tot_obj-opt_obj)/(1e-10+abs(opt_obj))
            is_optimal = gap <= self.absgaptol or relgap <= self.gaptol
            #------------------------------------------------
        elif heuristic=='parsimonious':
            """
            Especially useful for min sum y objective
            """
            #y0 = np.array([1. if v.Obj==0 else 0. for v in self._ys])
            ptol = 1e-9
            y0 = np.array([1. if v.objective_coefficient<=ptol else 0. for v in self._y0])
            obj_dict, feas_dict, sub_stats = self.check_feasible(y0)

            are_feas = feas_dict.values()
            tot_obj = sum(obj_dict.values())
            gap = abs(tot_obj-opt_obj)
            relgap = abs(tot_obj-opt_obj)/(1e-10+abs(opt_obj))
            is_optimal = gap <= self.absgaptol or relgap <= self.gaptol

            if np.all(are_feas):
                yfeas = y0
                feasUB = tot_obj

        else:
            print("Unknown heuristic: %s"%heuristic)

        return yfeas, feasUB, is_optimal


    def make_feasible(self, feasible_method='enumerate', alt_method='pool', max_alt=100,
            nogood_cuts=False):
        """
        Make (and check) feasible solution using various methods.
        """
        sub_dict = self.sub_dict
        # Compute UB from feasible 
        if feasible_method=='enumerate':
            # If there exists at least one primal feasible,
            # enumerate alternative optimal dual solutions until found.
            y0, feasUBk, is_optimal = self.enum_alt_opt(max_alt=max_alt, method=alt_method,
                    nogood_cuts=nogood_cuts)
        elif feasible_method=='best':
            subobj_dict = {}
            for sub_ind,sub in iteritems(sub_dict):
                y0 = sub.yopt
                obj_dict, feas_dict, sub_stats = self.check_feasible(y0)
                subobj_dict[sub_ind] = sum(obj_dict.values())

            minobj = min(subobj_dict.values())
            feasUB = minobj
            for sub_ind,sub in iteritems(sub_dict):
                if subobj_dict[sub_ind]==minobj:
                    y0 = sub.yopt
        elif feasible_method=='first':
            y0 = sub_dict[sub_dict.keys()[0]].yopt
            obj_dict, feas_dict, sub_stats = self.check_feasible(y0)
            feasUB = sum(obj_dict.values())
        elif feasible_method=='average':
            ymat = np.array([[v.X for v in sub._ys] for sub in sub_dict.values()])
            y0 = ymat.mean(axis=0).round()
            obj_dict, feas_dict, sub_stats = self.check_feasible(y0)
            feasUB = sum(obj_dict.values())
        elif feasible_method=='best_rounding':
            """
            Binary search to find the best rounding threshold
            """
            ymat = np.array([[v.X for v in sub._ys] for sub in sub_dict.values()])
            yr = ymat.mean(axis=0)
            #------------------------------------------------
            # Binary search to find best rounding threshold
            a = 0.
            b = 1.
            precision = 0.05
            feasUB = 1e100
            gap  = b-a
            y0 = yr.round()

            while gap > precision:
                mid = (a+b)/2.
                x1  = (a+mid)/2.
                x2  = (mid+b)/2.
                y1  = yr.copy()
                y1[yr<x1] = 0.
                y1[yr>=x1] = 1.
                y2  = yr.copy()
                y2[yr<x2] = 0.
                y2[yr>=x2] = 1.
                # Fix the set covering constraint if necessary...
                obj_dict1, feas_dict1, sub_stats1 = self.check_feasible(y1)
                obj_dict2, feas_dict2, sub_stats2 = self.check_feasible(y2)
                obj1 = sum(obj_dict1.values())
                obj2 = sum(obj_dict2.values())
                # Ensure feasible
                if not all(feas_dict1.values()) or not all(feas_dict2.values()):
                    # Then need to set more binaries to 1
                    b = mid
                else:
                    if obj1 < obj2:
                        b = mid
                        y0 = y1
                        feasUB = obj1
                    else:
                        a = mid
                        y0 = y2
                        feasUB = obj2

                gap = b-a
            #------------------------------------------------

        elif feasible_method is None:
            y0 = sub_dict[sub_dict.keys()[0]].yopt
            feasUB = UB
        else:
            raise Exception("Unknown feasibility restoring method: %s"%feasible_method)

        return y0

    def check_feasible(self, y0):
        """
        Check if y0 feasible for all subproblems.
        """
        sub_dict = self.sub_dict
        precision_sub = self.precision_sub
        var_dict = {}     # record orig bounds for later
        obj_dict   = {}
        stat_dict= {}
        feas_dict= {}
        for sub_ind,sub in iteritems(sub_dict):
            var_dict[sub_ind] = []
            for j,y in enumerate(sub._ys):
                var_dict[sub_ind].append({'LB':y.LB,'UB':y.UB,'VType':y.VType})
                y.LB = y0[j]
                y.UB = y0[j]
                y.VType = GRB.CONTINUOUS    # Since fixed
            sub.optimize()
            stat_dict[sub_ind] = sub.model.Status
            feas_dict[sub_ind] = sub.model.SolCount>0
            if sub.model.SolCount>0:
                obj = sub._weight*sub.model.ObjVal
            else:
                obj = np.nan
            obj_dict[sub_ind] = obj

        for sub_ind,sub in iteritems(sub_dict):
            # Reset bounds
            for j,y in enumerate(sub._ys):
                y.LB = var_dict[sub_ind][j]['LB']
                y.UB = var_dict[sub_ind][j]['UB']
                y.VType = var_dict[sub_ind][j]['VType']
            sub.model.update()

        return obj_dict, feas_dict, stat_dict

    def solve_subproblems(self, uk):
        """
        Subclass can implement parallel version.
        """
        sub_dict = self.sub_dict
        result_list = []
        for sub_ind,sub in iteritems(sub_dict):
            sub.update_obj(uk)
            obj = sub.optimize(yk=sub.yopt) * sub._weight
            if not np.isnan(obj):
                cut = self.make_multicut(sub)
            else:
                cut = None
            result_list.append({'id':sub_ind, 'obj':obj, 'cut':cut})

        return result_list

    def solve_master(self, delta, u0):
        """
        Subclass can implement parallel version.
        """
        model = self.model
        self.update_obj(delta, u0)
        model.optimize()
        if model.SolCount > 0:
            uk = np.array([u.X for u in self._us])
        else:
            uk = None

        return uk

    def update_proximal(self, uk, u0, delta, D_new, D_prox):
        """
        Update proximal point according to proximal bundle procedure.

        Proximal bundle for minimizing convex function D(u) given
        we can compute D(u) and subgradient g(u) at u in U,
        and polyhedral lower approximation,
        _Dk(u) = max_j {D(uj) + <g(uj), u-uj>}.

        Next trial point uk+1 is
        uk+1 \in arg min {_Dk(u) + 1/2*ck*||u-uc||^2: u \in U}.
        Descent step to xk+1 = yk+1 if

        D(uk+1) >= D(uc) + a*vk,

        else uk+1 = uk,
        where a \in (0,1) fixed and

        vk = _Dk(uk+1) - D(uc) >= 0,

        I.e., check if
        D(uk+1) - D(uc) >= a*( _Dk(uk+1) - D(uc) )

        if vk = 0 then uk is optimal.
        Here:
        D(u) is the Lagrangian at u
        _Dk(u) is the lower polyhedral approximation of Lagrangian D(u)
        """
        verbosity = self.verbosity
        if D_prox is None:
            D_prox = self.solve_lagrangian(u0)

        # Is the new approximated max D(uk+1) better enough to warrant
        # updating proximal point?
        z = self._z
        pred_ascent = z.X - D_prox     # Should always be better or the same
        ascent_tol = self.feastol
        if abs(pred_ascent) < ascent_tol:
            converged = True
        else:
            converged = False
            if D_new >= (D_prox + self.bundle_mult*pred_ascent):
                # Update proximal point
                u0 = uk
                # Compute Lagrangian at updated prox point
                D_prox = self.solve_lagrangian(u0)
            else:
                # Null move
                pass

        delta = max(self.delta_min, self.delta_mult*delta)

        return u0, delta, D_prox, pred_ascent

    def feasible_primal_decomp(self):
        """
        Recover feasible primal via primal decomposition
        """


class LagrangeMasterMPI(LagrangeMaster):
    """
    Solves subproblems in parallel.
    """
    def __init__(self,*args,**kwargs):
        from mpi4py import MPI

        self.comm = MPI.COMM_WORLD
        self.ROOT = 0
        self.sub_ids = []   # Need consistently ordered list of inds for parallel
        self.uk = None      # Need to bcast
        self.u0 = None
        self.best_dict = {'feasUB':1e100, 'bestUB':1e100, 'bestLB':1e100,
                'gap':1e100, 'relgap':1e100}
        super(LagrangeMasterMPI, self).__init__(*args,**kwargs)


    def optimize(self, feasible_methods=['heuristic','enumerate'], bundle=False,
            multicut=True, max_alt=10, alt_method='pool', nogood_cuts=False,
            early_heuristics=[]):
        """ Override to implement MPI """
        comm = self.comm
        rank = comm.Get_rank()
        ROOT = self.ROOT

        model = self.model

        sub_dict_all = self.sub_dict
        sub_ids = self.get_jobs(rank)
        sub_dict = {k:sub_dict_all[k] for k in sub_ids}

        max_iter = self.max_iter
        print_iter = self.print_iter
        time_limit = self.time_limit
        gaptol = self.gaptol
        absgaptol = self.absgaptol
        feastol = self.feastol
        penaltytol = self.penaltytol
        verbosity = self.verbosity
        delta_min = self.delta_min
        delta_mult= self.delta_mult
        delta = 1.
        bundle_mult = self.bundle_mult
        D_prox = None    # proximal point objval

        if feasible_methods is None:
            feasible_methods = []

        z = self._z
        us = self._us
        nu = len(us)

        feasUB = 1e100
        dualUB = 1e100  # Overestimator from master, with supergradient-based polyhedral approx
                        # Should decrease monotonically
        UB = 1e100      # UB from feasible solution. Not necessarily monotonic, depending
                        # method used to construct feasible solution.
        LB = -1e100     # Underestimator from dual relaxed subproblems
        bestLB = LB
        bestUB = UB     # Given by best feasible solution, not the master problem
        gap = bestUB-bestLB
        relgap = gap/(1e-10+abs(UB))
        uk = np.zeros(nu)
        self.uk = uk
        u0 = np.zeros(nu)   # Trust region center
        self.u0 = u0
        ubest = uk
        x_dict = {}
        self.log_rows = []

        status_dict[GRB.SUBOPTIMAL] = 'suboptimal'
        status_dict[GRB.NUMERIC] = 'numeric'

        tic = time.time()
        if rank==ROOT:
            print("%8.6s%22s%22s%10.8s%10.9s%10.8s%30s" % (
                'Iter','UB','LB','gap','relgap(%)','penalty','Time(s)'))
            print("%8.6s%22s%22s%10.8s%10.9s%10.8s%30s" % (
                '-'*7,'-'*19,'-'*19,'-'*9,'-'*9,'-'*9,'-'*29))
            print("%8.6s%11.8s%11.8s%11.8s%11.8s%10.8s%10.8s%10.8s%10.8s%10.8s%10.8s" % (
                '','Best','Feasible','Sub','Best','','','','total','master','sub'))

        # Don't pool solutions until gap below tolerance
        for sub in sub_dict.values():
            sub.model.Params.PoolSearchMode=0
            sub.model.Params.PoolSolutions =1

        for _iter in range(max_iter):
            #----------------------------------------------------
            # Solve Master
            #----------------------------------------------------
            tic_master = time.time()
            if rank==ROOT:
                uk = self.solve_master(delta, u0)
                self.uk = uk
                if model.Status != GRB.OPTIMAL:
                    warnings.warn("Solution available but Master solver status=%s (%s)."%(
                        status_dict[model.Status], model.Status))
                    if verbosity>1:
                        print("Master solver status=%s (%s)."%(
                            status_dict[model.Status], model.Status))
                if uk is None:
                    raise Exception("Master solver status=%s (%s). Aborting."%(
                        status_dict[model.Status], model.Status))
            else:
                uk = np.empty(nu, dtype='float64')

            toc_master = time.time()-tic_master
            comm.Bcast(uk, root=ROOT)
            # print("Broadcasted. rank=%s. uk=%s"%(rank,uk))

            #----------------------------------------------------
            # Solve Subproblems
            #----------------------------------------------------
            tic_sub = time.time()
            sub_results = self.solve_subproblems(uk)
            toc_sub = time.time()-tic_sub

            if rank==ROOT:
                sub_objs = []
                for sub_result in sub_results:
                    sub_ind = sub_result['id']
                    obj = sub_result['obj']
                    # print("rank=%s. obj=%s"%(rank,obj))
                    if np.isnan(obj):
                        if verbosity>0:
                            print("Relaxed subproblem %s has objval=%s."%(sub_ind, obj))
                        return None
                    else:
                        # cut = sub_result['cut']
                        sub = sub_dict_all[sub_ind]
                        x_dict =sub_result['x_dict']
                        cut = self.make_multicut(sub, x_dict)
                        model.addConstr(cut)
                    sub_objs.append(obj)

            if rank==ROOT:
                #----------------------------------------------------
                # Compute UB
                UB = z.X        # Lagrange dual UB
                dualUB = UB
                # if UB < bestUB:
                #     bestUB = UB
                bestUB = UB     # Since proximal penalty might bias it
                #----------------------------------------------------
                # Update bounds and check convergence
                LB = sum(sub_objs)
                if LB > bestLB:
                    bestLB = LB
                    ubest  = uk     # Record best multipliers that improved LB
                    self.uopt = ubest
                    self.x_dict = {v.VarName:v.X for v in model.getVars()}
                #----------------------------------------------------
                # Can use Heuristics before desired gap reached
                if early_heuristics:
                    for heuristic in early_heuristics:
                        yfeas, feasUBk, is_optimal = self.feasible_heuristics(heuristic)
                        if yfeas is not None:
                            if feasUBk < bestUB:
                                bestUB = feasUBk
                            if feasUBk < feasUB:
                                feasUB = feasUBk
                                ybest = yfeas
                                self.yopt = ybest
                                for j,yj in enumerate(ybest):
                                    x_dict[sub._ys[j].VarName] = yj
                                if verbosity>1:
                                    print("Best Heuristic solution has objval=%s"%(feasUB))
                                gap = abs(bestUB-bestLB)
                                if abs(gap)<absgaptol:
                                    gap = 0.
                                relgap = gap/(1e-10+abs(bestUB))
                                if relgap <= gaptol:
                                    #--------------------------------------------
                                    # Final QC that sum_k Hk*yk = 0
                                    Hys = []
                                    for sub_ind,sub in iteritems(sub_dict):
                                        yk = np.array([sub.x_dict[x.VarName] for x in sub._ys])
                                        Hyk = sub._H*yk
                                        Hys.append(Hyk)
                                    if verbosity>2:
                                        print("sum_k Hk*yk: %s"%(sum(Hys)))
                                        if sum([abs(x) for x in sum(Hys)]) > feastol:
                                            print("WARNING: Non-anticipativity constraints not satisfied.")
                                    #--------------------------------------------
                                    # Final Log
                                    toc = time.time()-tic
                                    res_u = delta/2.*sum((uk-u0)**2)    # total penalty term
                                    self.log_rows.append({'iter':_iter,'bestUB':bestUB,'feasUB':feasUB,
                                        'LB':LB,'bestLB':bestLB,'gap':gap,'relgap':relgap*100,'delta':delta,
                                        'res_u':res_u,'t_total':toc,'t_master':toc_master,'t_sub':toc_sub})
                                    break


            # Quadratic stabilization term can cause "best" LB and UB to change
            # non monotonically, not guaranteeing bestUB > bestLB
            bestUB = comm.bcast(bestUB, root=ROOT)
            bestLB = comm.bcast(bestLB, root=ROOT)

            gap = abs(bestUB-bestLB)
            if abs(gap)<absgaptol:
                gap = 0.
            relgap = gap/(1e-10+abs(bestUB))
            res_u = delta/2.*sum((uk-u0)**2)    # total penalty term
            #------------------------------------------------
            # Log progress
            toc = time.time()-tic
            if rank==ROOT:
                self.log_rows.append({'iter':_iter,'bestUB':bestUB,'feasUB':feasUB,
                    'LB':LB,'bestLB':bestLB,'gap':gap,'relgap':relgap*100,'delta':delta,
                    'res_u':res_u,'t_total':toc,'t_master':toc_master,'t_sub':toc_sub})

            #------------------------------------------------
            # Check convergence
            if relgap <= gaptol and abs(res_u)<penaltytol:
                #--------------------------------------------
                toc = time.time()-tic
                if verbosity>0:
                    print(
                    "%8.6s%11.4g%11.4g%11.4g%11.4g%10.4g%10.4g%10.3g%10.8s%10.8s%10.8s" % (
                    _iter,bestUB,feasUB,LB,bestLB,gap,relgap*100,res_u,toc,toc_master,toc_sub))
                #--------------------------------------------
                # Now, look for a feasible solution
                # Spend user-defined portion of total time on different strategies,
                # including heuristics, B&B, and enumeration.
                #--------------------------------------------
                if feasible_methods:
                    if 'heuristic' in feasible_methods:
                        yfeas, feasUBk, is_optimal = self.feasible_heuristics('average')
                        if yfeas is not None:
                            if feasUBk < feasUB:
                                feasUB = feasUBk
                                ybest = yfeas
                                self.yopt = ybest
                                for j,yj in enumerate(ybest):
                                    x_dict[sub._ys[j].VarName] = yj
                                if verbosity>1:
                                    print("Best Heuristic solution has objval=%s"%(feasUB))

                            if is_optimal:
                                #--------------------------------------------
                                # Final QC that sum_k Hk*yk = 0
                                Hys = []
                                for sub_ind,sub in iteritems(sub_dict):
                                    yk = np.array([sub.x_dict[x.VarName] for x in sub._ys])
                                    Hyk = sub._H*yk
                                    Hys.append(Hyk)
                                if verbosity>1:
                                    print("sum_k Hk*yk: %s"%(sum(Hys)))
                                    if sum([abs(x) for x in sum(Hys)]) > feastol:
                                        print("WARNING: Non-anticipativity constraints not satisfied.")
                                #--------------------------------------------
                                # Final Log
                                toc = time.time()-tic
                                self.log_rows.append({'iter':_iter,'bestUB':bestUB,'feasUB':feasUB,
                                    'LB':LB,'bestLB':bestLB,'gap':gap,'relgap':relgap*100,'delta':delta,
                                    'res_u':res_u,'t_total':toc,'t_master':toc_master,'t_sub':toc_sub})
                                break

                    # Move on to the next primal recovery method
                    if 'enumerate' in feasible_methods:
                        yfeas, feasUBk, is_optimal = self.enum_alt_opt(first_feasible=True,
                                max_alt=max_alt, method=alt_method, nogood_cuts=nogood_cuts)
                        if yfeas is not None:
                            #--------------------------------------------
                            # Final solution candidate
                            if feasUBk < feasUB:
                                feasUB = feasUBk
                                ybest = yfeas
                                self.yopt = ybest
                                for j,yj in enumerate(ybest):
                                    x_dict[sub._ys[j].VarName] = yj
                                if verbosity>1:
                                    print("Best feasible solution has objval=%s"%(feasUB))

                            if is_optimal:
                                #--------------------------------------------
                                # Final QC that sum_k Hk*yk = 0
                                Hys = []
                                for sub_ind,sub in iteritems(sub_dict):
                                    yk = np.array([sub.x_dict[x.VarName] for x in sub._ys])
                                    Hyk = sub._H*yk
                                    Hys.append(Hyk)
                                if verbosity>1:
                                    print("sum_k Hk*yk: %s"%(sum(Hys)))
                                    if sum([abs(x) for x in sum(Hys)]) > feastol:
                                        print("WARNING: Non-anticipativity constraints not satisfied.")
                                #--------------------------------------------
                                # Final Log
                                toc = time.time()-tic
                                self.log_rows.append({'iter':_iter,'bestUB':bestUB,'feasUB':feasUB,
                                    'LB':LB,'bestLB':bestLB,'gap':gap,'relgap':relgap*100,'delta':delta,
                                    'res_u':res_u,'t_total':toc,'t_master':toc_master,'t_sub':toc_sub})
                                break
                        else:
                            if verbosity>0:
                                print("Feasible solution not among alt opt. Trying next iteration.")
                    #--------------------------------------------
                    # END: Recover primal feasible solution.
                    #--------------------------------------------
                else:
                    break

            D_new  = LB
            if rank==ROOT:
                u0, delta, D_prox, pred_ascent = self.update_proximal(uk, u0, delta,
                        D_new, D_prox)
            else:
                u0 = np.empty(nu,'float64')
            comm.Bcast(u0, root=ROOT)

            if rank==ROOT:
                if verbosity>2:
                    print("pred_ascent=%s. Updating D_prox from %s to %s."%(
                        pred_ascent,D_prox,D_new))

            ascent_tol = self.feastol
            if abs(pred_ascent) < ascent_tol:
                toc = time.time()-tic
                if rank==ROOT:
                    print("Master ascent of %s < %s  below threshold. Stopping."%(
                        pred_ascent, ascent_tol))
                    # Final Log
                    self.log_rows.append({'iter':_iter,'bestUB':bestUB,'feasUB':feasUB,
                        'LB':LB,'bestLB':bestLB,'gap':gap,'relgap':relgap*100,'delta':delta,
                        'res_u':res_u,'t_total':toc,'t_master':toc_master,'t_sub':toc_sub})
                    if verbosity>0:
                        print(
                        "%8.6s%11.4g%11.4g%11.4g%11.4g%10.4g%10.4g%10.3g%10.8s%10.8s%10.8s" % (
                        _iter,bestUB,feasUB,LB,bestLB,gap,relgap*100,res_u,toc,toc_master,toc_sub))
                break
            #------------------------------------------------

            if rank==ROOT:
                if np.mod(_iter, print_iter)==0:
                    toc = time.time()-tic
                    if verbosity>0:
                        print(
                        "%8.6s%11.4g%11.4g%11.4g%11.4g%10.4g%10.4g%10.3g%10.8s%10.8s%10.8s" % (
                        _iter,bestUB,feasUB,LB,bestLB,gap,relgap*100,res_u,toc,toc_master,toc_sub))
            # Check timelimit
            toc = time.time()-tic
            if toc > time_limit:
                if verbosity>0:
                    print("Stopping due to time limit of %g seconds."%time_limit)
                break
            else:
                # Update time limits on all (sub)models
                time_left = time_limit-toc
                model.Params.TimeLimit = time_left
                for sub in sub_dict.values():
                    sub.model.Params.TimeLimit = time_left

        return x_dict

    def check_feasible_task(self, y0):
        """ subtask """
        sub_dict_all = self.sub_dict
        sub_ids = self.get_jobs(rank)
        sub_dict = {k:sub_dict_all[k] for k in sub_ids}
        precision_sub = self.precision_sub

        result_list = []

        var_dict = {}     # record orig bounds for later
        obj_dict   = {}
        stat_dict= {}
        feas_dict= {}
        for sub_ind,sub in iteritems(sub_dict):
            var_dict[sub_ind] = []
            for j,y in enumerate(sub._ys):
                var_dict[sub_ind].append({'LB':y.LB,'UB':y.UB,'VType':y.VType})
                y.LB = y0[j]
                y.UB = y0[j]
                y.VType = GRB.CONTINUOUS    # Since fixed
            sub.optimize()
            stat_dict[sub_ind] = sub.model.Status
            feas_dict[sub_ind] = sub.model.SolCount>0
            if sub.model.SolCount>0:
                obj = sub._weight*sub.model.ObjVal
            else:
                obj = np.nan
            obj_dict[sub_ind] = obj

        for sub_ind,sub in iteritems(sub_dict):
            # Reset bounds
            for j,y in enumerate(sub._ys):
                y.LB = var_dict[sub_ind][j]['LB']
                y.UB = var_dict[sub_ind][j]['UB']
                y.VType = var_dict[sub_ind][j]['VType']
            sub.model.update()

        result_list.append({'obj_dict':obj_dict,'stat_dict':stat_dict,'feas_dict':feas_dict})

        return result_list

    def check_feasible(self, y0):
        """ Override to implement MPI """
        comm = self.comm
        rank = comm.Get_rank()
        ROOT = self.ROOT

        results = self.check_feasible_task(y0)

        result_lists = comm.gather(results, root=ROOT)

        if rank==ROOT:
            result_list = [r for l in results for r in l]
            obj_dict = {}
            feas_dict = {}
            stat_dict = {}
            for result in result_list:
                obj_dictk = result['obj_dict']
                stat_dictk = result['stat_dict']
                feas_dictk = result['feas_dict']
                for k,v in iteritems(obj_dictk):
                    obj_dict[k] = v
                for k,v in iteritems(stat_dictk):
                    stat_dict[k] = v
                for k,v in iteritems(feas_dictk):
                    feas_dict[k] = v
        else:
            result_list = None
            obj_dict = None
            feas_dict = None
            stat_dict = None

        return obj_dict, feas_dict, stat_dict

    def solve_subproblems(self, uk):
        """ Override to implement MPI """
        comm = self.comm
        rank = comm.Get_rank()
        ROOT = self.ROOT
        # All nodes do work
        results = self.do_work(rank, uk)

        # If root, will gather results from all threads
        result_lists = comm.gather(results, root=ROOT)

        if rank==ROOT:
            # Flatten list of lists first
            result_list = [r for l in result_lists for r in l]
        else:
            result_list = None

        return result_list

    def solve_master(self, delta, u0):
        """ Override to implement MPI """
        comm = self.comm
        rank = comm.Get_rank()

        if rank==self.ROOT:
            uk = super(LagrangeMasterMPI,self).solve_master(delta, u0)
            self.uk = uk
        else:
            uk = None

        return uk

    def update_proximal(self, uk, u0, delta, D_new, D_prox):
        """ Override to implement MPI """
        comm = self.comm
        rank = comm.Get_rank()

        if rank==self.ROOT:
            u0, delta, D_prox, pred_ascent = super(
                    LagrangeMasterMPI,self).update_proximal(uk, u0, delta,  D_new, D_prox)
        else:
            pred_ascent=None

        return u0, delta, D_prox, pred_ascent

    def make_multicut(self, sub, x_dict):
        """ Override to implement MPI """
        tk  = self.model.getVarByName('tk_%s'%sub._id)
        fy  = sub._fy
        cx  = sub._cx
        Hk  = sub._H
        Q   = sub._Q
        xk = np.array([x_dict[v.VarName] for v in sub._xs])
        yk = np.array([x_dict[v.VarName] for v in sub._ys])

        yH  = Hk*yk
        us  = self._us
        if Q is None:
            quadTerm = 0.
        else:
            vk = np.array([x_dict[v.VarName] for v in sub.model.getVars()])
            quadTerm = 0.5*np.dot((Q*vk), vk)
        try:
            cut = tk <= sum(fy*yk) + sum(cx*xk) + LinExpr(yH,us) + quadTerm
        except GurobiError as e:
            print('Caught GurobiError (%s) in make_supercut()'%repr(e))

        return cut

    def feasible_heuristics(self, sub_objs, heuristic='average'):
        """ Override to implement MPI """
        yfeas = None
        feasUB = None
        is_optimal = False
        sub_dict = self.sub_dict

        opt_obj = sum(sub_objs)

        if heuristic=='average':
            ymat = np.array([[sub._weight*sub.x_dict[v.VarName] for v in sub._ys] for 
                sub in sub_dict.values()])
            #------------------------------------------------
            # Only need to round if not continuous
            # y0 = ymat.mean(axis=0).round()
            ym = ymat.mean(axis=0)
            y0 = np.array(
                    [ymi if y.VType==GRB.CONTINUOUS else ymi.round() for y,ymi in zip(sub._ys,ym)])
            #------------------------------------------------
            obj_dict, feas_dict, sub_stats = self.check_feasible(y0)

            are_feas = feas_dict.values()
            tot_obj = sum(obj_dict.values())
            gap = abs(tot_obj-opt_obj)
            relgap = abs(tot_obj-opt_obj)/(1e-10+abs(opt_obj))
            is_optimal = gap <= self.absgaptol or relgap <= self.gaptol

            if np.all(are_feas):
                yfeas = y0
                feasUB = tot_obj
        else:
            print("Unknown heuristic: %s"%heuristic)

        return yfeas, feasUB, is_optimal



    def do_work(self, rank, uk):
        """
        Do work on the subproblems assigned to this thread.
        """
        sub_dict_all = self.sub_dict

        sub_ids = self.get_jobs(rank)
        sub_dict = {k:sub_dict_all[k] for k in sub_ids}

        result_list = []
        for sub_ind,sub in iteritems(sub_dict):
            sub.update_obj(uk)
            obj = sub.optimize(yk=sub.yopt) * sub._weight
            # if not np.isnan(obj):
            #     cut = self.make_multicut(sub)
            # else:
            #     cut = None
            # result_list.append({'id':sub_ind, 'obj':obj, 'cut':cut})
            x_dict = sub.x_dict
            result_list.append({'id':sub_ind, 'obj':obj, 'x_dict':x_dict})

        return result_list

    def add_submodels(self, sub_dict, **kwargs):
        """
        Override to update list of submodel ids and jobs.
        """
        for sub_id in sub_dict.keys():
            self.sub_ids.append(sub_id)

        comm = self.comm
        size = comm.Get_size()
        self.worker_tasks = self.get_joblist(self.sub_ids, size)

        super(LagrangeMasterMPI, self).add_submodels(sub_dict, **kwargs)
        nu = len(self._us)
        self.uk = np.zeros(nu)

    def get_joblist(self, ids_all, n_workers):
        #------------------------------------------------------------
        # Split the work
        n_tasks = len(ids_all)
        tasks_per_worker = n_tasks/n_workers
        rem = n_tasks - tasks_per_worker * n_workers
        # Distributes remainder across workers
        worker_sizes = np.array(
                [tasks_per_worker + (1 if i<rem else 0) for i in range(0,n_workers)],'i')
        #------------------------------------------------------------
        # Thus, worker_tasks[rank] gives the actual indices to work on
        inds_end = np.cumsum(worker_sizes)
        inds_start = np.hstack( (0, inds_end[0:-1]) )
        worker_tasks = [ids_all[inds_start[i]:inds_end[i]] for i in range(0,n_workers)]

        return worker_tasks

    def get_jobs(self, rank):
        """
        Distribute submodels according to rank
        """
        worker_tasks = self.worker_tasks

        return worker_tasks[rank]


def binary_thresh(x, thresh):
    if x<thresh:
        y=0.
    else:
        y=1.
    return y
