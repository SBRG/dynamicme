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

from DecompModel import split_cobra
from dynamicme.callback_gurobi import cb_benders_multi

import gurobipy as grb
import numpy as np
import cobra
import warnings
import time


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
