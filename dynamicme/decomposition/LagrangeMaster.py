#============================================================
# File LagrangeMaster.py
#
# class LagrangeMaster
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

import gurobipy as grb
import numpy as np
import cobra
import warnings
import time


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
        self.ObjVal = None
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
        self.delta_min = 1e-20
        self.delta_mult = 0.5
        self.bundle_mult = 0.5
        self.max_max_alt = 200
        self.best_dict = {}
        self.uk = None
        self.x_dict = {}
        self.yopt = None
        self.covered_dict = {}

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

    def optimize(self, *args, **kwargs):
        """
        Optimize via two phase or otherwise
        """
        two_phase = kwargs.pop('two_phase',False)
        only_relaxed = kwargs.pop('only_relaxed',False)

        if two_phase or only_relaxed:
            sol = self.solve_relaxed(*args, **kwargs)
        if not only_relaxed:
            sol = self.solve_loop(*args, **kwargs)

        return sol

    def solve_loop(self, feasible_methods=['best_rounding','enumerate'],
            multicut=True, max_alt=10, alt_method='pool', nogood_cuts=False,
            early_heuristics=[], strictUB=False):
        """
        x_dict = optimize()

        method in LagrangeMaster //----------------------------------------------------

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
            '','Dual','Feasible','Sub','Best','','','','total','master','sub'))

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
                                toc = time.time()-tic
                                if verbosity>0:
                                    print(
                                    "%8.6s%11.4g%11.4g%11.4g%11.4g%10.4g%10.4g%10.3g%10.8s%10.8s%10.8s" % (
                                    _iter,bestUB,feasUB,LB,bestLB,gap,relgap*100,res_u,toc,toc_master,toc_sub))
                                return x_dict
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

        if hasattr(self._z,'X'):
            self.ObjVal = self._z.X

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
        x_dict = self.solve_loop(*args,**kwargs)

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

        elif heuristic=='reduced_cost':
            """
            Also good for min sum y objective. Uses reduced cost
            to select which (single) yj=1 within each covered set.
            Might be infeasible for other constraints, in which case
            need to make additional yj=1
            """
            if not hasattr(self,'covered_dict'):
                raise Exception("reduced_cost heuristic needs master.covered_dict")
            else:
                if not self.covered_dict:
                    raise Exception("reduced_cost heuristic needs non-empty master.covered_dict")
                else:
                    # Calculate reduced costs: fkj + (u'Hk)j
                    us = self.uopt
                    # or current us
                    rcs = [sub._fy + us*sub._H for sub in sub_dict.values()]
                    var_inds = {y.id:j for j,y in enumerate(self._y0)}
                    ny = len(self._y0)
                    ymat = np.array([[sub._weight*sub.x_dict[v.VarName] for v in sub._ys] for
                        sub in sub_dict.values()])
                    y0 = np.zeros(ny)
                    for group,vs in iteritems(self.covered_dict):
                        js  = [var_inds[v] for v in vs]
                        rc_set = np.array(
                                [[rcs[k][j] for j in js] for k,sub in enumerate(sub_dict.values())])
                        rc_sum = rc_set.sum(axis=0)
                        # Which ys inconsistent and need fixing?
                        ys_equal = [all(ymat[:-1,j]==ymat[1:,j]) for j in range(ny)]
                        for j,r in zip(js,rc_sum):
                            if ys_equal[j]:
                                y0[j]=ymat[0,j] # if equal can take any yk_j
                            else: # If not equal across k need to fix
                                if r==min(rc_sum):
                                    y0[j] = 1.
                                else:
                                    y0[j] = 0.

                    obj_dict, feas_dict, sub_stats = self.check_feasible(y0)
                    # Repair infeas if necessary

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
            #------------------------------------------------
            # Store previous solution
            yopt_orig = sub.yopt
            #------------------------------------------------
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
            #------------------------------------------------
            # Reset solution
            sub.yopt = yopt_orig
            #------------------------------------------------

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
