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
from cobra.core import Solution
from cobra import Reaction, Metabolite, Model
from cobra import DictList
from gurobipy import *
from cobra.solvers.gurobi_solver import variable_kind_dict

from DecompModel import split_cobra

import gurobipy as grb
import numpy as np
import cobra
import warnings
import time


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

    def update_obj(self, uk, scaling=1.):
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
        obj_lin = scaling*LinExpr(fy,ys) + scaling*LinExpr(cx,xs) + LinExpr(uH,ys)
        if Q is None:
            obj_fun = obj_lin
        else:
            var_dict = {i:v for i,v in enumerate(model.getVars())}
            obj_quad = grb.QuadExpr()
            for (ind0,ind1),val in Q.todok().items():
                x0 = var_dict[ind0]
                x1 = var_dict[ind1]
                obj_quad.addTerms(val/2., x0, x1)
            obj_fun = obj_lin + scaling*obj_quad

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
        xbest = None
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
            objval = self.optimize(bisection=False, yk=ybest, xk=xbest)

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
                xbest = [xd_best[x.VarName] for x in self._xs]

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
