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

import gurobipy as grb
import numpy as np
import cobra
import warnings
import time


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
