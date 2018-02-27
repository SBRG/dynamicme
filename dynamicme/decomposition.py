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
from cobra.core.Solution import Solution
from cobra import Reaction, Metabolite, Model
from gurobipy import *

import numpy as np
import warnings

class Decomposer(object):
    def __init__(self, milp):
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
        A,B,d,csenses,xs,ys = split_constraints(self.milp)
        self._A = A
        self._Acsc = A.tocsc()
        self._B = B
        self._Bcsc = B.tocsc()
        self._d = d
        self._csenses = csenses
        self._x0 = xs
        self._y0 = ys

    def make_master(self):
        """
        min     z
        y,z
        s.t.    z >= f'y + (dk-By)'wA_i,k + lk*wl_i,k - uk*wu_i,k,  i \in OptimalityCuts
                (dk-By)'wA_i,k + lk*wl_i,k - uk*wu_i,k <= 0,        i \in FeasibilityCuts
        """
        LB = -self._INF
        UB = self._INF
        if self._y0 is None:
            self._split_constraints()

        ys0 = self._y0
        ny = len(ys0)
        B = self._B
        fy = np.array([yj.Obj for yj in ys0])
        master = Model('master')
        z = master.addVar(LB, UB, 0., GRB.CONTINUOUS, 'z')
        ys = [master.addVar(y.LB, y.UB, y.Obj, y.VType, y.VarName) for y in ys0]
        rhs = LinExpr(fy, ys)
        master.addConstr(z, GRB.GREATER_EQUAL, rhs)
        master.setObjective(z, GRB.MINIMIZE)

        self._z = z
        self._ys = ys
        self._fy = fy

        master._decomposer = self   # Need this reference to access methods inside callback
        master._verbosity = 0

        master.Params.Presolve = 0          # To be safe, turn off
        master.Params.LazyConstraints = 1   # Required to use cbLazy
        master.Params.IntFeasTol = 1e-9

        master.update()

        return master

    def make_sub(self):
        """
        Constraint doesn't change.
        Objective changes with RMP solution, so yopt is optional when initiating.
        """
        LB = -self._INF
        UB = self._INF
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
        lb_dict = {GRB.EQUAL: -self._INF, GRB.GREATER_EQUAL: 0., GRB.LESS_EQUAL: -self._INF}
        ub_dict = {GRB.EQUAL: self._INF, GRB.GREATER_EQUAL: self._INF, GRB.LESS_EQUAL: 0.}
        wa = [sub.addVar(lb_dict[sense], ub_dict[sense], 0., GRB.CONTINUOUS, 'wa[%d]'%i) \
                for i,sense in enumerate(csenses)]
        wl = [sub.addVar(0., UB, 0., GRB.CONTINUOUS, 'wl[%d]'%i) for i in range(n)]
        wu = [sub.addVar(0., UB, 0., GRB.CONTINUOUS, 'wu[%d]'%i) for i in range(n)]
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

        return sub

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
        dBywa = LinExpr([dBy[j] for j in cinds], [wa[j] for j in cinds])
        obj = dBywa + LinExpr(xl,wl) - LinExpr(xu,wu)
        sub.setObjective(obj, GRB.MAXIMIZE)

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
        wa = np.array([w.X for w in self._wa])
        wl = np.array([w.X for w in self._wl])
        wu = np.array([w.X for w in self._wu])
        Bcsc = self._Bcsc

        # cut = z >= quicksum([fy[j]*ys[j] for j in range(ny)]) + \
        #         quicksum([d[i]*wa[i]-quicksum([B[i,j]*ys[j]*wa[i] for j in range(ny)]) \
        #         for i in range(m)]) + \
        #         quicksum([xl[j]*wl[j] for j in range(n)]) -\
        #         quicksum([xu[j]*wu[j] for j in range(n)])

        waB  = wa*Bcsc
        cinds = waB.nonzero()[0]
        waBy = LinExpr([waB[j] for j in cinds], [ys[j] for j in cinds])
        cut = z >= LinExpr(fy,ys) + sum(d*wa) - waBy + sum(xl*wl) - sum(xu*wu)

        return cut

    def make_feascut(self):
        """
        THIS MAY NOT WORK if dvar.X doesn't work for infeas problems.
        Instead, may need FarkasDuals for primal sub problem
        """
        # Get unbounded ray
        wa = [w.X for w in self._wa]
        wl = [w.X for w in self._wl]
        wu = [w.X for w in self._wu]
        ys = self._ys
        d = self._d
        B = self._B
        xl = self._xl
        xu = self._xu
        wa = self._wa
        wl = self._wl
        wu = self._wu
        m = len(d)
        n = len(xl)

        Bcsc = self._Bcsc
        waB  = wa*Bcsc
        cinds = waB.nonzero()[0]
        waBy = LinExpr([waB[j] for j in cinds], [ys[j] for j in cinds])
        cut = sum(d*wa) - waBy + sum(xl*wl) - sum(xu*wu) <= 0

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


def split_constraints(model):
    """
    Splits constraints into continuous and integer parts:
    Ax + By [<=>] d
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
    #A = csr_matrix((xdata, (xrow_inds, xcol_inds)), shape=(M,nx))
    #B = csr_matrix((ydata, (yrow_inds, ycol_inds)), shape=(M,ny))

    return A, B, d, csenses, xs, ys
