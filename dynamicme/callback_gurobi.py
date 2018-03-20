#============================================================
# File callback_gurobi.py
#
# Gurobi-specific callbacks
#
# Laurence Yang, SBRG, UCSD
#
# 23 Feb 2018:  first version
#============================================================

from six import iteritems
from builtins import range

from gurobipy import *

import numpy as np


def cb_benders(model, where):
    GAPTOL = model._gaptol
    precision_sub = model._precision_sub
    try:
        decomposer = model._decomposer
    except AttributeError:
        print('Need model._decomposer = decomposer')
        raise Exception('Need model._decomposer = decomposer')

    if where in (GRB.Callback.MIPSOL, GRB.Callback.MIPNODE):
        ### Lazy constraints only allowed for MIPNODE or MIPSOL
        ys = decomposer._ys
        z  = decomposer._z
        if where==GRB.Callback.MIPSOL:
            yopt = [model.cbGetSolution(y) for y in ys]
            zmaster = model.cbGetSolution(z)
            if np.nan in yopt:
                print('MIPSOL: nan in yopt!')
        elif where==GRB.Callback.MIPNODE:
            node_status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
            if node_status==GRB.OPTIMAL:
                yopt = [model.cbGetNodeRel(y) for y in ys]
                zmaster = model.cbGetNodeRel(z)
            else:
                if model._verbosity>0:
                    print('non-optimal cbGet(MIPNODE_STATUS)=%g'%node_status)
                return

        decomposer.update_subobj(yopt)
        sub = decomposer.get_sub()
        sub.optimize(precision=precision_sub)

        if sub.Status == GRB.Status.UNBOUNDED:
            # Add feasibility cut, ensuring that cut indeed eliminates current incumbent
            if model._verbosity > 0:
                print('*'*40)
                print('Adding Feasibility cut')

            feascut = decomposer.make_feascut()
            model.cbLazy(feascut)
        else:
            zsub = decomposer.calc_sub_objval(yopt)
            #gap = zmaster - zsub
            gap = zsub - zmaster    # UB - LB

            if model._verbosity > 1:
                print('#'*40)
                print('zmaster=%g. zsub=%g. gap=%g' % (zmaster, zsub, gap))
                #print('#'*40)

            if abs(gap) > GAPTOL:
                optcut = decomposer.make_optcut()
                model.cbLazy(optcut)

            else:
                # Accept as new incumbent
                pass


def cb_benders_multi(model, where):
    GAPTOL = model._gaptol
    precision_sub = model._precision_sub
    gap = 1e15
    try:
        master = model._master
    except AttributeError:
        print('Need model._master = master')
        raise Exception('Need model._master = master')

    if model._verbosity > 2:
        print('*'*40)
        print('In callback with where: %s'%where)

    if where in (GRB.Callback.MIPSOL, GRB.Callback.MIPNODE):
        ### Lazy constraints only allowed for MIPNODE or MIPSOL
        fy = master._fy
        ys = master._ys
        z  = master._z
        if where==GRB.Callback.MIPSOL:
            yopt = [model.cbGetSolution(y) for y in ys]
            zmaster = model.cbGetSolution(z)
        elif where==GRB.Callback.MIPNODE:
            node_status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
            if node_status==GRB.OPTIMAL:
                yopt = [model.cbGetNodeRel(y) for y in ys]
                zmaster = model.cbGetNodeRel(z)
            else:
                if model._verbosity>0:
                    print('non-optimal cbGet(MIPNODE_STATUS)=%g'%node_status)
                return

        sub_dict = master.sub_dict
        zsubs = []
        opt_sub_inds = []
        # zsub = fy*yopt + sum_k tk
        zsub_total = sum(fy*yopt)

        for sub_ind, sub in iteritems(sub_dict):
            sub.update_obj(yopt)
            sub.model.optimize(precision=precision_sub)
            if model._verbosity>1:
                print('Submodel %s status = %s'%(sub_ind, sub.model.Status))

            if sub.model.Status == GRB.Status.UNBOUNDED:
                # Add feasibility cut, ensuring that cut indeed elimi current incumbent
                # If even one submodel infeasible, original problem infeasible.
                IS_INFEAS = True
                if model._verbosity > 1:
                    #print('*'*40)
                    print('Adding Feasibility cut')
                feascut = master.make_feascut(sub)
                model.cbLazy(feascut)
            else:
                zsub_total += sub.model.ObjVal
                opt_sub_inds.append(sub_ind)

        gap = zsub_total - zmaster

        if model._verbosity > 0:
            print('#'*40)
            print('zmaster=%g. zsub=%g. gap=%g' % (zmaster, zsub_total, gap))

        if abs(gap) > GAPTOL:
            cutparts = []
            for sub_ind in opt_sub_inds:
                sub = sub_dict[sub_ind]
                optcut_part = master.make_optcut_part(sub)
                cutparts.append(optcut_part)
            multisum_cut = z >= LinExpr(fy,ys) + quicksum(cutparts)
            if model._verbosity > 1:
                #print('*'*40)
                print('Adding optimality multi-cut')
            model.cbLazy(multisum_cut)
        else:
            # Accept as new incumbent
            pass
