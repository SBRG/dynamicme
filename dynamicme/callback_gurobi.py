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
        elif where==GRB.Callback.MIPNODE:
            yopt = [model.cbGetNodeRel(y) for y in ys]
            zmaster = model.cbGetNodeRel(z)

        decomposer.update_subobj(yopt)
        sub = decomposer.get_sub()
        sub.optimize(precision=precision_sub)

        if sub.Status == GRB.Status.UNBOUNDED:
            # Add feasibility cut, ensuring that cut indeed eliminates current incumbent
            feascut = decomposer.make_feascut(yopt, zmaster)
            model.cbLazy(feascut)
        else:
            zsub = decomposer.calc_sub_objval(yopt)
            gap = zmaster - zsub

            if model._verbosity > 0:
                print('zmaster=%g. zsub=%g. gap=%g' % (zmaster, zsub, gap))

            if abs(gap) > GAPTOL:
                optcut = decomposer.make_optcut()
                model.cbLazy(optcut)

            else:
                # Accept as new incumbent
                pass
