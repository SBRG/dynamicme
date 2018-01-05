#============================================================
# File helpme.py
# 
# Helper classes & methods for tests.
#
# Laurence Yang, SBRG, UCSD
#
# 28 Sept 2017:  first version
#============================================================
"""
- test_ prefixed methods outside of class
- test_ prefixed methods inside Test prefixed classes (without an __init__ method)
"""

from qminos.me1 import ME_NLP1
from cobrame.core.MEModel import MEModel
import cPickle
import json
import os


class HelpME(object):
    """
    Used to quickly solve tiny ME including warm-starts.
    """
    def __init__(self, model_path=None, prototyping=True, protease=False):
        # Load the (default) test model
        if model_path is None:
            home = os.path.expanduser('~')
            if prototyping:
                filename = os.path.join(home,'ME','models','tinyME_nostress_v2.pickle')
                basisfile = None
                solverfile  = os.path.join(home,'ME','models','solver_tinyME_nostress_v2.pickle')
            else:
                if protease:
                    filename = os.path.join(home,'ME','models','iLE_protdegr.pickle')
                    basisfile = None
                    solverfile  = os.path.join(home,'ME','models','solver_iLE_protdegr.pickle')
                else:
                    filename = os.path.join(home,'ME','models','iLE_nostress.pickle')
                    basisfile = None
                    solverfile  = os.path.join(home,'ME','models','solver_iLE_nostress.pickle')

        else:
            filename = model_path
            basisfile = None
            solverfile  = None

        # Solver
        if solverfile is not None:
            with open(solverfile,'rb') as iofile:
                solver = cPickle.load(iofile)
            self.me = solver.me
            hs = solver.lp_hs # feas_basis
        else:
            with open(filename,'rb') as iofile:
                me = cPickle.load(iofile)
            solver = ME_NLP1(me, growth_key='mu')
            self.me = me
            # Basis
            if basisfile is not None:
                try:
                    with open(basisfile,'rb') as iofile:
                        hs = cPickle.load(iofile)
                except IOError:
                    hs = None
            else:
                hs = None

        self.solver = solver
        self.hs = hs
        self.tinyme = None


    def get_tinyme(self):
        if self.tinyme is None:
            home = os.path.expanduser('~')
            filename = os.path.join(home,'ME','models','tinyME_nostress_v0.pickle')
            with open(filename,'rb') as iofile:
                me = cPickle.load(iofile)
            self.tinyme = me

        return self.tinyme


    def get_flux(self, rxn_id):
        me = self.me
        solver = self.solve()

        rxn = me.reactions.get_by_id(rxn_id)
        return rxn.x
