#============================================================
# File estimate.py
#
# Parameter estimation interface.
#
# Laurence Yang, SBRG, UCSD
#
# 19 Mar 2018:  first version
#============================================================

from __future__ import division
from six import iteritems

from optimize import StackOptimizer, Optimizer
from decomposition import Decomposer

import numpy as np
import pandas as pd
import warnings


class Estimator(object):
    """
    Interface to estimators
    """

    def __init__(self):
        pass

    def fit(self, df_X, df_Y):
        """
        Fit parameters given input (df_X) and output (df_Y) dataframes.
        """
        pass

    def predict(self, df_X):
        """
        Predict using estimated model given input (df_X) dataframe.
        """
        pass

    def parameters(self):
        """
        Return fitted parameters
        """

class MulticutEstimator(Estimator):
    """
    Estimate using multi-cut Benders decomposition
    """
    def __init__(self):
        self.prevent_zero = True
        self.INF = 1e3

    def fit(self, base_model, df_X, df_Y, **kwargs):
        """
        Fit parameters given input (df_X) and output (df_Y) dataframes.
        """
        stacker = StackOptimizer()
        stacker.stack_models(base_model, df_X)

        radix = 2.
        powers = [-1,0,1]
        digits_per_power = radix
        digits = list(set(np.linspace(1, radix-1, digits_per_power)))
        fit_constraint_id = 'crowding'

        for k,v in iteritems(kwargs):
            if k=='radix':
                radix = v
            elif k=='digits_per_power':
                digits_per_power = v
                digits = list(set(np.linspace(1, radix-1, digits_per_power)))
            elif k=='fit_constraint_id':
                fit_constraint_id = v

        if kwargs.has_key('digits'):
            if kwargs.has_key('digits_per_power'):
                warnings.warn('Both kwargs digits and digits_per_power provided. Using digits.')
            digits = kwargs['digits']

        for mdl_ind, mdl in iteritems(stacker.model_dict):
            opt = Optimizer(mdl)
            gap = opt.add_duality_gap_constraint(INF=self.INF, inplace=True, index=mdl_ind)

        # Constraint-specific
        prevent_zero = self.prevent_zero
        cons_ref = base_model.metabolites.get_by_id(fit_constraint_id)
        var_cons_dict = {}
        for mdl_ind, mdl in iteritems(stacker.model_dict):
            for rxn_ref in cons_ref.reactions:
                constraint_p = mdl.metabolites.get_by_id(fit_constraint_id+'_%s'%mdl_ind)
                var_d = mdl.reactions.get_by_id('wa_%s'%constraint_p.id)
                rxn_p = mdl.reactions.get_by_id(rxn_ref.id+'_%s'%mdl_ind)
                # Get coeff in dual
                cons_ds = [m for m in var_d.metabolites.keys() if rxn_p.id==m.id]
                a0 = rxn_p.metabolites[constraint_p]
                if var_cons_dict.has_key(rxn_ref.id):
                    var_cons_dict[rxn_ref.id] += [(rxn_p, constraint_p, a0)] + \
                            [(var_d, cons_d, a0) for cons_d in cons_ds]
                else:
                    var_cons_dict[rxn_ref.id] = [(rxn_p, constraint_p, a0)] + \
                            [(var_d, cons_d, a0) for cons_d in cons_ds]

            opt = Optimizer(mdl)
            opt.to_radix(mdl,var_cons_dict,radix,powers,digits=digits,prevent_zero=prevent_zero)

        # Need
        # - master model
        # - dict of sub models



    def predict(self, df_X):
        """
        Predict using estimated model given input (df_X) dataframe.
        """
        pass

    def parameters(self):
        """
        Return fitted parameters
        """


class CrossEstimator(Estimator):
    """
    Estimate using cross decomposition
    """

