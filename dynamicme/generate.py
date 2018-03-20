#============================================================
# File generate.py
#
# Generate models to avoid running out of RAM for thousands
# of data sets
#
# Laurence Yang, SBRG, UCSD
#
# 19 Mar 2018:  first version
#============================================================

from __future__ import division
from six import iteritems
from cobra import Reaction, Metabolite, Model

import numpy as np
import pandas as pd
import warnings

class Generator(object):
    """
    Generate models given data on the sample
    """
    def __init__(self, base_model):
        self.base_model = base_model
        self.model_dict = {}

    def generate(self, df_X, col_cond='cond'):
        """
        Generate condition-specific model given input data
        """
        base_model = self.base_model
        conds = df_X[col_cond].unique()

        for cind,cond in enumerate(conds):
            dfi = df_X[ df_X[col_cond]==cond]
            # Create clone of reference model
            suffix = '_%s'%cond
            mdli = copy_model(base_model, suffix=suffix)
            # Modify its lb, ub
            for i,row in dfi.iterrows():
                rxn = mdli.reactions.get_by_id(row['rxn']+suffix)
                rxn.lower_bound = row['lb']
                rxn.upper_bound = row['ub']
                rxn.objective_coefficient = row['obj']

            yield mdli


def copy_model(model, suffix=''):
    clone  = Model(model.id)
    rxns = [Reaction(rxn.id+suffix, rxn.name, rxn.subsystem,
        rxn.lower_bound, rxn.upper_bound, rxn.objective_coefficient) for
        rxn in model.reactions]
    mets = [Metabolite(met.id+suffix, met.formula, met.name, met.charge, met.compartment) for
            met in model.metabolites]
    clone.add_reactions(rxns)
    clone.add_metabolites(mets)

    unique_rxn_attrs = ['_model','id','_metabolites','_genes']
    unique_met_attrs = ['_model','id','_reaction']

    # Add properties and stoich
    for rxn0 in model.reactions:
        rxn = clone.reactions.get_by_id(rxn0.id+suffix)
        for k,v in iteritems(rxn0.__dict__):
            if k not in unique_rxn_attrs:
                rxn.__dict__[k] = v

        stoich = {m.id+suffix:s for m,s in iteritems(rxn0.metabolites)}
        rxn.add_metabolites(stoich)

    for met0 in model.metabolites:
        met = clone.metabolites.get_by_id(met0.id+suffix)
        for k,v in iteritems(met0.__dict__):
            if k not in unique_met_attrs:
                met.__dict__[k] = v

    return clone
