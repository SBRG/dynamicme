#============================================================
# File location.py
#
# Location modeling
#
# Laurence Yang, SBRG, UCSD
#
# 18 Apr 2018:  first version
#============================================================

from __future__ import division
from six import iteritems

from optimize import StackOptimizer, SplitOptimizer, Optimizer
from optimize import Variable, Constraint
from optimize import clone_model
from cobra import Model, Reaction, Metabolite
from generate import copy_model
from cobra.solvers import gurobi_solver

import gurobipy as grb
import numpy as np
import pandas as pd
import warnings


class LocateMM(object):
    """
    Location model for microbiota given
    nutrient sources, i, and occupance nodes ,j.
    Each microbe (group) can also contribute to
    availability of a nutrient via biosynthesis.
    """
    def __init__(self, models, df_location, df_source, df_nutrient, df_organism, df_exchange,
            Tc):
        """
        Inputs
        models : list of models for community
        df_location : dataframe of allocatable node distances
            i (location 1)   j (location 2) d (distance)
        df_source : dataframe of distances
            i (source node)  j (location)  d (distance)
        df_nutrient : dataframe of (primary) nutrient sources
            i (source node)  l (nutrient)  vtot_il
        df_organism : dataframe of organisms and initial conditions
            id (model id)  j (initial location)  X0 (initial biomass)  biomass (biomass rxn)
        df_exchange : dataframe of organisms and initial conditions
            id (model id)  l (nutrient) rxn (exchange rxn)
        """
        self.model = None
        self.model_dict = {}
        self.models = models
        self.df_location = df_location
        self.df_source = df_source
        self.df_nutrient = df_nutrient
        self.df_organism = df_organism
        self.df_exchange = df_exchange
        self.Tc = Tc
        self.col_src = 'i'
        self.col_org = 'id'
        self.col_nutr= 'l'
        self.col_loc = 'j'
        self.col_dist= 'd'
        self.col_vtot='vtot'

    def optimize(self):
        """
        Solve optimal location problem.
        """

    def create_problem(self):
        """
        Create optimization problem
        """
        if len(self.model_dict)==0:
            self.stack_models()
        model_dict = self.model_dict

        df_location = self.df_location
        df_source   = self.df_source
        df_nutrient = self.df_nutrient
        df_organism = self.df_organism
        df_exchange = self.df_exchange

        col_src = self.col_src
        col_org = self.col_org
        col_nutr= self.col_nutr
        col_loc = self.col_loc
        col_vtot= self.col_vtot
        col_dist= self.col_dist

        Tc = self.Tc        # Flush cycle. Period available for cell division.
        # Locations
        J = np.union1d(df_location['i'], df_location['j'])
        for _i,row in df_organism.iterrows():
            mdl_id = row['id']
            X0     = row['X0']
            j0     = row['j0']
            mu_id  = row['biomass']
            mdl    = model_dict[mdl_id]
            rxn_mu = mdl.reactions.get_by_id(mu_id)

            yjk = [Variable('y_%s'%(j), lower_bound=0, upper_bound=1) for j in J]
            for yj in yjk:
                yj.variable_kind = 'integer'
            mdl.add_reactions(yjk)

            # ADD: sum_j ykj - Xk0*muk*Tc <= Xk0 
            cons = Constraint('cons_biomass')
            cons._bound = X0
            cons._constraint_sense = 'L'
            rxn_mu.add_metabolites({cons:-X0*Tc}, combine=False)
            for yj in yjk:
                yj.add_metabolites({cons:1.}, combine=False)

            # ADD: lkj*ykj <= vkj <= ukj*yjk
            for locj in J:
                yj = mdl.reactions.get_by_id('y_%s'%(locj))
                for rxn in mdl.reactions:
                    cons_lb = Constraint('binary_lb_%s'%rxn.id)
                    cons_lb._bound = 0.
                    cons_lb._constraint_sense = 'L'
                    rxn.add_metabolites({cons_lb:-1.}, combine=False)
                    yj.add_metabolites({cons_lb:rxn.lower_bound}, combine=False)

                    cons_ub = Constraint('binary_ub_%s'%rxn.id)
                    cons_ub._bound = 0.
                    cons_ub._constraint_sense = 'L'
                    rxn.add_metabolites({cons_ub:1.}, combine=False)
                    yj.add_metabolites({cons_ub:-rxn.upper_bound})

        #----------------------------------------------------
        # Inter-model linking constraints
        for locj in J:
            # ADD: sum_k ykj <= 1, j \in Locations
            cons_y = Constraint('cons_overlap_%s'%locj)
            cons_y._bound = 1.
            cons_y._constraint_sense = 'L'
            model.add_metabolites([cons_y])
            for mdl_id,mdl in iteritems(model_dict):
                yjk = mdl.reactions.get_by_id('binary_%s'%(locj))
                yjk.add_metabolites({cons_y:1.}, combine=False)

            # ADD: vklj >= sum_i vtot_lij/dij + sum_{p!=j} vsynth_lp/djp,  l \in Nutrients, j\in J
            #      vklj - sum_{p!=j} vsynth_lp/djp >= sum_i vtot_lij/dij,  l \in Nutrients, j\in J
            nutrients = df_nutrient[col_nutr].unique()
            df_nutr_src = pd.merge(df_nutrient, df_source, on=col_src)

            for nutr in nutrients:
                dflj = df_nutr_src[ (df_nutr_src[col_nutr]==nutr) &
                        (df_nutr_src[col_loc]==locj)]
                sum_vtot_d = sum(dflj[col_vtot]/dflj[col_dist])
                for ni,nrow in dfl.iterrows():
                    src = nrow[col_src]

                    cons_uptake_jl = Constraint('cons_uptake_%s_%s'%(locj,src))
                    cons_uptake_jl._constraint_sense = 'G'
                    cons_uptake_jl._bound = sum_vtot_d
                    dfe = df_exchange[df_exchange[col_nutr]==src]
                    for ei,erow in dfe.iterrows():
                        ex_id = erow['rxn']
                        mdl   = model_dict[erow[col_org]]
                        ex_rxn = mdl.reactions.get_by_id(ex_id)
                        ex_rxn.add_metabolites({cons_uptake_jl:1.}, combine=False)

                # Add contribution by other producers

    def stack_models(self):
        """
        Inputs

        Outputs

        max     sum_k sum_j ck'vkj
        vk,ykj
        s.t.    Sk*vkj = 0, j \in Locations
                sum_j ykj <= Xk0 + Xk0*muk*Tc
                lkj*ykj <= vkj <= ukj*yjk
                sum_k ykj <= 1, j \in Locations

                vljk >= sum_i vtot_lij/dij + sum_{p!=j} vsynth_lp/djp,
                    l \in Nutrients, k\in Orgs, j \in Locations

                vtot_il <= sum_k sum_j vklij * dij,  l \in Nutrients, i \in Sources (uptake v<=0)
                [Redundant: vtot_il / dij <= vklij,   l \in Nutrients (uptake v<=0)]
                yjk \in {0,1}
        """
        stacked_model = Model('stacked')
        df_location = self.df_location
        models = self.models
        J = np.union1d(df_location['i'], df_location['j'])

        for mdl in models:
            for j in J:
                clone_id = '%s_%s'%(mdl.id, j)
                mdlj = clone_model(mdl, stacked_model, suffix='_%s'%clone_id)
                self.model_dict[clone_id] = mdlj
        self.model = stacked_model


class CommunityPlotter(object):
    def __init__(self):
        pass

    def plot(self, dsplot, xcol='x',ycol='y',zcol='z', n_grid=100,
            colorbar=True, points=False):
        """
        Inputs:
        dsplot
            x  y  type  density
        """
        from scipy import interpolate
        import matplotlib.pyplot as plt

        x = dsplot[xcol]
        y = dsplot[ycol]
        z = dsplot[zcol]
        xx = np.linspace(x.min(),x.max(), n_grid)
        yy = np.linspace(x.min(),x.max(), n_grid)
        X,Y = np.meshgrid(xx,yy)
        rbf = interpolate.Rbf(x,y,z, function='linear')
        Z = rbf(X,Y)

        plot = plt.imshow(Z, vmin=z.min(), vmax=z.max(), origin='lower',
                extent=[x.min(), x.max(), y.min(), y.max()])
        if points:
            plt.scatter(x,y,c=z, edgecolors='#000000')
        if colorbar:
            plt.colorbar()

        return plot
