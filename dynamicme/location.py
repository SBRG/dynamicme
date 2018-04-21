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



class Locater(object):
    """
    Location model for microbiota given
    nutrient sources, i, and occupance nodes ,j.
    Each microbe (group) can also contribute to
    availability of a nutrient via biosynthesis.
    """
    def __init__(self):
        self.model = None
        self.model_dict = {}

    def create_problem(self, models, df_source, df_consumer, df_node, df_distance, df_organism):
        """
        Inputs
        models : list of models
        df_source
            met  source  rxn  node  vmax
        df_consumer
            met  id (model.id)  rxn
        df_node
            node  x  y  type  Tf
        df_distance
            i  j  d
        df_organism
            id  X0 biomass (rxn)

        max  sum_k sum_j ck' vkj
        vkj,ykj,sijl,epjl
        s.t. Sk vkj = 0                             (1)
             lk*ykj <= vkj <= uk*ykj                (2)
             sum_k ykj <= 1,  j in Nodes            (3)
             -vkjl_up <= sum_i s_ijl/dij + sum_p e_pjl/dpj, (4)
                k in Orgs, j in Nodes, l in Shared
             sum_j e_pjl <= sum_k vkex_pl,          (5)
                p in Nodes, l in Shared
             sum_j s_ijl <= smax_il,                (6)
                i in Sources, l in Shared
             sum_j ykj <= Xk0 + Xk0*sum_j mukj*Tfj, (7)
                k in Orgs
        """
        model = Model('stacked')
        # Only organism node locations
        locs = df_node[df_node.type != 'source']['node'].unique()

        # 1) Sk vkj = 0
        for mdl in models:
            for j in locs:
                clone_id = '%s_%s'%(mdl.id, j)
                mdlj = clone_model(mdl, model, suffix='_%s'%clone_id)
                self.model_dict[clone_id] = mdlj
        self.model = model

        # 2) lk*ykj <= vkj <= uk*ykj
        model_dict = self.model_dict
        for mdl_id,mdl in iteritems(model_dict):
            # mdl_id: modelk_nodej
            ykj = Variable('y_%s'%mdl_id, lower_bound=0, upper_bound=1)
            ykj.variable_kind = 'integer'
            mdl.add_reaction(ykj)
            for rxn in mdl.reactions:
                cons_lb = Constraint('binary_lb_%s'%rxn.id)
                cons_lb._bound = 0.
                cons_lb._constraint_sense = 'L'
                mdl.add_metabolites(cons_lb)
                rxn.add_metabolites({cons_lb:-1.}, combine=False)
                ykj.add_metabolites({cons_lb:rxn.lower_bound}, combine=False)
                cons_ub = Constraint('binary_ub_%s'%rxn.id)
                cons_ub._bound = 0.
                cons_ub._constraint_sense = 'L'
                mdl.add_metabolites(cons_ub)
                rxn.add_metabolites({cons_ub:1.}, combine=False)
                ykj.add_metabolites({cons_ub:-rxn.upper_bound}, combine=False)

        # 3) sum_k ykj <= 1,  j in Nodes
        organisms = [mdl.id for mdl in models]
        for locj in locs:
            cons = Constraint('sum_y_%s'%locj)
            cons._bound = 1.
            cons._constraint_sense = 'L'
            model.add_metabolites(cons)
            for org in organisms:
                ykj = model.reactions.get_by_id('y_%s_%s'%(org,locj))
                ykj.add_metabolites({cons:1.}, combine=False)

        # 4) -vkjl_up <= sum_i s_ijl/dij + sum_p e_pjl/dpj
        # 4) -vkjl_up - sum_i s_ijl/dij - sum_p e_pjl/dpj <= 0,
        #       k in Orgs, j in Nodes, l in Shared
        # svars = []  # Gather and add to model at the end
        # evars = []
        for ri,row in df_consumer.iterrows():
            met  = row['met']
            org  = row['id']
            rid  = row['rxn']
            for locj in locs:
                cons = Constraint('uptake_cap_%s_%s_%s'%(org,locj,met))
                cons._bound = 0.
                cons._constraint_sense = 'L'

                # Uptake rxn
                mdl_id = '%s_%s'%(org,locj)
                mdl = model_dict[mdl_id]
                rxn_id = '%s_%s_%s'%(rid,org,locj)
                ex_rxn = mdl.reactions.get_by_id(rxn_id)
                ex_rxn.add_metabolites({cons:-1.}, combine=False)

                # Primary sources
                df_src_l = df_source[ df_source['met']==met]
                df_primary = df_src_l[ df_src_l['source']=='primary']

                for si,srow in df_primary.iterrows():
                    loci = int(srow['node'])
                    dij  = df_distance[
                           (df_distance['i']==loci) & (df_distance['j']==locj)]['d'].iloc[0]
                    sid = 's_%s_%s_%s'%(loci, locj, met)
                    if model.reactions.has_id(sid):
                        sijl = model.reactions.get_by_id(sid)
                    else:
                        sijl = Variable(sid)
                        sijl.lower_bound = 0.
                        sijl.upper_bound = 1000.
                        model.add_reactions(sijl)
                    #svars.append(sijl)
                    sijl.add_metabolites({cons:-1./dij}, combine=False)

                # Cross-feed sources
                df_cross = df_src_l[ df_src_l['source']!='primary']
                for si,srow in df_cross.iterrows():
                    rid = srow['rxn']
                    org_x = srow['source']
                    for locp in locs:
                        if locp != locj:
                            dpj  = df_distance[
                                (df_distance['i']==locp) & (df_distance['j']==locj)]['d'].iloc[0]
                            eid ='e_%s_%s_%s'%(locp,locj,met)
                            if model.reactions.has_id(eid):
                                epjl = model.reactions.get_by_id(eid)
                            else:
                                epjl = Variable(eid)
                                epjl.lower_bound = 0.
                                epjl.upper_bound = 1000.
                                model.add_reactions(epjl)
                            # evars.append(epjl)
                            epjl.add_metabolites({cons:-1./dpj}, combine=False)

        # model.add_reactions(svars + evars)

        # 5) sum_j e_pjl <= sum_k vkex_pl, p in Nodes, l in Shared
        # 5) sum_j e_pjl - sum_k vkex_pl <= 0, p in Nodes, l in Shared
        df_cross_feed = df_source[~df_source['rxn'].isnull()]
        cross_mets = df_cross_feed['met'].unique()
        for met in cross_mets:
            df_cross = df_cross_feed[ df_cross_feed['met']==met]
            for locp in locs:
                cons = Constraint('crossfeed_capacity_%s_%s'%(locp, met))
                cons._bound = 0.
                cons._constraint_sense = 'L'
                model.add_metabolites(cons)
                for ir,row in df_cross.iterrows():
                    org = row['source']
                    rid = row['rxn']    # Exchange flux providing this met
                    # All orgs at p providing l
                    mdl_p_id = '%s_%s'%(org,locp)
                    mdl_p = model_dict[mdl_p_id]
                    rxn_id = '%s_%s_%s'%(rid,org,locp)
                    rxn = mdl_p.reactions.get_by_id(rxn_id)
                    rxn.add_metabolites({cons:-1.}, combine=False)
                # e_pjl: only cross-feed supplied
                for locj in locs:
                    if locj != locp:
                        e_id = 'e_%s_%s_%s'%(locp,locj,met)
                        if model.reactions.has_id(e_id):
                            e_pjl = model.reactions.get_by_id(e_id)
                            e_pjl.add_metabolites({cons:1.}, combine=False)

        # 6) sum_j s_ijl <= smax_il, i in Sources, l in Shared
        df_primary_feed = df_source[df_source['source']=='primary']
        primary_mets = df_primary_feed['met'].unique()
        for met in primary_mets:
            df_primary = df_primary_feed[ df_primary_feed['met']==met]
            for ir,row in df_primary.iterrows():
                loci = int(row['node'])
                vmax = row['vmax']
                cons = Constraint('source_capacity_%s_%s'%(loci,met))
                cons._bound = vmax
                cons._constraint_sense = 'L'
                model.add_metabolites(cons)
                for locj in locs:
                    s_id  = 's_%s_%s_%s'%(loci,locj,met)
                    s_ijl = model.reactions.get_by_id(s_id)
                    s_ijl.add_metabolites({cons:1.}, combine=False)

        # 7) sum_j ykj <= Xk0 + Xk0*sum_j mukj*Tfj
        # 7) sum_j ykj - Xk0*sum_j mukj*Tfj <= Xk0
        for oi,row in df_organism.iterrows():
            X0 = row['X0']
            org = row['id']
            mu_id = row['biomass']
            cons = Constraint('colonization_%s'%org)
            cons._bound = X0
            cons._constraint_sense = 'L'
            model.add_metabolites(cons)
            for locj in locs:
                Tf = df_node.loc[locj]['Tf']    # flow period is location-dependent
                rxn_id = '%s_%s_%s'%(mu_id,org,locj)
                rxn_mu = model.reactions.get_by_id(rxn_id)
                rxn_mu.add_metabolites({cons:-X0*Tf}, combine=False)
                yid = 'y_%s_%s'%(org,locj)
                ykj = model.reactions.get_by_id(yid)
                ykj.add_metabolites({cons:1.}, combine=False)



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
        model = self.model

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
        # Organisms
        org_ids = df_organism[col_org].unique()
        # Locations
        J = np.union1d(df_location['i'], df_location['j'])
        for mdl_id,mdl in iteritems(model_dict):
            ykj = [Variable('y_%s'%(mdl_id), lower_bound=0, upper_bound=1)]
            for yj in ykj:
                yj.variable_kind = 'integer'
            mdl.add_reactions(ykj)

        for _i,row in df_organism.iterrows():
            org_id = row['id']
            X0     = row['X0']
            j0     = row['j0']
            mu_id  = row['biomass']

            # ADD: sum_j ykj - Xk0*muk*Tc <= Xk0 
            cons_bio = Constraint('cons_biomass_%s'%org_id)
            cons_bio._bound = X0
            cons_bio._constraint_sense = 'L'
            for yj in ykj:
                yj.add_metabolites({cons_bio:1.}, combine=False)

            # ADD: lkj*ykj <= vkj <= ukj*ykj
            for locj in J:
                mdl    = model_dict["%s_%s"%(org_id, locj)]
                rxn_mu = mdl.reactions.get_by_id("%s_%s_%s"%(mu_id, org_id, locj))
                rxn_mu.add_metabolites({cons_bio:-X0*Tc}, combine=False)

                yj = model.reactions.get_by_id('y_%s_%s'%(org_id, locj))

                for rxn in mdl.reactions:
                    cons_lb = Constraint('binary_lb_%s_%s_%s'%(rxn.id, org_id, locj))
                    cons_lb._bound = 0.
                    cons_lb._constraint_sense = 'L'
                    rxn.add_metabolites({cons_lb:-1.}, combine=False)
                    yj.add_metabolites({cons_lb:rxn.lower_bound}, combine=False)

                    cons_ub = Constraint('binary_ub_%s_%s_%s'%(rxn.id, org_id, locj))
                    cons_ub._bound = 0.
                    cons_ub._constraint_sense = 'L'
                    rxn.add_metabolites({cons_ub:1.}, combine=False)
                    yj.add_metabolites({cons_ub:-rxn.upper_bound})

        #----------------------------------------------------
        # Inter-model linking constraints
        nutrients = df_nutrient[col_nutr].unique()
        df_nutr_src = pd.merge(df_nutrient, df_source, on=col_src)

        for locj in J:
            # ADD: sum_k ykj <= 1, j \in Locations
            cons_y = Constraint('cons_overlap_%s'%locj) # Only in the stacked model
            cons_y._bound = 1.
            cons_y._constraint_sense = 'L'
            model.add_metabolites([cons_y])
            for org_id in org_ids:
                ykj = model.reactions.get_by_id('y_%s_%s'%(org_id, locj))
                ykj.add_metabolites({cons_y:1.}, combine=False)

            # ADD: vklj >= sum_i vtot_lij/dij + sum_{p!=j} vsynth_lp/djp,  l \in Nutrients, j\in J
            #      vklj - sum_{p!=j} vsynth_lp/djp >= sum_i vtot_lij/dij,  l \in Nutrients, j\in J
            # All sources providing this nutrient
            for nutr in nutrients:
                # All primary sources
                dflj = df_nutr_src[ (df_nutr_src[col_nutr]==nutr) &
                        (df_nutr_src[col_loc]==locj)]
                sum_vtot_d = sum(dflj[col_vtot]/dflj[col_dist])
                for ni,nrow in dflj.iterrows():
                    src = nrow[col_src]
                    dfe = df_exchange[df_exchange[col_nutr]==src]
                    for ei,erow in dfe.iterrows():
                        ex_id = erow['rxn']
                        mdl   = model_dict[erow[col_org]]
                        cons_uptake_jl = Constraint('cons_uptake_%s_%s_%s_%s'%(nutr,src,org_id,locj))
                        cons_uptake_jl._constraint_sense = 'G'
                        cons_uptake_jl._bound = sum_vtot_d
                        model.add_metabolites(cons_uptake_jl)

                        ex_rxn = mdl.reactions.get_by_id(ex_id)
                        ex_rxn.add_metabolites({cons_uptake_jl:1.}, combine=False)

                        # Add contribution by other producers
                        for locp in J:
                            if locp != locj:
                                d_jp = df_location[(df_location.i==locj) &
                                        (df_location.j==locp)]['d'].iloc[0]
                                # Check which mdls at other locs can make this nutrient
                                for org_id in org_ids:
                                    ex_id = "EX_%s_%s_%s"%(nutr,org_id,locp)
                                    if model.reactions.has_id(ex_id):
                                        rxn_synth = model.reactions.get_by_id(ex_id)
                                        rxn_synth.add_metabolites({cons_uptake_jl:-1./d_jp}, combine=False)

        #----------------------------------------------------
        # Check all exchangable metabolites


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
