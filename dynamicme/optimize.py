#============================================================
# File optimize.py
#
# class  
#
# MI(N)LP methods 
#
# Laurence Yang, SBRG, UCSD
#
# 02 Feb 2018:  first version
#============================================================

from six import iteritems
from cobra.core.Solution import Solution
from cobra import Reaction, Metabolite, Model

import copy
import sys

solver_apis = ['gurobi_solver', 'cplex_solver']

for api in solver_apis:
    try:
        lib = __import__(api)
    except:
        print sys.exc_info()
    else:
        globals()[api] = lib

class Variable(Reaction):
    pass

class Constraint(Metabolite):
    pass


class Optimizer(object):
    """
    Methods for bilevel opt and others
    """
    def __init__(self, mdl, objective_sense='maximize'):
        self.mdl = mdl
        self.objective_sense = objective_sense

    def add_duality_gap_constraint(self, clear_obj=False):
        """
        Add duality gap as a constraint to current model
        Inputs:
        clear_obj : clear the new model's objective function
        """
        #   max     c'x
        #   s.t.    c'x - wa*b - wu*u + wl*l = 0
        #           Ax [<=>] b
        #           wa*A + wu - wl = c'
        #           l <= x <= u
        #           wl, wu >= 0
        #           wa >= 0 if ai*x <= bi
        #           wa <= 0 if ai*x >= bi
        #           wa \in R if ai*x == bi

        primal  = self.mdl
        dual = self.make_dual()
        mdl = Model('duality_gap')

        #----------------------------------------------------
        # Add Primal variables and constraints
        cons_gap = Constraint('duality_gap')
        cons_gap._constraint_sense = 'E'
        cons_gap._bound = 0.

        for rxn in primal.reactions:
            var = Variable(rxn.id)
            mdl.add_reaction(var)
            var.lower_bound = rxn.lower_bound
            var.upper_bound = rxn.upper_bound
            var.objective_coefficient = rxn.objective_coefficient
            for met,s in iteritems(rxn.metabolites):
                cons = Constraint(met.id)
                cons._constraint_sense = met._constraint_sense
                cons._bound = met._bound
                var.add_metabolites({cons:s})

            # Add duality gap, dual variables, and dual constraints
            if rxn.objective_coefficient != 0:
                var.add_metabolites({cons_gap:rxn.objective_coefficient})


        for rxn in dual.reactions:
            dvar = Variable(rxn.id)
            mdl.add_reaction(dvar)
            dvar.lower_bound = rxn.lower_bound
            dvar.upper_bound = rxn.upper_bound
            dvar.objective_coefficient = rxn.objective_coefficient
            dvar.add_metabolites({cons_gap:-rxn.objective_coefficient})

        #----------------------------------------------------
        # Add dual constraints
        for met in dual.metabolites:
            cons_dual = Constraint(met.id)
            cons_dual._constraint_sense = met._constraint_sense
            cons_dual._bound = met._bound
            for rxn in met.reactions:
                dvar = mdl.reactions.get_by_id(rxn.id)
                dvar.add_metabolites({cons_dual:rxn.metabolites[met]})

        return mdl

    def make_dual(self, LB=-1000, UB=1000):
        """
        Return dual of current model
        """
        mdl  = self.mdl
        objective_sense=self.objective_sense
        dual = Model('dual')
        # Primal:
        #   max     c'x
        #   s.t.    Ax [<=>] b
        #           l <= x <= u
        #
        # Dual:
        #   min     wa*b + wu*u - wl*l
        #   s.t.    wa*A + wu - wl = c'
        #           wl, wu >= 0

        wa_dict = {}

        for met in mdl.metabolites:
            wa = Variable('wa_'+met.id)
            wa_dict[met.id] = wa
            if met._constraint_sense == 'E':
                wa.lower_bound = LB
            else:
                wa.lower_bound = 0.
            if met._constraint_sense == 'G':
                wa.objective_coefficient = -met._bound
            else:
                wa.objective_coefficient = met._bound
        dual.add_reactions(wa_dict.values())

        wl_dict = {}
        wu_dict = {}
        for rxn in mdl.reactions:
            wl = Variable('wl_'+rxn.id)
            wu = Variable('wu_'+rxn.id)
            wl.lower_bound = 0.
            wu.lower_bound = 0.
            wl_dict[rxn.id] = wl
            wu_dict[rxn.id] = wu
            wl.objective_coefficient = -rxn.lower_bound
            wu.objective_coefficient = rxn.upper_bound

            #cons = Constraint('cons_'+rxn.id)
            cons = Constraint(rxn.id)
            if rxn.lower_bound < 0:
                cons._constraint_sense = 'E'
            else:
                cons._constraint_sense = 'G'
            cons._bound = rxn.objective_coefficient
            wl.add_metabolites({cons:-1.})
            wu.add_metabolites({cons:1.})

            for met,s in iteritems(rxn.metabolites):
                wa = wa_dict[met.id]
                if met._constraint_sense == 'G':
                    wa.add_metabolites({cons:-s})
                else:
                    wa.add_metabolites({cons:s})

        # Remember to minimize the problem
        dual.add_reactions(wl_dict.values())
        dual.add_reactions(wu_dict.values())

        return dual


    def make_disjunctive_primal_dual(self, mdl, a12_dict, M=1e4):
        """
        Make some constraints disjunctive for both primal and dual
        Inputs:
        a12_dict : {(met,rxn):(a1, a2)}
        """
        # Primal:
        #   max     c'x
        #   s.t.    sum_j (a1ij*(1-yij)*xj + a2ij*yij*xj) [<=>] bi
        #           l <= x <= u
        #           yij \in {0,1}
        #
        # Dual:
        #   min     wa*b + wu*u - wl*l
        #   s.t.    sum_i (wai*a1ij*(1-yij) + wai*a2ij*yij) + wuj - wlj = cj
        #           wl, wu >= 0
        #           yij \in {0,1}
        #
        #   max     c'x
        #   s.t.    c'x - wa*b - wu*u + wl*l = 0
        #           sum_j (a1ij*(1-yij)*xj + a2ij*yij*xj) [<=>] bi
        #           sum_i (wai*a1ij*(1-yij) + wai*a2ij*yij) + wuj - wlj = cj
        #           l <= x <= u
        #           wl, wu >= 0
        #           yij \in {0,1}
        #
        #   max     c'x
        #   s.t.    c'x - wa*b - wu*u + wl*l = 0
        #           sum_j a1ij*xj - a1ij*zij + a2ij*zij [<=>] bi
        #           l*yij <= zij <= u*yij
        #           -M*(1-yij) <= zij - xj <= M*(1-yij)
        #           sum_i a1ij*wai - a1ij*zaij + a2ij*zaij + wu - wl = cj
        #           wal*yij <= zaij <= wau*yij
        #           -M*(1-yij) <= zaij - wai <= M*(1-yij)
        #           l <= x <= u
        #           wl, wu >= 0
        #           yij \in {0,1}


        for met_rxn, a12 in iteritems(a12_dict):
            met = met_rxn[0]
            rxn = met_rxn[1]
            a1  = a12[0]
            a2  = a12[1]

            yij = Variable('binary_%s_%s'%(met.id,rxn.id))
            yij.variable_kind = 'integer'
            yij.lower_bound = 0.
            yij.upper_bound = 1.
            try:
                mdl.add_reaction(yij)
            except ValueError:
                yij = mdl.reactions.get_by_id(yij.id)
            zij = Variable('z_%s_%s'%(met.id,rxn.id))
            zij.lower_bound = rxn.lower_bound
            zij.upper_bound = rxn.upper_bound
            try:
                mdl.add_reaction(zij)
            except:
                zij = mdl.reactions.get_by_id(zij.id)
            # Used to be:
            #   sum_j aij*xj [<=>] bi
            # Change to:
            #   sum_j a1ij*xj - a1ij*zij + a2ij*zij [<=>] bi
            rxn._metabolites[met] = a1
            zij.add_metabolites({met:-a1+a2}, combine=False)
            # Add: l*yij <= zij <= u*yij
            cons_zl = Constraint('z_l_%s_%s'%(met.id,rxn.id))
            cons_zl._constraint_sense = 'L'
            cons_zl._bound = 0.
            yij.add_metabolites({cons_zl:rxn.lower_bound}, combine=False)
            zij.add_metabolites({cons_zl:-1.}, combine=False)
            cons_zu = Constraint('z_u_%s_%s'%(met.id,rxn.id))
            cons_zu._constraint_sense = 'L'
            cons_zu._bound = 0.
            yij.add_metabolites({cons_zu:-rxn.upper_bound}, combine=False)
            zij.add_metabolites({cons_zu:1.}, combine=False)
            # Add: -M*(1-yij) <= zij - xj <= M*(1-yij)
            cons_zl = Constraint('z_M_l_%s_%s'%(met.id,rxn.id))
            cons_zl._constraint_sense = 'L'
            cons_zl._bound = M
            rxn.add_metabolites({cons_zl:1.}, combine=False)
            zij.add_metabolites({cons_zl:-1.}, combine=False)
            yij.add_metabolites({cons_zl:M}, combine=False)
            cons_zu = Constraint('z_M_u_%s_%s'%(met.id,rxn.id))
            cons_zu._constraint_sense = 'L'
            cons_zu._bound = M
            rxn.add_metabolites({cons_zu:-1.}, combine=False)
            zij.add_metabolites({cons_zu:1.}, combine=False)
            yij.add_metabolites({cons_zu:M}, combine=False)
            # Used to be:
            # wa*A + wu - wl = c'
            # Change to:
            # sum_i a1ij*wai - a1ij*zaij + a2ij*zaij + wu - wl = cj
            cons = mdl.metabolites.get_by_id(rxn.id)
            wa = mdl.reactions.get_by_id('wa_'+met.id)
            wl = mdl.reactions.get_by_id('wl_'+rxn.id)
            wu = mdl.reactions.get_by_id('wu_'+rxn.id)
            zaij = Variable('za_%s_%s'%(met.id,rxn.id))
            zaij.lower_bound = wa.lower_bound
            zaij.upper_bound = wa.upper_bound
            try:
                mdl.add_reaction(zaij)
            except ValueError:
                zaij = mdl.reactions.get_by_id(zaij.id)
            wa._metabolites[cons] = a1
            zaij.add_metabolites({cons:-a1+a2}, combine=False)
            # wal*yij <= zaij <= wau*yij
            cons_zl = Constraint('za_l_%s_%s'%(met.id,rxn.id))
            cons_zl._constraint_sense = 'L'
            cons_zl._bound = 0.
            yij.add_metabolites({cons_zl:wa.lower_bound}, combine=False)
            zaij.add_metabolites({cons_zl:-1.}, combine=False)
            cons_zu = Constraint('za_u_%s_%s'%(met.id,rxn.id))
            cons_zu._constraint_sense = 'L'
            cons_zu._bound = 0.
            yij.add_metabolites({cons_zu:-wa.upper_bound}, combine=False)
            zaij.add_metabolites({cons_zu:1.}, combine=False)
            # -M*(1-yij) <= zaij - wai <= M*(1-yij)
            cons_zl = Constraint('za_M_l_%s_%s'%(met.id,rxn.id))
            cons_zl._constraint_sense = 'L'
            cons_zl._bound =  M
            wa.add_metabolites({cons_zl:1.}, combine=False)
            yij.add_metabolites({cons_zl:M}, combine=False)
            zaij.add_metabolites({cons_zl:-1.}, combine=False)
            cons_zu = Constraint('za_M_u_%s_%s'%(met.id,rxn.id))
            cons_zu._constraint_sense = 'L'
            cons_zu._bound = M
            wa.add_metabolites({cons_zu:-1.}, combine=False)
            yij.add_metabolites({cons_zu:M}, combine=False)
            zaij.add_metabolites({cons_zu:1.}, combine=False)

        return mdl

    def stack_disjunctive(self, mdls):
        """
        Stack models in mdls into disjunctive program.
        keff params are coupled by default
        """
        #   max     c'yi
        #   xk,wk,zk,yi
        #   s.t.    c'xk - wak*b - wuk*u + wlk*l = 0
        #           sum_j a1ij*xkj - a1ij*zij + a2ij*zij [<=>] bi
        #           l*yij <= zij <= u*yij
        #           -M*(1-yij) <= zij - xj <= M*(1-yij)
        #           sum_i a1ij*wai - a1ij*zaij + a2ij*zaij + wu - wl = cj
        #           wal*yij <= zaij <= wau*yij
        #           -M*(1-yij) <= zaij - wai <= M*(1-yij)
        #           l <= x <= u
        #           wl, wu >= 0
        #           yij \in {0,1}



