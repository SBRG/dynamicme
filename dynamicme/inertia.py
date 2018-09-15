#============================================================
# File dynamic.py
#
# class  DynamicME
# class  ParamOpt
# class  MMmodel
#   def  get_cplx_concs
#
# Class & methods for dynamic FBA with ME models.
#
# Laurence Yang, SBRG, UCSD
#
# 18 Mar 2016:  first version
# 28 Sep 2017:  migrated to separate module
#============================================================

from six import iteritems
from cobra.core.Solution import Solution
from cobra import DictList
from cobra import Reaction, Metabolite
from cobrame import mu
from cobrame import Constraint
from cobrame import MetabolicReaction, TranslationReaction, MEReaction
from cobrame import MEModel
from cobrame import Complex, ComplexFormation, GenericFormationReaction

from qminos.qnonlinme import ME_NLP
from qminos.me1 import ME_NLP1

from dynamicme.model import ComplexDegradation, PeptideDegradation

from cobrawe.me1tools import ME1tools
from sympy import Basic

import sympy
import numpy as np
import copy as cp
import pandas as pd
import time
import warnings
import cobra
import cobrame


#============================================================
# Error functions used in ParamOpt
def errfun_sae(x,y):
    """
    Sum of absolute errors
    """
    return sum(abs(x-y))


def errfun_sse(x,y):
    """
    Sum of absolute errors
    """
    return sum((x-y)**2)


def errfun_kld(x,y):
    """
    Kullback-Leibler divergence (relative entropy)
    sum( x * log(x/y) )
    """
    return sum( x * np.log( x/y ) )
#============================================================


class DynamicME(object):
    """
    Composite class of ME_NLP containing dynamic ME methods
    """

    def __init__(self, me, growth_key='mu', growth_rxn='biomass_dilution',
                 nlp_compat=False, exchange_one_rxn=None):
        self.me = me
        self.nlp_compat = nlp_compat
        is_me2 = isinstance(me, cobrame.core.MEModel.MEModel)
        if exchange_one_rxn is None:
            exchange_one_rxn = is_me2
        self.exchange_one_rxn = exchange_one_rxn

        if isinstance(me, cobrame.core.MEModel.MEModel):
            if nlp_compat:
                self.solver = ME_NLP(me, growth_key=growth_key, growth_rxn=growth_rxn)
            else:
                self.solver = ME_NLP1(me, growth_key=growth_key)
            self.growth_rxn = growth_rxn
        else:
            self.solver = ME_NLP1(me)
            self.growth_rxn = self.solver.growth_key
            if not hasattr(me,'translation_data'):
                me1tools = ME1tools(me)
                me1tools.make_translation_data()
        self.me_nlp = self.solver   # for backward compat

        self.mm_model = None    # Used for proteome-constrained sub simulation


    def __getattr__(self, attr):
        return getattr(self.solver, attr)

    def simulate_batch(self, T, c0_dict, X0, dt=0.1,
                       o2_e_id='o2_e', o2_head=0.21, kLa=7.5,
                       conc_dep_fluxes = False,
                       extra_rxns_tracked=[],
                       prec_bs=1e-6,
                       ZERO_CONC = 1e-3,
                       lb_dict={},
                       ub_dict={},
                       proteome_has_inertia=False,
                       cplx_conc_dict0={},
                       mm_model = None,
                       basis=None,
                       no_nlp=False,
                       verbosity=2,
                       LB_DEFAULT=-1000.,
                       UB_DEFAULT=1000.,
                       throttle_near_zero=True):
        """
        result = simulate_batch()

        Solve dynamic ME problem
        [Arguments]
        T:  batch time
        c0_dict: initial extracellular concentration dict
        X0: initial biomass density
        o2_e_id: oxygen (extracellular) metabolite ID
        o2_head: headspace O2 concentration
        kLa: mass transfer coefficient for O2
        dt: time step (h)
        conc_dep_fluxes: are uptake fluxes concentration dependent?
        prec_bs: precision of mu for bisection
        ZERO_CONC: (in mM) if below this concentration, consider depleted
        proteome_has_inertia: if True, track protein concentrations and
                              constrain catalyzed flux (default: False)
        cplx_conc_dict0: (initial) protein concentration dict.
                        Only the complexes in this dict will be constrained for
                        the rest of the simulation.
        mm_model : the metabolism and macromolecule model used to implement
                   proteome inertia constraints

        [Output]
        result
        ----------------------------------------------------
        Batch equations:
        dX/dt = mu*X
        dc/dt = A*v*X
        """
        # If uptake rate independent of concentration,
        # only recompute uptake rate once a substrate
        # depleted
        me = self.me
        solver = self.solver
        is_me2 = isinstance(me, cobrame.core.MEModel.MEModel)
        exchange_one_rxn = self.exchange_one_rxn

        # If constraining proteome "inertia" need extra constraints
        cplx_conc_dict = dict(cplx_conc_dict0)
        if proteome_has_inertia:
            # Initial proteome availability should be unconstrained
            # self.add_inertia_constraints(cplx_conc_dict)
            if mm_model is None:
                mm_model = MMmodel(solver, cplx_conc_dict)

        # Initialize concentrations & biomass
        conc_dict = c0_dict.copy()
        #prot_dict = prot0_dict.copy()
        X_biomass = X0
        mu_opt = 0.
        x_dict = None
        if exchange_one_rxn:
            ex_flux_dict = {self.get_exchange_rxn(metid).id:0. for metid in conc_dict.keys()}
        else:
            ex_flux_dict = {}
            for metid in conc_dict.keys():
                try:
                    rxn = self.get_exchange_rxn(metid, 'source', exchange_one_rxn)
                    ex_flux_dict[rxn.id] = 0.
                except ValueError:
                    pass
                try:
                    rxn = self.get_exchange_rxn(metid, 'sink', exchange_one_rxn)
                    ex_flux_dict[rxn.id] = 0.
                except ValueError:
                    pass

        #rxn_flux_dict = {rxn.id:0. for rxn in extra_rxns_tracked}
        rxn_flux_dict = {(r.id if hasattr(r,'id') else r):0. for r in extra_rxns_tracked}

        t_sim = 0.

        times = [t_sim]
        conc_profile = [conc_dict.copy()]
        cplx_profile = [cplx_conc_dict.copy()]
        #prot_profile = [prot_dict.copy()]
        biomass_profile = [X_biomass]
        ex_flux_profile = [ex_flux_dict.copy()]
        rxn_flux_profile= [rxn_flux_dict.copy()]

        iter_sim = 0
        recompute_fluxes = True     # In first iteration always compute
        while t_sim < T:
            # Determine available substrates given concentrations
            for metid,conc in conc_dict.iteritems():
                try:
                    ex_rxn = self.get_exchange_rxn(metid, exchange_one_rxn=exchange_one_rxn,
                                                   direction='source')
                    if conc <= ZERO_CONC:
                        if verbosity >= 1:
                            print 'Metabolite %s depleted.'%(metid)
                        if exchange_one_rxn:
                            lb0 = ex_rxn.lower_bound
                            lb1 = 0.
                            if lb1 != lb0:
                                recompute_fluxes = True
                            ex_rxn.lower_bound = 0.
                        else:
                            ub0 = ex_rxn.upper_bound
                            ub1 = 0.
                            if ub1 != ub0:
                                recompute_fluxes = True
                            ex_rxn.upper_bound = 0.
                    else:
                        # (re)-open exchange whenever concentration above
                        # threshold since, e.g., secreted products can be 
                        # re-consumed, too.
                        if verbosity >= 1:
                            print 'Metabolite %s available.'%(metid)
                        if exchange_one_rxn:
                            lb0 = ex_rxn.lower_bound
                            if lb_dict.has_key(ex_rxn.id):
                                lb1 = lb_dict[ex_rxn.id]
                            else:
                                if verbosity >= 1:
                                    print 'Using default LB=%g for %s'%(LB_DEFAULT, ex_rxn.id)
                                lb1 = LB_DEFAULT
                            if lb1 != lb0:
                                recompute_fluxes = True
                            ex_rxn.lower_bound = lb1
                        else:
                            ub0 = ex_rxn.upper_bound
                            if ub_dict.has_key(ex_rxn.id):
                                ub1 = ub_dict[ex_rxn.id]
                            else:
                                if verbosity >= 1:
                                    print 'Using default UB=%g for %s'%(UB_DEFAULT, ex_rxn.id)
                                ub1 = UB_DEFAULT
                            if ub1 != ub0:
                                recompute_fluxes = True
                            ex_rxn.upper_bound = ub1
                except:
                    if verbosity >= 2:
                        print 'No uptake rxn found for met:', metid

            # Recompute fluxes if any rxn bounds changed, which triggers
            # recompute_fluxes flag
            if recompute_fluxes:
                # Compute ME
                if verbosity >= 1:
                    print 'Computing new uptake rates'
                if self.nlp_compat:
                    if no_nlp:
                        mu_opt, hs_bs, x_opt, cache_opt = solver.bisectmu(prec_bs, basis=basis,
                                                                          verbosity=verbosity)
                    else:
                        x_opt, stat, hs_bs = solver.solvenlp(prec_bs, basis=basis)
                        cache_opt = None
                    if me.solution is None:
                        mu_opt = 0.
                    else:
                        mu_opt = me.solution.f
                else:
                    mu_opt, hs_bs, x_opt, cache_opt = solver.bisectmu(prec_bs, basis=basis,
                                                                      verbosity=verbosity)

                if proteome_has_inertia:
                    if verbosity >= 2:
                        print 'Constraining catalyzed rxns by proteome from prev time point'
                        #####################################
                        # **UPDATE PROTEOME CONSTRAINTS FOR NEXT TIME STEP**
                        # Calculate total complex concentration at this iteration
                        cplx_conc_dict = self.calc_cplx_concs(complexes_constrained,
                                me.solution.x_dict, mu_opt)

                        # Use to constrain fluxes at next iteration
                        self.update_inertia_constraints(cplx_conc_dict)

                        #####################################
                basis = hs_bs
                if me.solution is None:
                    x_dict = None
                else:
                    x_dict = me.solution.x_dict

            # Update biomass for next time step
            X_biomass_prime = X_biomass + mu_opt*X_biomass*dt
            # Update concentrations
            ex_flux_dict = {}
            conc_dict_prime = conc_dict.copy()
            #prot_dict_prime = prot_dict.copy()
            cplx_conc_dict_prime = cplx_conc_dict.copy()
            reset_run = False

            for metid, conc in conc_dict.iteritems():
                v = 0.
                # If ME 1.0, EX_ split into source and sink
                if exchange_one_rxn:
                    rxn = self.get_exchange_rxn(metid)
                    if x_dict is not None:
                        v = me.solution.x_dict[rxn.id]              # mmol/gDW/h
                    ex_flux_dict[rxn.id] = v
                else:
                    v_in  = 0.
                    v_out = 0.
                    try:
                        rxn_in  = self.get_exchange_rxn(metid, 'source', exchange_one_rxn)
                        v_in  = me.solution.x_dict[rxn_in.id]
                        ex_flux_dict[rxn_in.id] = v_in
                    except:
                        pass
                    try:
                        rxn_out = self.get_exchange_rxn(metid, 'sink', exchange_one_rxn)
                        v_out = me.solution.x_dict[rxn_out.id]
                        ex_flux_dict[rxn_out.id] = v_out
                    except:
                        pass
                    v = v_out - v_in

                if metid is not o2_e_id:
                    conc_dict_prime[metid] = conc + v*X_biomass_prime*dt    # mmol/L = mmol/gDW/h * gDW/L * h
                    if throttle_near_zero:
                        if conc_dict_prime[metid] < (ZERO_CONC - prec_bs):
                            # Set flag to negate this run and recompute fluxes again with a new lower bound if any of the
                            # metabolites end up with a negative concentration
                            if verbosity >= 1:
                                print metid, "below threshold, reset run flag triggered"
                            reset_run = True
                            lb_dict[rxn.id] = min(-conc_dict[metid] / (X_biomass_prime * dt), 0.)
                            if verbosity >= 1:
                                print 'Changing lower bounds %s to %.3f' % (metid, lb_dict[rxn.id])
                        elif -v*X_biomass_prime*dt > conc_dict_prime[metid]/2:
                            ### Update lower bounds as concentration is nearing 0
                            lb_dict[rxn.id] = min(-conc_dict_prime[metid]/(X_biomass*dt), 0.)
                            if verbosity >= 1:
                                print 'Changing lower bounds %s to %.3f' %(metid, lb_dict[rxn.id])
                else:
                    # Account for oxygen diffusion from headspace into medium
                    conc_dict_prime[metid] = conc + (v*X_biomass_prime + kLa*(o2_head - conc))*dt

            #------------------------------------------------
            # Update complex concentrations for next time step
            #------------------------------------------------
            """
            for a cell:
                Ej(t+1) = Ej(t) + v_formation*dt
                mmol/gDW = mmol/gDW + mmol/gDW/h * h
            """
            for cplx_id, conc in iteritems(cplx_conc_dict):
                cplx = me.metabolites.get_by_id(cplx_id)
                #data = me.complex_data.get_by_id(cplx_id)
                #data.formation
                v_cplx_net = 0.
                #for rxn in cplx.reactions:


                v_formation = me.solution.x_dict[rxn_form.id]
                cplx_conc_dict_prime[cplx_id] = conc + v_cplx_net*dt


            # Reset the run if the reset_run flag is triggered, if not update the new biomass and conc_dict
            if reset_run:
                if verbosity >= 1:
                    print "Resetting run"
                continue  # Skip the updating of time steps and go to the next loop while on the same time step
            else:
                X_biomass = X_biomass_prime
                conc_dict = conc_dict_prime.copy()
                cplx_dict = cplx_conc_dict_prime.copy()
                #prot_dict = prot_dict_prime.copy()

            ### Extra fluxes tracked
            for rxn in extra_rxns_tracked:
                v = 0.
                rid = rxn.id if hasattr(rxn,'id') else rxn
                if x_dict is not None:
                    v = x_dict[rid]
                rxn_flux_dict[rid] = v

            # ------------------------------------------------
            # Move to next time step
            t_sim = t_sim + dt
            iter_sim = iter_sim + 1
            times.append(t_sim)
            conc_profile.append(conc_dict.copy())
            biomass_profile.append(X_biomass)
            ex_flux_profile.append(ex_flux_dict.copy())
            rxn_flux_profile.append(rxn_flux_dict.copy())
            # Save protein concentrations
            cplx_profile.append(cplx_dict.copy())

            # Reset recompute_fluxes to false
            recompute_fluxes = False

            # ------------------------------------------------
            # Print some results
            if verbosity >= 1:
                print 'Biomass at t=%g: %g'%(t_sim, X_biomass)
                print 'Concentrations:', conc_dict


        result = {'biomass':biomass_profile,
                  'concentration':conc_profile,
                  'ex_flux':ex_flux_profile,
                  'rxn_flux':rxn_flux_profile,
                  'time':times,
                  'basis':basis,
                  'prot_concs':prot_concs}

        self.result = result

        return result


    def get_exchange_rxn(self, metid, direction='both', exchange_one_rxn=None):
        """
        Get exchange flux for metabolite with id metid
        """
        me = self.me
        met = me.metabolites.get_by_id(metid)
        is_me2 = isinstance(me, cobrame.core.MEModel.MEModel)
        if exchange_one_rxn is None:
            exchange_one_rxn = is_me2

        ex_rxn = None

        if exchange_one_rxn:
            ex_rxns = [rxn for rxn in met.reactions if
                       len(rxn.metabolites)==1 and rxn.metabolites[met]==-1.]
            if len(ex_rxns) < 1:
                raise ValueError('No exchange rxn for metabolite %s'%metid)
            else:
                ex_rxn = ex_rxns[0]
        else:
            ### If ME 1.0
            # Get the source or sink rxn?
            if direction is 'source':
                ex_rxns = [rxn for rxn in met.reactions if
                           len(rxn.metabolites)==1 and rxn.metabolites[met]==1.]
            elif direction is 'sink':
                ex_rxns = [rxn for rxn in met.reactions if
                           len(rxn.metabolites)==1 and rxn.metabolites[met]==-1.]
            else:
                raise ValueError("Direction must equal 'sink' or 'source' for ME 1.0 models.")

            if len(ex_rxns) < 1:
                raise ValueError('No exchange rxn for metabolite %s'%metid)
            else:
                ex_rxn = ex_rxns[0]

        return ex_rxn



    def simulate_fed_batch(self, T, c0_dict, X0, cplx_conc_dict0,
                       feed_schedule,
                       dt=0.1, H=None,
                       o2_e_id='o2_e', o2_head=0.21, kLa=7.5,
                       conc_dep_fluxes = False,
                       extra_rxns_tracked=[],
                       prec_bs=1e-6,
                       ZERO_CONC = 1e-3,
                       lb_dict={},
                       ub_dict={},
                       proteome_has_inertia=True,
                       mm_model = None,
                       basis=None,
                       no_nlp=False,
                       verbosity=2,
                       solver_verbosity=0,
                       LB_DEFAULT=-1000.,
                       UB_DEFAULT=1000.,
                       MU_MIN=0.,
                       MU_MAX=2):
        """
        result = simulate_fed_batch()

        Solve dynamic ME problem with proteome delay
        [Arguments]
        T:  batch time
        c0_dict: initial extracellular concentration dict
        X0: initial biomass density
        feed_schedule: feed schedule (amount added),
            dict = {time: {met: {'conc':concentration, 'vol':volume}}}
        o2_e_id: oxygen (extracellular) metabolite ID
        o2_head: headspace O2 concentration
        kLa: mass transfer coefficient for O2
        dt: time step (h)
        H:  prediction horizon. Default=None. In which case, sets equal to dt
        conc_dep_fluxes: are uptake fluxes concentration dependent?
        prec_bs: precision of mu for bisection
        ZERO_CONC: (in mM) if below this concentration, consider depleted
        proteome_has_inertia: if True, track protein concentrations and
                              constrain catalyzed flux (default: False)
        cplx_conc_dict0: (initial) protein concentration dict.
                        Only the complexes in this dict will be constrained for
                        the rest of the simulation.
        mm_model : the metabolism and macromolecule model used to implement
                   proteome inertia constraints

        [Output]
        result
        """


    def simulate_batch_par(self):
        """
        result = simulate_batch()

        [See simulate_batch]. Parallel version.
        Caution: requires considerable amount of RAM.
        """


    def change_uptake_kinetics(self, transport_classes={'PTS'}):
        """
        Update PTS (and other transporter) keff as function of extracellular metabolite concentration.
        Need to recompile the first time after doing this.

        Chassagnole PTS:
        r_PTS = 
            rmax_PTS * x_glc_e *x_pep_c/x_pyr_c
        ---------------------------------------------------------------------------------------
            Kpts_a1 + Kpts_a2*x_pep_c/x_pyr_c + Kpts_a3*x_glc_e + x_glc_e*x_pep_e/x_pyr_e)( 1+x_g6p^nPTS_g6p / Kpts_g6p)

        where rmax_PTS = kcat * [PTS].
        Thus, given metabolite concentrations, we derive a keff,
        such that v_PTS = keff(x_glc_e)*[PTS].
        We approximate x_pyr and x_pep and make x_glc_e the sole variable in computing keff.
        """
        # 
        me = self.me

        transport_classesL  = [c.lower() for c in transport_classes]
        if 'PTS' in transport_classesL:
            rxns_pts = me.reactions.query('ptspp')
            # Substitute concentration-dependent keff for each substrate


    def get_dilution_dict(self, cplx, extra_dil_prefix='extra_dilution_',
            excludes=['damage_','demetallation_'],
            rxn_types=[MetabolicReaction, TranslationReaction]):
        """
        get_dilution_dict
        Get total dilution for this rxn = sum_j vuse + extra_dilution
        """
        me = self.me

        # Just want the coefficient on mu (1/keff). Then, multiply mu back on.
        # I.e., don't want mu/keff + 1, etc. The +1 part does not contribute to dilution.
        # vdil = mu/keff * v
        dil_dict = {r:-r.metabolites[cplx].coeff(mu)*mu for r in cplx.reactions if
                r.metabolites[cplx]<0 and
                hasattr(r.metabolites[cplx],'subs') and
                any([isinstance(r,t) for t in rxn_types]) and
                all([s not in r.id for s in excludes])}
        rid_extra_dil = extra_dil_prefix + cplx.id

        # extra_dilution is just an extra sink for unused protein
        if me.reactions.has_id(rid_extra_dil):
            rxn = me.reactions.get_by_id(rid_extra_dil)
            dil_dict[rxn] = -rxn.metabolites[cplx]

        # Add explicit dilution rxns, too
        for rxn in cplx.reactions:
            if 'dilution_' in rxn.id and rxn.metabolites[cplx]<0:
                dil_dict[rxn] = -rxn.metabolites[cplx]

        return dil_dict

    def calc_proteome(self, mu_fix):
        """
        Get initial proteome concentration. 
        """
        me = self.me

        if me.solution is None:
            raise Exception('No solution exists. Solve the model for at least one time step first!')

        prot_conc_dict = {}
        for data in me.complex_data:
            # Sum up contribution from all enzyme-using rxns for this enzyme
            cplx = data.complex
            vdil_tot = self.calc_dilution(cplx, mu_fix)
            e_tot = vdil_tot / mu_fix
            prot_conc_dict[cplx] = float(e_tot)

        return prot_conc_dict

    def add_inertia_constraints(self, cplx_conc_dict={}, csense='L'):
        """
        add_inertia_constraints(self, cplx_conc_dict, csense='L')

        Inputs
        cplx_conc_dict : {cplx.id : concentration}.
                         Sets proteome inertia unconstrained if cplx.id not in dict.

        Add proteome inertia constraints

        Formulation:
        vj(t+1) <= keff_j*Ej(t)

        Ej(t) [mmol/gDW] is the enzyme concentration at timestep t
        """
        me = self.me
        solver = self.solver

        # Diluted complexes
        for cplx_id, conc in iteritems(cplx_conc_dict):
            cplx = me.metabolites.get_by_id(cplx_id)
            # Include cases like ribosome, which catalyzes but 
            # not MetabolicReactions
            for rxn in cplx.reactions:
                stoich = rxn.metabolites[cplx]
                if hasattr(stoich,'subs'):
                    keff = mu / stoich
                    # Add constraint
                    cons_id = 'cons_rate_'+rxn.id
                    if me.metabolites.has_id(cons_id):
                        cons = me.metabolites.get_by_id(cons_id)
                    else:
                        cons = Constraint(cons_id)
                        me.add_metabolites(cons)
                    cons._constraint_sense = csense
                    cons._bound = keff*cplx_conc_dict[cplx.id]
                    # And include the rxn in this constraint
                    rxn.add_metabolites({cons: 1}, combine=False)

                    ### Append to compiled expressions
                    mind = me.metabolites.index(cons)
                    expr = solver.compile_expr(cons._bound)
                    solver.compiled_expressions[(mind,None)] = (expr,
                            cons._constraint_sense)

        # Need to reset basis
        self.solver.lp_hs = None
        self.solver.feas_basis = None


    def update_inertia_constraints(self, cplx_conc_dict={}, csense='L'):
        """
        Update inertia constraints with new complex concentrations
        """
        me = self.me
        for cplx_id, conc in cplx_conc_dict.iteritems():
            cplx = me.metabolites.get_by_id(cplx_id)
            for rxn in cplx.reactions:
                cons_id = 'cons_rate_' + rxn.id
                if me.metabolites.has_id(cons_id):
                    stoich = rxn.metabolites[cplx]
                    keff = mu/stoich
                    cons = me.metabolites.get_by_id(cons_id)
                    cons._constraint_sense = csense
                    cons._bound = keff*conc


    def calc_cplx_concs(self, complexes, x_dict, muopt):
        """
        Calculate complex concentrations given solution x_dict and keffs of model

        conc = sum_(i\in rxns_catalyzed_by_cplx) mu / keffi
        """
        me = self.me
        cplx_conc_dict = {}
        subs_dict = dict(self.solver.substitution_dict)
        subs_dict['mu'] = muopt
        sub_vals = [subs_dict[k] for k in self.subs_keys_ordered]

        for cplx in complexes:
            concs = []
            for rxn in cplx.reactions:
                stoich = rxn.metabolites[cplx]
                irxn = me.reactions.index(rxn)
                if hasattr(stoich,'subs'):
                    # Make sure this converts to float!
                    imet = me.metabolites.index(cplx)
                    expr = solver.compiled_expressions[(imet,irxn)]
                    sval = expr(*sub_vals)

                    keff = float(muopt / sval)
                    conc = x_dict[rxn.id] / keff
                    concs.append(conc)

            conc_total = sum(concs)
            cplx_conc_dict[cplx.id] = conc_total

        return cplx_conc_dict



    def cplx_to_prot_concs(self, cplx_conc_dict):
        """
        Convert complex concentrations to protein concentrations
        """

    def prot_to_cplx_concs(self, prot_conc_dict):
        """
        Convert protein concentrations to complex concentrations
        """




# END of DynamicME
#============================================================


#============================================================
# Local move methods (modifies me in place)
class LocalMove(object):
    """
    Class providing local move method

    Must implement these methods:
    move
    unmove: resets ME model to before the move
    """
    def __init__(self, me):
        self.me = me
        # Default move type-parameter dict
        self.move_param_dict = {
            'uniform': {
                'min': 0.5,
                'max': 1.5
            },
            'lognormal': {
                'mean': 1.1,
                'std': 1.31,
                'min': 10.,
                'max': 1e6
            }
        }
        self.params0 = None

    def unmove(self, me):
        """
        Unmove to previous params
        """
        if self.params0 is None:
            print 'No pre-move params stored. Not doing anything'
        else:
            params0 = self.params0
            for rid,keff in params0.iteritems():
                rxn = me.reactions.get_by_id(rid)
                rxn.keff = keff
                rxn.update()

    def move(self, me, pert_rxns, method='uniform', group_rxn_dict=None, verbosity=0):
        """
        Randomly perturb me according to provided params

        pert_rxns: IDs of perturbed reactions
        group_rxn_dict: dict of group - perturbed reaction ID
        """
        from numpy.random import uniform

        n_pert = len(pert_rxns)
        param_dict = self.move_param_dict
        ### Save params before move
        self.params0 = {}

        if param_dict.has_key(method):
            params = param_dict[method]
            if method is 'uniform':
                rmin = params['min']
                rmax = params['max']
                rs = np.random.uniform(rmin,rmax,n_pert)
                # Perturb individually or in groups (all up/down)?
                if group_rxn_dict is None:
                    for j,rid in enumerate(pert_rxns):
                        rxn = me.reactions.get_by_id(rid)
                        self.params0[rxn.id] = rxn.keff

                        keff2 = rxn.keff * rs[j]

                        if verbosity >= 2:
                            print 'Rxn: %s\t keff_old=%g\t keff_new=%g'%(
                                    rxn.id, rxn.keff, keff2)

                        rxn.keff = keff2
                        rxn.update()
                else:
                    n_groups = len(group_rxn_dict.keys())
                    rs = uniform(rmin,rmax, n_groups)
                    for gind, (group,rids) in enumerate(group_rxn_dict.iteritems()):
                        #rand = np.random.uniform(rmin,rmax)
                        #rand = uniform(rmin,rmax)
                        rand = rs[gind]
                        for rid in rids:
                            if rid in pert_rxns:
                                rxn = me.reactions.get_by_id(rid)
                                self.params0[rxn.id] = rxn.keff
                                keff2 = rxn.keff * rand
                                if verbosity >= 2:
                                    print 'Group: %s\t Rxn: %s\t keff_old=%g\t keff_new=%g'%(
                                            group, rxn.id, rxn.keff, keff2)
                                rxn.keff = keff2
                                rxn.update()

            elif method is 'lognormal':
                norm_mean = params['mean']
                norm_std  = params['std']
                kmin  = params['min']
                kmax  = params['max']
                ks = 10**np.random.normal(norm_mean, norm_std, n_pert)
                ks[ks < kmin] = kmin
                ks[ks > kmax] = kmax
                for j,rid in enumerate(pert_rxns):
                    rxn = me.reactions.get_by_id(rid)
                    self.params0[rxn.id] = rxn.keff
                    rxn.keff = ks[j]
                    rxn.update()

            else:
                print 'Move method not implemented:', method

        else:
            warnings.warn('No parameters found for move: random')


class ParallelMove(object):
    """
    Handles parallel moves. Needs MPI.
    Samples in parallel, gathers samples, implements move.
    Also unoves.
    """
    import mpi4py

    def __init__(me, move_objects):
        self.me = me
        self.move_objects = move_objects

    def do_work(tasks):
        """
        Work performed by each thread
        """
        # Move
        mover.move(me, pert_rxns, group_rxn_dict=group_rxn_dict)

        # Simulate
        dyme = DynamicME(me, nlp_compat=nlp_compat,
                         growth_key=growth_key, growth_rxn=growth_rxn,
                         exchange_one_rxn=self.exchange_one_rxn)
        result = self.simulate_batch(dyme, basis=basis, no_nlp=no_nlp,
                                     verbosity=verbosity)

        # Compute objective value (error)
        df_sim = self.compute_conc_profile(result)
        objval = self.calc_error_conc(df_sim, df_meas, variables, error_fun=error_fun)

        result_dict = {'result':result, 'objval':objval}

        return result_dict


    def sample_move():
        """
        Main loop: sample, move, gather, return best
        """
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        nWorkers = size

        #----------------------------------------------------
        # Do work in parallel
        data = do_work(worker_tasks[rank])
        #----------------------------------------------------
        # Gather results by root
        data = comm.gather(data, root=0)

        # Keep moves within threshold


        # Return best move for next iteration
        objs = [d['objval'] for d in data]
        obj_best = min(objs)
        result_best = [d for d in data if d['objval']==obj_best]

        #----------------------------------------------------
        # If root: return best move for next iteration
        if rank==0:
            # Gather results and return move
            if verbosity >= 1:
                print 'Gathering samples by root'

#============================================================
class DelayME(object):
    """
    Delayed ME model that tries to maximize mu at next time step
    by optimizing proteome there.
    To do so, model determines optimal accumulation or
    decrease (dilution, degradation) of complexes.
    """
    def __init__(self, solver, dt, cplx_conc_dict=None,
            growth_key='mu', growth_rxn='biomass_dilution', undiluted_cplxs=None):
        #super(DelayME, self).__init__(*args, **kwargs)
        self.me_solver = solver
        me = solver.me
        self.me = me
        self.growth_key = growth_key
        self.growth_rxn = growth_rxn
        self.dt = dt

        mod_me = self.convert_model(dt, cplx_conc_dict, undiluted_cplxs=undiluted_cplxs)
        self.mod_me = mod_me
        self.solver = self.make_solver(mod_me, growth_key)
        self.cplx_conc_dict = cplx_conc_dict

    @property
    def cplx_conc_dict(self):
        return self._cplx_conc_dict

    @cplx_conc_dict.setter
    def cplx_conc_dict(self, value):
        self._cplx_conc_dict = value

    def make_solver(self, mod_me, growth_key='mu'):
        solver = ME_NLP1(mod_me, growth_key=growth_key)
        return solver

    def convert_model(self, dt, cplx_conc_dict=None, csense='L',
            undiluted_cplxs=None):
        """
        Make DelayME

        max  mu
        mu,v,E,vdEdt
        s.t. Sv = 0
             vform - mu*Ei - vdedt = 0  forall Complexes    [Complex]
             sum_j vij / keffj - Ei <= 0    [enzyme_capacity Constraint]
             Ei - E0i - vdEdt*dt = 0        [delayed_abundance Constraint]

        """
        me_solver = self.me_solver
        me = self.me
        # Set default complex abundances if necessary
        if cplx_conc_dict is None:
            # Initialize complex concentrations via ME sim
            if me.solution is None:
                warnings.warn('No initial solution! Solving now.')
                basis = me_solver.feas_basis
                me_solver.bisectmu(basis=basis)

            cplx_conc_dict = get_cplx_concs(me_solver, growth_rxn=self.growth_rxn,
                    undiluted_cplxs=undiluted_cplxs)
            self.cplx_conc_dict = cplx_conc_dict

        # Start with MMmodel
        mam = MMmodel(me_solver, cplx_conc_dict, self.growth_key, self.growth_rxn)
        dme = mam.mm
        # Now, just add additional constraints and rxns
        #for data in dme.complex_data:
        for dataid in cplx_conc_dict.keys():
            data = dme.complex_data.get_by_id(dataid)
            cplx = data.complex
            #------------------------------------------------
            # Can make more or less complex for next timestep
            #   vform - mu*Ei - vdedt = 0  forall Complexes
            #------------------------------------------------
            rxn_dedt_id = 'dedt_'+cplx.id
            try:
                rxn_dedt = ProteinDifferential(rxn_dedt_id)
                dme.add_reaction(rxn_dedt)
            except Exception:
                rxn_dedt = dme.reactions.get_by_id(rxn_dedt_id)

            rxn_dedt.lower_bound = -1000
            rxn_dedt.upper_bound = 1000
            rxn_dedt.add_metabolites({cplx:-1})
            #------------------------------------------------
            # Add the delayed enzyme abundance constraint:
            #   Ei - vdEdt*dt = E0i
            #------------------------------------------------
            cons_id = 'delayed_abundance_%s' % cplx.id
            try:
                cons_delay = AbundanceConstraint(cons_id)
                dme.add_metabolites(cons_delay)
            except Exception:
                cons_delay = dme.metabolites.get_by_id(cons_id)

            cons_delay._bound = cplx_conc_dict[cplx.id]
            #cons_delay._bound = cplx_conc_dict[cplx.id]*sympy.exp(-mu*dt)
            cons_delay._constraint_sense = 'E'

            rxn_conc = dme.reactions.get_by_id('abundance_%s' % cplx.id)
            rxn_conc.add_metabolites({cons_delay:1}, combine=False)
            rxn_dedt.add_metabolites({cons_delay:-dt}, combine=False)
            # rxn_form = data.formation
            # rxn_form.add_metabolites({cons_delay:-dt}, combine=False)
            # for rxn in cplx.reactions:
            #     if isinstance(rxn,ComplexDegradation):
            #         rxn.add_metabolites({cons_delay:dt}, combine=False)

            #rxn_conc.add_metabolites({cons_delay:1}, combine=False)
            # need to relax the bounds on abundance_cplx variables
            # since we move the prior abundance into the constraint rhs now. 
            rxn_conc.lower_bound = 0.
            rxn_conc.upper_bound = 1000.

        return dme


    def update_horizon(self, dt, cplx_conc_dict=None):
        """
        Update timestep used in simulation & constraints
        """
        mm = self.mod_me
        self.dt = dt
        if cplx_conc_dict is None:
            cplx_conc_dict = self.cplx_conc_dict
        ### Update:
        # cons_delay:
        #   1) ._bound
        #   2) rxn_form
        #   3) ComplexDegradation rxns
        #for dataid,conc in iteritems(cplx_conc_dict):
        #    data = mm.complex_data.get_by_id(dataid)
        #    cplx = data.complex
        #    cons_id = 'delayed_abundance_%s' % cplx.id
        #    cons = mm.metabolites.get_by_id(cons_id)
        #    cons._bound = conc*sympy.exp(-mu*dt)

        rxns_dedt = [r for r in mm.reactions if isinstance(r,ProteinDifferential)]
        for rxn in rxns_dedt:
            for met in rxn.metabolites.keys():
                if isinstance(met, AbundanceConstraint):
                    rxn._metabolites[met] = -dt


    def update_cplx_concs(self, cplx_conc_dict, dt=None):
        """
        Update complex concentrations in the constraints.
        (Different from MMmodel, which changes variable bounds.)
        """
        self.cplx_conc_dict = cplx_conc_dict
        mm = self.mod_me
        if dt is None:
            dt = self.dt
        for dataid,conc in iteritems(cplx_conc_dict):
            data = mm.complex_data.get_by_id(dataid)
            cplx = data.complex
            cons_id = 'delayed_abundance_%s' % cplx.id
            cons = mm.metabolites.get_by_id(cons_id)
            # cons._bound = conc*sympy.exp(-mu*dt)
            cons._bound = conc


#============================================================
class MMmodel(object):
    """
    M & M model:
    Proteome constrained M model (Metabolism and Macromolecule model)

    max  mu
    mu,v
    s.t. Sv = 0
         vform - vdil - vdegr + vdelta = 0  forall Complexes
         vdil = mu*Ei
         sum_j vij / keffj <= Ei
         Ei = E0i + vdeltai*dt

    """
    def __init__(self, solver, cplx_conc_dict=None,
            growth_key='mu', growth_rxn='biomass_dilution'):
        self.me_solver = solver
        me = solver.me
        self.me = me
        self.growth_key = growth_key
        self.growth_rxn = growth_rxn
        self.cplx_rxn_keff = {}

        mm = self.convert_model(cplx_conc_dict)
        self.mm = mm
        self.mod_me = mm
        self.solver = self.make_mm_solver(mm, growth_key)
        self.cplx_conc_dict = cplx_conc_dict

    @property
    def cplx_conc_dict(self):
        return self._cplx_conc_dict

    @cplx_conc_dict.setter
    def cplx_conc_dict(self, value):
        self._cplx_conc_dict = value

    def make_mm_solver(self, mm, growth_key='mu'):
        solver = ME_NLP1(mm, growth_key=growth_key)
        return solver

    def update_cplx_bounds(self, cplx_conc_dict):
        """
        Update complex concentration bounds.
        Can do this directly, too, but have this for convenience.
        """
        mm = self.mod_me
        for cplx_id,conc in iteritems(cplx_conc_dict):
            rxn_id = 'abundance_%s' % cplx_id
            rxn = mm.reactions.get_by_id(rxn_id)
            rxn.lower_bound = conc
            rxn.upper_bound = conc

    def convert_model(self, cplx_conc_dict=None, csense='L', undiluted_cplxs=None,
            convert_mRNA=True):
        """
        Make M&M from ME
        - replace complexes with complex-constrained flux constraints:
          sum_j vj(t+1)/keff_j <= Ei(t)
        - removes many mu-dependent constraints in the process
        """
        me = self.me
        me_solver = self.me_solver
        # Don't want to change the ME passed in
        mm = cp.deepcopy(me)
        cplx_rxn_keff = self.cplx_rxn_keff

        if cplx_conc_dict is None:
            # Initialize complex concentrations via ME sim
            if me.solution is None:
                warnings.warn('No initial solution! Solving now.')
                basis = me_solver.feas_basis
                me_solver.bisectmu(basis=basis)

            cplx_conc_dict = get_cplx_concs(me_solver, growth_rxn=self.growth_rxn,
                    undiluted_cplxs=undiluted_cplxs)

            self.cplx_conc_dict = cplx_conc_dict

        #----------------------------------------------------
        """
        1. Make constraint for each complex
           sum_j vj(t+1)/keff_j - Ei(t) <= 0
        2. Remove complex formation and dilution
        """
        rxns_form = []
        #for data in mm.complex_data:
        #    cplx = data.complex
        for cplx_id,abundance in iteritems(cplx_conc_dict):
            if hasattr(cplx_id, 'id'):
                cplx = cplx_id
            else:
                cplx = mm.metabolites.get_by_id(cplx_id)

            #------------------------------------------------
            # Constrain this enzyme's concentration
            #------------------------------------------------
            cons_id = 'enzyme_capacity_%s' % (cplx.id)
            try:
                cons = Constraint(cons_id)
                mm.add_metabolites(cons)
            except Exception:
                cons = mm.metabolites.get_by_id(cons_id)

            cons._bound = 0
            cons._constraint_sense = csense

            # Create new rxn representing this complex's concentration
            rxn_conc_id = 'abundance_%s' % cplx.id
            try:
                rxn_conc = ProteinAbundance(rxn_conc_id)
                mm.add_reaction(rxn_conc)
            except Exception:
                rxn_conc = mm.reactions.get_by_id(rxn_conc_id)

            rxn_conc.add_metabolites({cons:-1})
            # Can add some slack if necessary
            rxn_conc.lower_bound = cplx_conc_dict[cplx.id]
            rxn_conc.upper_bound = cplx_conc_dict[cplx.id]

            # All rxns catalyzed by this complex
            for rxn in cplx.reactions:
                stoich = rxn.metabolites[cplx]
                if stoich<0 and hasattr(stoich,'free_symbols') and mu in stoich.free_symbols:
                    ci = stoich.coeff(mu)
                    if not ci.free_symbols:
                        keff_inv = -float(ci)
                        rxn.add_metabolites({cons:keff_inv}, combine=False)

            # Remove complex from reaction usage so we don't double count
            # its dilution
            for rxn in cplx.reactions:
                stoich = rxn.metabolites[cplx]
                if stoich<0 and \
                        hasattr(rxn.metabolites[cplx],'free_symbols') and \
                        mu in stoich.free_symbols and \
                        not isinstance(rxn, ProteinAbundance):
                    rxn.subtract_metabolites({cplx:rxn.metabolites[cplx]})

            #------------------------------------------------
            # Add dilution rxn and constraint
            # Complex: vform - mu*E = 0
            #------------------------------------------------
            rxn_conc.add_metabolites({cplx: -mu})

            #================================================
            # Also convert mRNA?
            if convert_mRNA:
                mm = self.make_mRNA_dynamic(mm)

        return mm

    def make_mRNA_dynamic(self, mm):
        """
        Convert model to track dynamic RNA concentrations over time.
        Inputs:
            mm : MMmodel
        """
        #----------------------------------------------------
        """
        1. Add explicit dilution and degradation (coupled to degradosome) rxns:
        d[mRNA]/dt = s1*vtrsc - mu*[mRNA] - vdeg
        vdeg >= kdeg*[mRNA]  (an empirical fit--might be able to drop and recapitulate)
        vdeg <= keff*[Degradosome]
        2. Remove mRNA degradation and dilution stoichs from translation rxns
        """
        warnings.warn('Dynamic mRNA not yet implemented!')
        return mm



#============================================================
class ParamOpt(object):
    """
    Methods for fitting parameters to measured conc or flux profiles 
    Constructor:
    ParamOpt(me, sim_params)

    me:         ME model (1.0 or 2.0)
    sim_params: dict of simulation parameters:
                T, c0_dict, ZERO_CONC, extra_rxns_tracked, lb_dict
    """

    def __init__(self, me, sim_params,
                 growth_key='mu', growth_rxn='biomass_dilution',
                 exchange_one_rxn=None):
        self.me = me
        self.growth_key = growth_key
        self.growth_rxn = growth_rxn
        self.sim_params = sim_params
        self.exchange_one_rxn = exchange_one_rxn
        random_move = LocalMove(me)
        self.move_objects = [random_move]


    def update_keffs(self, keff_dict):
        me = self.me
        for rid,keff in keff_dict.iteritems():
            rxn = me.reactions.get_by_id(rid)
            rxn.keff = keff
            rxn.update()


    def calc_threshold(self, objval0, objval):
        T_rel = (objval - objval0) / abs(objval0 + 1.0)
        return T_rel


    def fit_profile(self, df_meas, pert_rxns, variables,
                    Thresh0=1.0, result0=None,
                    basis=None,
                    max_iter_phase1=10,
                    max_iter_phase2=100,
                    max_reject = 10,
                    nlp_compat=False,
                    group_rxn_dict=None,
                    no_nlp=False,
                    verbosity=2,
                    error_fun=None):
        """
        Tune parameters (e.g., keffs) to fit flux or conc profile
        """
        #----------------------------------------------------
        # LBTA
        #----------------------------------------------------
        # Phase I: list filling phase
        # 1. Select initial threshold T > 0 and initial solution
        # 2. Local search moves to fill the list
        #    1) generate neighbor of current solution using a
        #       local search move (e.g., randomly selected from
        #       a set of possible moves)
        #    2) calc relative cost deviation between proposed
        #       and current solution:
        #       T(s,s') = [c(s') - c(s)] / c(s) 
        #    3) if 0 < T(s,s') < max(List), insert T(s,s')
        #       into List
        #    4) repeat for each move until list exhausted
        #       List stores variability of local function values
        #       as binary tree with the new threshold value 
        #       as the key.
        # Phase II: optimization
        # 3. Optimize
        #   1) generate s0 (initial solution)
        #   2) generate s' (neighbor) via local search move
        #   3) compute threshold value, check move acceptance
        #      criterion using max element of List:
        #      If T_new = [c(s')-c(s)]/c(s) < T_max:
        #           set s = s'
        #           if c(s) < c(s_best):
        #               s_best = s
        #           insert T_new in List
        #           pop T_max from List
        #   4) repeat until a number of feasible moves rejected
        #     
        # 4. report best solution found
        #----------------------------------------------------
        # L: length of list
        # T: number of iterations
        # L + T sims
        #----------------------------------------------------

        opt_stats = []
        #----------------------------------------------------
        # Phase I: list filling
        #----------------------------------------------------
        Thresh = Thresh0
        Ts = [Thresh]

        me = self.me
        growth_key = self.growth_key
        growth_rxn = self.growth_rxn
        dyme = DynamicME(me, nlp_compat=nlp_compat,
                         growth_key=growth_key, growth_rxn=growth_rxn,
                         exchange_one_rxn=self.exchange_one_rxn)

        # Get initial solution
        if result0 is None:
            result0 = self.simulate_batch(dyme, basis=basis, no_nlp=no_nlp,
                                          verbosity=verbosity)
        df_sim0 = self.compute_conc_profile(result0)
        objval0 = self.calc_error_conc(df_sim0, df_meas, variables, error_fun=error_fun)

        # Perform local moves
        move_objects = self.move_objects
        n_iter = 0
        obj_best = objval0
        sol = df_sim0
        sol_best = sol
        result = result0
        result_best = result

        while n_iter < max_iter_phase1:
            n_iter = n_iter + 1
            for mover in move_objects:
                #--------------------------------------------
                tic = time.time()
                #--------------------------------------------
                # TODO: PARALLEL sampling and moves
                # Local move

                if verbosity >= 1:
                    print '[Phase I] Iter %d:\t Performing local move:'%n_iter, type(mover)
                mover.move(me, pert_rxns, group_rxn_dict=group_rxn_dict)

                # Simulate
                dyme = DynamicME(me, nlp_compat=nlp_compat,
                                 growth_key=growth_key, growth_rxn=growth_rxn,
                                 exchange_one_rxn=self.exchange_one_rxn)
                result = self.simulate_batch(dyme, basis=basis, no_nlp=no_nlp,
                                             verbosity=verbosity)
                # Unmove: generate samples surrounding initial point
                # TODO: PARALLEL unmoves
                mover.unmove(me)

                # Compute objective value (error)
                df_sim = self.compute_conc_profile(result)
                objval = self.calc_error_conc(df_sim, df_meas, variables, error_fun=error_fun)
                if objval < obj_best:
                    obj_best = objval
                    sol_best = sol
                    result_best = result

                # Calc relative cost deviation
                #T_rel = (objval - objval0) / (objval0 + 1.0)
                T_rel = self.calc_threshold(objval0, objval)

                Tmax = max(Ts)
                if T_rel <= Tmax and T_rel > 0:
                    Ts.append(T_rel)
                    Tmax = max(Ts)

                opt_stats.append({'phase':1, 'iter':n_iter,
                                  'obj':objval, 'objbest':obj_best,
                                  'Tmax':Tmax, 'Tk':T_rel})

                #--------------------------------------------
                toc = time.time()-tic
                #--------------------------------------------
                if verbosity >= 1:
                    print 'Obj:%g \t Best Obj: %g \t Tmax:%g \t T:%g \t Time:%g secs'%(
                        objval, obj_best, Tmax, T_rel, toc)
                    print '//============================================'

        #----------------------------------------------------
        # Phase II: optimization
        #----------------------------------------------------
        n_reject = 0
        n_iter = 0
        while (n_iter < max_iter_phase2) and (n_reject < max_reject):
            n_iter = n_iter + 1
            for mover in move_objects:
                #--------------------------------------------
                tic = time.time()
                #--------------------------------------------
                # Local move
                # TODO: PARALLEL sampling and moves
                if verbosity >= 1:
                    print '[Phase II] Iter %d:\t Performing local move:'%n_iter, type(mover)

                mover.move(me, pert_rxns, group_rxn_dict=group_rxn_dict)

                # Simulate
                dyme = DynamicME(me, nlp_compat=nlp_compat,
                                 growth_key=growth_key, growth_rxn=growth_rxn,
                                 exchange_one_rxn=self.exchange_one_rxn)
                result = self.simulate_batch(dyme, basis=basis, no_nlp=no_nlp,
                                             verbosity=verbosity)

                # Compute objective value (error)
                df_sim = self.compute_conc_profile(result)
                objval = self.calc_error_conc(df_sim, df_meas, variables, error_fun=error_fun)

                # Calc threshold and accept or reject move
                #T_new = (objval - objval0) / (objval0+1.0)
                T_new = self.calc_threshold(objval0, objval)
                T_max = max(Ts)
                move_str = ''
                if T_new <= T_max:
                    # Move if under threshold
                    objval0 = objval
                    sol = df_sim
                    if T_new > 0:
                        Ts.remove(max(Ts))
                        Ts.append(T_new)
                        Tmax = max(Ts)
                    if objval < obj_best:
                        sol_best = sol
                        obj_best = objval
                        result_best = result
                    move_str = 'accept'
                else:
                    n_reject = n_reject + 1
                    # Reject move: reset the model via unmove
                    # TODO: PARALLEL unmoves
                    mover.unmove(me)
                    move_str = 'reject'

                opt_stats.append({'phase':2, 'iter':n_iter,
                                  'obj':objval, 'objbest':obj_best,
                                  'Tmax':Tmax, 'Tk':T_new})
                #--------------------------------------------
                toc = time.time()-tic
                #--------------------------------------------
                if verbosity >= 1:
                    print 'Obj:%g \t Best Obj: %g \t Tmax:%g \t T:%g \t Move:%s\t n_reject:%d\t Time:%g secs'%(
                        objval, obj_best, Tmax, T_new, move_str, n_reject, toc)
                    print '//============================================'

        return sol_best, opt_stats, result_best


    def simulate_batch(self, dyme, basis=None, prec_bs=1e-3, no_nlp=False,
                       verbosity=2):
        """
        Compute error in concentration profile given params

        [Inputs]
        dyme:   DynamicME object

        [Outputs]
        """
        import copy as cp

        sim_params = self.sim_params
        # Provide copy of params and containers
        # so we can re-simulate after local moves
        T = cp.copy(sim_params['T'])
        X0 = cp.copy(sim_params['X0'])
        c0_dict = cp.deepcopy(sim_params['c0_dict'])
        lb_dict = cp.deepcopy(sim_params['lb_dict'])
        ub_dict = cp.deepcopy(sim_params['ub_dict'])
        extra_rxns_tracked = sim_params['extra_rxns_tracked']
        ZERO_CONC = sim_params['ZERO_CONC']

        result = dyme.simulate_batch(T, c0_dict, X0, prec_bs=prec_bs,
                                     ZERO_CONC=ZERO_CONC,
                                     extra_rxns_tracked=extra_rxns_tracked,
                                     lb_dict=lb_dict,
                                     ub_dict=ub_dict,
                                     no_nlp=no_nlp,
                                     verbosity=verbosity)
        self.result = result
        return result


    def compute_conc_profile(self, result):
        """
        Generate concentration profile from simulation result
        """
        df_conc = pd.DataFrame(result['concentration'])
        df_time = pd.DataFrame({'time':t, 'biomass':b} for t,b in zip(
            result['time'], result['biomass']))
        df_flux = pd.DataFrame(result['ex_flux'])
        df_result = pd.concat([df_time, df_conc, df_flux], axis=1)

        return df_result

    def get_time_ss(self, df, cols_fit, colT='time', ZERO_SS=0):
        T_ss = min( df[colT][df[cols_fit].diff().abs().sum(axis=1)<=ZERO_SS])
        return T_ss


    def calc_error_conc(self, df_sim0, df_meas0, cols_fit,
                        error_fun=None,
                        col_weights={},
                        normalize_time=False,
                        ZERO_SS=0.,
                        LAG_MEAS=1.,
                        LAG_SIM=1.,
                        verbosity=0):
        """
        Compute error in concentration profile given params

        [Inputs]
        result:   output of dyme.simulate_batch
        normalize_time: normalize simulated and measured time so
            relative phases/modes are compared instead of absolute
            time-points.

        [Outputs]
        """
        # Align timesteps of measured & simulated conc profiles,
        # interpolating where necessary
        # All timepoints
        if error_fun is None:
            error_fun = errfun_sae

        if normalize_time:
            df_meas = df_meas0.copy()
            df_sim  = df_sim0.copy()

            T_end = self.get_time_ss(df_meas.loc[df_meas['time']>LAG_MEAS,:], cols_fit)
            if verbosity > 0:
                print 'T_end(meas):', T_end
            df_meas['time'] = df_meas['time'] / T_end
            df_meas = df_meas.loc[ df_meas['time'] <= 1, :]
            T_end = self.get_time_ss(df_sim.loc[df_sim['time']>LAG_SIM,:], cols_fit)
            if verbosity > 0:
                print 'T_end(sim):', T_end
            df_sim['time'] = df_sim['time'] / T_end
            df_sim = df_sim.loc[ df_sim['time'] <= 1, :]
        else:
            df_meas = df_meas0
            df_sim  = df_sim0

        t_sim = df_sim['time']
        t_meas= df_meas['time']
        tt = np.union1d(t_sim, t_meas)

        weighted_errors = []
        for col in cols_fit:
            y_sim = df_sim[col]
            y_meas = df_meas[col]
            yy_sim = np.interp(tt, t_sim, y_sim)
            yy_meas= np.interp(tt, t_meas,y_meas)
            error = error_fun(yy_meas, yy_sim)
            if col_weights.has_key(col):
                error = error * col_weights[col]
            weighted_errors.append(error)

        error_tot = sum(weighted_errors)
        return error_tot


    def fit_profile_abc(self):
        """
        Tune parameters (e.g., keffs) to fit flux or conc profile
        """

    def compute_proteome_profile(self, result, rxns_trsl):
        """
        df_prot = compute_proteome_profile(result, rxns_trsl) 
        Return proteome profile
        """
        df_rxn = pd.DataFrame(result['rxn_flux'])
        cols_trsl = [r.id for r in rxns_trsl if r.id in df_rxn.columns]
        df_trsl = df_rxn[cols_trsl]
        df_time = pd.DataFrame([{'time':t} for t in result['time']])
        df_prot = pd.concat([ df_time, df_trsl], axis=1)

        return df_prot

# END of ParamOpt
#============================================================

class DelayedDynamicME(object):
    """
    DynamicME with proteome delay
    """
    def __init__(self, solver, cplx_conc_dict=None, mm_model=None, delay_model=None, dt=0.1,
            nlp_compat=False, exchange_one_rxn=None):

        me = solver.me
        self.nlp_compat = nlp_compat
        self.me = me
        is_me2 = isinstance(me, cobrame.core.MEModel.MEModel)
        if exchange_one_rxn is None:
            exchange_one_rxn = is_me2
        self.exchange_one_rxn = exchange_one_rxn

        # Make MMmodel 
        #if mm_model is None:
        #    mm_model = MMmodel(solver, cplx_conc_dict)
        #self.mm_model = mm_model
        self.mm_model = None

        # Make DelayME
        if delay_model is None:
            delay_model = DelayME(solver, dt=dt)
        self.delay_model = delay_model

    def set_timestep(self, dt):
        self.delay_model.update_horizon(dt)

    def simulate_batch(self, T, c0_dict, X0, cplx_conc_dict0,
                       dt=0.1, H=None,
                       o2_e_id='o2_e', o2_head=0.21, kLa=7.5,
                       conc_dep_fluxes = False,
                       extra_rxns_tracked=[],
                       prec_bs=1e-6,
                       ZERO_CONC = 1e-3,
                       lb_dict={},
                       ub_dict={},
                       proteome_has_inertia=True,
                       mm_model = None,
                       basis=None,
                       no_nlp=False,
                       verbosity=2,
                       solver_verbosity=0,
                       LB_DEFAULT=-1000.,
                       UB_DEFAULT=1000.,
                       MU_MIN=0.,
                       MU_MAX=2,
                       handle_negative_conc='timestep',
                       ZERO=1e-15):
        """
        result = simulate_batch()

        Solve dynamic ME problem with proteome delay
        [Arguments]
        T:  batch time
        c0_dict: initial extracellular concentration dict
        X0: initial biomass density
        o2_e_id: oxygen (extracellular) metabolite ID
        o2_head: headspace O2 concentration
        kLa: mass transfer coefficient for O2
        dt: time step (h)
        H:  prediction horizon. Default=None. In which case, sets equal to dt
        conc_dep_fluxes: are uptake fluxes concentration dependent?
        prec_bs: precision of mu for bisection
        ZERO_CONC: (in mM) if below this concentration, consider depleted
        proteome_has_inertia: if True, track protein concentrations and
                              constrain catalyzed flux (default: False)
        cplx_conc_dict0: (initial) protein concentration dict.
                        Only the complexes in this dict will be constrained for
                        the rest of the simulation.
        mm_model : the metabolism and macromolecule model used to implement
                   proteome inertia constraints
        handle_negative_conc : 'throttle' or 'timestep'
                How to handle concentrations about to become negative in next timestep.
                'throttle': update lower_bound of the exchange rxn to estimated value
                    that would prevent negative concentration (not always reliable
                    since it can change the optimal solution)
                'timestep': estimate the largest timestep where concentration is not negative.

        [Output]
        result
        ----------------------------------------------------
        Batch equations:
        dX/dt = mu*X
        dc/dt = A*v*X
        ----------------------------------------------------
        Procedure:
            1.  Initialize proteome (complex abundances)
            2.  Solve DelayME to maximize mu at next timestep,
                allowing protein accumulation or depletion by
                synthesizing more or letting it dilute
            3.  Update actual protein abundance using current
                synthesis & dilution rates
            4.  Update biomass and metabolite concentrations
            5.  Iterate
        """
        # If uptake rate independent of concentration,
        # only recompute uptake rate once a substrate
        # depleted
        #mm_model = self.mm_model
        #mm_solver = mm_model.solver
        #mm = mm_model.mod_me

        delay_model = self.delay_model
        delay_solver = delay_model.solver
        dme = delay_model.mod_me
        exchange_one_rxn = self.exchange_one_rxn

        cplx_conc_dict = dict(cplx_conc_dict0)
        # Initialize proteome state in DelayME 
        if H is None:
            H = dt
        delay_model.update_cplx_concs(cplx_conc_dict0)
        delay_model.update_horizon(H)

        # Initialize concentrations & biomass
        conc_dict = c0_dict.copy()
        X_biomass = X0
        mu_opt = 0.
        x_dict = None
        if exchange_one_rxn:
            ex_flux_dict = {get_exchange_rxn(dme, metid).id:0. for metid in conc_dict.keys()}
        else:
            ex_flux_dict = {}
            for metid in conc_dict.keys():
                try:
                    rxn = get_exchange_rxn(dme, metid, 'source', exchange_one_rxn)
                    ex_flux_dict[rxn.id] = 0.
                except ValueError:
                    pass
                try:
                    rxn = get_exchange_rxn(dme, metid, 'sink', exchange_one_rxn)
                    ex_flux_dict[rxn.id] = 0.
                except ValueError:
                    pass

        rxn_flux_dict = {(r.id if hasattr(r,'id') else r):0. for r in extra_rxns_tracked}

        t_sim = 0.

        times = [t_sim]
        conc_profile = [conc_dict.copy()]
        cplx_profile = [cplx_conc_dict.copy()]
        biomass_profile = [X_biomass]
        ex_flux_profile = [ex_flux_dict.copy()]
        rxn_flux_profile= [rxn_flux_dict.copy()]
        mu_profile = []

        iter_sim = 0
        recompute_fluxes = True     # In first iteration always compute
        while t_sim < T:
            # Determine available substrates given concentrations
            for metid,conc in conc_dict.iteritems():
                try:
                    ex_rxn = get_exchange_rxn(dme, metid, exchange_one_rxn=exchange_one_rxn,
                                                   direction='source')
                    if conc <= ZERO_CONC:
                        if verbosity >= 1:
                            print 'Metabolite %s depleted.'%(metid)
                        if exchange_one_rxn:
                            lb0 = ex_rxn.lower_bound
                            lb1 = 0.
                            if lb1 != lb0:
                                recompute_fluxes = True
                            ex_rxn.lower_bound = 0.
                        else:
                            ub0 = ex_rxn.upper_bound
                            ub1 = 0.
                            if ub1 != ub0:
                                recompute_fluxes = True
                            ex_rxn.upper_bound = 0.
                    else:
                        # (re)-open exchange whenever concentration above
                        # threshold since, e.g., secreted products can be 
                        # re-consumed, too.
                        if verbosity >= 1:
                            print 'Metabolite %s available.'%(metid)
                        if exchange_one_rxn:
                            lb0 = ex_rxn.lower_bound
                            if lb_dict.has_key(ex_rxn.id):
                                lb1 = lb_dict[ex_rxn.id]
                            else:
                                if verbosity >= 1:
                                    print 'Using default LB=%g for %s'%(LB_DEFAULT, ex_rxn.id)
                                lb1 = LB_DEFAULT
                            if lb1 != lb0:
                                recompute_fluxes = True
                            ex_rxn.lower_bound = lb1
                        else:
                            ub0 = ex_rxn.upper_bound
                            if ub_dict.has_key(ex_rxn.id):
                                ub1 = ub_dict[ex_rxn.id]
                            else:
                                if verbosity >= 1:
                                    print 'Using default UB=%g for %s'%(UB_DEFAULT, ex_rxn.id)
                                ub1 = UB_DEFAULT
                            if ub1 != ub0:
                                recompute_fluxes = True
                            ex_rxn.upper_bound = ub1
                except:
                    if verbosity >= 2:
                        print 'No uptake rxn found for met:', metid

            # Recompute fluxes if any rxn bounds changed, which triggers
            # recompute_fluxes flag
            if recompute_fluxes:
                # Compute ME
                if verbosity >= 1:
                    print 'Computing new uptake rates'
                if self.nlp_compat:
                    if no_nlp:
                        mu_opt, hs_bs, x_opt, cache_opt = delay_solver.bisectmu(
                                prec_bs, basis=basis,
                                #mumin=mu_feas*0.999, mumax=10*mu_feas,
                                mumin=MU_MIN, mumax=MU_MAX,
                                verbosity=solver_verbosity)
                    else:
                        x_opt, stat, hs_bs = delay_solver.solvenlp(prec_bs, basis=basis)
                        cache_opt = None
                    if dme.solution is None:
                        mu_opt = 0.
                    else:
                        mu_opt = dme.solution.f
                else:
                    mu_opt, hs_bs, x_opt, cache_opt = delay_solver.bisectmu(
                            prec_bs, basis=basis,
                            #mumin=mu_feas*0.999, mumax=10*mu_feas,
                            mumin=MU_MIN, mumax=MU_MAX,
                            verbosity=solver_verbosity)

                basis = hs_bs
                if dme.solution is None:
                    x_dict = None
                else:
                    x_dict = dme.solution.x_dict

            #------------------------------------------------
            # Update concentrations
            ex_flux_dict = {}
            conc_dict_prime = conc_dict.copy()
            cplx_conc_dict_prime = cplx_conc_dict.copy()
            #------------------------------------------------


            #================================================
            # Determine if smaller timestep needed to prevent
            # concentration of metabolite or complex < 0
            # Update actual concentration after all concentrations
            # checked and the final timestep decided.
            #================================================
            dt_new = dt     # If need to change timestep
            # Get biomass at next timestep assuming default timestep used
            #X_biomass_prime = X_biomass + mu_opt*X_biomass*dt_new
            X_biomass_prime = X_biomass*np.exp(mu_opt*dt_new)
            X_biomass_trpzd = (X_biomass + X_biomass_prime)/2
            """
            for a cell:
                Ej(t+1) = Ej(t) + v_formation*dt
                mmol/gDW = mmol/gDW + mmol/gDW/h * h
            """
            for cplx_id, conc in iteritems(cplx_conc_dict):
                cplx = dme.metabolites.get_by_id(cplx_id)
                dedt = dme.reactions.get_by_id('dedt_'+cplx_id)
                v_dedt = 0.
                try:
                    v_dedt = dme.solution.x_dict[dedt.id]
                except:
                    continue
                conc_new = conc + v_dedt*dt_new
                if conc_new < -ZERO:
                    if v_dedt < -ZERO:
                        dt_new = min(dt_new, -conc/v_dedt)

            #------------------------------------------------
            # Determine if smaller timestep needed for metabolite concentrations
            #------------------------------------------------
            for metid, conc in conc_dict.iteritems():
                v = 0.
                # If cobrame, exchange fluxes are just one -1000 <= EX_... <= 1000, etc.
                if exchange_one_rxn:
                    rxn = get_exchange_rxn(dme, metid)
                    if x_dict is not None:
                        v = dme.solution.x_dict[rxn.id]              # mmol/gDW/h
                # If ME 1.0, EX_ split into source and sink
                else:
                    v_in  = 0.
                    v_out = 0.
                    try:
                        rxn_in  = get_exchange_rxn(dme, metid, 'source', exchange_one_rxn)
                        v_in  = dme.solution.x_dict[rxn_in.id]
                    except:
                        pass
                    try:
                        rxn_out = get_exchange_rxn(dme, metid, 'sink', exchange_one_rxn)
                        v_out = dme.solution.x_dict[rxn_out.id]
                    except:
                        pass
                    v = v_out - v_in

                if metid is not o2_e_id:
                    # Update concentration for next timestep
                    # mmol/L = mmol/gDW/h * gDW/L * h
                    conc_new = conc + v*X_biomass_trpzd*dt_new
                    if conc_new < (ZERO_CONC - prec_bs):
                        if v < -ZERO:
                            # If multiple concentrations below 0,
                            # need to take the smallest timestep.
                            dt_new = min(dt_new, -conc / (v*X_biomass_trpzd))

            #------------------------------------------------
            # Actually update complex concentrations for next time step
            # using smallest required timestep.
            #------------------------------------------------
            dme.dt = dt_new
            if verbosity >= 1:
                if dt_new != dt:
                    print 'Changed timestep to %g' % (dt_new)

            for cplx_id, conc in iteritems(cplx_conc_dict):
                cplx = dme.metabolites.get_by_id(cplx_id)
                # E - vform*dt + vdegr*dt = E0*exp(-mu*dt)
                rxn_conc = dme.reactions.get_by_id('abundance_%s'%cplx.id)
                conc_new = conc
                try:
                    conc_new = dme.solution.x_dict[rxn_conc.id]
                except:
                    continue
                #dedt = dme.reactions.get_by_id('dedt_'+cplx_id)
                #v_dedt = 0.
                #try:
                #    v_dedt = dme.solution.x_dict[dedt.id]
                #except:
                #    continue
                ##conc_new = conc + v_dedt*dt_new
                #formation = dme.reactions.get_by_id('formation_'+cplx_id)
                #v_form = 0.
                #try:
                #    v_form = dme.solution.x_dict[formation.id]
                #except:
                #    continue
                #degr_id = 'degradation_'+cplx_id
                #v_degr = 0.
                #try:
                #    v_degr = dme.solution.x_dict[degr_id]
                #except:
                #    continue
                #conc_new = conc*np.exp(-mu_opt*dt_new) + (v_form-v_degr)*dt_new
                cplx_conc_dict_prime[cplx_id] = conc_new

            delay_model.update_cplx_concs(cplx_conc_dict_prime)
            #------------------------------------------------
            # Actually update metabolite concentrations using smallest required timestep.
            #------------------------------------------------
            for metid, conc in conc_dict.iteritems():
                v = 0.
                # If cobrame, exchange fluxes are just one -1000 <= EX_... <= 1000, etc.
                if exchange_one_rxn:
                    rxn = get_exchange_rxn(dme, metid)
                    if x_dict is not None:
                        v = dme.solution.x_dict[rxn.id]              # mmol/gDW/h
                    ex_flux_dict[rxn.id] = v
                # If ME 1.0, EX_ split into source and sink
                else:
                    v_in  = 0.
                    v_out = 0.
                    try:
                        rxn_in  = get_exchange_rxn(dme, metid, 'source', exchange_one_rxn)
                        v_in  = dme.solution.x_dict[rxn_in.id]
                        ex_flux_dict[rxn_in.id] = v_in
                    except Exception:
                        continue
                    try:
                        rxn_out = get_exchange_rxn(dme, metid, 'sink', exchange_one_rxn)
                        v_out = dme.solution.x_dict[rxn_out.id]
                        ex_flux_dict[rxn_out.id] = v_out
                    except Exception:
                        continue
                    v = v_out - v_in

                if metid is not o2_e_id:
                    # Update concentration for next timestep
                    # mmol/L = mmol/gDW/h * gDW/L * h
                    conc_new = conc + v*X_biomass_trpzd*dt_new
                    conc_dict_prime[metid] = conc_new
                else:
                    # Account for oxygen diffusion from headspace into medium
                    conc_dict_prime[metid] = conc+(v*X_biomass_trpzd + kLa*(o2_head - conc))*dt_new

            # Actually update biomass for next timepoint using smallest required timestep.
            X_biomass_prime = X_biomass + mu_opt*X_biomass*dt_new

            #------------------------------------------------
            # Copy updated values to be the previous values
            X_biomass = X_biomass_prime
            conc_dict = conc_dict_prime.copy()
            cplx_conc_dict = cplx_conc_dict_prime.copy()

            ### Extra fluxes tracked
            for rxn in extra_rxns_tracked:
                v = 0.
                rid = rxn.id if hasattr(rxn,'id') else rxn
                if x_dict is not None:
                    v = x_dict[rid]
                rxn_flux_dict[rid] = v

            # ------------------------------------------------
            # Move to next time step
            t_sim = t_sim + dt_new
            iter_sim = iter_sim + 1
            times.append(t_sim)
            conc_profile.append(conc_dict.copy())
            biomass_profile.append(X_biomass)
            ex_flux_profile.append(ex_flux_dict.copy())
            rxn_flux_profile.append(rxn_flux_dict.copy())
            mu_profile.append(mu_opt)
            # Save protein concentrations
            cplx_profile.append(cplx_conc_dict.copy())

            # ------------------------------------------------
            # Print some results
            if verbosity >= 1:
                print 'Biomass at t=%g: %g'%(t_sim, X_biomass)
                print 'Concentrations:', conc_dict
                print 'Growth rate:', mu_opt


        # Append last growth rate once more
        mu_profile.append(mu_opt)

        result = {'biomass':biomass_profile,
                  'concentration':conc_profile,
                  'ex_flux':ex_flux_profile,
                  'rxn_flux':rxn_flux_profile,
                  'growth_rate':mu_profile,
                  'complex':cplx_profile,
                  'time':times,
                  'basis':basis}

        self.result = result

        return result

    def get_exchange_rxn(self, metid, direction='both', exchange_one_rxn=None): 
        """
        Exchange rxn for DelayME
        """
        dme = self.delay_model.mod_me
        rxn_dme = get_exchange_rxn(dme, metid, direction, exchange_one_rxn)

        return rxn_dme

    def get_exchange_rxns(self, metid, direction='both', exchange_one_rxn=None):
        """
        Get exchange rxns from all the models
        """
        me = self.me
        mm = self.mm_model.mod_me
        dme = self.delay_model.mod_me

        rxn_me = get_exchange_rxn(me, metid, direction, exchange_one_rxn)
        rxn_mm = get_exchange_rxn(mm, metid, direction, exchange_one_rxn)
        rxn_dme = get_exchange_rxn(dme, metid, direction, exchange_one_rxn)

        result = {'me':rxn_me, 'mm':rxn_mm, 'delay':rxn_dme}

        return result



def get_exchange_rxn(me, metid, direction='both', exchange_one_rxn=None):
    """
    Get exchange fluxes for metabolite with id metid
    """
    met = me.metabolites.get_by_id(metid)
    is_me2 = isinstance(me, cobrame.core.MEModel.MEModel)
    if exchange_one_rxn is None:
        exchange_one_rxn = is_me2

    ex_rxn = None

    if exchange_one_rxn:
        ex_rxns = [rxn for rxn in met.reactions if
                   len(rxn.metabolites)==1 and rxn.metabolites[met]==-1.]
        if len(ex_rxns) < 1:
            raise ValueError('No exchange rxn for metabolite %s'%metid)
        else:
            ex_rxn = ex_rxns[0]
    else:
        ### If ME 1.0
        # Get the source or sink rxn?
        if direction is 'source':
            ex_rxns = [rxn for rxn in met.reactions if
                       len(rxn.metabolites)==1 and rxn.metabolites[met]==1.]
        elif direction is 'sink':
            ex_rxns = [rxn for rxn in met.reactions if
                       len(rxn.metabolites)==1 and rxn.metabolites[met]==-1.]
        else:
            raise ValueError("Direction must equal 'sink' or 'source' for ME 1.0 models.")

        if len(ex_rxns) < 1:
            raise ValueError('No exchange rxn for metabolite %s'%metid)
        else:
            ex_rxn = ex_rxns[0]

    return ex_rxn


def get_undiluted_cplxs(solver, exclude_types=[
    ComplexFormation, GenericFormationReaction, ComplexDegradation, PeptideDegradation]):
    """
    Find cplxs that are not diluted

    Inputs
    exclude_types : Reaction types that are allowed to not have complex dilution coupling
    """
    me = solver.me
    undiluted_cplxs = []
    for data in me.complex_data:
        met = data.complex
        for rxn in met.reactions:
            if not any([isinstance(rxn,t) for t in exclude_types]):
                try:
                    if rxn.metabolites[met]<0:
                        if not hasattr(rxn.metabolites[met],'subs'):
                            undiluted_cplxs.append(data)
                except TypeError:
                    continue

    undiluted_cplxs = list(set(undiluted_cplxs))

    return undiluted_cplxs


def get_cplx_concs(solver, muopt=None, growth_rxn='biomass_dilution', undiluted_cplxs=None,
        ZERO=1e-20):
    """
    Get complex concentrations (mmol/gDW) from solution:
    [E_i] = sum_j v_j / keff_ij

    undiluted_cplxs: skip the complexes that are not diluted--i.e.,. treated as metabolites
    """
    me = solver.me
    x_dict = me.solution.x_dict
    if muopt is None:
        #muopt = solver.substitution_dict['mu']
        muopt = x_dict[growth_rxn]

    if undiluted_cplxs is None:
        undiluted_cplxs = get_undiluted_cplxs(solver)

    solver.substitution_dict['mu'] = muopt
    sub_vals = [solver.substitution_dict[k] for k in solver.subs_keys_ordered]
    cplxs = [data.complex for data in me.complex_data if data not in undiluted_cplxs]

    cplx_conc_dict = {}
    for cplx in cplxs:
        #----------------------------------------------------
        # Just get the coefficient on mu to avoid:  -1/keff*mu - 1
        concs = []
        for rxn in cplx.reactions:
            stoich = rxn.metabolites[cplx]
            if stoich<0 and hasattr(stoich,'free_symbols') and mu in stoich.free_symbols:
                ci = stoich.coeff(mu)
                if not ci.free_symbols:
                    conci = x_dict[rxn.id] * -float(ci)
                    concs.append(conci)
        conc = sum(concs)
        if conc < ZERO:
            conc = 0
        cplx_conc_dict[cplx.id] = conc

    return cplx_conc_dict

class ProteinDifferential(MEReaction):
    """
    Protein accumulation or depletion rate
    """
    pass

class ProteinAbundance(MEReaction):
    """
    Protein (enzyme complex) concentration
    Units: [mmol/gDW]
    (note units are different from a metabolite concentration since not normalized to volume)
    """
    pass

class AbundanceConstraint(Constraint):
    """
    Constraint on protein abundance
    """

class DilutionReaction(MEReaction):
    """
    Complex dilution rate
    """
    pass
