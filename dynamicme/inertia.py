#============================================================
# File dynamic.py
#
# class DelayME (InertiaME model in paper)
# class IMEmodel (part of the InertiaME model)
# class DelayedDynamicME
# class ProteinDifferential
# class ProteinAbundance
# class AbundanceConstraint
# class DilutionReaction
#
# Class & methods for dynamic FBA with ME models.
#
# Laurence Yang, SBRG, UCSD
#
# 18 Mar 2016:  first version
# 28 Sep 2017:  migrated to separate module
# 30 Sep 2018:  cleaned up and moved some redundant classes to dynamic.py
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
from dynamicme.dynamic import DynamicME
from dynamicme.dynamic import get_cplx_concs

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
class DelayME(object):
    """
    Delayed ME model that tries to maximize mu at next time step
    by optimizing proteome there.
    To do so, model determines optimal accumulation or
    decrease (dilution, degradation) of complexes.
    """
    def __init__(self, solver, dt, cplx_conc_dict=None,
            growth_key='mu', growth_rxn='biomass_dilution', undiluted_cplxs=None):

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

        # Start with IMEmodel
        mam = IMEmodel(me_solver, cplx_conc_dict, self.growth_key, self.growth_rxn)
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
            cons_delay._constraint_sense = 'E'

            rxn_conc = dme.reactions.get_by_id('abundance_%s' % cplx.id)
            rxn_conc.add_metabolites({cons_delay:1}, combine=False)
            rxn_dedt.add_metabolites({cons_delay:-dt}, combine=False)

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

        rxns_dedt = [r for r in mm.reactions if isinstance(r,ProteinDifferential)]
        for rxn in rxns_dedt:
            for met in rxn.metabolites.keys():
                if isinstance(met, AbundanceConstraint):
                    rxn._metabolites[met] = -dt


    def update_cplx_concs(self, cplx_conc_dict, dt=None):
        """
        Update complex concentrations in the constraints.
        (Different from IMEmodel, which changes variable bounds.)
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
class IMEmodel(object):
    """
    DelayME (InertiaME) model's inner part:

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
                try:
                    if stoich<0 and hasattr(stoich,'free_symbols') and mu in stoich.free_symbols:
                        ci = stoich.coeff(mu)
                        if not ci.free_symbols:
                            keff_inv = -float(ci)
                            rxn.add_metabolites({cons:keff_inv}, combine=False)
                except:
                    warnings.warn('get_cplx_cons: problem with cplx %s and rxn %s'%(
                        cplx.id, rxn.id))

            # Remove complex from reaction usage so we don't double count
            # its dilution
            for rxn in cplx.reactions:
                stoich = rxn.metabolites[cplx]
                try:
                    if stoich<0 and \
                            hasattr(rxn.metabolites[cplx],'free_symbols') and \
                            mu in stoich.free_symbols and \
                            not isinstance(rxn, ProteinAbundance):
                        rxn.subtract_metabolites({cplx:rxn.metabolites[cplx]})
                except:
                    warnings.warn('get_cplx_cons: problem with cplx %s and rxn %s'%(
                        cplx.id, rxn.id))

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
            mm : IMEmodel
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

        # InertiaME model
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
            ex_flux_dict = {get_exchange_rxn(dme, metid, exchange_one_rxn=exchange_one_rxn).id:0.
                    for metid in conc_dict.keys()}
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
                    rxn = get_exchange_rxn(dme, metid, exchange_one_rxn=exchange_one_rxn)
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
                cplx_conc_dict_prime[cplx_id] = conc_new

            delay_model.update_cplx_concs(cplx_conc_dict_prime)
            #------------------------------------------------
            # Actually update metabolite concentrations using smallest required timestep.
            #------------------------------------------------
            for metid, conc in conc_dict.iteritems():
                v = 0.
                # If cobrame, exchange fluxes are just one -1000 <= EX_... <= 1000, etc.
                if exchange_one_rxn:
                    rxn = get_exchange_rxn(dme, metid, exchange_one_rxn=exchange_one_rxn)
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
