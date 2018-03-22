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
from optimize import Variable, Constraint
from decomposition import Decomposer
from cobra.solvers import gurobi_solver

import gurobipy as grb
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

class RadixEstimator(Estimator):
    """
    Radix-based discretization estimator.
    """
    def __init__(self):
        self.prevent_zero = True
        self._INF = 1e3
        self.stacker = None
        self.var_cons_dict = {}
        self.radix = None
        self.powers = None
        self.digits = None
        self.kfit_dict = None
        self.col_ind = None


    def fit(self, base_model, df_X, df_Y, **kwargs):
        """
        Fit parameters given input (df_X) and output (df_Y) dataframes.

        Solver parameters in kwargs:
            radix (default: 2)
            powers (default: [-1,0,1])
            digits_per_power (default: radix)
            digits (default: list(set(np.linspace(1, radix-1, digits_per_power))) )
            fit_constraint_id (default: 'crowding')
            solver (default: 'gurobi')
            objective (default: 'minerr')
            col_meas_id (default: 'output_id')
            col_meas_val (default: 'output')
            col_ind (default: 'cond')
            reg_weight (default: 1e-3)
            max_nonzero_binaries (default: None)
            optimize : if True, solve now, else just build model (default: True).
        """

        radix = 2.
        powers = [-1,0,1]
        digits_per_power = radix
        digits = list(set(np.linspace(1, radix-1, digits_per_power)))
        fit_constraint_id = 'crowding'
        solver = 'gurobi'
        objective = 'minerr'
        col_meas_val = 'output'
        col_meas_id  = 'output_id'
        col_ind  = 'cond'
        reg_weight = 1e-3
        max_nonzero_binaries = None
        optimize = True

        #----------------------------------------------------
        # Parse kw arguments
        for k,v in iteritems(kwargs):
            if k=='radix':
                radix = v
            if k=='powers':
                powers = v
            elif k=='digits_per_power':
                digits_per_power = v
                digits = list(set(np.linspace(1, radix-1, digits_per_power)))
            elif k=='fit_constraint_id':
                fit_constraint_id = v
            elif k=='solver':
                solver = v
            elif k=='objective':
                objective = v
            elif k=='col_meas_id':
                col_meas_id = v
            elif k=='col_meas_val':
                col_meas_val = v
            elif k=='col_ind':
                col_ind = v
            elif k=='reg_weight':
                reg_weight = v
            elif k=='max_nonzero_binaries':
                max_nonzero_binaries = v
            elif k=='optimize':
                optimize = v

        if kwargs.has_key('digits'):
            if kwargs.has_key('digits_per_power'):
                warnings.warn('Both kwargs digits and digits_per_power provided. Using digits.')
            digits = kwargs['digits']
        #----------------------------------------------------
        self.digits = digits
        self.powers = powers
        self.radix  = radix
        self.col_ind= col_ind

        stacker = StackOptimizer()
        stacker.stack_models(base_model, df_X, col_ind)

        for mdl_ind, mdl in iteritems(stacker.model_dict):
            opt = Optimizer(mdl)
            gap = opt.add_duality_gap_constraint(INF=self._INF, inplace=True, index=mdl_ind)

        # Constraint-specific
        prevent_zero = self.prevent_zero
        cons_ref = base_model.metabolites.get_by_id(fit_constraint_id)
        var_cons_dict = {}
        for rxn_ref in cons_ref.reactions:
            for mdl_ind, mdl in iteritems(stacker.model_dict):
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

        opt.to_radix(stacker.model,var_cons_dict,
                radix,powers,digits=digits,prevent_zero=prevent_zero)

        if max_nonzero_binaries is not None:
            self.limit_nonzero_binaries(stacker.model, max_nonzero_binaries)

        #----------------------------------------------------
        # Update self
        self.var_cons_dict = var_cons_dict
        self.stacker = stacker
        self.make_objective(df_Y, col_meas_id, col_meas_val, col_ind,
                objective, reg_weight)
        # self.optimize(solver=solver)
        #----------------------------------------------------
        if solver=='gurobi':
            milp = gurobi_solver.create_problem(stacker.model)
            milp.ModelSense = grb.GRB.MINIMIZE
            milp.Params.IntFeasTol = 1e-9
            milp.Params.OutputFlag = 1
            # milp.Params.NodefileStart = 3
            # milp.Params.TimeLimit = 2*3600
            self.milp = milp
        else:
            raise ValueError("Currently need gurobi. Solver=%s not supported yet."%solver)

        if optimize:
            self.optimize(solver=solver)

    def optimize(self, solver='gurobi'):
        # Make solver-specific model (OPTIONAL)
        if solver=='gurobi':
            milp = self.milp
            milp.optimize()

            if milp.SolCount>0:
                solution = gurobi_solver.format_solution(milp, self.stacker.model)
                self.solution = solution
                self.stacker.model.solution = solution
                return solution
        else:
            raise ValueError("Currently need gurobi. Solver=%s not supported yet."%solver)


    def make_objective(self, df_meas, col_meas_id, col_meas_val, col_ind,
            objective='minerr', reg_weight=1e-3,
            reset_obj=True):
        """
        Make objective function of type objective.
        """
        INF = self._INF
        stacker = self.stacker
        model_dict = stacker.model_dict
        var_cons_dict = self.var_cons_dict
        digits = self.digits
        powers = self.powers
        radix  = self.radix

        if objective == 'minerr':
            for mdl_ind,mdl in iteritems(model_dict):
                if reset_obj:
                    for rxn in mdl.reactions:
                        rxn.objective_coefficient = 0.
                dfi = df_meas[ df_meas[col_ind]==mdl_ind]
                for rind,row in dfi.iterrows():
                    x_meas = row[col_meas_val]
                    meas_id = row[col_meas_id]
                    rxn_meas = mdl.reactions.get_by_id(meas_id+"_%s"%mdl_ind)

                    sp = Variable('sp_%s'%mdl_ind, lower_bound=0., upper_bound=INF)
                    sn = Variable('sn_%s'%mdl_ind, lower_bound=0., upper_bound=INF)
                    weight = (1-reg_weight)/(abs(x_meas) + 1) 
                    sp.objective_coefficient = weight
                    sn.objective_coefficient = weight
                    cons = Constraint('abs_err_%s'%mdl_ind)
                    cons._constraint_sense = 'E'
                    cons._bound = x_meas
                    mdl.add_metabolites(cons)
                    mdl.add_reactions([sp,sn])
                    sp.add_metabolites({cons:-1.})
                    sn.add_metabolites({cons:1.})
                    rxn_meas.add_metabolites({cons:1.})

            for group_id in var_cons_dict.keys():
                for l,pwr in enumerate(powers):
                    for k,digit in enumerate(digits):
                        yid = 'binary_%s%s%s'%(group_id,k,l)
                        y   = stacker.model.reactions.get_by_id(yid)
                        # Prefer pwr=0, digit=1
                        if pwr==0 and digit==1:
                            y.objective_coefficient=0.
                        else:
                            y.objective_coefficient=reg_weight
        else:
            raise Exception("Objective=%s not supported!"%(objective))

    def limit_nonzero_binaries(self, model, max_nonzero):
        """
        Add constraint to limit number of nonzero binaries
        """
        var_cons_dict = self.var_cons_dict
        cid = 'limit_binaries'
        if model.metabolites.has_id(cid):
            cons = model.metabolites.get_by_id(cid)
        else:
            cons = Constraint(cid)
        model.add_metabolites(cons)
        # sum y <= maxy
        miny = len(var_cons_dict)
        maxy = max(miny,max_nonzero)
        cons._bound = maxy
        cons._constraint_sense = 'L'
        for rxn in model.reactions.query('binary_'):
            rxn.add_metabolites({cons:1.})

    def get_params(self):
        """
        Extract estimated parameters.
        """
        kfit_dict = None
        milp = self.milp
        if milp.SolCount>0:
            x_dict = self.solution.x_dict
            kfit_dict = {}
            var_cons_dict = self.var_cons_dict
            for group_id,var_dict in iteritems(var_cons_dict):
                var = var_dict[0]
                cons= var_dict[1]
                a0  = var_dict[0][2]
                kfit= 0.
                for l,pwr in enumerate(self.powers):
                    for k,digit in enumerate(self.digits):
                        yid = 'binary_%s%s%s'%(group_id,k,l)
                        y   = x_dict[yid]
                        kfit+= y*a0*self.radix**pwr*digit
                kfit_dict[group_id] = kfit
            self.kfit_dict = kfit_dict

        else:
            raise Exception("Model not solved!")

        return kfit_dict

    def predict(self, df_X, base_model, meas_id=None,
            cons_fit_id='crowding',
            col_ind=None, kfit_dict=None):
        """
        Predict using estimated model given input (df_X) dataframe.
        """
        if col_ind is None:
            col_ind = self.col_ind

        if kfit_dict is None:
            kfit_dict = self.kfit_dict
            if kfit_dict is None:
                kfit_dict = self.get_params()

        rows = []

        if kfit_dict is None:
            raise Exception("No estimated parameters available")
        else:
            conds = df_X[col_ind].unique()
            for cond in conds:
                mdl_fit = base_model.copy()
                cons_fit = mdl_fit.metabolites.get_by_id(cons_fit_id)
                dfi = df_X[df_X[col_ind]==cond]
                for i,row in dfi.iterrows():
                    rid = row['rxn']
                    rxn = mdl_fit.reactions.get_by_id(rid)
                    rxn.lower_bound = row['lb']
                    rxn.upper_bound = row['ub']

                for rid,kfit in iteritems(kfit_dict):
                    rxn = mdl_fit.reactions.get_by_id(rid)
                    rxn.add_metabolites({cons_fit:kfit}, combine=False)

                mdl_fit.optimize()
                rxns = mdl_fit.reactions
                if meas_id is not None:
                    rxns = [mdl_fit.reactions.get_by_id(meas_id)]
                for rxn in rxns:
                    rows.append({'cond':cond, 'rxn':rxn.id, 'x':rxn.x})

        df_pred = pd.DataFrame(rows)

        return df_pred

    def parameters(self):
        """
        Return fitted parameters
        """


class CrossEstimator(Estimator):
    """
    Estimate using cross decomposition
    """

