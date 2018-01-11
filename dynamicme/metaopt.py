#============================================================
# File metaopt.py
#
# class  MetaOpt
#
# Generic meta-heuristic optimizer used also for 
# coarse-grained (surrogate) model
#
# Laurence Yang, SBRG, UCSD
#
# 08 Jan 2018:  first version
#============================================================

from six import iteritems
from cobra.core.Solution import Solution
from cobra import Reaction, Metabolite

import numpy as np
import copy as cp
import pandas as pd
import scipy.stats as stats
import time
import warnings
import cobra


#============================================================
class MetaOpt(object):
    """
    Methods for fitting parameters to measured conc or flux profiles
    Constructor:
    MetaOpt(mdl, sim_fun, obj_fun, get_param_fun, set_param_fun)

    mdl:            COBRAme or cobra model
    sim_fun:        Function to simulate
    obj_fun:        Objective function to minimize
    get_param_fun:  Function to get parameters
    set_param_fun:  Function to set parameters
    """

    def __init__(self, mdl, sim_fun, obj_fun, get_param_fun, set_param_fun):
        self.mdl = mdl
        self.sim_fun = sim_fun
        self.obj_fun = obj_fun
        self.get_param_fun = get_param_fun
        self.set_param_fun = set_param_fun

        random_move = LocalMove(get_param_fun, set_param_fun)
        self.move_objects = [random_move]
        self.stats = {}


    def calc_threshold(self, objval0, objval):
        T_rel = (objval - objval0) / abs(objval0 + 1.0)
        return T_rel


    def optimize(self, y_meas, Thresh0=1.0,
                max_iter_phase1=10, max_iter_phase2=100, max_reject=10,
                verbosity=2, param_confidence=0.95, use_gradient=True):
        """
        Tune parameters (e.g., keffs) to fit vector of measurements
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
        mdl = self.mdl
        opt_stats = []
        #----------------------------------------------------
        # Phase I: list filling
        #----------------------------------------------------
        Thresh = Thresh0
        Ts = [Thresh]

        # Get initial solution
        y_sim0 = self.sim_fun(mdl)
        objval0 = self.obj_fun(y_sim0, y_meas)

        # Perform local moves
        move_objects = self.move_objects
        n_iter = 0
        objval = objval0
        obj_best = objval0
        x0 = self.get_param_fun(mdl)
        x_pert = x0
        x_best = cp.copy(x0)
        y_sim = y_sim0
        y_best = cp.copy(y_sim)

        nx = len(x0)
        dzdxs_list = [np.zeros((max_iter_phase1, nx)) for mover in move_objects]

        while n_iter < max_iter_phase1:
            for i_mover, mover in enumerate(move_objects):
                #--------------------------------------------
                tic = time.time()
                #--------------------------------------------
                dzdxs1 = dzdxs_list[i_mover]
                # Local move
                if verbosity >= 1:
                    print '[Phase I] Iter %d:\t Performing local move:'%n_iter, type(mover)

                mover.move(mdl)
                x_prev = x_pert
                x_pert = self.get_param_fun(mdl)
                # Simulate
                y_sim = self.sim_fun(mdl)
                # Compute objective value (error)
                objval_prev = objval
                objval = self.obj_fun(y_sim, y_meas)
                # Unmove: generate samples surrounding initial point
                mover.unmove(mdl)

                # Keep track of dzdx
                dz = objval - objval_prev
                dx = np.array(x_pert) - np.array(x_prev)
                dzdx = dz/dx
                #try:
                dzdxs1[n_iter,:] = dzdx
                # except:
                #     print dzdx.shape
                #     print dzdxs1.shape
                #     raise Exception

                if objval < obj_best:
                    obj_best = objval
                    x_best = cp.copy(x_pert)
                    y_best = cp.copy(y_sim)

                # Calc relative cost deviation
                #T_rel = (objval - objval0) / (objval0 + 1.0)
                T_rel = self.calc_threshold(objval0, objval)

                Tmax = max(Ts)
                if T_rel <= Tmax and T_rel > 0:
                    Ts.append(T_rel)
                    Tmax = max(Ts)

                opt_stats.append({'phase':1, 'iter':n_iter, #'x_best':x_best,
                                  'obj':objval, 'objbest':obj_best,
                                  'Tmax':Tmax, 'Tk':T_rel})

                #--------------------------------------------
                toc = time.time()-tic
                #--------------------------------------------
                if verbosity >= 1:
                    print 'Obj:%g \t Best Obj: %g \t Tmax:%g \t T:%g \t Time:%g secs'%(
                        objval, obj_best, Tmax, T_rel, toc)
                    if verbosity >=2:
                        print 'y=', y_sim
                    print '//============================================'

            #------------------------------------------------
            n_iter = n_iter + 1
            #------------------------------------------------

        #----------------------------------------------------
        # Phase II: optimization
        #----------------------------------------------------
        n_reject = 0
        n_iter = 0
        while (n_iter < max_iter_phase2) and (n_reject < max_reject):
            for i_mover, mover in enumerate(move_objects):
                #--------------------------------------------
                tic = time.time()
                #--------------------------------------------
                # Local move
                # TODO: PARALLEL sampling and moves
                if verbosity >= 1:
                    print '[Phase II] Iter %d:\t Performing local move:'%n_iter, type(mover)

                dzdxs = dzdxs_list[i_mover]
                # Any dzdx consistently >0 or <0 ?
                m_cis = [mean_ci( dzdxs[:,i], confidence=param_confidence) for i in range(nx)]
                dzdx_m = np.array([m_ci[0] for m_ci in m_cis])
                dzdx_l = [m_ci[1] for m_ci in m_cis]
                dzdx_u = [m_ci[2] for m_ci in m_cis]

                for i in range(nx):
                    l = dzdx_l[i]
                    u = dzdx_u[i]
                    if np.sign(l) != np.sign(u):
                        dzdx_m[i] = 0.

                if not self.stats.has_key('gradient'):
                    self.stats['gradient'] = [None for i in move_objects]

                self.stats['gradient'][i_mover] = dzdx_m

                if use_gradient:
                    mover.move(mdl, dzdx=dzdx_m)
                else:
                    mover.move(mdl)

                x_pert = self.get_param_fun(mdl)
                # Simulate
                y_sim = self.sim_fun(mdl)
                objval = self.obj_fun(y_sim, y_meas)

                # Calc threshold and accept or reject move
                #T_new = (objval - objval0) / (objval0+1.0)
                T_new = self.calc_threshold(objval0, objval)
                T_max = max(Ts)
                move_str = ''
                if T_new <= T_max:
                    # Move if under threshold
                    objval0 = objval
                    if T_new > 0:
                        Ts.remove(max(Ts))
                        Ts.append(T_new)
                        Tmax = max(Ts)
                    if objval < obj_best:
                        x_best = cp.copy(x_pert)
                        obj_best = objval
                        y_best = cp.copy(y_sim)
                    move_str = 'accept'
                else:
                    n_reject = n_reject + 1
                    # Reject move: reset the model via unmove
                    # TODO: PARALLEL unmoves
                    mover.unmove(mdl)
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

            #------------------------------------------------
            n_iter += 1
            #------------------------------------------------

        # Save stats
        self.stats['gradients'] = dzdxs_list

        return y_best, opt_stats, x_best


#============================================================
# Local move methods (modifies model in place)
class LocalMove(object):
    """
    Class providing local move method

    Must implement these methods:
    move
    unmove: resets ME model to before the move
    """
    def __init__(self, get_param_fun, set_param_fun):
        self.get_param_fun = get_param_fun
        self.set_param_fun = set_param_fun
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
        # Memory to rollback to previous parameters
        self.x_rollback = None
        self.dydx = None

    def unmove(self, mdl):
        """
        Unmove to previous params
        """
        if self.x_rollback is None:
            print 'No pre-move params stored. Not doing anything'
        else:
            x0 = self.x_rollback

    def move(self, mdl, method='uniform', dzdx=None):
        """
        Randomly perturb me according to provided params
        """
        from numpy.random import uniform

        x0 = self.get_param_fun(mdl)
        n_pert = len(x0)
        ### Save params before move
        self.x_rollback = cp.copy(x0)
        param_dict = self.move_param_dict

        if param_dict.has_key(method):
            params = param_dict[method]
            if method is 'uniform':
                rmin = params['min']
                rmax = params['max']
                rs = np.random.uniform(rmin, rmax, n_pert)
                # Correct with gradient information if available
                if dzdx is not None:
                    # If obj increases with x, want to decrease x
                    rs[dzdx>0] = 10**(-abs(np.log10(rs[dzdx>0])))
                    rs[dzdx<0] = 10**(abs(np.log10(rs[dzdx<0])))

                # Perturb individually or in groups (all up/down)?
                self.set_param_fun(mdl, rs*x0)

            elif method is 'lognormal':
                norm_mean = params['mean']
                norm_std  = params['std']
                kmin  = params['min']
                kmax  = params['max']
                # Correct with gradient information if available
                es = np.random.normal(norm_mean, norm_std, n_pert)
                if dzdx is not None:
                    es[dzdx>0] = -abs(es[dzdx>0])
                    es[dzdx<0] = abs(es[dzdx<0])

                ks = 10**es
                ks[ks < kmin] = kmin
                ks[ks > kmax] = kmax
                self.set_param_fun(mdl, ks*x0)

            else:
                print 'Move method not implemented:', method

        else:
            warnings.warn('No parameters found for move: random')


def mean_ci(x, confidence=0.95):
    # Return mean and confidence interval
    n = len(x)
    m,se = np.mean(x), stats.sem(x)
    h = se*stats.t._ppf((1+confidence)/2., n-1)

    return m, m-h, m+h
