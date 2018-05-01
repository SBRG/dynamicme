#============================================================
# File branch_bound.py
#
#
# Branch and Bound for LR 
#
# Laurence Yang, SBRG, UCSD
#
# 27 Apr 2018:  first version
#============================================================

from __future__ import division
from six import iteritems
from builtins import range
from cobra.core import Solution
from cobra import Reaction, Metabolite, Model
from cobra import DictList
from cobra.solvers import gurobi_solver

from dynamicme.decomposition.LagrangeMaster import LagrangeMaster
from dynamicme.decomposition.LagrangeSubmodel import LagrangeSubmodel
from dynamicme.decomposition.ProblemTree import ProblemTree
from dynamicme.decomposition.ProblemTree import ProblemNode

import gurobipy as grb
import numpy as np
import cobra
import warnings
import time

ZERO = 1e-9

class LagrangeBB(object):
    def __init__(self, master, heuristics=['best_rounding']):
        """
        LagrangeBB()

        Implements branch and bound to recover primal feasible solution
        from Lagrangian Relaxation solutions.
        """
        self.tree  = ProblemTree()
        node = ProblemNode(master)
        self.tree.add(node)
        self.heuristics = heuristics
        self.verbosity = 1
        self.node_verbosity = 0
        self.sol_best = None

    def optimize(self, *args, **kwargs):
        """
        optimize()
        """
        INF = 1e100
        ZERO = 1e-9
        feasUB = INF
        heuristics = self.heuristics
        tree = self.tree
        verbosity = self.verbosity
        node_verbosity = self.node_verbosity

        _iter=0
        while not tree.isempty():
            if verbosity>0:
                print("%-10.8s. NODES TO EXPLORE: %d"%(_iter,tree.size()))
            _iter+=1
            node = tree.pop()

            #--------------------------------------------
            # Drop all cuts from before before solving.
            # TODO: But keep appropriate parent cuts.
            node.problem.reset()
            #--------------------------------------------

            node.verbosity = node_verbosity
            sol = node.optimize(*args, **kwargs)
            if node.feasible:
                zLD = node.ObjVal
            else:
                #--------------------------------------------
                # If infeasible drop this node
                continue

            #------------------------------------------------
            # Lagrange dual worse or same? Then drop this node.
            if zLD >= feasUB:
                continue
            else:
                problem = node.problem
                #Hy = self.calc_inconsistency(problem)
                #if sum(abs(Hy)) <= ZERO:
                incons = self.calc_inconsistency(problem)
                if sum(incons) <= ZERO:
                    # Already consistent
                    sub_ids = problem.sub_dict.keys()
                    sub0 = problem.sub_dict[sub_ids[0]]
                    obj_dict, feas_dict, sub_stats = self.check_feasible(sub0.yopt)
                    feasUBk = sum(obj_dict.values())
                    if feasUBk <= feasUB:
                        self.sol_best = sub0.x_dict
                        problem.yopt = sub0.yopt
                        feasUB = feasUBk
                else:
                    # Need to make consistent
                    for heuristic in heuristics:
                        yfeas, feasUBk, is_optimal = problem.feasible_heuristics(heuristic)
                        if yfeas is not None:
                            if feasUBk <= feasUB:
                                problem.yopt = yfeas
                                self.sol_best = {y.id:yj for y,yj in zip(problem._y0,yfeas)}
                                feasUB = feasUBk
                # Prune tree
                n_nodes = tree.size()
                tree.prune(feasUB, drop_below_cutoff=False)
                n_dropped = n_nodes - tree.size()
                if verbosity>0:
                    print("%d NODES DROPPED BY PRUNING: z > %g"%(n_dropped, feasUB))

                # Branch and add problems
                children = self.branch(node)
                if children:
                    tree.add(children)

        return self.sol_best

    def calc_inconsistency(self, master):
        """
        Check if consistency (non-anticipativity) constraints satisfied.
        Return amount of dispersion.
        """
        # Hys = [sub._H*sub.yopt for sub in master.sub_dict.values()]
        # Hy  = np.array(sum(Hys))
        yk_mat = np.array([sub.yopt for sub in master.sub_dict.values()])
        incons = np.var(yk_mat, axis=0)

        return incons


    def branch(self, node, rule="dispersion"):
        """
        Branch according to branching rule.
        Create branched problems approrpriately depending
        on binary, integer, continuous variable.
        """
        problem = node.problem
        verbosity = self.verbosity
        # Choose branching variable
        children = []
        if rule=='dispersion':
            #Hy = self.calc_inconsistency(problem)
            #max_disp = max(abs(Hy))
            incons = self.calc_inconsistency(problem)
            max_disp = max(incons)
            if max_disp > ZERO:
                #ind = np.where(Hy==max_disp)[0][0]
                ind = np.where(incons==max_disp)[0][0]
                yval = problem.yopt[ind]
                if verbosity>0:
                    print("y[%d] = %s"%(ind,yval))

                yj = problem._y0[ind]
                #yl = yj.lower_bound     # Unchanged
                #yu = yj.upper_bound     # Unchanged
                if node.bound_dict.has_key(yj.id):
                    yl,yu = node.bound_dict[yj.id]
                else:
                    yl = yj.lower_bound
                    yu = yj.upper_bound

                if yu <= 1:
                    # If binary
                    child0 = ProblemNode(problem, parent=node)
                    child0.bound_dict = {yj.id:(yl, 0)}
                    child1 = ProblemNode(problem, parent=node)
                    child1.bound_dict = {yj.id:(1, yu)}
                    children.append(child0)
                    children.append(child1)
                else:
                    # Just integer
                    child0 = ProblemNode(problem, parent=node)
                    child0.bound_dict = {yj.id:(yl, yval)}
                    child1 = ProblemNode(problem, parent=node)
                    child1.bound_dict = {yj.id:(yval+1, yu)}
                    children.append(child0)
                    children.append(child1)
                if verbosity>0:
                    print("BRANCHING ON y[%d]: %s"%(ind,yj.id))
        else:
            raise ValueError("Unknown branching rule: %s"%rule)

        return children
