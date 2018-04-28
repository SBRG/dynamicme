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

import gurobipy as grb
import numpy as np
import cobra
import warnings
import time


class ProblemNode(object):
    """
    Node pointing to base Problem without
    actually copying Problem object.
    """
    def __init__(self, problem, parent=None):
        self.problem = problem
        self.parent  = parent
        self.value   = None
        self.bound_dict = {}

    def optimize(self, *args, **kwargs):
        """
        Recursively implement all problem modifications of parent-parent...
        Finally solve modified problem.
        """
        # Inherit changes to problem including this node
        self.inherit_changes()

        # Solve
        sol = self.problem.optimize(*args, **kwargs)

        # Store value only the first time problem is solved
        if self.value is None:
            self.value = self.ObjVal

        return sol

    def update_problem(self):
        problem = self.problem
        for sub in problem.sub_dict.values():
            for k,(lb,ub) in iteritems(self.bound_dict):
                v = sub.model.getVarByName(k)
                v.LB = lb
                v.UB = ub

    def reset_problem(self):
        problem = self.problem
        for sub in problem.sub_dict.values():
            for y in sub._y0:
                v = sub.model.getVarByName(y.id)
                v.LB = y.lower_bound
                v.UB = y.upper_bound

    def inherit_changes(self):
        """
        Recursively inherit changes from parent
        """
        # Reset bounds
        self.reset_problem()

        # Inherit all changes along this node lineage
        self.update_problem()
        parent = self.parent
        if parent is not None:
            parent.inherit_changes()

    @property
    def feasible(self):
        return self.problem.model.SolCount > 0

    @property
    def ObjVal(self):
        return self.problem.ObjVal


    @property
    def current_value(self):
        # Value updated when problem solved.
        problem = self.problem
        value = problem.ObjVal
        return value

    @property
    def children(self):
        pass


class ProblemTree(object):
    """
    Tree grown from base Problem without
    actually copying Problem object.
    """
    def __init__(self):
        self.nodes = []

    def add(self, nodes):
        if hasattr(nodes,'__iter__'):
            self.nodes += nodes
        else:
            self.nodes.append(nodes)

    def pop(self):
        node = self.nodes.pop()
        return node

    def isempty(self):
        return len(self.nodes)==0

    def prune(self, cutoff, drop_below_cutoff=True):
        """
        Drop nodes with values below/above cutoff
        cutoff :  cutoff value for dropping nodes
        sense : 'below' or 'above' to drop nodes with values
                below or above
        """
        if drop_below_cutoff:
            kept_nodes = [k for k in self.nodes if k.value >= cutoff]
        else:
            kept_nodes = [k for k in self.nodes if k.value <= cutoff]

        self.nodes = kept_nodes

    def size(self):
        return len(self.nodes)
