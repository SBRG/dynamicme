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
        self.verbosity = 0

    def optimize(self, *args, **kwargs):
        """
        Recursively implement all problem modifications of parent-parent...
        Finally solve modified problem.
        """
        # Reset bounds
        self.reset_problem()

        # Inherit changes to problem including this node
        self.inherit_changes()

        # Solve
        sol = self.problem.optimize(*args, **kwargs)

        # Store value only the first time problem is solved
        if self.value is None:
            self.value = self.ObjVal

        return sol

    def update_problem(self, exclude_list=[]):
        problem = self.problem
        for k,(lb,ub) in iteritems(self.bound_dict):
            if k not in exclude_list:
                for sub in problem.sub_dict.values():
                    v = sub.model.getVarByName(k)
                    v.LB = lb
                    v.UB = ub
                    sub.model.update()
                    if self.verbosity>0:
                        print("sub=%s. k=%s. lb=%s. ub=%s"%(sub._id,k,lb,ub))


    def reset_problem(self):
        problem = self.problem
        for sub in problem.sub_dict.values():
            for y in sub._y0:
                v = sub.model.getVarByName(y.id)
                v.LB = y.lower_bound
                v.UB = y.upper_bound
                sub.model.update()

    def get_changelist(self, changelist):
        """
        Get change list. Should implemnt changes in reverse order.
        First call should provide changelist=[]
        """
        changelist.append(self.bound_dict)
        if self.parent is not None:
            changelist = self.parent.get_changelist(changelist)

        return changelist


    def inherit_changes(self):
        """
        Recursively inherit changes from parent
        """
        changelist = self.get_changelist([])
        # Implement parent changes first,
        # since children may overwrite them.
        problem = self.problem
        for c in reversed(changelist):
            for k,vs in iteritems(c):
                for sub in problem.sub_dict.values():
                    v = sub.model.getVarByName(k)
                    v.LB = vs[0]
                    v.UB = vs[1]
                    sub.model.update()

        # Inherit all changes along this node lineage
        # self.update_problem(exclude_list)
        # exclude_list += self.bound_dict.keys()
        # print("exclude_list=%s"%exclude_list)
        # parent = self.parent
        # if parent is not None:
        #     print("inheriting changes from parent=%s"%parent)
        #     parent.inherit_changes(exclude_list=exclude_list)

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
