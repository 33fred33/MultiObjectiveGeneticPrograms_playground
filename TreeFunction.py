#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 10:52:52 2019

@author: 33fred33 and Kevin Galligan

Deep specifications:

safe divide returns numerator if denominator = 0
terminals can be integers used as indexes for features, or a random value between -1 and 1
operators: +, -, *, safe divide

mutation:
    subtree method:
        mutation point is a random node is selected from the indiviidual to mutate
        mutation point is replaced with a random tree generated with koze's grow method and same max depth used for the initial population
        while expected new depthafter mutation is greater than the maximum allowed, the new mutation point is now the parent of the old mutation point

crossover:
    a random subtree is taken from the second parent
    a random node is selected in the first parent (excluding the root node to avoid depth=1 offsprings)
    this random node is to be swapped with the second parent's subtree
    if expected offspring's depth is greater than the maximum allowed, the second parent's subtree is replaced with a random subtree of itself
"""



import operator
from inspect import signature
import random as rd
import math
import numpy as np

def safe_divide(a, b):
    """
    Arguments:
        a is a number
        b is a number
    Executes a/b. If b=0, returns a
    """
    if b == 0 :
        return a
    else:
        return a/b

class TreeFunctionClass:
    def __init__(self,
            features,
            evaluate_fitness_f,
            max_initial_depth = 3,
            max_depth = 15,
            initialisation_method = "ramped half half",
            mutation_method = "subtree"):
        """
        Positional arguments:
            features: number of features the tree will expect
            evaluate_fitness_f: function to be called when evaluation is needed
                Positional arguments:
                    population: population to be evaluated.
                Returns the list of evaluations
        Keyword arguments:
            max_initial_depth: restricts depth of generated functions in their tree representation. Default is 3
            max_depth: stablishes the limit for the tree depth, to be tested after every mutation or crossover. Default is 15
            initialisation_method: can be "ramped half half", "grow", "full", "ramped full", "ramped grow", "half half". Default is ramped half half
            mutation_method: can be "subtree" or "unit". Default is subtree
        """
        self.features = features
        self.evaluate = evaluate_fitness_f
        self.max_initial_depth = max_initial_depth
        self.max_depth = max_depth
        self.initialisation_method = initialisation_method
        self.mutation_method = mutation_method
    
    def _generate_terminal(self):
        """
        Generates a number that can be an index for the features vector, or a random number between -1 and 1
        """
        value = rd.randint(0, self.features)
        if value == self.features:
            value = np.random.uniform(low=-1, high=1)
        return value

    def _generate_operator(self):
        """
        Generates an operator.
        Operators can be add, sub, mul or safe divide
        """
        return rd.choice([
            operator.add,
            operator.sub,
            operator.mul,
            safe_divide])
        
    def _generate_individual_full(self, max_depth, parent=None, depth=0): #can be mixed with full
        """
        Generates a random individual using Koza's full method
        """
        if depth == max_depth - 1:
            terminal = self._generate_terminal()
            return Node(terminal, parent = parent)
        else:
            operator = self._generate_operator()
            sig = signature(operator)
            arity = len(sig.parameters)
            node = Node(operator, parent = parent)
            for _ in range(arity):
                node.children.append(self._generate_individual_full(max_depth, parent=node, depth=depth+1))
            return node

    def _generate_individual_grow(self, max_depth, parent=None, depth=0):
        """
        Generates a random individual using Koza's grow method
        """
        if depth == max_depth - 1:
            terminal = self._generate_terminal()
            return Node(terminal, parent = parent)
        else:
            if rd.choice([True, False]) or depth == 0:
                operator = self._generate_operator()
                sig = signature(operator)
                arity = len(sig.parameters)
                node = Node(operator, parent = parent)
                for _ in range(arity):
                    node.children.append(self._generate_individual_full(max_depth, parent=node, depth=depth+1))
                return node
            else:
                terminal = self._generate_terminal()
                return Node(terminal, parent = parent)
        
    def generate_population(self, size):
        """
        Positional arguments:
            size: is an integer
        Generates a population of size: size
        """
        if self.initialisation_method == "half half":
            first_half = math.ceil(size/2)
            population = [self._generate_individual_grow(self.max_initial_depth) for _ in range(first_half)]
            second_half = size - first_half
            population.extend([self._generate_individual_full(self.max_initial_depth) for _ in range(second_half)])

        elif self.initialisation_method == "grow":
            population = [self._generate_individual_grow(self.max_initial_depth) for _ in range(size)]

        elif self.initialisation_method == "full":
            population = [self._generate_individual_full(self.max_initial_depth) for _ in range(size)]

        elif self.initialisation_method == "ramped half half":
            parts = self.max_initial_depth - 1
            part_size = int(size / parts)
            remainder_size = size % parts
            population = [self._generate_individual_full(self.max_initial_depth) for _ in range(remainder_size)]
            for max_depth in range(2, self.max_initial_depth+1):
                first_half = math.ceil(part_size/2)
                population.extend([self._generate_individual_grow(max_depth) for _ in range(first_half)])
                second_half = part_size - first_half
                population.extend([self._generate_individual_full(max_depth) for _ in range(second_half)])

        elif self.initialisation_method == "ramped grow":
            parts = self.max_initial_depth - 1
            part_size = int(size / parts)
            remainder_size = size % parts
            population = [self._generate_individual_grow(self.max_initial_depth) for _ in range(remainder_size)]
            for max_depth in range(2, self.max_initial_depth+1):
                population = [self._generate_individual_grow(max_depth) for _ in range(part_size)]

        elif self.initialisation_method == "full":
            parts = self.max_initial_depth - 1
            part_size = int(size / parts)
            remainder_size = size % parts
            population = [self._generate_individual_full(self.max_initial_depth) for _ in range(remainder_size)]
            for max_depth in range(2, self.max_initial_depth+1):
                population = [self._generate_individual_full(max_depth) for _ in range(part_size)]

        else:
            print("No correct initialisation method was given")
        return population
    
    def mutate(self, parent):
        """
        Positional arguments:
            parent: node
        Returns the same tree with a mutation applied
        """
        new_individual = parent.copy()

        if self.mutation_method == "unit":
            mutation_point = rd.choice(new_individual.subtree_nodes())
            if mutation_point.is_terminal():
                mutation_point.content = self._generate_terminal()
            else:
                mutation_point.content = self._generate_operator() #MISSING: specify arity
                #MISSING: match with return

        elif self.mutation_method == "subtree":
            node_to_overwrite = rd.choice(new_individual.subtree_nodes())
            subtree = self._generate_individual_grow(self.max_initial_depth, parent = node_to_overwrite.parent)

            #max depth handling
            while parent.my_depth() + (subtree.my_depth() - node_to_overwrite.my_depth()) > self.max_depth:
                node_to_overwrite = node_to_overwrite.parent

            #node replacing
            if node_to_overwrite.is_root():
                return subtree
            else:
                fooled_parent = node_to_overwrite.parent
                for i,child in enumerate(fooled_parent.children):
                    if child == node_to_overwrite:
                        fooled_parent.children[i] = subtree
                        break

            while fooled_parent.parent is not None:
                fooled_parent = fooled_parent.parent

        return fooled_parent
        
    def crossover(self, first_parent, second_parent):
        """
        Positional arguments:
            first_parent: node
            second_parent: node
        Returns the offspring tree resulting from the crossover between the first_parent and the second_parent
        """
        crossover_section = rd.choice(second_parent.subtree_nodes()).copy()
        new_individual = first_parent.copy()
        node_to_overwrite = rd.choice(new_individual.subtree_nodes()[1:])

        #max depth handling
        while crossover_section.my_depth() + node_to_overwrite.my_depth() > self.max_depth + 1:
            if crossover_section.is_terminal():
                print("In crossover node choice. This should never be reached")
                break
            print("NEEDED TO CHANGE!")
            print("crossover_section.my_depth()",crossover_section.my_depth())
            print("node_to_overwrite.my_depth()",node_to_overwrite.my_depth())
            crossover_section = rd.choice(crossover_section.children)

        if node_to_overwrite.is_root():
            print("In crossover overwrite. This should never be reached")
            return crossover_section
        else:
            parent = node_to_overwrite.parent
            for i,child in enumerate(parent.children):
                if child == node_to_overwrite:
                    parent.children[i] = crossover_section
                    crossover_section.parent = parent
                    break

            while crossover_section.parent is not None:
                crossover_section = crossover_section.parent

        return crossover_section
    
class Node:
    def __init__(self, content, *children, parent = None,):
        """
        Positional arguments:
            content: can be an operator or a number
            childs: expects any number of nodes from this same class
        Keyword arguments:
            parent: expects a node from this same class
                default: None
        """
        self.content = content
        self.parent = parent
        self.children = []
        for child in children:
            self.children.append(child)
    
    def is_terminal(self):
        return self.children == []
    
    def is_root(self):
        return self.parent is None

    def subtree_nodes(self):
        """
        Returns a list with all the nodes of the subtree with self as the root node.
        Includes self in the list
        """
        nodes = [self]
        i = 0
        while i < len(nodes):
            if not nodes[i].is_terminal():
                nodes.extend(nodes[i].children)
            i += 1
        return nodes

    def my_depth(self, depth = 0):
        new_depth = depth + 1
        if self.is_terminal():
            return new_depth
        else:
            return max([child.my_depth(new_depth) for child in self.children])
    
    def copy(self, parent=None):
        """
        Don't give arguments
        Returns an unrelated new individual with the same characteristics
        """       
        the_copy = Node(self.content, parent = parent)
        if not self.is_terminal():
            for child in self.children:
               the_copy.children.append(child.copy(parent = the_copy))
        return the_copy

    def __eq__(self, other):
        if self.is_terminal and other.is_terminal:
            return self.content == other.content
        else:
            children_length = len(self.children)
            if children_length != len(other.children):
                return False
            else:
                for i in range(children_length):
                    if not self.__eq__(self.children[i], other.children[i]):
                        return False
                return True

    def __str__(self):
        if self.is_terminal():
            return str(self.content)
        else:
            name_string = "(" + self.content.__name__
            for child in self.children:
                name_string += " " + str(child)
            name_string += ")"
            return name_string





