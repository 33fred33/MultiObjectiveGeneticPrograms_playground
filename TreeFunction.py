#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 10:52:52 2019

@author: 33fred33 and Kevin Galligan

Especifications:

safe divide returns numerator if denominator = 0
terminals can be integers used as indexes for features, or a random value between -1 and 1
operators: +, -, *, safe divide

mutation:
    subtree method:
        mutation point is a random node from the indiviidual to mutate
        mutation point is replaced with a random tree generated with koza's grow method, same max depth used for the initial population (3)
        while expected new depth after mutation is greater than the maximum allowed, a new mutation point is selected: the parent of the old mutation point

crossover:
    a random subtree is taken from the second parent
    a random node is selected in the first parent (excluding the root node to avoid depth=1 offsprings)
    this random node from the first parent is to be swapped with the second parent's random subtree
    if expected offspring's depth is greater than the maximum allowed, the second parent's subtree is replaced with a random subtree from the second parent's subtree
"""

from inspect import signature
import random as rd
import numpy as np
import math

class TreeFunctionClass:
    def __init__(self,
            features,
            operators,
            max_initial_depth = 5,
            min_initial_depth = 3,
            max_depth = 15,
            max_nodes = None,
            initialisation_method = "ramped half half",
            mutation_method = "subtree",
            bloat_control = "iteration"):
        """
        Positional arguments:
            features: number of features the tree will expect
            operators: a list with all the operations to be considered
        Keyword arguments:
            max_initial_depth: restricts depth of generated functions in their tree representation.
            min_initial_depth: restricts depth of generated functions in their tree representation.
            max_depth: stablishes the limit for the tree depth, to be tested after every mutation or crossover. Default is 15
            max_nodes: stablishes the limit for the tree nodes, to be tested after every mutation or crossover. Default is None and will be calculated with max_depth
            initialisation_method: can be "ramped half half", "grow", "full", "ramped full", "ramped grow", "half half". Default is ramped half half
            mutation_method: can be "subtree" or "unit". Default is subtree
            bloat_control can be iteration or minimum decrease
        """
        self.features = features
        self.operators = operators
        arities = []
        for operator in operators:
            sig = signature(operator)
            arity = len(sig.parameters)
            arities.append(arity)
        self.max_arity = max(arities)
        self.max_initial_depth = max_initial_depth
        self.min_initial_depth = min_initial_depth
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        if self.max_nodes is None:
            self.max_nodes = sum([math.pow(self.max_arity,i) for i in range(self.max_depth)])

        self.initialisation_method = initialisation_method
        self.mutation_method = mutation_method
        self.bloat_control = bloat_control
    
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
        Operators can commented if not needed as an option
        """
        return rd.choice(self.operators)
        
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
            for max_depth in range(self.min_initial_depth, self.max_initial_depth+1):
                first_half = math.ceil(part_size/2)
                population.extend([self._generate_individual_grow(max_depth) for _ in range(first_half)])
                second_half = part_size - first_half
                population.extend([self._generate_individual_full(max_depth) for _ in range(second_half)])

        elif self.initialisation_method == "ramped grow":
            parts = self.max_initial_depth - 1
            part_size = int(size / parts)
            remainder_size = size % parts
            population = [self._generate_individual_grow(self.max_initial_depth) for _ in range(remainder_size)]
            for max_depth in range(self.min_initial_depth, self.max_initial_depth+1):
                population = [self._generate_individual_grow(max_depth, self.min_initial_depth) for _ in range(part_size)]

        elif self.initialisation_method == "full":
            parts = self.max_initial_depth - 1
            part_size = int(size / parts)
            remainder_size = size % parts
            population = [self._generate_individual_full(self.max_initial_depth) for _ in range(remainder_size)]
            for max_depth in range(self.min_initial_depth, self.max_initial_depth+1):
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
            while parent.my_depth() + (subtree.my_depth() - node_to_overwrite.my_depth()) > self.max_depth or parent.nodes_count() + (subtree.nodes_count() - node_to_overwrite.nodes_count()) > self.max_nodes:
                if self.bloat_control == "iteration":
                    node_to_overwrite = rd.choice(new_individual.subtree_nodes())
                    subtree = self._generate_individual_grow(self.max_initial_depth, parent = node_to_overwrite.parent)
                elif self.bloat_control == "minimum decrease":
                    node_to_overwrite = node_to_overwrite.parent
                else:
                    print("No correct bloat control")
                    break

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
        while new_individual.my_depth() + (crossover_section.my_depth() - node_to_overwrite.my_depth()) > self.max_depth or new_individual.nodes_count() + (crossover_section.nodes_count() - node_to_overwrite.nodes_count()) > self.max_nodes:
            if crossover_section.is_terminal():
                print("In crossover node choice. This should never be reached")
                break
            if self.bloat_control == "iteration":
                node_to_overwrite = rd.choice(new_individual.subtree_nodes()[1:])
                crossover_section = rd.choice(second_parent.subtree_nodes()).copy()
            elif self.bloat_control == "minimum decrease":
                crossover_section = rd.choice(crossover_section.children)
            else:
                print("No correct bloat control")
                break

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

    def evaluate(self, node, x, threshold = 0, class_over_threshold = 1, class_below_threshold = 0):
        """
        Positional arguments:
            node: root node from the tree to be evaluated
            x: is the set of features
        Returns all outputs for each row in x
        """
        y = [self.evaluate_single_sample(node, sample) for sample in x]
        #y = [class_over_threshold if self.evaluate_single_sample(node, sample) >= threshold else class_below_threshold for sample in x]
        return y

    def evaluate_single_sample(self, node, sample):
        """
        Positional arguments:
            node: is the root of the tree to be evaluated
            sample: is the single set of features from x to obtain a single y value
        Returns a single value
        """
        if not node.is_terminal():
            arguments = [self.evaluate_single_sample(child, sample) for child in node.children]
            return node.content(*arguments)
        elif isinstance(node.content, int):
            return sample[node.content]
        else:
            return node.content

    def generate_from_string(self, string_representation, parent = None): #deprecated
        """
        Positional arguments:
            string_representation is a string with the str returned from this 
        Returns:
            Generated tree's root node
        """ 
        bracket = string_representation.split("(", 1)
        between_brackets = bracket[1].rsplit(")", 1)[0]
        print("between_brackets", between_brackets)
        split_space = between_brackets.split(" ")
        content = split_space[0].strip()
        for op in self.operators:
            if op.__name__ ==  content:
                sig = signature(op)
                arity = len(sig.parameters)
                root_node = Node(op, parent = parent)
                print("Match: op", op.__name__, "arity:", arity)
                for idx in range(arity):
                    child_idx = idx + 1
                    if split_space[child_idx][0] == "(":
                        print(between_brackets.split(" ", child_idx)[child_idx])
                        input("yay")
                        child_node = self.generate_from_string(between_brackets.split(" ", child_idx)[child_idx], root_node)
                        root_node.children.append(child_node)
                    elif split_space[child_idx][0] == "x":
                        value = int(split_space[child_idx][1:]) 
                        root_node.children.append(value)
                    else:
                        value = float(split_space[child_idx])
                        root_node.children.append(value)
                return root_node
    
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

    def nodes_count(self):
        """
        Returns the number of nodes in this tree (including this node as the root) as an int
        """
        return len(self.subtree_nodes())

    def my_depth(self, depth = 0):
        """
        Returns the max depth of this tree as an int
        """
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
            if isinstance(self.content, int):
                return "x" + str(self.content)
            else:
                return str(self.content)
        else:
            name_string = "(" + self.content.__name__
            for child in self.children:
                name_string += " " + str(child)
            name_string += ")"
            return name_string



