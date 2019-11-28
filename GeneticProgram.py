#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 10:52:52 2019

@author: 33fred33
"""
import math
import random as rd

#Global variables
logs_level = 0

class IndividualClass:
    def __init__(self, fenotype):
        self.fenotype = fenotype
        self.evaluation = None
        
    def __lt__(self, other): #less than
        for eval_ind in range(len(self.evaluation)):
            if self.evaluation[eval_ind] > other.evaluation[eval_ind]:
                return False
        return True
    
    def __str__(self):
        return str(self.fenotype)

class GeneticProgram:
    def __init__(
            self,
            population_size,
            generations,
            Model,
            multi_objective = False, 
            sampling_method="tournament",
            mutation_ratio=0.02,
            tournament_size=2,
            checkpoint_file_name = None):
        """
        Arguments:
        population_size
        generations
        Model: class containing the methods:
            generate_individual(): initialises a random individual
        multi_objective
        sampling_method: can be tournament / weighted_random / random
        mutation_ratio is the ratio of the next generation non-elite population to be filled with mutation-generated individuals
        tournament_size only used if sampling method is tournament. Best out from randomly selected individuals will be selected for sampling
        checkpoint_file_name
        """  
        
        #Variables assignment
        self.population_size = population_size
        self.generations = generations
        self.Model = Model
        self.multi_objective = multi_objective
        self.sampling_method = sampling_method
        self.mutation_ratio = mutation_ratio
        self.tournament_size = tournament_size
        self.checkpoint_file_name = checkpoint_file_name
        
        #General variables initialisation
        self.darwin_champion = None
        
    def fit(self):
        """
        
            
        """
        
        #Initial population initialisation
        self.population = [self.Model.generate_individual(self.population_size)]
        self._evaluate_population()
        
        #amounts of each population type and procedence     
        mutations = math.ceil(self.population_size * self.mutation_ratio)
        crossovers = self.population_size - mutations
            
        for generation in range(self.generations):
                
            #Parents selection
            if self.sampling_method == "tournament":    
                selected_first_parents = self._tournament_selection(self.population, crossovers)
                selected_second_parents = self._tournament_selection(self.population, crossovers)
                selected_mutations = self._tournament_selection(self.population, mutations) 
            elif self.sampling_method == "weighted_random":
                selected_first_parents = self._weighted_random_sample(self.population, crossovers)
                selected_second_parents = self._weighted_random_sample(self.population, crossovers)
                selected_mutations = self._weighted_random_sample(self.population, mutations)         
            
            #increase population
            self.population.extend([IndividualClass(self.Model.mutate(individual.fenotype)) for individual in selected_mutations])
            self.population.extend([IndividualClass(self.Model.crossover(selected_first_parents[i].fenotype, selected_second_parents[i].fenotype)) for i in range(crossovers)])
            
            #evaluate population
            self._evaluate_population()
            
            #select next generation's population
            self.population = sorted(self.population)[:self.population_size]
        
        #final individual selection
        self.darwin_champion = self.population[0]
        
        return self.darwin_champion
    
    def _evaluate_population(self):
        for individual in self.population:
            if individual.evaluation is None:
                individual.evaluation = self.Model.evaluate(individual.fenotype)
    
    def _weighted_random_sample(self, parent_population, amount, probabilities = None):
        """
        returns randomly selected individuals
        """
        sorted_population = sorted(parent_population)
        if probabilities is None:
            total_proportion = len(sorted_population)
            probabilities = []
            for i in range(total_proportion):
                probability = total_proportion/(total_proportion - i)
                probabilities.append(probability)
        sample = rd.choices(sorted_population, weights = probabilities, k = amount)
        return sample
    
    def _tournament_selection(self, parent_population, amount):
        selection = []
        for i in range(amount):
            competitors = rd.choices(parent_population, amount = self.tournament_size)
            winner = sorted(competitors)[0]
            selection.append(winner)               
        return selection
    
    def __str__(self):
        return str(self.__dict__)
    
    