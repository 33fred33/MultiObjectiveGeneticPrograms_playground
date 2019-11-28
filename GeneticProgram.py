#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 10:52:52 2019

@author: 33fred33
"""
import math
import random as rd
import time

#Global variables
logs_level = 1

class IndividualClass:
    def __init__(self, fenotype):
        self.fenotype = fenotype
        self.evaluation = None
        
    def __lt__(self, other): #less than
        """
        Each evaluation in this individual is used as a tiebreak for the previous one
        """ #hola
        if isinstance(self.evaluation, list):
            for eval_ind in range(len(self.evaluation)):
                if self.evaluation[eval_ind] > other.evaluation[eval_ind]:
                    return False
                if self.evaluation[eval_ind] < other.evaluation[eval_ind]:
                    return True
        else:
            return self.evaluation < other.evaluation
    
    def __eq__(self, other):
        if isinstance(self.evaluation, list):
            for eval_ind in range(len(self.evaluation)):
                if self.evaluation[eval_ind] != other.evaluation[eval_ind]:
                    return False
            return True
        else:
            return self.evaluation == other.evaluation
    
    def __str__(self):
        return str(self.fenotype)

class GeneticProgramClass:
    def __init__(
            self,
            population_size,
            generations,
            Model,
            sampling_method="tournament",
            mutation_ratio=0.02,
            tournament_size=2,
            checkpoint_file_name = None):
        """
        Arguments:
        population_size
        generations
        Model: class containing the methods:
            generate_individual(n): initialises n random individuals
            mutate(individual): returns the mutated the individual
            crossover(individual1, individual2): returns the crossover offspring individual from the given individuals
        sampling_method: can be tournament / weighted_random / random
        mutation_ratio is the ratio of the next generation non-elite population to be filled with mutation-generated individuals
        tournament_size only used if sampling method is tournament. Best out from randomly selected individuals will be selected for sampling
        checkpoint_file_name
        """  
        
        #Variables assignment
        self.population_size = population_size
        self.generations = generations
        self.Model = Model
        self.sampling_method = sampling_method
        self.mutation_ratio = mutation_ratio
        self.tournament_size = tournament_size
        self.checkpoint_file_name = checkpoint_file_name
        
        #General variables initialisation
        self.darwin_champion = None
        self.population = []
        
    def fit(self):
        """
        Executes the genetic program
            
        """
        
        #Initial population initialisation
        self.population = [IndividualClass(individual) for individual in self.Model.generate_population(self.population_size)]
        self._evaluate_population()
        
        #amounts of each population type and procedence     
        mutations = math.ceil(self.population_size * self.mutation_ratio)
        crossovers = self.population_size - mutations
        
        if logs_level >= 1:
            print("population_size: ", self.population_size)
            print("mutations per gen: ", mutations)
            print("crossovers per gen: ", crossovers)
            
        for generation in range(self.generations):
            
            if logs_level >= 1:
                print("Generation: ", generation)
                start_time = time.time()
                
            #Parents selection
            if self.sampling_method == "tournament":    
                selected_first_parents = self._tournament_selection(self.population, crossovers)
                selected_second_parents = self._tournament_selection(self.population, crossovers)
                selected_mutations = self._tournament_selection(self.population, mutations) 
            elif self.sampling_method == "weighted_random":
                selected_first_parents = self._weighted_random_sample(self.population, crossovers)
                selected_second_parents = self._weighted_random_sample(self.population, crossovers)
                selected_mutations = self._weighted_random_sample(self.population, mutations)
            else:
                selected_first_parents = rd.choices(self.population, k = crossovers)
                selected_second_parents = rd.choices(self.population,  k = crossovers)
                selected_mutations = rd.choices(self.population,  k = mutations)
            
            #increase population
            self.population.extend([IndividualClass(self.Model.mutate(individual.fenotype)) for individual in selected_mutations])
            self.population.extend([IndividualClass(self.Model.crossover(selected_first_parents[i].fenotype, selected_second_parents[i].fenotype)) for i in range(crossovers)])
            
            #evaluate population
            self._evaluate_population()
            
            #select next generation's population
            self.population = sorted(self.population)[:self.population_size]
            
            if logs_level >= 1: 
                print("Generation time: ", str(time.time() - start_time))
                #print("Best individual so far: ", self.population[0].fenotype)
        
        #final individual selection
        self.darwin_champion = self.population[0]
        
        return self.darwin_champion
    
    def _evaluate_population(self):
        individuals_fitness = self.Model.evaluate([individual.fenotype for individual in self.population])
        for i,individual in enumerate(self.population):
            individual.evaluation = individuals_fitness[i]
    
    def _weighted_random_sample(self, parent_population, amount, probabilities = None):
        """
        returns randomly selected individuals 
        """
        sorted_population = sorted(parent_population)
        if probabilities is None:
            total_proportion = len(sorted_population)
            probabilities = []
            for i in range(total_proportion):
                probability = (total_proportion-i)/total_proportion
                probabilities.append(probability)
        sample = rd.choices(sorted_population, weights = probabilities, k = amount)
        return sample
    
    def _tournament_selection(self, parent_population, amount):
        selection = []
        for i in range(amount):
            competitors = rd.choices(parent_population, k = self.tournament_size)
            winner = sorted(competitors)[0]
            selection.append(winner)               
        return selection
    
    def __str__(self):
        return str(self.__dict__)
    
    