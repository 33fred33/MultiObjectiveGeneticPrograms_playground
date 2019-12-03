#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 10:52:52 2019

@author: 33fred33
"""
import math
import random as rd
import time

class IndividualClass:
    def __init__(self, fenotype):
        self.fenotype = fenotype
        self.evaluation = None
        
    def __lt__(self, other): #less than
        """
        Each evaluation value is the tiebreak for the previous one in the same individual
        """
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
            mutation_ratio=0.4,
            tournament_size=2,
            checkpoint_file_name = None):
        """
        Positional arguments:
            population_size
            generations
            Model: class containing the methods:
                generate_individual(n): initialises n random individuals
                mutate(individual): returns the mutated the individual
                crossover(individual1, individual2): returns the crossover offspring individual from the given individuals
        Keyword arguments:
            sampling_method: can be tournament / weighted_random / random. Default is tournament
            mutation_ratio is the ratio of the next generation non-elite population to be filled with mutation-generated individuals
            tournament_size only used if sampling method is tournament. Best out from randomly selected individuals will be selected for sampling. Default is 2
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
        self.logs_level = 1
        self.darwin_champion = None
        self.population = []
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.fitness_method = None
        
    def fit(self
        , x_train
        , y_train
        , fitness_method = "MSE"
        ):
        """
        Positional arguments:
            x_train
            y_train
        Keyword arguments:
            fitness_method can be MSE, SPEA2, NSGA2
        """  
        #variables assignment
        self.x_train = x_train
        self.y_train = y_train
        self.fitness_method = fitness_method
        
        #Initial population initialisation
        self.population = [IndividualClass(individual) for individual in self.Model.generate_population(self.population_size)]
        self._evaluate_population()
        if self.logs_level > 1:
            for i,ind in enumerate(self.population):
                print("\n ", str(i), "th individual's evaluation = ", ind.evaluation)
            input("wait!")
        
        #amounts of each population type and procedence     
        mutations = math.ceil(self.population_size * self.mutation_ratio)
        crossovers = self.population_size - mutations
        
        if self.logs_level >= 1:
            print("population_size: ", self.population_size)
            print("mutations per gen: ", mutations)
            print("crossovers per gen: ", crossovers)
            
        for generation in range(self.generations):
            
            if self.logs_level >= 1:
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
            
            if self.logs_level >= 1: 
                print("Generation time: ", str(time.time() - start_time))
                print("Darwin champion: ", self.population[0].fenotype)
                print("Darwin champion evaluations: ", self.population[0].evaluation)
                #print("Best individual so far: ", self.population[0].fenotype)
                if self.logs_level >= 2: 
                    for i,ind in enumerate(self.population):
                        print("\n ", str(i), "th individual's evaluation = ", ind.evaluation)
                    input("wait!") 
        
        #final individual selection
        self.darwin_champion = self.population[0].fenotype
        
        return self.darwin_champion
    
    def _evaluate_population(self, test = False): #needs to be changed
        if test:
            x = self.x_test
            y = self.y_test
        else:
            x = self.x_train
            y = self.y_train

        if self.fitness_method == "MSE":
            evaluations = [self.Model.evaluate(individual.fenotype, x) for individual in self.population]
            individuals_fitness = [self._MSE(y, y_predicted) for y_predicted in evaluations]

        else: #old
            individuals_fitness = self.Model.evaluate([individual.fenotype for individual in self.population])

        for i,individual in enumerate(self.population):
            individual.evaluation = individuals_fitness[i]

    def _MSE(self, y, y_predicted):
        """
        Positional arguments:
            y is a list of labels
            y_predicted is a predicted list of labels
        Returns: Mean Squared Error as a float
        """
        n = len(y)
        MSE = sum([pow(y[i]-y_predicted[i],2) for i in range(n)]) / n
        return MSE

    def _single_goal_accuracy(self, y, y_predicted, goal_class):
        """
        Positional arguments:
            y is a list of labels
            y_predicted is a predicted list of labels
            goal_class is the label to be compared
        Returns: y_predicted match ration with y, over the given goal class only, as a single float 
        """
        corrects = sum([1 if y_predicted[i] == self.y[i] and self.y[i] == goal_class else 0 for i in range(len(y))])
        total = sum([1 if self.y[i] == goal_class else 0 for i in range(len(y))])
        accuracy = corrects / total
        return accuracy

    def _crowding_distance(self, objective_values):
        """
        Positional arguments:
            objective_values is a list of lists, with ordered values for each objective to be considered
        Returns: a list of values with a crowding distance as a float
        """
        items = list(zip(*objective_values, list(range(len(objective_values[0])))))
        distances = defaultdict(list)
        for objective_idx in range(len(objective_values)):
            items.sort(key=lambda item: item[objective_idx])
            distances[items[0][-1]].append(-np.inf)
            distances[items[-1][-1]].append(-np.inf)
            for i in range(1, len(items) - 1):
                distances[items[i][-1]].append(items[i + 1][objective_idx] - items[i - 1][objective_idx])
        indexes_mean_distances = [(item_index, sum(ds) / len(objective_values)) for item_index, ds in distances.items()]
        indexes_mean_distances.sort(key=lambda t: t[0])
        crowding_distances = [d for i, d in indexes_mean_distances]
        return crowding_distances  
    
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
    
    