#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 10:52:52 2019

@author: 33fred33


Especifications:

GP structure:
    - Initial population is obtained and evaluated
    - The amount of mutations and crossovers to happen at each generation is established according to the population size
    - Generation structure:
        Offsprings are calculated using crossover and mutation only
        Parents and offsprings are evaluated together
        The best ones stay in the next generation's population

Tournament selection of size n: n individuals are uniformly randomly picked from the population. The best one is returned

"""

#TO DO:
#GENERIC VOTATION
#NSGA2
#Bloat control by lenght, not depth only
#no tiebreak if doesnt matter
# max tree depth and size to set between 0 and 1
#store tree evaluations?

import math
import random as rd
import time
import numpy as np
import matplotlib.pyplot as plt
import pylab as py
from collections import defaultdict
import datetime
import os
import errno
import csv
import pickle
#from scipy.interpolate import interp1d

class IndividualClass:
    def __init__(self, fenotype, objective_values = None):
        self.fenotype = fenotype
        self.evaluation = []
        self.objective_values = objective_values # should be between 0 and 1 (0 is the best)
        #self.relative_objective_values = None #to test
        
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
        return "Fenotype: " + str(self.fenotype) + " Evaluations: " + str(self.evaluation)

class GeneticProgramClass:
    def __init__(
            self,
            population_size,
            generations,
            Model,
            objective_functions,
            objective_functions_arguments = None, #[[f1_arg1, ..., f1_argn], [f2_arg1, ..., f2_argn], ..., [fn_arg1, ..., fn_argn]]
            multiobjective_fitness = "SPEA2",
            sampling_method="tournament",
            mutation_ratio=0.4,
            tournament_size=2,
            experiment_name = None,
            ensemble_type = "baseline"):
        """
        Positional arguments:
            population_size
            generations
            Model: class containing the methods:
                generate_individual(n): initialises n random individuals
                mutate(individual): returns the mutated the individual
                crossover(individual1, individual2): returns the crossover offspring individual from the given individuals
            objective_functions: expects an array of functions with the following characteristics:
                Positional arguments:
                    y: labels or classes to be achieved
                    y_predicted: a list of results to be compared with
                    Returns a single float
        Keyword arguments:
            objective_functions_arguments: list of list as: [[f1_arg1, ..., f1_argn], [f2_arg1, ..., f2_argn], ..., [fn_arg1, ..., fn_argn]]
            multiobjective_fitness: can be SPEA2 or NSGA2. Default is SPEA2
            sampling_method: can be tournament / weighted_random / random. Default is tournament
            mutation_ratio is the ratio of the next generation non-elite population to be filled with mutation-generated individuals
            tournament_size only used if sampling method is tournament. Best out from randomly selected individuals will be selected for sampling. Default is 2
            experiment_name is a string, used to create a new folder to store the outputs
            ensemble_type can be rpf or baseline. Default is baseline, which considers the full pareto
        """  
        
        #Positional arguments variables assignment
        self.population_size = population_size
        self.generations = generations
        self.Model = Model
        if not isinstance(objective_functions, list):
            objective_functions = [objective_functions]
        self.objective_functions_names = objective_functions
        self.objective_functions = []
        for objective_function_name in objective_functions:
            self.objective_functions.append(self._objective_function_parser(objective_function_name))

        #Keyword arguments variables assignment
        if objective_functions_arguments is None:
            self.objective_functions_arguments = [[] for _ in range(len(self.objective_functions))]
        else:
            self.objective_functions_arguments = objective_functions_arguments
        self.multiobjective_fitness = multiobjective_fitness
        self.sampling_method = sampling_method
        self.mutation_ratio = mutation_ratio
        self.tournament_size = tournament_size
        if experiment_name is None:
            now = datetime.datetime.now()
            self.experiment_name = str(now.year) + "-" + str(now.month) + "-" + str(now.day) + "-" + str(now.hour) + "-" + str(now.minute)
        else:
            self.experiment_name = experiment_name
        self.ensemble_type = ensemble_type
        
        #General variables initialisation
        self.logs_level = 0
        self.objectives = len(self.objective_functions)
        self.darwin_champion = None
        self.population = []
        self.x = []
        self.y = []
        self.logs = {}
        self.genlogs = {}
        self.ran_generations = 0
        self.last_gen_time = 0

        
    def fit(self, x, y):
        """
        Positional arguments:
            x is data as a list of lists
            y is a list of labels
        """  
        #variables assignment
        self.x = x
        self.y = y
        self.ran_generations = 0
        
        #Initial population initialisation
        self.population = [IndividualClass(individual) for individual in self.Model.generate_population(self.population_size)]
        self._evaluate_population()
        self.population = sorted(self.population)

        self.logs_checkpoint()

        if self.logs_level > 1:
            for i,ind in enumerate(self.population):
                print("\n ", str(i), "th individual's evaluation = ", ind.evaluation)
            input("wait!")
        
        self.train(self.generations)
        #return self.logs, self.genlogs
    
    def get_logs(self):
        """
        Returns logs, genlogs
        logs contains data by individual from all generations
        genlogs contains data by generation from all generations
        """
        return self.logs, self.genlogs
    
    def train(self, generations):
        """
        Positional arguments:
            generations is an int
        """
        mutations = math.ceil(self.population_size * self.mutation_ratio)
        crossovers = self.population_size - mutations
            
        for generation in range(generations):
            self.ran_generations += 1
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
            self.last_gen_time = time.time() - start_time
            self.logs_checkpoint()

            #print("\nGeneration ", generation, " time: ", str(time.time() - start_time))
            print("\nGeneration ", generation)
            for key, value in self.genlogs.items():
                if key[0] == generation:
                    print(key, value)

    def get_best_individual(self, objective_index=None):
        if objective_index is None:
            best_individual = self.population[0]
        else:
            best_individual = sorted(self.population, key=lambda ind: ind.objective_values[objective_index])[0]
        return best_individual

    def get_ensemble(self, ensemble_type=None, rpf_objective_indexes_to_care=None):
        """
        Keyword arguments:
            ensemble types are baseline or rpf
            rpf_objective_indexes_to_care is a list of objective indexes as integers
        Returns ensemble of individuals in the pareto front in a list
        """
        if ensemble_type is None:
            ensemble_type = self.ensemble_type
        ensemble = []

        if ensemble_type == "baseline":
            for individual in self.population:
                if individual.evaluation[0] == 0:  # choose only pareto front individuals
                    ensemble.append(individual)

        elif ensemble_type == "rpf":
            if rpf_objective_indexes_to_care is None: 
                rpf_objective_indexes_to_care = [i for i in range(self.objectives)]
            
            for individual in self.population:
                useful_individual = True
                if individual.evaluation[0] > 0:
                    useful_individual = False
                else:
                    for obj_idx, obj_value in enumerate(individual.objective_values):
                        if obj_idx in rpf_objective_indexes_to_care:
                            if obj_value >= 0.5: #Minimize or maximize dependency
                                useful_individual = False
                                break
                if useful_individual:
                    ensemble.append(individual)
        else:
            print("wrong ensemble type")
        if len(ensemble)==0:
            print("Not a single individual did meet ensemble criteria")
        
        return ensemble

    def get_ensemble_votations(self, x=None, y=None, ensemble=None, ensemble_decision="votation"):
        """
        Assumes binary classification between class 0 and 1 (ints)
        Returns a list with votation rates for class 0 and class 1
        """
        if x is None or y is None:
            x = self.x 
            y = self.y
        samples = len(y)
        if ensemble is None:
            ensemble = self.get_ensemble()

        y_predicted_collection = []
        for individual in ensemble:
            values = self.Model.evaluate(individual.fenotype, x)
            y_predicted_collection.append(map_to_binary(values))
        y_predicted_collection = np.array(y_predicted_collection)

        if ensemble_decision == "votation":
            y_predicted_votations = [[0,0] for _ in range(samples)]
            for sample_idx in range(samples):
                votations_list = list(y_predicted_collection[:,sample_idx])
                y_predicted_votations[sample_idx] = [votations_list.count(0)/len(ensemble), votations_list.count(1)/len(ensemble)]
            
        return y_predicted_votations

    def evaluate_ensemble_accuracy(self, x=None, y=None, ensemble=None, ensemble_decision="votation"):
        """
        Returns overall ensemble accuracy
        """
        if x is None or y is None:
            x = self.x 
            y = self.y
        if ensemble is None:
            ensemble = self.get_ensemble()

        y_predicted_votations = self.get_ensemble_votations(x, y, ensemble, ensemble_decision)
        y_predicted = []
        for votations in y_predicted_votations:
            if votations[0] > votations[1]: #if tied, 1
                y_predicted.append(0)
            else:
                y_predicted.append(1)
        corrects = sum([1 if y_predicted[i] == y[i] else 0 for i in range(len(y))])
        accuracy = corrects / len(y)

        return accuracy

    def _objective_function_parser(self,name):
        if name == "single_goal_accuracy":
            objective_function = self.single_goal_accuracy
        elif name == "mse":
            objective_function = self.mse
        elif name == "rmse":
            objective_function = self.rmse
        elif name == "errors_by_threshold":
            objective_function = self.errors_by_threshold
        elif name == "accuracy":
            objective_function = self.accuracy
        elif name == "tree_depth":
            objective_function = self.tree_depth
        elif name == "tree_size":
            objective_function = self.tree_size
        elif name == "sum_absolute_errors":
            objective_function = self.sum_absolute_errors
        elif name == "mean_absolute_errors":
            objective_function = self.mean_absolute_errors
        else:
            print("wrong objective function name")
        return objective_function

    
    def _evaluate_population(self):
        """
        Evaluates the entire population
        """

        for ind_idx, individual in enumerate(self.population):
            if individual.objective_values is None:
                individual.objective_values = []
                for obj_idx, objective_function in enumerate(self.objective_functions):
                    individual.objective_values.append(objective_function(individual, *self.objective_functions_arguments[obj_idx]))
                    
        if self.objectives == 1:
            for ind_idx, individual in enumerate(self.population):
                individual.evaluation = individual.objective_values
        else:
            if self.multiobjective_fitness == "SPEA2":
                objective_values = [[individual.objective_values[obj_idx] for individual in self.population] for obj_idx in range(self.objectives)] #added
                evaluations = self._spea2(objective_values)

                for ind_idx, individual in enumerate(self.population):
                    individual.evaluation = [evaluations[ind_idx]]

                ### for plotting
                logaritmical_plots = [False,False]
                if self.objective_functions_names[0] == "mse" or self.objective_functions_names[0] == "rmse":
                    logaritmical_plots[0] = True
                if self.objective_functions_names[1] == "mse" or self.objective_functions_names[1] == "rmse":
                    logaritmical_plots[1] = True

                title = "Gen " + str(self.ran_generations)
                colored_plot(objective_values[0], 
                                  objective_values[1], 
                                  evaluations, 
                                  title = title, 
                                  colormap = "cool", 
                                  markers = evaluations,
                                  marker_size = 200,
                                  save = True,
                                  path = self.experiment_name,
                                  logaritmical = logaritmical_plots)
                ### end plotting
        
            elif self.multiobjective_fitness == "NSGA2": # pending
                pass 

            crowding_distances = self._crowding_distance()
            for ind_idx, individual in enumerate(self.population):
                individual.evaluation.append(crowding_distances[ind_idx])

    def _crowding_distance(self, objective_indexes_to_ignore=None):
        """
        Positional arguments:
            objective_indexes_to_ignore is a list of lists, with ordered values for each objective to be considered
        Returns: a list of values with a crowding distance as a float. 0 means the one with higher distance
        """
        crowding_distances = [[0 for _ in range(self.objectives)] for _ in range(len(self.population))]
        objective_values_list = [[] for _ in range(self.objectives)]
        for obj_idx in range(self.objectives):
            #interpolate
            temp_objective_list = [ind.objective_values[obj_idx] for ind in self.population]
            #This to interpolate and give same importance?
            max_ov = max(temp_objective_list)
            min_ov = min(temp_objective_list)
            if max_ov == min_ov:
                print("In crowding distance: objective values are in the same range, skipping interpolation")
            elif abs(max_ov) == np.inf or abs(min_ov) == np.inf:
                print("In crowding distance: objective value is infinite, skipping interpolation")
            else:
                #interpolate_function = interp1d([min_ov, max_ov],[0,1])
                #temp_objective_list = interpolate_function(temp_objective_list)
                temp_objective_list = [np.interp(value,[min_ov, max_ov],[0,1]) for value in temp_objective_list]
            
            objective_values_list[obj_idx] = sorted([(obj_v, ind_idx) for ind_idx, obj_v in enumerate(temp_objective_list)], key = lambda x: x[0])

        for obj_idx in range(self.objectives):
            for sorted_idx, (objective_value, ind_idx) in enumerate(objective_values_list[obj_idx]):
                if sorted_idx == 0 or sorted_idx == len(objective_values_list[obj_idx]) - 1:
                    crowding_distances[ind_idx][obj_idx] = np.inf
                else:
                    abs1 = abs(objective_values_list[obj_idx][sorted_idx-1][0] - objective_value)
                    abs2 = abs(objective_values_list[obj_idx][sorted_idx+1][0] - objective_value)
                    crowding_distances[ind_idx][obj_idx] = (abs1 + abs2)/2

        crowding_distance = [np.mean(crowding_distances[ind_idx]) for ind_idx in range(len(self.population))]
        max_cd = max([x for x in crowding_distance if x != np.inf])
        inverted_crowding_distances = [max_cd - cd if cd != np.inf else -np.inf for cd in crowding_distance]

        return inverted_crowding_distances

    def _spea2(self, objective_values):

        #strengths calculation
        individuals = len(objective_values[0])
        strengths = []
        for ind_idx in range(individuals):
            dominated_solutions = 0
            for comparison_ind_idx in range(individuals):
                dominated = True
                for obj_idx in range(self.objectives):
                    #if objective_values[obj_idx][ind_idx] < objective_values[obj_idx][comparison_ind_idx]:
                    if objective_values[obj_idx][ind_idx] > objective_values[obj_idx][comparison_ind_idx]:
                        dominated = False
                        break
                if dominated:
                    dominated_solutions += 1   
            strengths.append(dominated_solutions - 1)

        #strengths sum
        evaluations = []
        for ind_idx in range(individuals):
            total_strengths = 0
            for comparison_ind_idx in range(individuals):
                dominates_me = True
                for obj_idx in range(self.objectives):
                    if objective_values[obj_idx][ind_idx] <= objective_values[obj_idx][comparison_ind_idx]:
                    #if objective_values[obj_idx][ind_idx] >= objective_values[obj_idx][comparison_ind_idx]:
                        dominates_me = False
                        break
                if dominates_me:
                    total_strengths += strengths[comparison_ind_idx]
            evaluations.append(total_strengths)

        return evaluations
    
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

    def logs_checkpoint(self):
        """
        Updates self.logs and self.genlogs
        """
        for ind_idx, individual in enumerate(self.population):
            self.logs[(self.ran_generations,ind_idx)] = [
										                str(individual.fenotype) 
										                ,individual.fenotype.my_depth()
										                ,individual.fenotype.nodes_count()
										                ,*individual.evaluation
										                ,*individual.objective_values]
        logs_to_file(self.logs, self.experiment_name)

        self.genlogs[(self.ran_generations,"execution_time")] = [self.last_gen_time]
        self.genlogs[(self.ran_generations,"mean_tree_size")] = [np.mean([ind.fenotype.nodes_count() for ind in self.population])]
        self.genlogs[(self.ran_generations,"mean_tree_depth")] = [np.mean([ind.fenotype.my_depth() for ind in self.population])]
        self.genlogs[(self.ran_generations,"std_tree_size")] = [np.std([ind.fenotype.nodes_count() for ind in self.population])]
        self.genlogs[(self.ran_generations,"std_tree_depth")] = [np.std([ind.fenotype.my_depth() for ind in self.population])]
        self.genlogs[(self.ran_generations,"non_dominated_solutions")] = [str(len([0 for ind in self.population if ind.evaluation[0] == 0]))]
        for obj_idx in range(self.objectives):
            self.genlogs[(self.ran_generations,"objective_" + str(obj_idx + 1) + "_name")] = [self.objective_functions_names[obj_idx]]
            self.genlogs[(self.ran_generations,"mean_objective_value_" + str(obj_idx + 1))] = [np.mean([ind.objective_values[obj_idx] for ind in self.population])]
            self.genlogs[(self.ran_generations,"std_objective_value_" + str(obj_idx + 1))] = [np.std([ind.objective_values[obj_idx] for ind in self.population])]
            best_by_obj = sorted(self.population, key=lambda x: x.objective_values[obj_idx])[0]
            #self.genlogs[(self.ran_generations,"best_individual_for_objective_" + str(obj_idx + 1))] = [best_by_obj.fenotype]
            self.genlogs[(self.ran_generations,"best_value_reached_for_objective_" + str(obj_idx + 1) + "_(min_is_best)")] = [best_by_obj.objective_values[obj_idx]]
            self.genlogs[(self.ran_generations,"best_individual_for_objective_" + str(obj_idx + 1) + "_all_objective_values")] = [best_by_obj.objective_values]
            self.genlogs[(self.ran_generations,"best_individual_for_objective_" + str(obj_idx + 1) + "_tree_size")] = [best_by_obj.fenotype.nodes_count()]
            self.genlogs[(self.ran_generations,"best_individual_for_objective_" + str(obj_idx + 1) + "_tree_depth")] = [best_by_obj.fenotype.my_depth()]

        #temporal

        error_by_threshold_0_1 = [self.errors_by_threshold(ind,threshold=0.1) for ind in self.population]
        error_by_threshold_0_01 = [self.errors_by_threshold(ind,threshold=0.01) for ind in self.population]
        error_by_threshold_0_001 = [self.errors_by_threshold(ind,threshold=0.001) for ind in self.population]
        self.genlogs[(self.ran_generations,"mean_error_by_threshold_0.1")] = [np.mean(error_by_threshold_0_1)]
        self.genlogs[(self.ran_generations,"mean_error_by_threshold_0.01")] = [np.mean(error_by_threshold_0_01)]
        self.genlogs[(self.ran_generations,"mean_error_by_threshold_0.001")] = [np.mean(error_by_threshold_0_001)]
        self.genlogs[(self.ran_generations,"best_error_by_threshold_0.1")] = [min(error_by_threshold_0_1)]
        self.genlogs[(self.ran_generations,"best_error_by_threshold_0.01")] = [min(error_by_threshold_0_01)]
        self.genlogs[(self.ran_generations,"best_error_by_threshold_0.001")] = [min(error_by_threshold_0_001)]

        logs_to_file(self.genlogs, self.experiment_name, logs_by_gen = True)
            
    def __str__(self):
        return str(self.__dict__)

    ##### Objective functions

    def single_goal_accuracy(self, individual, goal_class):
        y = self.y
        values = self.Model.evaluate(individual.fenotype, self.x)
        y_predicted = map_to_binary(values)
        corrects = sum([1 if y_predicted[i] == y[i] and y[i] == goal_class else 0 for i in range(len(y))])
        accuracy = corrects / y.count(goal_class)
        return 1-accuracy

    def mse(self, individual):
        y = self.y
        y_predicted = self.Model.evaluate(individual.fenotype, self.x)
        n = len(y)
        MSE = sum([pow(y[i]-y_predicted[i],2) for i in range(n)]) / n
        #if MSE > 10000: return 10000
        return MSE

    def rmse(self, individual):
        return math.sqrt(self.mse(individual))

    def tree_size(self, individual):
        return individual.fenotype.nodes_count()#/self.max_population_size

    def tree_depth(self, individual):
        return individual.fenotype.my_depth()#/self.max_population_depth

    def accuracy(self, individual):
        y = self.y
        values = self.Model.evaluate(individual.fenotype, self.x)
        y_predicted = map_to_binary(values)
        corrects = sum([1 if y_predicted[i] == y[i] else 0 for i in range(len(y))])
        accuracy = corrects / len(y)
        return 1-accuracy

    def errors_by_threshold(self, individual, threshold = 0.1):
        y = self.y
        y_predicted = self.Model.evaluate(individual.fenotype, self.x)
        errors = sum([1 if abs(y_predicted[i]-y[i])>threshold else 0 for i in range(len(y))])
        error_rate = errors/len(y)
        return error_rate

    def sum_absolute_errors(self, individual):
        y = self.y
        y_predicted = self.Model.evaluate(individual.fenotype, self.x)
        sum_absolute_errors = sum([abs(y_predicted[i]-y[i]) for i in range(len(y))])
        return sum_absolute_errors

    def mean_absolute_errors(self, individual):
        y = self.y
        y_predicted = self.Model.evaluate(individual.fenotype, self.x)
        sum_absolute_errors = sum([abs(y_predicted[i]-y[i]) for i in range(len(y))])
        mean_absolute_errors = sum_absolute_errors/len(y)
        return mean_absolute_errors

def map_to_binary(values, threshold = 0, class_over_threshold = 1, class_below_threshold = 0):
    """
    Positional arguments:
        values: list of numbers to be mapped.
    Returns a list with 0 if value was negative, or 1 otherwise
    """
    y = [class_over_threshold if value >= threshold else class_below_threshold for value in values]
    return y

def verify_path(tpath):
    if tpath is None:
        return ""
    else:
        if tpath[-1] != "/":
            tpath = "outputs/" + tpath + "/"

        if not os.path.exists(os.path.dirname(tpath)):
            try:
                os.makedirs(os.path.dirname(tpath))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        return tpath

def logs_to_file(logs, path, logs_by_gen = False): #some hardcoded rules
    """
    logs is a dictionary with a key that can be split into two
    """
    path = verify_path(path)
    if logs_by_gen:
        path += "_bygen_"
    with open(path + "logs.csv", mode='w') as logs_file:
        logs_writer = csv.writer(logs_file, delimiter=',')
        if logs_by_gen:
            logs_writer.writerow(['generation', 'indicator', 'value'])
        else:
            logs_writer.writerow(['generation', 'individual_index', 'fenotype', 'depth', 'nodes', 'evaluation1', 'evaluation2', 'objective_value1', 'objective_value2'])
        for key, value in logs.items():
            values = [str(v) for v in value]
            logs_writer.writerow([str(key[0]), str(key[1]), *values])


def colored_plot(x, y, values, title = "default_title", colormap = "cool", markers = None, marker_size = 50, save = False, path = None, logaritmical = [False,False]):  #some hardcoded rules
        path = verify_path(path)
        f = plt.figure()   
        f, axes = plt.subplots(nrows = 1, ncols = 1, sharex=True, sharey = True, figsize=(10,10))
        """points are x, y pairs, values are used for graduated coloring"""
        max_value = max(values)
        min_value = min(values)
        colors = [(1 - (value - min_value)) / (max_value - min_value + 0.001) for value in values]
        log_string = ["",""]
        if logaritmical[0]:
            x = [math.log(xi) if math.log(xi)>0 or xi != 0 else 0 for xi in x]
            log_string[0] += " (log)"
        if logaritmical[1]:
            y = [math.log(yi) if math.log(yi)>0 or yi != 0 else 0 for yi in y]
            log_string[1] += " (log)"

        nondoms = np.array([[x[i],y[i]] for i in range(len(y)) if markers[i] == 0])
        doms = np.array([[x[i],y[i]] for i in range(len(y)) if markers[i] != 0])
        plt.scatter(nondoms[:,0], nondoms[:,1], 
                        marker = "o",
                        s = 40,
                        edgecolors="blue",
                        facecolors='none',
                        #c = color, 
                        #cmap = colormap, 
                        alpha = 0.9)
        plt.scatter(doms[:,0], doms[:,1], 
                        marker = "x",
                        #s = marker_size,
                        #edgecolors=color,
                        #facecolors='none',
                        c = "black", 
                        #cmap = colormap, 
                        alpha = 0.6)

        """
        data = [[x[i], y[i]] for i in range(len(x))]
        for i, d in enumerate(data):
            if markers[i] == 0:
                plt.scatter(d[0], d[1], 
                        marker = "o",
                        #s = marker_size,
                        edgecolors="blue",
                        facecolors='none',
                        #c = color, 
                        #cmap = colormap, 
                        alpha = 0.9)
            else:
                
                plt.scatter(d[0], d[1], 
                        marker = "x",
                        #s = marker_size,
                        #edgecolors=color,
                        #facecolors='none',
                        c = "black", 
                        #cmap = colormap, 
                        alpha = 0.6)

        if markers is None:
            
            plt.scatter(x, y, 
                        c = colors, 
                        cmap = colormap, 
                        alpha = 0.6)
            
            
        else:
            markers = [str(marker) for marker in markers]
            data = [[x[i], y[i], markers[i]] for i in range(len(x))]
            for i, d in enumerate(data):
                py.scatter(d[0], d[1], 
                            marker = r"$ {} $".format(d[2]),
                            s = marker_size,
                            edgecolors='none',
                            #c = "blue", 
                            cmap = colormap, 
                            alpha = 0.9)
        """
        plt.title(title)
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.xlabel("Objective 1" + log_string[0])
        plt.ylabel("Objective 2" + log_string[1])
        plt.grid()
        
        if save:
            name = path + title + ".png"
            plt.savefig(name)
        plt.close('all')
    
    