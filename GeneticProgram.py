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
#GENERATIONAL LOGS
#NSGA2
#Bloat control by lenght, not depth only
#no tiebreak if doesnt matter
#crowding distance on standardized opbjective values?

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
        self.objective_functions = objective_functions

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
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.logs = {}
        self.genlogs = {}
        self.ran_generations = 0
        self.last_gen_time = 0
        
    def fit(self
        , x_train
        , y_train
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
        return self.logs, self.genlogs
            
    
    def train(self, generations):
        """
        Positional arguments:
            generations is an int
        """
        mutations = math.ceil(self.population_size * self.mutation_ratio)
        crossovers = self.population_size - mutations
        
        if self.logs_level >= 1:
            print("population_size: ", self.population_size)
            print("mutations per gen: ", mutations)
            print("crossovers per gen: ", crossovers)
            
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

    def predict(self, x, rpf_objective_indexes=None):
        """
        Positional arguments:
            x is a dataset to to evaluate the fit with
        Returns:
            the predicion as an array
        Assummes the best individual is the result of the genetic algorithm, and evaluates him
        """
        if len(self.objective_functions) == 1:
            prediction = self.Model.evaluate(self.population[0], x) # assummes population is sorted
        else:
            if self.ensemble_type == "rpf":
                ensemble = self._rpf_ensemble(rpf_objective_indexes)
            else: #default is baseline
                ensemble=self._baseline_ensemble()

            ensemble_size = len(ensemble)
            if ensemble_size == 0:
                return None

            voting_threshold = ensemble_size/2
            predictions = np.array([self.Model.evaluate(individual.fenotype, x) for individual in ensemble])
            prediction = [1 # most voted class value, minority class preferred
                        if sum(predictions[:,pred_idx]) > voting_threshold 
                        else 0 # less voted class value
                        for pred_idx in range(len(predictions[0]))] # WRONG this needs to be changed if classes are not 1 or 0

        return prediction

    def _rpf_ensemble(self, objective_indexes=None):
        """
        Keyword arguments:
            objective_indexes are the indexes of the objectives to be considered as filter
        Returns:
            the ensemble of individuals as an array
        """
        if objective_indexes is None: 
            objective_indexes = [i for i in range(len(self.objective_functions))]
        ensemble = []
        for individual in self.population:
            if individual.evaluation[0] > 0: useful_individual = False
            else: useful_individual = True
            for obj_idx, obj_value in enumerate(individual.objective_values):
                if obj_idx in objective_indexes:
                    if obj_value <= 0.5:
                        useful_individual = False
                        break
            if useful_individual:
                ensemble.append(individual)
        return ensemble

    def _baseline_ensemble(self):
        """
            the ensemble of individuals from the pareto front as an array
        """
        ensemble = []
        for individual in self.population:
            if individual.evaluation[0] == 0:  # choose only pareto front individuals
                ensemble.append(individual)
        return ensemble

    def load_from_file(self, file_name, desired_generation = None):
        """
        Positional arguments:
            file_name: expect the path + file name in string format
        Retrieves the population for the last generation from the file.
        It will overwrite current population.
        File format:
            Each row is an individual
            First row is headers
            Minimum column names: generation, fenotype
        """
        print("In load")
        with open(file_name, mode="r") as read_file:
            reader = csv.DictReader(read_file)
            logs_dict = {}
            for row in reader:
                logs_dict[(row["generation"], row["individual_index"])] = row["fenotype"]
            
        if desired_generation is None:
            desired_generation = max([x[0] for x in logs_dict.keys()])
        ind_indexes = [x[1] for x in logs_dict.keys()]

        for ind_idx in ind_indexes:
            print(logs_dict[(desired_generation, ind_idx)])
            fenotype = self.Model.generate_from_string(logs_dict[(desired_generation, ind_idx)])
            individual = IndividualClass(fenotype)

    
    def _evaluate_population(self):
        """
        Evaluates the entire population
        """
        for ind_idx, individual in enumerate(self.population):
            if individual.objective_values is None:
                individual.objective_values = []
                for obj_idx, objective_function in enumerate(self.objective_functions):
                    if objective_function == "single_goal_accuracy":
                        objective_function = self.single_goal_accuracy
                    elif objective_function == "mse":
                        objective_function = self.mse
                    elif objective_function == "rmse":
                        objective_function = self.rmse
                    elif objective_function == "errors_by_threshold":
                        objective_function = self.errors_by_threshold
                    elif objective_function == "accuracy":
                        objective_function = self.accuracy
                    elif objective_function == "tree_depth":
                        objective_function = self.tree_depth
                    elif objective_function == "tree_size":
                        objective_function = self.tree_size
                    else:
                        print("wrong objective function name")
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
                logaritmical_plots = False
                if "mse" in self.objective_functions:
                    logaritmical_plots = True

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

            crowding_distances = self._crowding_distance(objective_values)
            for ind_idx, individual in enumerate(self.population):
                individual.evaluation.append(crowding_distances[ind_idx])

    def _crowding_distance(self, objective_values):
        """
        Positional arguments:
            objective_values is a list of lists, with ordered values for each objective to be considered
        Returns: a list of values with a crowding distance as a float. 0 means the one with higher distance
        """
        crowding_distances = [[0 for _ in range(self.objectives)] for _ in range(len(self.population))]
        #print(np.array(crowding_distances).shape)
        objective_values_list = [[] for _ in range(self.objectives)]
        for obj_idx in range(self.objectives):
            #interpolate
            temp_objective_list = [ind.objective_values[obj_idx] for ind in self.population]
            """ This to interpolate and give same importance?
            max_ov = max(temp_objective_list)
            min_ov = min(temp_objective_list)

            if max_ov == min_ov:
                print("In crowding distance: objective values are in the same range, skipping interpolation")
            elif abs(max_ov) == np.inf or abs(min_ov) == np.inf:
                print("In crowding distance: objective value is infinite, skipping interpolation")
            else:
                interpolate_function = interp1d([max_ov, min_ov],[0,1])
                temp_objective_list = interpolate_function(temp_objective_list)
            """

            #objective_values_list[obj_idx] = sorted([(ind.objective_values[obj_idx], ind_idx) for ind_idx, ind in enumerate(self.population)], key = lambda x: x[0])
            objective_values_list[obj_idx] = sorted([(obj_v, ind_idx) for ind_idx, obj_v in enumerate(temp_objective_list)], key = lambda x: x[0])
            #print("obj_idx",obj_idx,
            #    "min_ov",min_ov, "0", objective_values_list[obj_idx][0][0],
            #    "max_ov",max_ov, "1", objective_values_list[obj_idx][-1][0],
            #    "min_ind_idx", objective_values_list[obj_idx][0][1],
            #    "max_ind_idx", objective_values_list[obj_idx][-1][1],)
        #print(np.array(objective_values_list).shape)

            

        for obj_idx in range(self.objectives):
            for sorted_idx, (objective_value, ind_idx) in enumerate(objective_values_list[obj_idx]):
                if sorted_idx == 0 or sorted_idx == len(objective_values_list[obj_idx]) - 1:
                    crowding_distances[ind_idx][obj_idx] = np.inf
                else:
                    abs1 = abs(objective_values_list[obj_idx][sorted_idx-1][0] - objective_value)
                    abs2 = abs(objective_values_list[obj_idx][sorted_idx+1][0] - objective_value)
                    crowding_distances[ind_idx][obj_idx] = (abs1 + abs2)/2

        crowding_distance = [np.mean(crowding_distances[ind_idx]) for ind_idx in range(len(self.population))]
        #print(crowding_distance)
        max_cd = max([x for x in crowding_distance if x != np.inf])
        inverted_crowding_distances = [max_cd - cd if cd != np.inf else -np.inf for cd in crowding_distance]


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
        max_cd = max(crowding_distances)
        inverted_crowding_distances = [max_cd - cd for cd in crowding_distances]
        """
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
        #individual wise logs
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
            self.genlogs[(self.ran_generations,"objective_" + str(obj_idx + 1) + "_name")] = [self.objective_functions[obj_idx]]
            self.genlogs[(self.ran_generations,"mean_objective_value_" + str(obj_idx + 1))] = [np.mean([ind.objective_values[obj_idx] for ind in self.population])]
            self.genlogs[(self.ran_generations,"std_objective_value_" + str(obj_idx + 1))] = [np.std([ind.objective_values[obj_idx] for ind in self.population])]
            best_by_obj = sorted(self.population, key=lambda x: x.objective_values[obj_idx])[0]
            #self.genlogs[(self.ran_generations,"best_individual_for_objective_" + str(obj_idx + 1))] = [best_by_obj.fenotype]
            self.genlogs[(self.ran_generations,"best_value_reached_for_objective_" + str(obj_idx + 1) + "_(min_is_best)")] = [best_by_obj.objective_values[obj_idx]]
            self.genlogs[(self.ran_generations,"best_individual_for_objective_" + str(obj_idx + 1) + "_all_objective_values")] = [best_by_obj.objective_values]
            self.genlogs[(self.ran_generations,"best_individual_for_objective_" + str(obj_idx + 1) + "_tree_size")] = [best_by_obj.fenotype.nodes_count()]
            self.genlogs[(self.ran_generations,"best_individual_for_objective_" + str(obj_idx + 1) + "_tree_depth")] = [best_by_obj.fenotype.my_depth()]

        logs_to_file(self.genlogs, self.experiment_name, logs_by_gen = True)
            
    def __str__(self):
        return str(self.__dict__)

    ##### Objective functions

    def single_goal_accuracy(self, individual, goal_class):
        y = self.y_train
        values = self.Model.evaluate(individual.fenotype, self.x_train)
        y_predicted = map_to_binary(values)
        corrects = sum([1 if y_predicted[i] == y[i] and y[i] == goal_class else 0 for i in range(len(y))])
        accuracy = corrects / y.count(goal_class)
        return 1-accuracy

    def mse(self, individual):
        y = self.y_train
        y_predicted = self.Model.evaluate(individual.fenotype, self.x_train)
        n = len(y)
        MSE = sum([pow(y[i]-y_predicted[i],2) for i in range(n)]) / n
        #if MSE > 10000: return 10000
        return MSE

    def rmse(self, individual):
        return math.sqrt(self.mse(individual))

    def tree_size(self, individual):
        return individual.fenotype.nodes_count()

    def tree_depth(self, individual):
        return individual.fenotype.my_depth()

    def accuracy(self, individual):
        y = self.y_train
        values = self.Model.evaluate(individual.fenotype, self.x_train)
        y_predicted = map_to_binary(values)
        corrects = sum([1 if y_predicted[i] == y[i] else 0 for i in range(len(y))])
        accuracy = corrects / len(y)
        return 1-accuracy

    def errors_by_threshold(self, individual, threshold = 0.01):
        y = self.y_train
        y_predicted = self.Model.evaluate(individual.fenotype, self.x_train)
        errors = sum([1 if abs(y_predicted[i]-y[i])>threshold else 0 for i in range(len(y))])
        error_rate = errors/len(y)
        return error_rate

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


def colored_plot(x, y, values, title = "default_title", colormap = "cool", markers = None, marker_size = 50, save = False, path = None, logaritmical = False):  #some hardcoded rules
        path = verify_path(path)
        f = plt.figure()   
        f, axes = plt.subplots(nrows = 1, ncols = 1, sharex=True, sharey = True, figsize=(10,10))
        """points are x, y pairs, values are used for graduated coloring"""
        max_value = max(values)
        min_value = min(values)
        colors = [(1 - (value - min_value)) / (max_value - min_value + 0.001) for value in values]
        log_string = ""
        if logaritmical:
            x = [math.log(xi) if math.log(xi)>0 or xi != 0 else 0 for xi in x]
            y = [math.log(yi) if math.log(yi)>0 or yi != 0 else 0 for yi in y]
            log_string += " (log)"

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
        plt.xlabel("Objective 1" + log_string)
        plt.ylabel("Objective 2" + log_string)
        plt.grid()
        
        if save:
            name = path + title + ".png"
            plt.savefig(name)
        plt.close('all')
    
    