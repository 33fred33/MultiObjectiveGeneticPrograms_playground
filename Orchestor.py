#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 05 10:52:52 2019

@author: 33fred33
"""

#include mnist db
#receive arguments

import csv
import TreeFunction as tf
import GeneticProgram as gp
import ObjectiveFunctions as of
import Operators as ops
import operator
import time
import math
import sys
import pickle
from sklearn.model_selection import train_test_split
from datasets.load_pedestrian import load_pedestrian_data
from skimage.feature import hog
import numpy as np
import argparse
import random as rd

def single_variable_polynomial(x, coefficients):
    value = 0
    variable = x[0]
    max_exponent = len(coefficients)
    for i,coefficient in enumerate(coefficients):
        value += coefficient * math.pow(variable, max_exponent - i)
    return value

def feature_extractor(image, box_size = 6, stride = 6, ignore_borders = 2):
    """
    Receives a numpy matrix as image, returns mean and standard deviation for every squared box in a list
    """
    image = np.array(image)
    features = []
    if ignore_borders > 0:
        image = image[ ignore_borders : -ignore_borders , ignore_borders : -ignore_borders]
    horizontal_jumps = int((image.shape[0] - box_size) / stride)
    vertical_jumps = int((image.shape[1] - box_size) / stride)
    for i in range(vertical_jumps):
        for j in range(horizontal_jumps):
            vertical_start = j * stride
            horizontal_start = i * stride
            box = image[ vertical_start : vertical_start + box_size , horizontal_start : horizontal_start + box_size]
            features.append(np.mean(box))
            features.append(np.std(box))
    return np.array(features)

def load_from_csv(name, ints = False):
    """
    Positional arguments:

    """
    with open(name + ".csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        data = []
        for row in csv_reader:
            if ints:
                data.append(*[int(x) for x in row])
            else:
                data.append([float(x) for x in row])
    return data


#Arguments handling:
parser = argparse.ArgumentParser()
parser.add_argument("-p",
                    "--problem",
                    default="pedestrian",
                    type=str,
                    help="problem to be solved: pedestrian, pedestrian_old, MNIST, symbollic_regression")
parser.add_argument("-pv",
                    "--problem_variable",
                    default="0",
                    type=str,
                    help="problem variable if needed")
parser.add_argument("-ps",
                    "--population_size",
                    default=100,
                    type=int,
                    help="genetic program's initial population size")
parser.add_argument("-en",
                    "--experiment_name",
                    default="default_experiment_name",
                    type=str,
                    help="outputs will be sent to outputs/experiment_name folder")
parser.add_argument("-mid",
                    "--max_initial_depth",
                    default=5,
                    type=int,
                    help="tree function initial population max depth")
parser.add_argument("-md",
                    "--max_depth",
                    default=15,
                    type=int,
                    help="tree function max depth allowed")
parser.add_argument("-im",
                    "--initialisation_method",
                    default="ramped half half",
                    type=str,
                    help="genetic program's initial population generation method")
parser.add_argument("-mm",
                    "--mutation_method",
                    default="subtree",
                    type=str,
                    help="genetic program's mutation method")
parser.add_argument("-g",
                    "--generations",
                    default=30,
                    type=int,
                    help="genetic program's generations")
parser.add_argument("-sm",
                    "--sampling_method",
                    default="tournament",
                    type=str,
                    help="genetic program's population selection method")
parser.add_argument("-mr",
                    "--mutation_ratio",
                    default=0.4,
                    type=float,
                    help="genetic program's population portion to be mutated (between 0 and 1)")
parser.add_argument("-ts",
                    "--tournament_size",
                    default=3,
                    type=int,
                    help="genetic program's tournament size, if tournament selection method enabled")
parser.add_argument("-op",
                    "--operators",
                    default="add,sub,mul,safe_divide_numerator",
                    type=str,
                    help="tree function operators available in this format: add,sub,mul,safe_divide_numerator,signed_if,sin,cos")
parser.add_argument("-et",
                    "--ensemble_type",
                    default="baseline",
                    type=str,
                    help="can be rpf or baseline")
parser.add_argument("-of",
                    "--objective_functions",
                    default="single_goal_accuracy,single_goal_accuracy",
                    type=str,
                    help="can be single_goal_accuracy, mse, accuracy, tree_depth, tree_size")
parser.add_argument("-ofa",
                    "--objective_functions_arguments",
                    default="",
                    type=str,
                    help="f1_a1,f1_a2,...,f1_an_f2_a1,f2_a2,...,f2_an_..._fm_a1,fm_a2,...,fm_an") #','to separate arguments, '_' to separate functions'
args=parser.parse_args()

# data loading
if args.problem == "pedestrian":
    x_train = load_from_csv("datasets/pedestrian_x_train")
    x_test = load_from_csv("datasets/pedestrian_x_test")
    y_train = load_from_csv("datasets/pedestrian_y_train", True)
    y_test = load_from_csv("datasets/pedestrian_y_test", True)
elif args.problem == "pedestrian_old":
    x_train = load_from_csv("datasets/x_train_0-6-6-pedestrian-features")
    x_test = load_from_csv("datasets/x_test_0-6-6-pedestrian-features")
    y_train = load_from_csv("datasets/y_train_pedestrian-features", True)
    y_test = load_from_csv("datasets/y_test_pedestrian-features", True)
elif args.problem == "MNIST":
    x_train = load_from_csv("datasets/MNIST_x_train")
    x_test = load_from_csv("datasets/MNIST_x_test")
    y_train = load_from_csv("datasets/MNIST" + args.problem_variable + "_y_train", True)
    y_test = load_from_csv("datasets/MNIST" + args.problem_variable + "_y_test", True)
elif args.problem == "symbollic_regression":
    coefficients = [1,1,1]
    if args.problem_variable == "1":
        coefficients = [1,1,1]
    elif args.problem_variable == "2":
        coefficients = [1,1,1,1]
    elif args.problem_variable == "3":
        coefficients = [1,1,1,1,1]
    elif args.problem_variable == "4":
        coefficients = [1,1,1,1,1,1]
    fitness_cases = 41
    train_interval = [-10,10]
    test_interval = [-10,10]
    x_train = [[x] for x in np.linspace(train_interval[0],train_interval[1],fitness_cases)]
    x_test = [[rd.uniform(test_interval[0], test_interval[1])] for _ in range(fitness_cases)]
    y_train = [single_variable_polynomial(x, coefficients) for x in x_train]
    y_test = [single_variable_polynomial(x, coefficients) for x in x_test]
    #print(x_train,"\n",y_train)


operators = []
for operator_string in [operator_string for operator_string in args.operators.split(',')]:
    if operator_string == "add":
        operators.append(operator.add)
    elif operator_string == "sub":
        operators.append(operator.sub)
    elif operator_string == "mul":
        operators.append(operator.mul)
    elif operator_string == "safe_divide_numerator":
        operators.append(ops.safe_divide_numerator)
    elif operator_string == "signed_if":
        operators.append(ops.signed_if)
    elif operator_string == "sin":
        operators.append(math.sin)
    elif operator_string == "cos":
        operators.append(math.cos)

#objects creation
features = len(x_train[0])
TF = tf.TreeFunctionClass(
            features = features,
            operators = operators,
            max_initial_depth = args.max_initial_depth,
            max_depth = args.max_depth,
            initialisation_method = args.initialisation_method,
            mutation_method = args.mutation_method)
    
objective_functions = [objective_function_string for objective_function_string in args.objective_functions.split(',')]
if len(args.objective_functions_arguments) > 0:
    objective_functions_arguments = [[int(argument) for argument in function_arguments.split(",")] for function_arguments in args.objective_functions_arguments.split('_')]
else: objective_functions_arguments = []
while len(objective_functions_arguments) < len(objective_functions):
    objective_functions_arguments.append([])
print("objective_functions",objective_functions)
print("objective_functions_arguments",objective_functions_arguments)

GP = gp.GeneticProgramClass(
            population_size = args.population_size,
            generations = args.generations,
            Model = TF,
            objective_functions = objective_functions,
            objective_functions_arguments = objective_functions_arguments,
            sampling_method = args.sampling_method,
            mutation_ratio = args.mutation_ratio,
            tournament_size = args.tournament_size,
            experiment_name = args.experiment_name,
            ensemble_type = args.ensemble_type)

#Execution
start_time = time.time()
GP.fit(x_train, y_train)
run_time = time.time() - start_time
print(GP.population[0].evaluation)
print(GP.population[0].objective_values)
print("run_time", run_time)

path = gp.verify_path(args.experiment_name)
with open(path + "gp.p", "wb") as f:
    pickle.dump(GP, f) 

with open(path + "parameters.csv", mode = "w") as f:
    writer = csv.writer(f, delimiter = ",")
    writer.writerow(["experiment_name",str(args.experiment_name)])
    writer.writerow(["features",str(features)])
    writer.writerow(["model","Tree function"])
    writer.writerow(["max_initial_depth",str(args.max_initial_depth)])
    writer.writerow(["max_depth" ,str(args.max_depth)])
    writer.writerow(["initialisation_method" ,str(args.initialisation_method)])
    writer.writerow(["population_size" ,str(args.population_size)])
    writer.writerow(["generations" ,str(args.generations)])
    writer.writerow(["sampling_method" ,str(args.sampling_method)])
    writer.writerow(["tournament_size" ,str(args.tournament_size)])
    writer.writerow(["mutation_ratio" ,str(args.mutation_ratio)])
    writer.writerow(["operators" ,str(args.operators)])
    writer.writerow(["objective_functions" ,str(objective_functions)])
    writer.writerow(["objective_functions_arguments" ,str(objective_functions_arguments)])
    writer.writerow(["run_time" ,str(run_time)])





