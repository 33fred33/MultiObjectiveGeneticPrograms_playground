#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 05 10:52:52 2019

@author: 33fred33
"""

import csv
import TreeFunction as tf
import GeneticProgram as gp
import ObjectiveFunctions as of
import Operators as ops
import operator
import time

#Pedestrian dataset
def load_from_csv(name, ints = False):
    with open(name + ".csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        data = []
        for row in csv_reader:
            if ints:
                data.append(*[int(x) for x in row])
            else:
                data.append([float(x) for x in row])
    return data

x_train = load_from_csv("datasets/x_train_0-6-6-pedestrian-features")
x_test = load_from_csv("datasets/x_test_0-6-6-pedestrian-features")
y_train = load_from_csv("datasets/y_train_pedestrian-features", True)
y_test = load_from_csv("datasets/y_test_pedestrian-features", True)

#Variables assignment
operators = [
    operator.add
    ,operator.sub
    ,operator.mul
    ,ops.safe_divide_numerator
    #,ops.signed_if
    #,math.sin
    ]
objective_functions = [
    of.single_goal_accuracy
    ,of.single_goal_accuracy
    ]
objective_functions_arguments = [
    [0]
    ,[1]
    ]

#objects creation
TF = tf.TreeFunctionClass(
            features = len(x_train[0]),
            operators = operators,
            max_initial_depth = 3,
            max_depth = 15,
            initialisation_method = "ramped half half",
            mutation_method = "subtree")

NewGP = gp.GeneticProgramClass(population_size = 100,
            generations = 50,
            Model = TF,
            objective_functions = objective_functions,
            objective_functions_arguments = objective_functions_arguments, #[[f1_arg1, ..., f1_argn], [f2_arg1, ..., f2_argn], ..., [fn_arg1, ..., fn_argn]]
            sampling_method="tournament",
            mutation_ratio=0.4,
            tournament_size=2,
            checkpoint_file_name = None)

#Execution
start_time = time.time()
dc = NewGP.fit(x_train, y_train)
run_time = time.time() - start_time
print(dc)
print(NewGP.population[0].evaluation)
print("run_time", run_time)