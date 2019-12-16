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

def feature_extractor(image, box_size = 6, stride = 6, ignore_borders = 2):
    """Receives a numpy matrix as image, returns mean and standard deviation for every squared box in a list"""
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

#MNIST dataset (each digit)
"""
from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = np.array([image.reshape((28,28)) for image in X])
y = np.array(y)

X_features = [feature_extractor(x) for x in X]
max_feature = max([max(x) for x in X_features])
X_normalised_features = [x / max_feature for x in X_features]
y_temp = [1 if label == "2" else 0 for label in y] #hardcoded to class 2

x_train, x_test, y_train, y_test = train_test_split(X_normalised_features, y_temp, test_size = 0.8)
print(len(y_test))
"""

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
            max_initial_depth = 5,
            max_depth = 15,
            initialisation_method = "ramped half half",
            mutation_method = "subtree")

NewGP = gp.GeneticProgramClass(population_size = 100,
            generations = 40,
            Model = TF,
            objective_functions = objective_functions,
            objective_functions_arguments = objective_functions_arguments, #[[f1_arg1, ..., f1_argn], [f2_arg1, ..., f2_argn], ..., [fn_arg1, ..., fn_argn]]
            sampling_method="tournament",
            mutation_ratio=0.4,
            tournament_size=2,
            experiment_name = "Pedestrian_6_6_0_pop100_mr40_ts2-fast")


#Execution
start_time = time.time()
dc = NewGP.fit(x_train, y_train)
run_time = time.time() - start_time
print(dc)
print(NewGP.population[0].evaluation)
print("run_time", run_time)


with open('Outputs/Pedestrian_6_6_0_pop100_mr40_ts2-fast/gp.p', 'wb') as f:
    pickle.dump(NewGP, f) 