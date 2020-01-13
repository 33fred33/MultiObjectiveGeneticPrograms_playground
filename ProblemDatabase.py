#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 10:52:52 2020

@author: 33fred33
"""
import math
import numpy as np
import csv
import random as rd
import pickle
from skimage.feature import hog

def load_from_csv(name, ints = False):
    """
    Positional arguments:
        name is a relative file path as a string
    Keyword arguments:
        inst is a boolean indicating if data type to load are ints
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

def single_variable_polynomial(x, coefficients):
    """
    Evaluates x in the polynomial with coefficients
    """
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

def extract_hog(X):
            return np.array([
                hog(x,
                    pixels_per_cell=(6, 6),
                    cells_per_block=(3, 3),
                    transform_sqrt=True)
                for x in X])

def read_data_file(filename):
    with open(f"./datasets/cifar-10/{filename}", "rb") as file:
        d = pickle.load(file, encoding="bytes")
        raw_data = np.array(d[b"data"])
        return (
            extract_hog(
                np.rollaxis(
                    raw_data.reshape((-1, 3, 32, 32)),
                    1,
                    4)),
            np.array(d[b"labels"]))

class Problem:
    def __init__(self, name, variant):
        self.name = name
        self.variant = variant

        if name == "pedestrian":
            self.x_train = load_from_csv("datasets/pedestrian_x_train")
            self.x_test = load_from_csv("datasets/pedestrian_x_test")
            self.y_train = load_from_csv("datasets/pedestrian_y_train", True)
            self.y_test = load_from_csv("datasets/pedestrian_y_test", True)
            self.problem_type = "classification"
            self.problem_labels = [1,0]

        elif name == "pedestrian_old":
            self.x_train = load_from_csv("datasets/x_train_0-6-6-pedestrian-features")
            self.x_test = load_from_csv("datasets/x_test_0-6-6-pedestrian-features")
            self.y_train = load_from_csv("datasets/y_train_pedestrian-features", True)
            self.y_test = load_from_csv("datasets/y_test_pedestrian-features", True)
            self.problem_type = "classification"
            self.problem_labels = [1,0]

        elif name == "MNIST": #test and train are swapped in the csv's!
            self.x_train = load_from_csv("datasets/MNIST_x_test")
            self.x_test = load_from_csv("datasets/MNIST_x_train")
            self.y_train = load_from_csv("datasets/MNIST" + variant + "_y_test", True)
            self.y_test = load_from_csv("datasets/MNIST" + variant + "_y_train", True)
            self.problem_type = "classification"
            self.problem_labels = ["0","1","2","3","4","5","6","7","8","9"]

        elif name == "symbollic_regression":
            coefficients = [1,1,1]
            if variant == "1":
                coefficients = [1,1,1]
            elif variant == "2":
                coefficients = [1,1,1,1]
            elif variant == "3":
                coefficients = [1,1,1,1,1]
            elif variant == "4":
                coefficients = [1,1,1,1,1,1]
            fitness_cases = 201
            train_interval = [-5,5]
            test_interval = [-5,5]
            self.x_train = [[x] for x in np.linspace(train_interval[0],train_interval[1],fitness_cases)]
            self.x_test = [[rd.uniform(test_interval[0], test_interval[1])] for _ in range(fitness_cases)]
            self.y_train = [single_variable_polynomial(x, coefficients) for x in self.x_train]
            self.y_test = [single_variable_polynomial(x, coefficients) for x in self.x_test]
            self.problem_type = "approximation"
            self.problem_labels = None

        elif name == "CIFAR":
            TRAIN_FILES = [f"data_batch_{n}" for n in range(1, 6)]
            X_trains, y_trains = zip(*[read_data_file(filename) for filename in TRAIN_FILES])
            self.x_train, y_train = np.concatenate(X_trains), np.concatenate(y_trains)
            self.x_test, y_test = read_data_file("test_batch")
            self.y_train = [1 if str(y)==variant else 0 for y in y_train]
            self.y_test = [1 if str(y)==variant else 0 for y in y_test]



            self.problem_type = "classification"
