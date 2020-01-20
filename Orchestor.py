#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 05 10:52:52 2019

@author: 33fred33
"""

#to do:
#time limit as argument
#min initial depth as argument
#max tree size as argument
#minimize or maximize for each objective?
#tree size 0 to 1 setting max or calculating it

import os
import csv
import TreeFunction as tf
import GeneticProgram as gp
import Operators as ops
import ProblemDatabase as problems
import operator
import time
import math
import sys
import pickle
import numpy as np
import argparse
import random as rd
import matplotlib.pyplot as plt

def verify_path(tpath):
    """
    Positional arguments:
        tpath is a relative file path as a string
    verifies tpath format as "outputs/tpath/"
    verifies target folder exists
    """
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

def plot_this(x, y, title, path, xlabel, ylabel):
    path = verify_path(path)
    f = plt.figure()   
    f, axes = plt.subplots(nrows = 1, ncols = 1, sharex=True, sharey = True, figsize=(10,10))


    plt.plot(x,y)

    plt.title(title)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()

    name = path + title + ".png"
    plt.savefig(name)
    plt.close('all')
    


#Arguments handling:
parser = argparse.ArgumentParser()
parser.add_argument("-p",
                    "--problem",
                    default="pedestrian",
                    type=str,
                    help="problem to be solved: pedestrian, pedestrian_old, MNIST, symbollic_regression")
parser.add_argument("-pv",
                    "--problem_variable",
                    default=None, #it was "0"
                    type=str,
                    help="problem variable if needed (in symbollic_regression: 1,2,3 or 4)")
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
                    help="can be single_goal_accuracy, mse, accuracy, tree_depth, tree_size, rmse, errors_by_threshold")
parser.add_argument("-ofa",
                    "--objective_functions_arguments",
                    default="",
                    type=str,
                    help="f1_a1,f1_a2,...,f1_an_f2_a1,f2_a2,...,f2_an_..._fm_a1,fm_a2,...,fm_an") #','to separate arguments, '_' to separate functions'
parser.add_argument("-r",
                    "--runs",
                    default=1,
                    type=int,
                    help="times to run same genetic program")
args=parser.parse_args()


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

objective_functions = [objective_function_string for objective_function_string in args.objective_functions.split(',')]
if len(args.objective_functions_arguments) > 0:
    objective_functions_arguments = [[int(argument) for argument in function_arguments.split(",")] for function_arguments in args.objective_functions_arguments.split('_')]
else: objective_functions_arguments = []
while len(objective_functions_arguments) < len(objective_functions):
    objective_functions_arguments.append([])

path = verify_path(args.experiment_name)


Problem = problems.Problem(
                            name =args.problem, 
                            variant = args.problem_variable)

x_train = Problem.x_train
x_test = Problem.x_test
y_train = Problem.y_train
y_test = Problem.y_test


features = len(x_train[0])
TF = tf.TreeFunctionClass(
                            features = features,
                            operators = operators,
                            max_initial_depth = args.max_initial_depth,
                            max_depth = args.max_depth,
                            initialisation_method = args.initialisation_method,
                            mutation_method = args.mutation_method)

#Runs
all_genlogs = []
run_times = []
run_time = 0

if args.problem == "symbollic_regression":
    tree_size_by_gen = {}
    rmse_by_gen = {}
    best_rmse_by_gen = {}
    best_rmse_tree_size_by_gen = {}
    for gen in range(args.generations + 1):
        tree_size_by_gen[gen] = []
        rmse_by_gen[gen] = []
        best_rmse_by_gen[gen] = []
        best_rmse_tree_size_by_gen[gen] = []

for run in range(args.runs):
    ename = args.experiment_name + "/run" + str(run)
    GP = gp.GeneticProgramClass(
                                population_size = args.population_size,
                                generations = args.generations,
                                Model = TF,
                                objective_functions = objective_functions,
                                objective_functions_arguments = objective_functions_arguments,
                                sampling_method = args.sampling_method,
                                mutation_ratio = args.mutation_ratio,
                                tournament_size = args.tournament_size,
                                experiment_name = ename,
                                ensemble_type = args.ensemble_type)

    #Single execution
    start_time = time.time()
    GP.fit(x_train, y_train)
    run_logs, run_genlogs = GP.get_logs()
    run_time += time.time() - start_time
    run_times.append(time.time() - start_time)
    
    #Run logs
    if args.problem == "symbollic_regression":
        for obj_idx in range(len(objective_functions)):
            run_genlogs[(args.generations, "best_ind_for_obj_"+str(obj_idx)+ "_errors_by_threshold_in_symb_reg")] = [GP.errors_by_threshold(sorted(GP.population, key=lambda x: x.objective_values[obj_idx])[0])]

        for gen in range(args.generations + 1):
            tree_size_by_gen[gen].append(run_genlogs[(gen,"mean_tree_size")])
            rmse_by_gen[gen].append(run_genlogs[(gen,"mean_objective_value_1")])
            best_rmse_by_gen[gen].append(run_genlogs[(gen,"best_value_reached_for_objective_1_(min_is_best)")])
            best_rmse_tree_size_by_gen[gen].append(run_genlogs[(gen,"best_individual_for_objective_1_tree_size")])

        plot_this(x=list(range(args.generations + 1)),
                y=[np.mean(value) for key,value in tree_size_by_gen.items()], 
                title = "Tree_size", 
                path = args.experiment_name, 
                xlabel = "Generation", 
                ylabel = "Average tree size")

        plot_this(x=list(range(args.generations + 1)),
                y=[math.log(np.mean(value)) for key,value in rmse_by_gen.items()], 
                title = "RMSE", 
                path = args.experiment_name, 
                xlabel = "Generation", 
                ylabel = "Average RMSE (log)")

        plot_this(x=list(range(args.generations + 1)),
                y=[np.mean(value) for key,value in best_rmse_by_gen.items()], 
                title = "Best_RMSE", 
                path = args.experiment_name, 
                xlabel = "Generation", 
                ylabel = "Average best RMSE")

        plot_this(x=list(range(args.generations + 1)),
                y=[np.mean(value) for key,value in best_rmse_tree_size_by_gen.items()], 
                title = "Best_RMSE_tree_size", 
                path = args.experiment_name, 
                xlabel = "Generation", 
                ylabel = "Average tree size")

    print("\nRun ",run," time", run_time)
    all_genlogs.append(run_genlogs)

    #Data dump
    ename = verify_path(ename)
    with open(ename + "gp.p", "wb") as f:
        pickle.dump(GP, f) 

#Final logs
final_lists = {key[1]:[] for key, _ in all_genlogs[0].items()}
for genlogs in all_genlogs:
    for run_idx, (key, value) in enumerate(genlogs.items()):
        temp_list = []
        if str(key[0]) == str(args.generations):
            final_lists[key[1]].append(value)

with open(path + "results_file.csv", mode = "w") as f:
    writer = csv.writer(f, delimiter = ",")
    writer.writerow(["experiment_name",str(args.experiment_name)])
    writer.writerow(["execution_command"," ".join(sys.argv[1:])])
    writer.writerow(["full_execution_time",str(run_time)])
    writer.writerow(["problem",str(args.problem)])
    writer.writerow(["problem_variable",str(args.problem_variable)])
    writer.writerow(["runs",str(args.runs)])
    writer.writerow(["model","Tree function"])
    writer.writerow(["tree_features",str(features)])
    writer.writerow(["tree_operators" ,str(args.operators)])
    writer.writerow(["min_initial_tree_depth","2"])
    writer.writerow(["max_initial_tree_depth",str(args.max_initial_depth)])
    writer.writerow(["max_tree_depth" ,str(args.max_depth)])
    writer.writerow(["max_tree_size" ,"not set"])
    writer.writerow(["initialisation_method" ,str(args.initialisation_method)])
    writer.writerow(["population_size" ,str(args.population_size)])
    writer.writerow(["generations" ,str(args.generations)])
    writer.writerow(["sampling_method" ,str(args.sampling_method)])
    writer.writerow(["tournament_size" ,str(args.tournament_size)])
    writer.writerow(["mutation_ratio" ,str(args.mutation_ratio)])
    writer.writerow(["objective_functions" ,str(objective_functions)])
    writer.writerow(["objective_functions_goals" ,["minimize" for _ in range(len(objective_functions))]])
    writer.writerow(["objective_functions_arguments" ,str(objective_functions_arguments)])
    writer.writerow(["runs_avg_run_time" ,str(np.mean(run_times))])
    for obj_idx in range(len(objective_functions)):
        writer.writerow(["best_overall_individual_obj_"+str(obj_idx + 1)+ "_value" ,str(min(final_lists["best_value_reached_for_objective_" + str(obj_idx + 1) + "_(min_is_best)"]))]) #min max dependent
        writer.writerow(["best_obj_"+str(obj_idx + 1)+ "_values_by_run" ,str(final_lists["best_value_reached_for_objective_" + str(obj_idx + 1) + "_(min_is_best)"])]) #min max dependent      
    for key, value in final_lists.items():
        if not isinstance(value[0][0], list) and not isinstance(value[0][0], str):
            values = [str(v) for v in value]
            writer.writerow(["avg_last_gen_" + str(key), str(np.mean([float(x[0]) for x in value]))])
            writer.writerow(["std_last_gen_" + str(key), str(np.std([float(x[0]) for x in value]))])

with open(path + "results_by_run.csv", mode='w') as last_file:
    last_writer = csv.writer(last_file, delimiter = ",")
    for run_idx, genlogs in enumerate(all_genlogs):
        last_writer.writerow(["results_from experiment_name: ",str(args.experiment_name)])
        last_writer.writerow(["run",str(run_idx)])
        last_writer.writerow(["last_generation_results:"])
        for key, value in genlogs.items():
            if str(key[0]) == str(args.generations):
                values = [str(v) for v in value]
                last_writer.writerow([str(key[1]), *values])




