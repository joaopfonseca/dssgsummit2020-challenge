from dssg_challenge.ga.algorithm.genetic_algorithm import GeneticAlgorithm
from dssg_challenge.ga.custom_problem.alskeyboard_problem import AlsKeyboardProblem

from dssg_challenge.ga.algorithm.ga_operators import (
    initialize_using_random, initialize_using_hc, initialize_using_greedy, initialize_using_multiple,
    roulettewheel_selection, rank_selection, tournament_selection, 
    cycle_crossover, pmx_crossover, order1_crossover, heuristic_crossover, multiple_crossover,
    swap_mutation, insert_mutation, inversion_mutation, scramble_mutation, greedy_mutation, multiple_mutation,
    elitism_replacement, standard_replacement 
)    

from dssg_challenge.ga.util.observer import LocalSearchObserver
from copy import deepcopy

from os import listdir, path, mkdir
from os.path import isfile, join, splitext
from pandas import pandas as pd
from sklearn.model_selection import ParameterGrid

# Problem
#--------------------------------------------------------------------------------------------------
# Decision Variables
with open(join(".", "data", "raw", "en-corpus.txt")) as file:
    en_corpus = file.read()[:-1]  # get rid of "\n"

with open(join(".", "data", "raw", "en-keys.txt")) as file:
    en_keys = file.read()[:-1] + "_"  # get rid of "\n" and add "_"

en_key_decision_variables = {
    "Corpus": en_corpus,
    "Valid_keys": en_keys
}

with open(join(".", "data", "raw", "pt-corpus.txt")) as file:
    pt_corpus = file.read()[:-1]  # get rid of "\n"

with open(join(".", "data", "raw", "pt-keys.txt")) as file:
    pt_keys = file.read()[:-1] + "_"  # get rid of "\n" and add "_"

pt_key_decision_variables = {
    "Corpus": pt_corpus,
    "Valid_keys": pt_keys
}

# Problem Instance
en_alskeyboard_problem_instance = AlsKeyboardProblem(decision_variables=en_key_decision_variables)
pt_alskeyboard_problem_instance = AlsKeyboardProblem(decision_variables=pt_key_decision_variables)


def one_combination(problem_instance, params, param_labels, sample_size=30, 
                    log_run_dir=join(".", "data", "log_run"), log_all_dir=join(".", "data", "log_all") ):
    """
    Actually runs the algorithm with one set of parameters.
    Names the resume file from parameters of search
    Creates the resume file from all the runs for a set of parameters
    """
    if not path.exists(log_run_dir):
        mkdir(log_run_dir)
    
    if not path.exists(log_all_dir):
        mkdir(log_all_dir)
    
    log_labels = {i: param_labels[j] if i in ["Initialization-Approach", "Selection-Approach", "Crossover-Approach", "Mutation-Approach", "Replacement-Approach"] else str(j) for i, j in params.items()}
    log_name = "I-{Initialization-Approach}_S-{Selection-Approach}_C-{Crossover-Approach}_M-{Mutation-Approach}_R-{Replacement-Approach}_CP-{Crossover-Probability}_MP-{Mutation-Probability}_TS-{Tournament-Size}_PS-{Population-Size}_NG-{Number-of-Generations}".format(**log_labels)
    resume_name= join(log_all_dir, f"{log_name}.xlsx")

    # Checks if the configuration has already been performed (all sample_size runs), if so, exits.
    # --------------------------------------------------------------------------------------------------
    runs_made = [f for f in listdir(log_all_dir) if isfile(join(log_all_dir, f))]
    if f"{log_name}.xlsx" in runs_made:
        print (f"Configuration {log_name} already performed, skipping...")
        return None  # exit one_combination call

    # Perform several runs of the same configuration (get sample distribution)
    #--------------------------------------------------------------------------------------------------
    overall_best_solution = None
    for run in range(1, sample_size + 1):
        # Genetic Algorithm
        ga = GeneticAlgorithm(
            problem_instance=problem_instance,
            params=params,
            run=run,
            log_name=log_name,
            log_dir=log_run_dir
            )

        ga_observer = LocalSearchObserver(ga)
        ga.register_observer(ga_observer)
        ga.search()
        ga.save_log()

        # find the best solution over the runs
        if run == 1:
            overall_best_solution = deepcopy(ga.best_solution)
        else:
            if ga.best_solution.fitness < overall_best_solution.fitness:
                overall_best_solution = deepcopy(ga.best_solution)

        print('overall_best_solution: ', overall_best_solution.representation)
        print('overall_best_solution fitness: ', overall_best_solution.fitness)

    # Consolidate the runs
    #--------------------------------------------------------------------------------------------------
    sub_log_dir = join(log_run_dir, log_name)
    log_files = [f for f in listdir(sub_log_dir) if isfile(join(sub_log_dir, f))]
    fitness_runs = []
    columns_name = []
    counter = 0
    generations = []

    # Going to each run for a given parameter configuration and extracting fitness for each generation
    for file_name in log_files:
        if file_name.startswith('run_'):
            df = pd.read_excel(join(sub_log_dir, file_name))
            fitness_runs.append(list(df["Fitness"]))
            columns_name.append(splitext(file_name)[0])
            counter += 1
            if not generations:
                generations = list(df["Generation"])

    df = pd.DataFrame(list(zip(*fitness_runs)), columns=columns_name)
    fitness_std = list(df.std(axis=1))
    fitness_mean = list(df.mean(axis=1))

    df["Generation"] = generations
    df["Fitness_STD"] = fitness_std
    df["Fitness_Mean"] = fitness_mean
    df["Fitness_Lower"] = df["Fitness_Mean"] - 1.96 * df["Fitness_STD"] / (sample_size ** 0.5)
    df["Fitness_Upper"] = df["Fitness_Mean"] + 1.96 * df["Fitness_STD"] / (sample_size ** 0.5)

    # Exporting summary of configuration with best solution
    with pd.ExcelWriter(join(log_all_dir, f"{log_name}.xlsx")) as writer:
        df.to_excel(writer, sheet_name='Fitness', index=False, encoding='utf-8')
        pd.DataFrame([[overall_best_solution.representation, overall_best_solution.fitness]], columns=["Representation", "Fitness"]).\
            to_excel(writer, sheet_name='Overall_Best_Solution', index=False)
    
    print("\nEnding one combination run...\n")


# Parameter Configuration
#--------------------------------------------------------------------------------------------------
# TODO: NEED TO IMPLEMENT NEIGHBORHOOD FUNCTION - test_init = [initialize_using_multiple, initialize_using_hc, initialize_using_random]
param_grid = ParameterGrid(
    [
        {
        "Initialization-Approach": [initialize_using_random],
        "Selection-Approach": [tournament_selection],
        "Crossover-Approach": [cycle_crossover, pmx_crossover, order1_crossover],  # multiple_crossover, heuristic_crossover, 
        "Mutation-Approach": [inversion_mutation, multiple_mutation, swap_mutation, insert_mutation, scramble_mutation],  # multiple_mutation, greedy_mutation, 
        "Replacement-Approach": [elitism_replacement, standard_replacement],
        "Crossover-Probability": [0.1, 0.9, 0.95, 0.05],
        "Mutation-Probability": [0.95, 0.9, 0.1, 0.05],
        "Tournament-Size": [15, 10, 5],
        "Population-Size": [10],  # 20
        "Number-of-Generations": [50]  # 1000
        },
        {
        "Initialization-Approach": [initialize_using_random],
        "Selection-Approach": [roulettewheel_selection, rank_selection],
        "Crossover-Approach": [cycle_crossover, pmx_crossover, order1_crossover],  # multiple_crossover, heuristic_crossover, 
        "Mutation-Approach": [inversion_mutation, swap_mutation, insert_mutation, scramble_mutation],  # multiple_mutation, greedy_mutation, 
        "Replacement-Approach": [elitism_replacement, standard_replacement],
        "Crossover-Probability": [0.1, 0.9, 0.95, 0.05],
        "Mutation-Probability": [0.95, 0.9, 0.1, 0.05],
        "Population-Size": [10],  # 20
        "Number-of-Generations": [50]  # 1000
        }
    ]
)

param_labels = {
    initialize_using_random: "rand",
    initialize_using_hc: "hc",
    initialize_using_greedy: "greedyI", 
    initialize_using_multiple: "mixIB",
    roulettewheel_selection: "rol",
    tournament_selection: "tourn",
    rank_selection: "rank",
    cycle_crossover: "cycle",
    pmx_crossover: "pmx",
    order1_crossover: "order1",
    heuristic_crossover: "heur",
    multiple_crossover: "mixCB",
    swap_mutation: "swap",
    insert_mutation: "insert",
    inversion_mutation: "invert",
    scramble_mutation: "scramble",
    greedy_mutation: "greedyM",
    multiple_mutation: "mixMB",
    elitism_replacement: "elit",
    standard_replacement: "std"
}

# Performs Grid Search over possible parameters
# -------------------------------------------------------------------------------------------------
num_comb = len(list(param_grid))
print("The Parameter Grid has {} combinations".format(num_comb))
for i, params in enumerate(param_grid):
    print(f"Starting run number {i}")
    one_combination(pt_alskeyboard_problem_instance, params, param_labels, sample_size=3, 
                    log_run_dir=join(".", "data", "log_run_pt"), log_all_dir=join(".", "data", "log_all_pt"))
    print("\n--------------------------------------------------------------------------------------------------\n")
    one_combination(en_alskeyboard_problem_instance, params, param_labels, sample_size=3, 
                    log_run_dir=join(".", "data", "log_run_en"), log_all_dir=join(".", "data", "log_all_en"))
    # TODO: Figure out why we get invalid solutions in en but not in pt
    print(f"\n{i}/{num_comb} runs done.")
