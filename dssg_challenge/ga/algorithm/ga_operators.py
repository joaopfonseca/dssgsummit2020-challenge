from random import uniform, randint, choices, sample, shuffle, random
from copy import deepcopy
import numpy as np
import math
import heapq

from dssg_challenge.ga.problem.objective import ProblemObjective
from dssg_challenge.ga.problem.solution import EncodingDataType, LinearSolution
from dssg_challenge.ga.problem.population import Population


###################################################################################################
# INITIALIZATION APPROACHES
###################################################################################################

# (!) REMARK:
# Initialization signature: <method_name>(problem, population_size):

# -------------------------------------------------------------------------------------------------
# Initialization Random
# -------------------------------------------------------------------------------------------------
def initialize_using_random(problem, population_size):
    """
    Initialize a population of solutions (feasible solution) for an evolutionary algorithm using Random method

    Required:
    @ problem - problem's build solution function knows how to create an individual in accordance with the encoding.
    @ population_size - to define the size of the population to be returned.
    """
    solution_list = []

    # generate a population of admissible solutions (individuals)
    for i in range(0, population_size):
        s = problem.build_solution(method='Random')

        # check if the solution is admissible
        while not problem.is_admissible(s):
            s = problem.build_solution(method='Random')

        s.id = [0, i]

        problem.evaluate_solution(s)

        solution_list.append(s)

    population = Population(
        problem = problem,
        maximum_size = population_size,
        solution_list = solution_list
    )

    return population

# -------------------------------------------------------------------------------------------------
# Initialization using Hill Climbing
# -------------------------------------------------------------------------------------------------
def initialize_using_hc(problem, population_size):
    """
    Initialize a population of solutions (feasible solution) for an evolutionary algorithm using Hill Climbing

    Required:
    @ problem - problem's build solution function knows how to create an individual in accordance with the encoding.
    @ population_size - to define the size of the population to be returned.
    """
    solution_list = []

    # generate a population of admissible solutions (individuals)
    for i in range(0, population_size):
        s = problem.build_solution(method='Hill Climbing')

        # check if the solution is admissible
        while not problem.is_admissible(s):
            s = problem.build_solution(method='Hill Climbing')

        s.id = [0, i]

        problem.evaluate_solution(s)

        solution_list.append(s)

    population = Population(
        problem = problem,
        maximum_size = population_size,
        solution_list = solution_list
    )

    return population

# -------------------------------------------------------------------------------------------------
# Initialization using Multiple Initializations
# -------------------------------------------------------------------------------------------------
def initialize_using_multiple(problem, population_size):
    """
    Initialize a population of solutions (feasible solution) for an evolutionary algorithm using multiple algorithms:
    - One solution from greedy initialization
    - One solution from hill climbing initialization
    - The rest of the solutions from random initialization

    Required:
    @ problem - problem's build solution function knows how to create an individual in accordance with the encoding.
    @ population_size - to define the size of the population to be returned.
    """
    solution_list = []

    i = 0
    # generate a population of admissible solutions (individuals)
    if population_size >= 10:
        while i in range(0, population_size):
            if i > 10:
                s = problem.build_solution(method='Random')
            elif i < 5:
                s = problem.build_solution(method='Greedy')
            else:
                s = problem.build_solution(method='Hill Climbing')

            if problem.is_admissible(s):
                s.id = [0, i]

                problem.evaluate_solution(s)

                solution_list.append(s)

                i += 1
    else:
        while i in range(0, population_size):
            if i > 2:
                s = problem.build_solution(method='Random')
            elif i == 0:
                s = problem.build_solution(method='Greedy')
            else:
                s = problem.build_solution(method='Hill Climbing')

            if problem.is_admissible(s):
                s.id = [0, i]

                problem.evaluate_solution(s)

                solution_list.append(s)

                i += 1

    population = Population(
        problem = problem,
        maximum_size = population_size,
        solution_list = solution_list
    )

    return population

# -------------------------------------------------------------------------------------------------
# Initialization using Simulated Annealing
# -------------------------------------------------------------------------------------------------
#TODO: OPTIONAL, implement a initialization based on Simulated Annealing
# Remark: remember, you will need a neighborhood function for each problem
def initialize_using_sa(problem, population_size):
    """
    Initialize a population of solutions (feasible solution) for an evolutionary algorithm using Simulated Annealing

    Required:
    @ problem - problem's build solution function knows how to create an individual in accordance with the encoding.
    @ population_size - to define the size of the population to be returned.
    """
    solution_list = []

    # generate a population of admissible solutions (individuals)
    for i in range(0, population_size):
        s = problem.build_solution(method='Simulated Annealing')

        # check if the solution is admissible
        while not problem.is_admissible(s):
            s = problem.build_solution(method='Simulated Annealing')

        s.id = [0, i]

        problem.evaluate_solution(s)

        solution_list.append(s)

    population = Population(
        problem=problem,
        maximum_size=population_size,
        solution_list=solution_list
    )

    return population

# -------------------------------------------------------------------------------------------------
# Initialization using Greedy Method
# -------------------------------------------------------------------------------------------------
def initialize_using_greedy(problem, population_size):
    """
    Initialize a population of solutions (feasible solution) for an evolutionary algorithm using a Greedy Algorithm

    Required:
    @ problem - problem's build solution function knows how to create an individual in accordance with the encoding.
    @ population_size - to define the size of the population to be returned.
    """
    solution_list = []

    # generate a population of admissible solutions (individuals)
    for i in range(0, population_size):
        s = problem.build_solution(method='Greedy')

        # check if the solution is admissible
        while not problem.is_admissible(s):
            s = problem.build_solution(method='Greedy')

        s.id = [0, i]

        problem.evaluate_solution(s)

        solution_list.append(s)

    population = Population(
        problem=problem,
        maximum_size=population_size,
        solution_list=solution_list
    )

    return population


###################################################################################################
# SELECTION APPROACHES
###################################################################################################
# -------------------------------------------------------------------------------------------------
# Roulettewheel_Selection Function
# -------------------------------------------------------------------------------------------------
def roulettewheel_selection(population, objective, params):
    """
    Main idea: better individuals get higher chance
    The chances are proportional to the fitness
    Implementation: roulette wheel technique
    Assign to each individual a part of the roulette wheel
    Spin the wheel n times to select n individuals

    REMARK: This implementation does not consider minimization problem
    """
    def _select_index_max(population):
        # Get the Total Fitness (all solutions in the population) to calculate the chances proportional to fitness
        total_fitness = 0
        for solution in population.solutions:
            total_fitness += solution.fitness

        # spin the wheel
        wheel_position = uniform(0, 1)

        # calculate the position which wheel should stop
        stop_position = 0
        index = 0
        for solution in population.solutions:
            stop_position += (solution.fitness / total_fitness)
            if stop_position > wheel_position:
                break
            index += 1

        return index

    def _select_index_min(population):
        # Get the Total Fitness (all solutions in the population) to calculate the chances proportional to fitness
        total_fitness = 0
        pop_size = population.size  # get the population size for the formula

        for solution in population.solutions:
            total_fitness += solution.fitness

        # spin the wheel
        wheel_position = uniform(0, 1)

        # calculate the position which wheel should stop
        stop_position = 0
        index = 0
        for solution in population.solutions:
            stop_position += (1 - (solution.fitness / total_fitness))/(pop_size - 1)
            if stop_position > wheel_position:
                break
            index += 1

        return index

    # select two different parents using roulette wheel
    if objective == ProblemObjective.Maximization:
        index1 = _select_index_max(population=population)
        index2 = index1

        while index2 == index1:
            index2 = _select_index_max(population=population)

    elif objective == ProblemObjective.Minimization:
        index1 = _select_index_min(population=population)
        index2 = index1

        while index2 == index1:
            index2 = _select_index_min(population=population)
    else:
        print('The code does not handle multiobjective problems yet.')
        exit(code=1)

    return population.get(index1), population.get(index2)


# -------------------------------------------------------------------------------------------------
# Rank Selection Function
# -------------------------------------------------------------------------------------------------
def rank_selection(population, objective, params):
    """
    Rank Selection sorts the population first according to fitness value and ranks them. Then every chromosome is
    allocated selection probability with respect to its rank. Individuals are selected as per their selection
    probability. Rank selection is an exploration technique of selection.
    """
    def _sort(population, objective):
        if objective == ProblemObjective.Maximization: # in maximization, the solution with higher fitness is at the end
            for i in range(0, population.size):
                for j in range(i, population.size):
                    if population.solutions[i].fitness > population.solutions[j].fitness:
                        population.solutions[i], population.solutions[j] = population.solutions[j], population.solutions[i]

        elif objective == ProblemObjective.Minimization: # in minimization, the solution with smaller fitness is at the end
            for i in range(0, population.size):
                for j in range(i, population.size):
                    if population.solutions[i].fitness < population.solutions[j].fitness:
                        population.solutions[i], population.solutions[j] = population.solutions[j], population.solutions[i]

        else:
            print('The code does not handle multiobjective problems yet.')
            exit(code=1)

        return population

    # Step 1: Sort / Rank
    population = _sort(population, objective)

    # Step 2: Create a rank list [0, 1, 1, 2, 2, 2, ...]

    # New Step 2: get the triangular number instead of len list
    def triangular_number(n):
        return n * (n + 1) // 2

    # Step 3: Select solution index
    index1, index2 = get_two_diff_order_index(start=0, stop=(triangular_number(population.size)-1), order=False, diff=True)

    def undo_triangular_number(n):
        return int(abs(math.ceil(((math.sqrt(8 * n +1) - 1) / 2)-1)))

    return population.get(undo_triangular_number(index1)), population.get(undo_triangular_number(index2))

# -------------------------------------------------------------------------------------------------
# Tournament Selection Function
# -------------------------------------------------------------------------------------------------
def tournament_selection(population, objective, params):
    """
    """
    tournament_size = 2
    if "Tournament-Size" in params:
        tournament_size = params["Tournament-Size"]

    def _select_index(objective, population, tournament_size):
        index_temp      = -1
        index_selected  = randint(0, population.size - 1)

        if objective == ProblemObjective.Maximization:
            for _ in range(0, tournament_size):
                index_temp = randint(0, population.size - 1)

                if population.solutions[index_temp].fitness > population.solutions[index_selected].fitness:
                    index_selected = index_temp
        elif objective == ProblemObjective.Minimization:
            for _ in range(0, tournament_size):
                index_temp = randint(0, population.size - 1)

                if population.solutions[index_temp].fitness < population.solutions[index_selected].fitness:
                    index_selected = index_temp
        else:
            print('The code does not handle multiobjective problems yet.')
            exit(code=1)

        return index_selected

    index1 = _select_index(objective, population, tournament_size)
    index2 = index1

    while index2 == index1:
        index2 = _select_index(objective, population, tournament_size)

    return population.solutions[index1], population.solutions[index2]


###################################################################################################
# CROSSOVER APPROACHES
###################################################################################################
# -------------------------------------------------------------------------------------------------
# Singlepoint crossover
# -------------------------------------------------------------------------------------------------
def singlepoint_crossover(problem, solution1, solution2):
    singlepoint = randint(0, len(solution1.representation)-1)
    #print(f" >> singlepoint: {singlepoint}")

    offspring1 = deepcopy(solution1) #solution1.clone()
    offspring2 = deepcopy(solution2) #.clone()

    for i in range(singlepoint, len(solution2.representation)):
        offspring1.representation[i] = solution2.representation[i]
        offspring2.representation[i] = solution1.representation[i]

    return offspring1, offspring2

# -------------------------------------------------------------------------------------------------
# Partially Mapped Crossover (PMX)
# -------------------------------------------------------------------------------------------------
def pmx_crossover(problem, solution1, solution2):
    # copy the parents to the children because we only need to change the middle part and the repeated elements
    offspring1 = deepcopy(solution1)
    offspring2 = deepcopy(solution2)

    # choose the random crossover points - get two different and ordered indexes
    crosspoint1, crosspoint2 = get_two_diff_order_index(0, (len(solution1.representation) - 1))

    # change the middle part of the children
    offspring2.representation[crosspoint1:crosspoint2] = solution1.representation[crosspoint1:crosspoint2]
    offspring1.representation[crosspoint1:crosspoint2] = solution2.representation[crosspoint1:crosspoint2]

    # indexes of the elements out of the middle part (they will be equal for both solutions)
    out_index = [item for item in list(range(0, len(solution1.representation))) if
                 item not in list(range(crosspoint1, crosspoint2))]

    repeated_index1 = []            # indexes of the repeated elements solution1
    repeated_index2 = []            # indexes of the repeated elements solution2
    for i in out_index:             # get the indexes of the repeated elements
        if offspring1.representation[i] in offspring1.representation[crosspoint1:crosspoint2]:  # repeated elements
            repeated_index1.append(i)                                                           # solution 1
        if offspring2.representation[i] in offspring2.representation[crosspoint1:crosspoint2]:  # repeated elements
            repeated_index2.append(i)                                                           # solution 2


    for i in repeated_index1: # for each repeated element in offspring1
        # replace i if the element in i of the opposite solution is not in offspring1
        if solution2.representation[i] not in offspring1.representation:
            offspring1.representation[i] = solution2.representation[i]
        else:
            idx = i
            while True:
                # get the index of solution1 of the element in i of solution2
                idx = solution1.representation.index(solution2.representation[idx])

                # replace i if the element in i of the opposite solution is not in offspring1
                if solution2.representation[idx] not in offspring1.representation:
                    offspring1.representation[i] = solution2.representation[idx]
                    break

    for i in repeated_index2:   # for each repeated element in offspring2
        # replace i if the element in i of the opposite solution is not in offspring2
        if solution1.representation[i] not in offspring2.representation:
            offspring2.representation[i] = solution1.representation[i]
        else:
            idx = i
            while True:
                # get the index of solution2 of the element in i of solution1
                idx = solution2.representation.index(solution1.representation[idx])

                # replace i if the element in i of the opposite solution is not in offspring2
                if solution1.representation[idx] not in offspring2.representation:
                    offspring2.representation[i] = solution1.representation[idx]
                    break   # break the while cycle because we already finished the cycle finding the element

    return offspring1, offspring2

# -------------------------------------------------------------------------------------------------
# Cycle Crossover
# -------------------------------------------------------------------------------------------------
def cycle_crossover(problem, solution1, solution2):
    cycle = 1   # number of cycles

    # only changes when the cycle is even (considering it starts at 1)
    offspring1 = deepcopy(solution1)  # solution1.clone()
    offspring2 = deepcopy(solution2)  # .clone()

    index_visited = []  # list of visited indexes during the cycles
    idx = None

    while len(index_visited) < len(solution1.representation):  # the loop only ends when all the indexes were visited
        for i in range(0, len(solution1.representation)):  # get the smallest index of the solution not visited
            if i not in index_visited:
                idx = i
                index_visited.append(idx)
                break

        if idx == None:
            print('Warning: idx equals None')

        if cycle % 2 == 0:    # when the cycle is even
            while True:
                offspring1.representation[idx] = solution2.representation[idx]    # save the swaped elements
                offspring2.representation[idx] = solution1.representation[idx]

                # get the respective index of the solution 1 for the element in solution 2
                idx = solution1.representation.index(solution2.representation[idx])

                if idx not in index_visited:    # if the index was already visited, the cycle ends
                    index_visited.append(idx)   # if not, append to the list of visited indexes
                else:
                    cycle += 1      # go to the next cycle
                    break

        else:       # when the cycle is odd
            while True:
                # get the respective index of the solution 1 for the element in solution 2
                idx = solution1.representation.index(solution2.representation[idx])

                if idx not in index_visited:    # if the index was already visited, the cycle ends
                    index_visited.append(idx)   # if not, append to the list of visited indexes
                else:
                    cycle += 1      # go to the next cycle
                    break

    return offspring1, offspring2

# -------------------------------------------------------------------------------------------------
# Order1 Crossover
# -------------------------------------------------------------------------------------------------
def order1_crossover(problem, solution1, solution2):
    # copy the parents to the children because we only need to change the middle part and the repeated elements
    offspring1 = deepcopy(solution1)
    offspring2 = deepcopy(solution2)

    # get two different, ordered, indexes
    crosspoint1, crosspoint2 = get_two_diff_order_index(0, (len(solution1.representation) - 1))

    # indexes of the elements not in the middle
    sub = [*range((crosspoint2+1), len(offspring1.representation))]+[*range(0, crosspoint1)]
    len_sub = len(sub)
    j = 0
    k = 0

    if crosspoint2 == (len(solution1.representation)-1) and crosspoint1 == 0:
        return solution1, solution2
    else:
                        # order by which elements must be considered
        for i in [*range((crosspoint2+1), len(offspring1.representation))]+[*range(0, (crosspoint2+1))]:

            # replace offspring1 if element is not in the middle
            if solution2.representation[i] not in solution1.representation[crosspoint1:(crosspoint2+1)]:
                offspring1.representation[sub[j]] = solution2.representation[i]
                j += 1

            # replace offspring2 if element is not in the middle
            if solution1.representation[i] not in solution2.representation[crosspoint1:(crosspoint2+1)]:
                offspring2.representation[sub[k]] = solution1.representation[i]
                k += 1

            if j == len_sub and k == len_sub:
                break

        return offspring1, offspring2

# -------------------------------------------------------------------------------------------------
# Heuristic Crossover
# -------------------------------------------------------------------------------------------------
def heuristic_crossover(problem, solution1, solution2):
    offspring1 = deepcopy(solution1)
    offspring2 = deepcopy(solution2)

    distances = problem.distances

    def apply_crossover_individually(offspring, matrix):
        copy = offspring.representation.copy()

        # get the element that will be in the first position of the offspring
        first = randint(1, len(solution1.representation))

        offspring.representation[0] = first  # put the element in the first position

        element = first
        i = 1
        while True:
            idx1 = solution1.representation.index(element) + 1  # get the index of the city next to element in solution 1
            idx2 = solution2.representation.index(element) + 1  # get the index of the city next to element in solution 2

            if idx1 == len(solution1.representation):  # if the index corresponds to the length of the solution
                idx1 = 0                               # replace it with the first index of all
            if idx2 == len(solution2.representation):  # if the index corresponds to the length of the solution
                idx2 = 0                               # replace it with the first index of all

            # get the elements in each parent next to the first element
            element1 = solution1.representation[idx1]
            element2 = solution2.representation[idx2]

            # if element1 is equal to element2 and element is not one of the changed elements in offspring1
            if (element1 == element2) and (element2 not in offspring.representation[0:i]):
                offspring.representation[i] = element2    # if the elements are equal or element 2 is already one of the changed elements
                element = element2          # in offspring1 (if it is will close the cycle too early)

            # if element1 is already one of the changed elements in offspring1 and element2 is not one of the
            # changed elements in offspring1
            elif (element1 in offspring.representation[0:i]) and (element2 not in offspring.representation[0:i]):
                offspring.representation[i] = element2        # (if it is will close the cycle too early)
                element = element2

            # if element2 is already one of the changed elements in offspring1 and element1 is not one of the
            # changed elements in offspring1
            elif (element2 in offspring.representation[0:i]) and (element1 not in offspring.representation[0:i]):
                offspring.representation[i] = element1        # (if it is will close the cycle too early)
                element = element1

            # evaluate which element has the shortest path
            elif (element2 not in offspring.representation[0:i]) and (element1 not in offspring.representation[0:i]):
                distance1 = matrix[element][element1]  # get the distance between the city in offspring and the element1
                distance2 = matrix[element][element2]  # get the distance between the city in offspring and the element1

                if distance1 <= distance2:  # if distance1 is the shortest path
                    offspring.representation[i] = element1  # replace the index i in offspring with the element 1
                    element = element1  # set the next element to look for

                else:  # if distance2 is the shortest path
                    offspring.representation[i] = element2  # replace the index i in offspring with the element 1
                    element = element2  # set the next element to look for
            else:
                element = sample((set(copy) - set(offspring.representation[0:i])), 1)[0]
                offspring.representation[i] = element  # replace the index i in offspring with the element

            i += 1  # iterate to the next index in the offspring to replace it with the correct element

            if i == (len(solution1.representation)-1):
                break

        offspring.representation[-1] = list(set(copy) - set(offspring.representation[0:-1]))[0]

        return offspring

    offspring1 = apply_crossover_individually(offspring1, distances)
    offspring2 = apply_crossover_individually(offspring2, distances)

    return offspring1, offspring2

# -------------------------------------------------------------------------------------------------
# Multiple Crossover
# -------------------------------------------------------------------------------------------------
def multiple_crossover(problem, solution1, solution2):
    prob = uniform(0, 1)

    if prob < (0.1):
        return pmx_crossover(problem, solution1, solution2)
    elif (0.1) <= prob < (0.2):
        return order1_crossover(problem, solution1, solution2)
    else:
        return heuristic_crossover(problem, solution1, solution2)

# -------------------------------------------------------------------------------------------------
# Single Arithmetic Crossover
# -------------------------------------------------------------------------------------------------
def single_arithmetic_crossover(problem, solution1, solution2):
    offspring1 = deepcopy(solution1)
    offspring2 = deepcopy(solution2)

    crosspoint = randint(0, (len(solution1.representation)-1))
    alpha = uniform(0, 1)

    offspring1.representation[crosspoint] = alpha*offspring1.representation[crosspoint] + \
                                            (1-alpha)*offspring2.representation[crosspoint]
    offspring2.representation[crosspoint] = alpha*offspring1.representation[crosspoint] + \
                                            (1-alpha)*offspring2.representation[crosspoint]

    offspring1.representation = list(map(lambda x: x/sum(offspring1.representation), offspring1.representation))
    offspring2.representation = list(map(lambda x: x / sum(offspring2.representation), offspring2.representation))

    offspring1.representation = np.array(offspring1.representation)
    offspring2.representation = np.array(offspring2.representation)

    return offspring1, offspring2

# -------------------------------------------------------------------------------------------------
# Simple Arithmetic Crossover
# -------------------------------------------------------------------------------------------------
def simple_arithmetic_crossover(problem, solution1, solution2):
    offspring1 = deepcopy(solution1)
    offspring2 = deepcopy(solution2)

    crosspoint = randint(0, (len(solution1.representation)-1))
    alpha = uniform(0, 1)

    for i in range(crosspoint, len(solution1.representation)):
        offspring1.representation[i] = alpha*offspring1.representation[i] + (1-alpha)*offspring2.representation[i]
        offspring2.representation[i] = alpha*offspring1.representation[i] + (1-alpha)*offspring2.representation[i]

    offspring1.representation = list(map(lambda x: x / sum(offspring1.representation), offspring1.representation))
    offspring2.representation = list(map(lambda x: x / sum(offspring2.representation), offspring2.representation))

    offspring1.representation = np.array(offspring1.representation)
    offspring2.representation = np.array(offspring2.representation)

    return offspring1, offspring2

# -------------------------------------------------------------------------------------------------
# Whole Arithmetic Crossover
# -------------------------------------------------------------------------------------------------
def whole_arithmetic_crossover(problem, solution1, solution2):
    offspring1 = deepcopy(solution1)
    offspring2 = deepcopy(solution2)

    alpha = uniform(0, 1)

    for i in range(0, len(solution1.representation)):
        offspring1.representation[i] = alpha*offspring1.representation[i] + (1-alpha)*offspring2.representation[i]
        offspring2.representation[i] = alpha*offspring1.representation[i] + (1-alpha)*offspring2.representation[i]

    offspring1.representation = np.array(offspring1.representation)
    offspring2.representation = np.array(offspring2.representation)

    return offspring1, offspring2

# -------------------------------------------------------------------------------------------------
# Multiple Arithmetic Crossover
# -------------------------------------------------------------------------------------------------
def multiple_arithmetic_crossover(problem, solution1, solution2):
    prob = uniform(0, 1)

    if prob < (1/3):
        return single_arithmetic_crossover(problem, solution1, solution2)
    elif (1/3) <= prob < (2/3):
        return simple_arithmetic_crossover(problem, solution1, solution2)
    else:
        return whole_arithmetic_crossover(problem, solution1, solution2)

# -------------------------------------------------------------------------------------------------
# PIP crossover
# -------------------------------------------------------------------------------------------------
def pip_crossover(problem, solution1, solution2):
    singlepoint = randint(0, len(solution1.representation)-1)

    offspring1 = deepcopy(solution1)
    offspring2 = deepcopy(solution2)

    for i in range(singlepoint, len(solution2.representation)):
        offspring1.representation[i] = solution2.representation[i]
        offspring2.representation[i] = solution1.representation[i]

    while offspring1.representation.sum() != offspring1.encoding.precision:
        if offspring1.representation.sum() < offspring1.encoding.precision:
            index = randint(0, (len(offspring1.representation)-1))
            offspring1.representation[index] = offspring1.representation[index] +\
                                               (offspring1.encoding.precision - offspring1.representation.sum())

        elif offspring1.representation.sum() > offspring1.encoding.precision:
            nonzero = np.nonzero(offspring1.representation)
            index = np.random.choice(nonzero[0], 1)[0]
            offspring1.representation[index] = 0

    while offspring2.representation.sum() != offspring2.encoding.precision:
        if offspring2.representation.sum() < offspring2.encoding.precision:
            index = randint(0, (len(offspring2.representation)-1))
            offspring2.representation[index] = offspring2.representation[index] +\
                                               (offspring2.encoding.precision - offspring2.representation.sum())

        elif offspring2.representation.sum() > offspring2.encoding.precision:
            nonzero = np.nonzero(offspring2.representation)
            index = np.random.choice(nonzero[0], 1)[0]
            offspring2.representation[index] = 0

    return offspring1, offspring2

###################################################################################################
# MUTATION APPROACHES
###################################################################################################
# -------------------------------------------------------------------------------------------------
# Singlepoint mutation
# -----------------------------------------------------------------------------------------------
def single_point_mutation(problem, solution):
    singlepoint = randint(0, len(solution.representation)-1)
    #print(f" >> singlepoint: {singlepoint}")

    encoding = problem.encoding

    if encoding.encoding_type == EncodingDataType.choices:
        try:
            temp = deepcopy(encoding.encoding_data)

            temp.pop(solution.representation[singlepoint])

            gene = temp[0]
            if len(temp) > 1:
                gene = choices(temp)

            solution.representation[singlepoint] = gene

            return solution
        except:
            print('(!) Error: singlepoint mutation encoding.data issues)')

    # return solution           

# -------------------------------------------------------------------------------------------------
# Swap mutation
# -----------------------------------------------------------------------------------------------
def swap_mutation(problem, solution):
    index = sample(range(0, len(solution.representation)), 2)

    solution.representation[index[0]], solution.representation[index[1]] = solution.representation[index[1]], \
                                                                           solution.representation[index[0]]

    return solution

# -------------------------------------------------------------------------------------------------
# Insert mutation
# -----------------------------------------------------------------------------------------------
def insert_mutation(problem, solution):
    solution2 = deepcopy(solution)

    mutpoint1, mutpoint2 = get_two_diff_order_index(0,(len(solution.representation) - 1))  # get two indexes

    # inserting the second value in the index after 1st mutation point
    solution2.representation[(mutpoint1 + 1)] = solution.representation[mutpoint2]

    # passing the rest of the elements by their original order to after the number inserted
    for i in range((mutpoint1 + 2), (mutpoint2 + 1)):
        solution2.representation[i] = solution.representation[(i - 1)]

    return solution2

# -------------------------------------------------------------------------------------------------
# Inversion mutation
# -----------------------------------------------------------------------------------------------
def inversion_mutation(problem, solution):
    solution2 = deepcopy(solution)  # create a copy that we will edit and return

    mutpoint1, mutpoint2 = get_two_diff_order_index(0, (len(solution.representation) - 1))  # get two indexes

    for i in range(mutpoint1, (mutpoint2 + 1)):  # replace the indexes by inverse order
        solution2.representation[i] = solution.representation[mutpoint2]
        mutpoint2 -= 1

    return solution2

# -------------------------------------------------------------------------------------------------
# Scramble mutation
# -----------------------------------------------------------------------------------------------
def scramble_mutation(problem, solution):
    mutpoint1, mutpoint2 = get_two_diff_order_index(0, (len(solution.representation) - 1))  # get two indexes

    # store the part of the solution to be shuffled in separate list
    shuffle_part = solution.representation[mutpoint1: (mutpoint2+1)]

    shuffle(shuffle_part)  # shuffle the middle part (shuffle works inplace)

    solution.representation[mutpoint1: (mutpoint2 + 1)] = shuffle_part  # replace the segment with shuffled part

    return solution

# -------------------------------------------------------------------------------------------------
# Greedy Swap mutation
# -----------------------------------------------------------------------------------------------
def greedy_mutation(problem, solution):
    offspring = deepcopy(solution)

    mutpoint1 = randint(0, (len(solution.representation) - 2))

    element = solution.representation[mutpoint1]

    distances = problem.distances
    distance_array = np.array(heapq.nsmallest(3, distances[element]))

    # get the index (item name) of the second smallest distance between mutpoint1 and all the other cities
    closest_city = np.argwhere(distances[element] == distance_array[1])[0][0]

    if closest_city == 0:  # the closest city can not be our first city on the matrix
        closest_city = np.argwhere(distances[element] == distance_array[2])[0][0]

    mutpoint2 = np.argwhere(solution.representation == closest_city)[0][0]

    offspring.representation[mutpoint1+1], offspring.representation[mutpoint2] = offspring.representation[mutpoint2], \
                                                                           offspring.representation[mutpoint1+1]

    # if final offspring equals the initial parent call the function again
    if offspring.representation == solution.representation:
        offspring = greedy_mutation(problem, solution)

    return offspring

# -------------------------------------------------------------------------------------------------
# Multiple mutation
# -----------------------------------------------------------------------------------------------
def multiple_mutation(problem, solution):
    prob = uniform(0, 1)

    if prob < (0.1):
        return greedy_mutation(problem, solution)
    elif (0.1) <= prob < (0.2):
        return scramble_mutation(problem, solution)
    else:
        return inversion_mutation(problem, solution)

# -------------------------------------------------------------------------------------------------
# Multiple Real Mutation
# -----------------------------------------------------------------------------------------------
def multiple_real_mutation(problem, solution):
    index = sample(range(0, len(solution.representation)), 2)

    param = random()

    if param < 0.3:
        solution.representation[index[0]], solution.representation[index[1]] = solution.representation[index[1]], \
                                                                               solution.representation[index[0]]

    elif param < 0.6:
        avg_num = math.floor(random() * (len(solution.representation) - 1))
        l = sample(range(0, (len(solution.representation) - 1)), avg_num)
        sum = 0
        for i in l:
            sum += solution.representation[i]
        for i in l:
            solution.representation[i] = sum / avg_num

    else:
        rand_num = round(random(), 1)
        solution.representation[index[0]], solution.representation[index[1]] = \
            round(rand_num * (solution.representation[index[0]] + solution.representation[index[1]]), 2), round(
            (1 - rand_num) * (solution.representation[index[0]] + solution.representation[index[1]]), 2)

    solution.representation = np.array(solution.representation)

    return solution

###################################################################################################
# REPLACEMENT APPROACHES
###################################################################################################
# -------------------------------------------------------------------------------------------------
# Standard replacement
# -----------------------------------------------------------------------------------------------
def standard_replacement(problem, current_population, new_population):
    return deepcopy(new_population)

# -------------------------------------------------------------------------------------------------
# Elitism replacement
# -----------------------------------------------------------------------------------------------
def elitism_replacement(problem, current_population, new_population):
    if problem.objective == ProblemObjective.Minimization:
        if current_population.fittest.fitness < new_population.fittest.fitness:
           new_population.solutions[0] = current_population.solutions[-1]

    elif problem.objective == ProblemObjective.Maximization:
        if current_population.fittest.fitness > new_population.fittest.fitness:
           new_population.solutions[0] = current_population.solutions[-1]

    return deepcopy(new_population)

###################################################################################################
# HELPER FUNCTIONS
####################################################################################################
def get_two_diff_order_index(start=0, stop=1, order=True, diff=True):
    """
    Returns two integers from a range, they can be:
        put in order (default) or unordered
        always different(default) or can be repeated
    start - integer (default = 0)
    stop - integer (default= 1)
    order - boolean ( default= True)
    """
    my_range = stop - start
    first = int(my_range * random())+start
    second = int(my_range * random())+start

    #first = randint(start, stop)
    #second = randint(start, stop)

    if diff:
        while first == second:
            second = int( my_range * random()) + start
            #second = randint(start, stop)
    if order:
        if first > second:
            second, first = first, second

    return first, second


