from copy import deepcopy
import numpy as np

from ga_proj.cifo.problem.problem_template import ProblemTemplate
from cifo.problem.objective import ProblemObjective
from cifo.problem.solution import LinearSolution
from cifo.algorithm.hill_climbing import HillClimbing

key_encoding_rule = {
    "Size"         : 37,  # Number of keyboard possible keys
    "Is ordered"   : True,
    "Can repeat"   : True,
    "Data"         : [0, 0],  # must be defined by the data
    "Data Type"    : "Pattern"
}

# Default corpus
input_en = []
input_pt = []

tsp_decision_variables_example = {
    "Distances": np.array(input),
    "Item-Name": [i for i in range(1, len(input[0]))]  # << String, Optional, The names of the items, can be used to
                                                        # present the solution.
        # It starts at 1 reserving the 0 for the static city
        # We are setting the departure point, so we only have to arrange the 12 remaining cities
}

# REMARK: There is no constraint

 # -------------------------------------------------------------------------------------------------
# TSP - Travel Salesman Problem
# -------------------------------------------------------------------------------------------------
class TravelSalesmanProblem(ProblemTemplate):
    """
    Travel Salesman Problem: Given a list of cities and the distances between each pair of cities, what is the shortest
    possible route that visits each city and returns to the origin city?
    """

    # Constructor
    #----------------------------------------------------------------------------------------------
    def __init__(self, decision_variables=tsp_decision_variables_example, constraints=None,
                 encoding_rule=tsp_encoding_rule):
        """
        - Defines:
            - the Size and the Encoding characters (Data) according to the input distances matrix
            - the objective type (Max or Min)
            - the problem name
        - Optimizes the access to the decision variables
        """
        # optimize the access to the decision variables
        self._distances = np.array([])
        if "Distances" in decision_variables:
            self._distances = decision_variables["Distances"]

        encoding_rule["Size"] = len(self._distances) - 1
        encoding_rule["Data"] = decision_variables["Item-Name"]

        # Call the Parent-class constructor to store these values and to execute any other logic to be implemented by
        # the constructor of the super-class
        super().__init__(
            decision_variables = decision_variables,
            constraints = constraints,
            encoding_rule = encoding_rule
        )

        # 1. Define the Name of the Problem
        self._name = "Travel Salesman Problem"

        # 2. Define the Problem Objective
        self._objective = ProblemObjective.Minimization

    @property
    def distances(self):
        return self._distances

    # Build Solution for Travel Salesman Problem
    #----------------------------------------------------------------------------------------------
    def build_solution(self, method='Random'):
        """
        Creates a initial solution and returns it according to the encoding rule and the chosen method from the
        following available methods:
            - 'Hill Climbing'
            - 'Random' (default method)
            - 'Greedy'
        """
        if method == 'Random':  # shuffles the list and then instantiates it as a Linear Solution
            encoding_data = self._encoding.encoding_data

            solution_representation = encoding_data.copy()
            np.random.shuffle(solution_representation)  # inplace shuffle

            solution = LinearSolution(
                representation=solution_representation,
                encoding_rule=self._encoding_rule
            )

            return solution
        elif method == 'Hill Climbing':

            solution = HillClimbing(
                        problem_instance=self,
                        neighborhood_function=self.tsp_get_neighbors_np
                        ).search()
            #TODO: allow changes in params for Hill Climbing

            return solution
        #elif method == 'Simulated Annealing': TODO: build SA initialization
        #    return solution
        elif method == 'Greedy':
            # get a random solution where we will keep the first element
            initial_solution = np.array(self.build_solution(method='Random').representation)
            copy = initial_solution.copy()

            element = initial_solution[0]

            # i can not reach the last position because there is no more cities to evaluate
            for i in range(1, (len(initial_solution)-1)):
                distance_array = np.array(heapq.nsmallest(len(initial_solution), self._distances[element]))

                # get the index (item name) of the second smallest distance between element in index i and all the other cities
                closest_city = np.argwhere(self._distances[element] == distance_array[1])[0][0]

                if closest_city == 0:  # the closest city can not be our first city on the matrix
                    closest_city = np.argwhere(self._distances[element] == distance_array[2])[0][0]

                n = 2  # let us to go through the distance ascending list

                # while the closest city is already in the changed slice of the initial_solution, we have to keep looking
                while closest_city in initial_solution[0:i]:
                    # get the next closest city in the distance ascending list with the element evaluated
                    closest_city = np.argwhere(self._distances[element] == distance_array[n])[0][0]

                    if closest_city == 0:  # the closest city can not be our first city on the matrix
                        closest_city = np.argwhere(self._distances[element] == distance_array[n+1])[0][0]

                        n += 1  # if it is not a valid closest city the n should be plus one than the usual

                    n += 1

                # change the current position in initial_solution with the closest_city
                initial_solution[i] = closest_city

                # get the next element to evaluate the closest city
                element = initial_solution[i]

            # change the last index with the missing element
            initial_solution[-1] = np.setdiff1d(copy, initial_solution[0:-1], assume_unique=True)[0]

            #print('greedy solutions: ', list(initial_solution))
            solution = LinearSolution(
                representation=list(initial_solution),
                encoding_rule=self._encoding_rule
            )

            return solution
        else:
            print('Please choose one of these methods: Hill Climbing, Greedy, Random or Multiple. It will use the Random method')
            return self.build_solution()


    # Solution Admissibility Function - is_admissible()
    #----------------------------------------------------------------------------------------------
    def is_admissible(self, solution, debug=True):  # << use this signature in the sub classes, the meta-heuristic
        """
        Checks if the solution:
            - has unique values equal to the size of the encoding_rule
            - has length equal to the size of the encoding_rule
            - has all values within the defined range of item-names
        If all these conditions are true, the solution is admissible and it returns True.
        """
        if debug:
            result = False

            if (len(set(solution.representation)) == self.encoding_rule["Size"]) & \
                    (len(solution.representation) == self.encoding_rule["Size"]) & \
                    all([True if i in range(1, (self.encoding_rule["Size"]+1))
                         else False for i in solution.representation]):
                result = True
            else:
                print('Error: Solution ' + str(solution.representation) + ' Result ' + str(result))

            return result
        else:
            return True

    # Evaluate_solution()
    #-------------------------------------------------------------------------------------------------------------
    # It should be seen as an abstract method 
    def evaluate_solution(self, solution, feedback=None):  # << This method does not need to be extended, it already
                                        # automated solutions evaluation, for Single-Objective and for Multi-Objective
        """
        Calculates the total distance of the defined route
        """
        distances = self._distances
        rep = solution.representation

        fitness = distances[0, rep[0]] + distances[rep[-1], 0]  # the distances are assymmetric
            # distance from the departure point (position 0 in the matrix) to the first city(position 0 in the solution)
            # plus
            # distance from the last city to the departure point (position 0 in the solution)

        for i in range(0, (len(rep)-1)):
            fitness += distances[rep[i], rep[i+1]]

        solution._fitness = fitness
        solution._is_fitness_calculated = True

        return solution

# -------------------------------------------------------------------------------------------------
# OPTIONAL - it is only needed if you will implement Local Search Methods
#            (Hill Climbing and Simulated Annealing)
# -------------------------------------------------------------------------------------------------
    def tsp_get_neighbors_np(self, solution, problem, neighborhood_size=0, n_changes=3):
        initial_sol = np.asarray(solution.representation)       # change to numpy array for performance
        neighbors_np = [initial_sol]                            # list of numpy arrays for performance

        def n_change(list_solutions):
            neighbors = []

            for k in list_solutions:                            # find all neighbors
                for l in range(0, len(k)):
                    for j in range((l + 1), len(k)):
                        neighbor = np.copy(k)
                        neighbor[l], neighbor[j] = neighbor[j], neighbor[l]
                        neighbors.append(neighbor)

            return neighbors

        for i in range(0, n_changes):                       # How many swaps to allow,
            neighbors_np = n_change(neighbors_np)           # This escalates fast!!! one should be more than enough

                                                            # convert back to linear solution for evaluation
        neighbors_final = []
        for sol in neighbors_np:
            sol = sol.tolist()
            solution = LinearSolution(
                representation=sol,
                encoding_rule=self._encoding_rule
            )
            neighbors_final.append(solution)

        return neighbors_final                              # return a list of solutions

def tsp_get_neighbors(solution, problem, neighborhood_size = 0, n_changes=1):
    neighbors_final = [solution]

    def n_change(list_solutions):
        neighbors = []

        for k in list_solutions:
            for l in range(0, len(k.representation)):
                for j in range((l+1), len(k.representation)):
                    neighbor = deepcopy(k)

                    neighbor.representation[l], neighbor.representation[j] = neighbor.representation[j], \
                                                                             neighbor.representation[l]
                    #if neighbor.representation not in list(map(lambda x: x.representation, neighbors)):
                    neighbors.append(neighbor)

        #neighbors = [n for n in neighbors if list(map(lambda x: x.representation, neighbors)).count(n.representation) > 1]

        return neighbors

    for i in range(0, n_changes):
        neighbors_final = n_change(neighbors_final)
        print(str(len(neighbors_final))+"i="+str(i))

    #if solution in neighbors_final:                # slower than evaluating the solution also
    #    neighbors_final.remove(solution)

    return neighbors_final

