from copy import deepcopy
import logging
import random
import os

from dssg_challenge import compute_cost, utils
from dssg_challenge.ga.problem.problem_template import ProblemTemplate
from dssg_challenge.ga.problem.objective import ProblemObjective
from dssg_challenge.ga.problem.solution import LinearSolution
from dssg_challenge.ga.algorithm.hill_climbing import HillClimbing
from dssg_challenge.ga.algorithm.ga_operators import swap_mutation, insert_mutation, inversion_mutation, scramble_mutation

# TODO: set logs instead of prints

key_encoding_rule = {
    "Size"         : 37,  # Number of keyboard possible keys - fixed
    "Is ordered"   : True,
    "Can repeat"   : True,
    "Data"         : "",  # must be updated by the decision variable Valid_keys
    "Data Type"    : "Pattern"
}

# Default corpus
default_corpus = "THIS IS K<THE LIST OF VALID CHARS #ABCDEFGHIJKLMNOPQ RSTUVWXYZ0.#,^?<#0THIS COMPETITION P<IS ORGANIZED TOGETHER WITH THE DSSG SUMMIT ####.0^ITS GOAL IS TO OPTIMIZE A KEYBOARD LAYOUT TO MINIMIZE THE Y<WORKLOAD FOR USAGE BY AN ALS PATIENT. ^THIS WAS MOTIVATEM<D BY ANTHONY CARBAJAL, A FULL#TIME DAILY LIFE HACKER THAT AIMS TO FIND INNOVATIVE Z<WAYS TO IMPROVE HIS #<AND OTHER ALS#PATIENT LIVES, WITH WHOM K<WE WORKED TOGETHER FOR 0<DEVELOPING A FIRST VERSION J<OF THIS SOLUTION.0^YOU WILL N<WORK IN CREATING INNOVATIVE P<SOLUTIONS TO K<THE PROBLEM WE TRIED TO SOLVE#0"
default_keys = "ABCDEFGHIJKLMNOPQ RSTUVWXYZ0.#,^?<"

key_decision_variables_example = {
    "Corpus": default_corpus,
    "Valid_keys": default_keys
}

key_constraints_example = {
    "Exaustiveness" : True  # Constrains solutions to have at least one of each valid character
}

# -------------------------------------------------------------------------------------------------
# ALS Keyboard Problem
# -------------------------------------------------------------------------------------------------
class AlsKeyboardProblem(ProblemTemplate):
    """
    ALS Keyboard Problem: Given an input corpus, what is the arrangement of keys of a given keyboard layout that
    minimizes the writting effort?
    """

    # Constructor
    #----------------------------------------------------------------------------------------------
    def __init__(self, decision_variables=key_decision_variables_example, constraints=key_constraints_example,
                 encoding_rule=key_encoding_rule):
        """
        - Defines:
            - Data according to the input corpus
            - Objective type (Max or Min)
            - Problem name
        - Optimizes the access to the decision variables
        """
        # optimize the access to the decision variables
        self._corpus = ""
        if "Corpus" in decision_variables:
            self._corpus = decision_variables["Corpus"]

        self._valid_keys = ""
        if "Valid_keys" in decision_variables:
            self._valid_keys = decision_variables["Valid_keys"]

        # optimize the access to the constraints
        self._exaustive = True
        if "Exaustiveness" in constraints:
            assert constraints["Exaustiveness"]

        # update encoding_rule given decision variables to pass to Parent's constructor
        encoding_rule["Data"] = decision_variables["Valid_keys"] + "_"

        # call the Parent-class constructor
        super().__init__(
            decision_variables = decision_variables,
            constraints = constraints,
            encoding_rule = encoding_rule
        )

        # Additional attributes
        self._name = "ALS Keyboard Problem"
        self._objective = ProblemObjective.Minimization

    # Build Solution for ALS Keyboard Problem
    #----------------------------------------------------------------------------------------------
    def build_solution(self, method='Random'):
        """
        Creates a initial solution and returns it according to the encoding rule and the chosen method from the
        following available methods:
            - 'Hill Climbing'
            - 'Random' (default method)
        """
        if method == 'Random':

            # Creates a list of characters with the correct number of characters and at least one of each of the valid characters
            solution_representation = list(self._valid_keys) + \
                random.choices(self._encoding.encoding_data, k=(self._encoding.size - len(self._valid_keys)))
            random.shuffle(solution_representation)  # shuffles the string characters

            solution = LinearSolution(
                representation=solution_representation,
                encoding_rule=self._encoding_rule
            )
            return solution

        elif method == "Heuristic":
            initial_solution = LinearSolution(
                representation=list('EINOA TCGVDURL<^SWH_Z__XJQFPBMY,#.0K?'),
                encoding_rule=self._encoding_rule
            )
            if not self.is_admissible(initial_solution):
                raise Exception("Initial solution not valid given valid keys passed.")

            possible_mutations = [swap_mutation, insert_mutation, inversion_mutation, scramble_mutation]
            mutation = random.choice(possible_mutations)
            final_solution = mutation(self, initial_solution)
            
            return final_solution

        else:
            raise Exception("No {} method available. Currently only 'Random' and 'Heuristic' method is implemented.".format(method))
        # elif method == 'Hill Climbing':

        #     solution = HillClimbing(
        #                 problem_instance=self,
        #                 neighborhood_function=self.tsp_get_neighbors_np  # CODE HERE (pass neighborhood function)
        #                 ).search()
        #     #TODO: allow changes in params for Hill Climbing

        #     return solution
        #elif method == 'Simulated Annealing': TODO: build SA initialization
        #    return solution

    # Solution Admissibility Function - is_admissible()
    #----------------------------------------------------------------------------------------------
    def is_admissible(self, solution, debug=False):
        """
        Checks if:
            - the solution has the correct size
            - the solution has every valid character
            - the set of valid characters has every character from the solution
        If all these conditions are true, the solution is admissible and it returns True.
        """
        try:
            utils.check_keyboard(solution.representation, self._valid_keys, self._encoding.size)
        except AssertionError as e:
            if debug:
                print(e)
            return False
        else:
            return True

    # Evaluate_solution()
    #-------------------------------------------------------------------------------------------------------------
    def evaluate_solution(self, solution, feedback=None):
        """
        Evaluates the solution provided
        """
        rep = solution.representation
        fitness = compute_cost(rep, self._corpus)
        solution._fitness = fitness
        solution._is_fitness_calculated = True

        return solution

    @property
    def corpus(self):
        return self._corpus
# -------------------------------------------------------------------------------------------------
# OPTIONAL - it is only needed if you will implement Local Search Methods
#            (Hill Climbing and Simulated Annealing)
# -------------------------------------------------------------------------------------------------
#     def tsp_get_neighbors_np(self, solution, problem, neighborhood_size=0, n_changes=3):
#         initial_sol = np.asarray(solution.representation)       # change to numpy array for performance
#         neighbors_np = [initial_sol]                            # list of numpy arrays for performance

#         def n_change(list_solutions):
#             neighbors = []

#             for k in list_solutions:                            # find all neighbors
#                 for l in range(0, len(k)):
#                     for j in range((l + 1), len(k)):
#                         neighbor = np.copy(k)
#                         neighbor[l], neighbor[j] = neighbor[j], neighbor[l]
#                         neighbors.append(neighbor)

#             return neighbors

#         for i in range(0, n_changes):                       # How many swaps to allow,
#             neighbors_np = n_change(neighbors_np)           # This escalates fast!!! one should be more than enough

#                                                             # convert back to linear solution for evaluation
#         neighbors_final = []
#         for sol in neighbors_np:
#             sol = sol.tolist()
#             solution = LinearSolution(
#                 representation=sol,
#                 encoding_rule=self._encoding_rule
#             )
#             neighbors_final.append(solution)

#         return neighbors_final                              # return a list of solutions

# def tsp_get_neighbors(solution, problem, neighborhood_size = 0, n_changes=1):
#     neighbors_final = [solution]

#     def n_change(list_solutions):
#         neighbors = []

#         for k in list_solutions:
#             for l in range(0, len(k.representation)):
#                 for j in range((l+1), len(k.representation)):
#                     neighbor = deepcopy(k)

#                     neighbor.representation[l], neighbor.representation[j] = neighbor.representation[j], \
#                                                                              neighbor.representation[l]
#                     #if neighbor.representation not in list(map(lambda x: x.representation, neighbors)):
#                     neighbors.append(neighbor)

#         #neighbors = [n for n in neighbors if list(map(lambda x: x.representation, neighbors)).count(n.representation) > 1]

#         return neighbors

#     for i in range(0, n_changes):
#         neighbors_final = n_change(neighbors_final)
#         print(str(len(neighbors_final))+"i="+str(i))

#     #if solution in neighbors_final:                # slower than evaluating the solution also
#     #    neighbors_final.remove(solution)

#     return neighbors_final

