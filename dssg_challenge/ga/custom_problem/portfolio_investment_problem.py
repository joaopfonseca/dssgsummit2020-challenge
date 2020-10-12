from random import randint
import numpy as np
from pandas import read_excel

from dssg_challenge.ga.problem.problem_template import ProblemTemplate
from dssg_challenge.ga.problem.objective import ProblemObjective
from dssg_challenge.ga.problem.solution import PIP_Solution

pip_encoding_rule = {
    "Size": -1,  # It must be defined by the size of DV (Number of products)
    "Is ordered": False,
    "Can repeat": True,
    "Data": [0, 0],
    "Data Type": "",
    "Precision": 1/0.01
}

closing_prices = read_excel(r".\data\sp_12_weeks.xlsx")  # importing data example

def get_dv_pip(prices):
    est_ret = np.log(prices.values[1:] / prices.values[:-1])  # returns for each period in column
    exp_ret = np.sum(est_ret, axis=0)  # expected return over time interval
    cov_ret = np.cov(est_ret, rowvar=False)  # estimated covariance matrix over time interval
    return exp_ret, cov_ret

exp, cov = get_dv_pip(closing_prices)
# pip_decision_variables_example = {
# #     "Closing_Prices" : closing_prices,  # The closing prices for each period
# # }

# Just like in knapsack, each portfolio (bag) is composed by stocks (items) which have an expected return (value) and
# a risk (weight). The main difference is that the stocks (items) are not independent and we need to take into account
# the covariance between them.
pip_decision_variables_example = {
    "Expected_Returns" : exp,  # The closing prices for each period
    "Covariance_Returns": cov,
    "Risk_Free_Return": 1.53  # US Treasury Bonds according to Bloomberg on 08/01/2020, 18:08
}

pip_constraints_example = {
    "Risk-Tolerance": 1,
    "Budget": 100000
}

# -------------------------------------------------------------------------------------------------
# PIP - Portfolio Investment Problem 
# -------------------------------------------------------------------------------------------------
class PortfolioInvestmentProblem(ProblemTemplate):
    """
    """
    # Constructor
    # ----------------------------------------------------------------------------------------------
    def __init__(self, decision_variables=pip_decision_variables_example, constraints=pip_constraints_example,
                 encoding_rule=pip_encoding_rule):
        """
        """
        # optimize the access to the decision variables
        # self._prices = np.array([])
        # if "Closing_Prices" in decision_variables:
        #     self._prices = decision_variables["Closing_Prices"]
        #
        # encoding_rule["Size"] = self._prices.shape[1]  # the encoding will be a string with as many elements as stocks
        # encoding_rule["Data"] = decision_variables["Closing_Prices"]
        self._exp_returns = np.array([])
        if "Expected_Returns" in decision_variables:
            self._exp_returns = decision_variables["Expected_Returns"]

        self._cov_returns = np.array([])
        if "Covariance_Returns" in decision_variables:
            self._cov_returns = decision_variables["Covariance_Returns"]

        self._risk_free_return = 0
        if "Risk_Free_Return" in decision_variables:
            self._risk_free_return = decision_variables["Risk_Free_Return"]

        self._risk_tol = 0  # "Tolerance"
        if "Tolerance" in constraints:
            self._risk_tol = constraints["Tolerance"]

        self._budget = 0  # "Budget"
        if "Budget" in constraints:
            self._budget = constraints["Budget"]

        # the encoding will be a string with as many elements as stocks
        if encoding_rule["Size"] == -1:  # if the user doesn't give a limit of stocks for the portfolio
            encoding_rule["Size"] = self._exp_returns.shape[0]

        # encoding data doesn't need to be defined (just like in knapsack)

        # Call the Parent-class constructor to store these values and to execute  any other logic to be implemented by the constructor of the super-class
        super().__init__(
            decision_variables=decision_variables,
            constraints=constraints,
            encoding_rule=encoding_rule
        )

        # 1. Define the Name of the Problem
        self._name = "Portfolio Investment Problem"

        # 2. Define the Problem Objective
        self._objective = ProblemObjective.Maximization

    # Build Solution
    # ----------------------------------------------------------------------------------------------
    def build_solution(self, method='Random'):
        """
        We generate a random solution ensuring that the sum of the weights equals to 100
        """
        if method == 'Random':  # shuffles the list and then instantiates it as a Linear Solution
            solution_representation = np.zeros(self._encoding.size, dtype=int)

            indexes_visited = []
            sum = 0
            while sum < self._encoding.precision:
                index = randint(0, (self._encoding.size-1))

                while index in indexes_visited:
                    index = randint(0, (self._encoding.size - 1))

                element = randint(0, (self._encoding.precision-sum))

                solution_representation[index] = element
                sum += element
                indexes_visited.append(index)

                if len(indexes_visited) == (self._encoding.size-1):
                    if solution_representation.sum() != self._encoding.precision:
                        index = randint(0, (self._encoding.size - 1))
                        solution_representation[index] = solution_representation[index] + (self._encoding.precision-sum)
                    break

            solution = PIP_Solution(
                representation = solution_representation,
                encoding_rule = self._encoding_rule,
                problem = self
            )

            return solution
        else:
            print("Not yet implemented")
            return None

    # Solution Admissibility Function - is_admissible()
    # ----------------------------------------------------------------------------------------------
    def is_admissible(self, solution):  # << use this signature in the sub classes, the meta-heuristic
        """
        The solution is admissible if its sharpe ratio is bigger or equal to the risk tolerance and the sum of the
        weights have to be equal to 1
        """
        #if (solution.sharpe_ratio < self._risk_tol) | (solution.representation.sum() != 1):

        if solution.representation.sum() != self.encoding.precision:
            print('ERROR: the solution is not admissible!')
            return False
        else:
            return True

    # Evaluate_solution()
    # -------------------------------------------------------------------------------------------------------------
    # It should be seen as an abstract method 
    def evaluate_solution(self, solution, feedback=None):  # << This method does not need to be extended, it already
        # automated solutions evaluation, for Single-Objective and for Multi-Objective
        """
        Our objective is to maximize the sharpe ratio so the fitness is the sharpe ratio. Besides that for each new
        solution we have to update the expected return, standard deviation and the sharpe ratio in order to get the
        new fitness.
        """
        # weights = solution.representation / self.encoding.precision
        solution._exp_return = solution.representation @ self._exp_returns  # Portfolio Expected Return formula
        solution._risk = np.sqrt(solution.representation.T @ self._cov_returns @ solution.representation)  # Portfolio Standard Deviation formula
        solution._sharpe_ratio = (solution._exp_return - self._risk_free_return) / solution._risk

        solution._fitness = solution.sharpe_ratio
        solution._is_fitness_calculated = True

        return solution


###################################################################################################
# INPUT SPACE REDUCTION PIP
###################################################################################################
def input_space_red_pip(size, data):
    ex_return, covariance = get_dv_pip(data)
    risk = np.sqrt(np.diag(covariance))
    corr = data.corr(method='pearson')
    ratio = ex_return/risk

    indexes = np.argsort(ratio)

    keep = list(indexes[-size:])
    keep4ever = []

    for i in keep:
        if i not in keep4ever:
            if len(keep4ever) == (size-1):
                keep4ever.append(i)
                break
            else:
                keep4ever.append(i)
                correlated = list(corr.iloc[:, (i-1)]).index(corr.iloc[:, (i-1)].min())
                if correlated not in keep4ever:
                    keep4ever.append(correlated)

                if len(keep4ever) == size:
                    break

    print('Indexes of the Stocks: ', keep4ever)
    df = data.iloc[:, keep4ever]

    return df

