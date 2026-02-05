"""
Simple base class for all of the solvers.
All solvers - lower bound, upper bound, candidate generator
"""
# suppress warnings when loading infeasible models
import logging
logging.getLogger('pyomo.core').setLevel(logging.ERROR) 
import pyomo as pyomo

class BoundingProblemBase():
    """
    Initalizes all of the common elements of the bounding problems.
    Mostly solver information.
    """

    def __init__(self, 
                 solver) -> None:
        """
        Initializes the solver information.

        Parameters
        -----------
        solver : pyo.SolverFactory
            initialized Pyomo solver factory object
        """
        # init the opt object + save solver name
        self.opt = solver
        if (solver != None): 
            if type(solver) == pyomo.solvers.plugins.solvers.GAMS.GAMSShell:
                self.solver = f"GAMS, {solver.options}"
            else: self.solver = solver.name
        # if we do not require a solver for CG, might be None