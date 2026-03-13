import heapq
from collections.abc import Iterable
from enum import Enum
import pyomo.environ as pyo

import snoglode.utils.compute as compute

class QueueStrategy(Enum):
    lifo = 'lifo'
    fifo = 'fifo'
    bound = 'bound'
    bound_k_solutions = "bound_k_solutions"


class NodeQueue(Iterable):
    def __init__(self) -> None:
        self._q = []

    def __iter__(self):
        for i in self._q:
            yield i

    def __len__(self):
        return len(self._q)
    
    def push(self, node, *args, **kwargs) -> None:
        raise NotImplementedError('should be implemented by derived classes')
    
    def pop(self):
        raise NotImplementedError('should be implemented by derived classes')


class LIFONodeQueue(NodeQueue):
    def __init__(self) -> None:
        super().__init__()
        self._ndx = 0

    def push(self, node, *args, **kwargs) -> None:
        heapq.heappush(self._q, (self._ndx, node))
        self._ndx -= 1

    def pop(self):
        _, node = heapq.heappop(self._q)
        return node


class FIFONodeQueue(NodeQueue):
    def __init__(self) -> None:
        super().__init__()
        self._ndx = 0

    def push(self, node, *args, **kwargs):
        heapq.heappush(self._q, (self._ndx, node))
        self._ndx += 1

    def pop(self):
        _, node = heapq.heappop(self._q)
        return node


class WorstBoundNodeQueue(NodeQueue):
    def push(self, node, *args, **kwargs):
        heapq.heappush(self._q, (node.lb_problem.objective, node))

    def pop(self):
        _, node = heapq.heappop(self._q)
        return node

# ============================== AOS/diversity Queuing ==============================

class DiversiTree(NodeQueue):
    def __init__(self):
        super().__init__()
        self.alpha = 0.5

        try:
            from pyomo.contrib.alternative_solutions.solnpool import (
                PoolCounter,
                SolutionPool_KeepBest,
                PoolManager,
            )
            from pyomo.contrib.alternative_solutions.solution import Solution, PyomoSolution
            from pyomo.contrib.alternative_solutions import Objective, Variable
            print("DiversiTree: successfully imported pyomo.contrib.alternative_solutions")
        except:
            raise ImportError(
                "Alternative solutions package from pyomo.contrib is unavailable.\n" +\
                "Please install the pyomo.contrib package to use DiversiTree.\n" +\
                "Otherwise, consider using alternative queues."
            )
        
        self.solution_pool = SolutionPool_KeepBest(max_pool_size=10, 
                                                   sense_is_min=True,
                                                   counter = PoolCounter(),
                                                   rel_tolerance=0.01,
                                                   abs_tolerance=1000)
        # self.solution_pool.add_pool("pool", policy="keep_best", max_pool_size=10, sense_is_min=True,
                                    # rel_tolerance=0.01, abs_tolerance=1e-4)

    def DAll(self, node):
        """
        Using the generalized diveresity metric 
        (can alternatively use Hamming distance in the case of only binaries)

        Given a set of solutions (S) the diversity Dall(S) is given by:

        DAll(S) = (1/len S) * sum(var(s in S) variance(s))
        """
        num_solutions = self.solution_pool.__len__()
        raise NotImplementedError
    

class WorstBoundkSolutions(DiversiTree):
    def push(self, node, candidate, candidate_obj, *args, **kwargs):
        from pyomo.contrib.alternative_solutions import Objective, Variable
        from pyomo.contrib.alternative_solutions.solution import Solution
        if candidate != None:
            # save solution
            variables = []
            for var in candidate:
                variables.append(Variable(name = var, value = candidate[var]))

            self.solution_pool.add(
                        variables=variables,
                        objectives=[Objective(value=pyo.value(candidate_obj))],
                )
        
        # push to queue based on worst bound 
        heapq.heappush(self._q, (node.lb_problem.objective, node))

    def pop(self):
        _, node = heapq.heappop(self._q)
        return node