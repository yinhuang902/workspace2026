"""
Cut Pool for Lagrangean Branch-and-Cut

This module provides data structures for storing and managing Lagrangean cuts
in a global (or node-shared) cut pool, consistent with the Karuppiah-Grossmann
(2008) paper-style branch-and-cut algorithm.

Key classes:
- LagrangeanCut: A single Lagrangean cut with associated metadata
- CutPool: A collection of cuts with retrieval by node domain
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import copy


@dataclass
class LagrangeanCut:
    """
    Represents a single Lagrangean cut of the form:
        η_ω >= v_val + μ^T y
    
    where:
        - v_val is the subproblem objective value (global LB on min [f_ω(y) - μ^T y])
        - μ is the multiplier vector at the time of cut generation
        - The cut is valid in the space where y ∈ [bounds at generation time]
    
    Attributes
    ----------
    scenario_name : str
        Name of the scenario this cut corresponds to
    mu_vector : Dict[str, float]
        Multiplier values at cut generation, keyed by variable ID
    v_val : float
        Subproblem objective value (must be a proven global lower bound)
    iteration : int
        Iteration number when cut was generated
    node_id : int
        Node ID where cut was generated (for domain tracking)
    y_bounds : Dict[str, Tuple[float, float]]
        Bounds on y variables at generation time: {var_id: (lb, ub)}
    is_global : bool
        If True, cut is valid globally (not domain-restricted)
    """
    scenario_name: str
    mu_vector: Dict[str, float]
    v_val: float
    iteration: int = 0
    node_id: int = 0
    y_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    is_global: bool = False
    
    def is_valid_for_domain(self, 
                            current_bounds: Dict[str, Tuple[float, float]]) -> bool:
        """
        Check if this cut is valid for a given node domain.
        
        A cut generated at node N is valid for node M if:
        - M's domain is a subset of N's domain (M is a descendant of N), OR
        - The cut is marked as global
        
        For safety, we check: current_bounds ⊆ self.y_bounds
        i.e., for each var: current_lb >= stored_lb AND current_ub <= stored_ub
        
        Parameters
        ----------
        current_bounds : Dict[str, Tuple[float, float]]
            Bounds at the node where we want to use the cut
            
        Returns
        -------
        bool
            True if cut can be safely used at the given node
        """
        if self.is_global:
            return True
        
        # If no bounds were stored (legacy cut), assume not valid
        if not self.y_bounds:
            return False
        
        for var_id, (cur_lb, cur_ub) in current_bounds.items():
            if var_id not in self.y_bounds:
                # Variable not in cut's domain - conservative: reject
                return False
            stored_lb, stored_ub = self.y_bounds[var_id]
            # Current domain must be contained in stored domain
            if cur_lb < stored_lb - 1e-9 or cur_ub > stored_ub + 1e-9:
                return False
        
        return True
    
    def to_constraint_data(self) -> Tuple[str, Dict[str, float], float]:
        """
        Return the data needed to add this cut as a constraint.
        
        Returns: (scenario_name, mu_vector, v_val)
        """
        return (self.scenario_name, self.mu_vector, self.v_val)
    
    def signature(self, precision: int = 4) -> Tuple:
        """
        Generate a signature for duplicate detection.
        
        Parameters
        ----------
        precision : int
            Number of decimal places to round to for comparison
            
        Returns
        -------
        Tuple
            (scenario_name, rounded_v_val, tuple of rounded mu values)
        """
        rounded_v = round(self.v_val, precision)
        # Sort mu_vector keys for consistent signature
        sorted_mu = tuple(sorted((k, round(v, precision)) for k, v in self.mu_vector.items()))
        return (self.scenario_name, rounded_v, sorted_mu)


class CutPool:
    """
    Global cut pool for Lagrangean branch-and-cut.
    
    Manages a collection of Lagrangean cuts and provides methods for:
    - Adding new cuts
    - Retrieving cuts valid for a specific node domain
    - Pruning old/dominated cuts (optional)
    
    Attributes
    ----------
    cuts : List[LagrangeanCut]
        All stored cuts
    max_cuts_per_scenario : int
        Maximum number of cuts to keep per scenario (oldest pruned first)
    """
    
    def __init__(self, max_cuts_per_scenario: int = 100):
        """
        Initialize an empty cut pool.
        
        Parameters
        ----------
        max_cuts_per_scenario : int
            Maximum cuts to store per scenario. Set to -1 for unlimited.
        """
        self.cuts: List[LagrangeanCut] = []
        self.max_cuts_per_scenario = max_cuts_per_scenario
        self._cut_count_by_scenario: Dict[str, int] = {}
        self._cut_signatures: set = set()  # For duplicate detection
        self._duplicates_skipped: int = 0  # Counter for diagnostics
        # Constant cuts (mu=0, CZ-style) stored per (node_id, scenario)
        self._constant_cuts: Dict[Tuple[int, str], LagrangeanCut] = {}
    
    # ---- Constant (CZ-style) cut management --------------------------------

    def add_or_replace_constant_cut(self,
                                     node_id: int,
                                     scenario_name: str,
                                     mu_vector: Dict[str, float],
                                     v_val: float,
                                     y_bounds: Dict[str, Tuple[float, float]],
                                     iteration: int = -1) -> None:
        """
        Store a constant cut (mu=0) for a given node and scenario.
        Overwrites any previous constant cut for the same (node_id, scenario).
        
        These cuts are kept separate from the main pool to avoid
        polluting the pruning / deduplication logic.
        """
        cut = LagrangeanCut(
            scenario_name=scenario_name,
            mu_vector=copy.deepcopy(mu_vector),
            v_val=v_val,
            iteration=iteration,
            node_id=node_id,
            y_bounds=copy.deepcopy(y_bounds),
            is_global=(node_id == 0)
        )
        self._constant_cuts[(node_id, scenario_name)] = cut

    def get_constant_cuts_for_node(self,
                                    node_id: int) -> List[Tuple[str, Dict[str, float], float]]:
        """
        Return all constant cuts for a given node_id in RMP-ready format.
        
        Constant cuts are always valid for the node they were computed on
        (same y_bounds), so no domain check is needed.
        """
        result = []
        for (nid, sname), cut in self._constant_cuts.items():
            if nid == node_id:
                result.append(cut.to_constraint_data())
        return result
    
    def add_cut(self, cut: LagrangeanCut) -> bool:
        """
        Add a cut to the pool if not a duplicate.
        
        Parameters
        ----------
        cut : LagrangeanCut
            The cut to add
            
        Returns
        -------
        bool
            True if cut was added, False if skipped as duplicate
        """
        # Check for duplicate
        sig = cut.signature()
        if sig in self._cut_signatures:
            self._duplicates_skipped += 1
            return False
        
        self._cut_signatures.add(sig)
        self.cuts.append(cut)
        
        # Track count per scenario
        scenario = cut.scenario_name
        self._cut_count_by_scenario[scenario] = \
            self._cut_count_by_scenario.get(scenario, 0) + 1
        
        # Prune if over limit
        if self.max_cuts_per_scenario > 0:
            if self._cut_count_by_scenario[scenario] > self.max_cuts_per_scenario:
                self._prune_oldest_for_scenario(scenario)
        
        return True
    
    def add_cuts_from_iteration(self,
                                 cuts_data: List[Tuple[str, Dict[str, float], float]],
                                 iteration: int,
                                 node_id: int,
                                 y_bounds: Dict[str, Tuple[float, float]]) -> None:
        """
        Add multiple cuts from a single iteration.
        
        Parameters
        ----------
        cuts_data : List[Tuple[scenario_name, mu_vector, v_val]]
            Cut data from LG iteration
        iteration : int
            Current iteration number
        node_id : int
            Current node ID
        y_bounds : Dict[str, Tuple[float, float]]
            Bounds on y at this node
        """
        for (scenario_name, mu_vector, v_val) in cuts_data:
            # Root node cuts (node_id=0) are marked as global since they're valid 
            # for all descendants. This prevents LB oscillation when the pool fills
            # up and starts pruning old cuts.
            cut = LagrangeanCut(
                scenario_name=scenario_name,
                mu_vector=copy.deepcopy(mu_vector),
                v_val=v_val,
                iteration=iteration,
                node_id=node_id,
                y_bounds=copy.deepcopy(y_bounds),
                is_global=(node_id == 0)  # Root node cuts are globally valid
            )
            self.add_cut(cut)
    
    def get_valid_cuts(self, 
                       current_bounds: Dict[str, Tuple[float, float]],
                       scenario_filter: Optional[List[str]] = None) -> List[LagrangeanCut]:
        """
        Get all cuts valid for the given node domain.
        
        Parameters
        ----------
        current_bounds : Dict[str, Tuple[float, float]]
            Bounds on y at the current node
        scenario_filter : Optional[List[str]]
            If provided, only return cuts for these scenarios
            
        Returns
        -------
        List[LagrangeanCut]
            Cuts that can be safely used at this node
        """
        valid_cuts = []
        for cut in self.cuts:
            if scenario_filter and cut.scenario_name not in scenario_filter:
                continue
            if cut.is_valid_for_domain(current_bounds):
                valid_cuts.append(cut)
        return valid_cuts
    
    def get_cuts_for_rmp(self,
                         current_bounds: Dict[str, Tuple[float, float]],
                         all_scenario_names: List[str]) -> List[Tuple[str, Dict[str, float], float]]:
        """
        Get cuts in the format expected by _solve_rmp.
        
        Parameters
        ----------
        current_bounds : Dict[str, Tuple[float, float]]
            Bounds on y at the current node
        all_scenario_names : List[str]
            All scenario names (for filtering)
            
        Returns
        -------
        List[Tuple[scenario_name, mu_vector, v_val]]
            Cut data ready for RMP
        """
        valid_cuts = self.get_valid_cuts(current_bounds, all_scenario_names)
        return [cut.to_constraint_data() for cut in valid_cuts]
    
    def _prune_oldest_for_scenario(self, scenario: str) -> None:
        """Remove the oldest non-global cut for a given scenario.
        
        Global cuts (from root node) are never pruned since they're valid
        for all descendant nodes.
        """
        for i, cut in enumerate(self.cuts):
            if cut.scenario_name == scenario and not cut.is_global:
                del self.cuts[i]
                self._cut_count_by_scenario[scenario] -= 1
                break
    
    def clear(self) -> None:
        """Remove all cuts from the pool."""
        self.cuts.clear()
        self._cut_count_by_scenario.clear()
    
    def __len__(self) -> int:
        return len(self.cuts)
    
    def __repr__(self) -> str:
        return f"CutPool({len(self.cuts)} cuts)"
