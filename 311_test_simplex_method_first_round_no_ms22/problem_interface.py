"""
problem_interface.py — General two-stage stochastic programming problem interface.

Responsibility boundary (IMPORTANT — keep narrow):
  This module is a *problem-data / metadata container only*.
  It does NOT own mesh state, solver state, iteration logs, EF caches,
  or any algorithm-specific mutable state.

Provides:
  - ScenarioSubproblem : structured return type from subproblem_creator
  - ProblemSetup       : problem descriptor (dim, scenarios, bounds, variable mappings)
  - assign_point_to_vars / extract_point_from_vars : safe point <-> Pyomo variable transfer

Forward-compatibility note (Patches 5–6):
  ScenarioSubproblem already carries fields for objective_expr and metadata
  that are *optional* in Patch 1. Later patches will use them for EF upper-
  bound computation and weighted objective aggregation.

Accepted Pyomo variable types:
  first_stage_vars entries may be any Pyomo component that has .value
  (settable) and .name attributes — including ScalarVar, VarData, indexed
  Var slices, etc. We do not restrict to pyo.Var specifically.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import pyomo.environ as pyo
except ImportError:
    pyo = None  # allow import without Pyomo for type-checking only

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured return type from subproblem_creator
# ---------------------------------------------------------------------------

@dataclass
class ScenarioSubproblem:
    """Structured result returned by a ``subproblem_creator`` callable.

    Attributes
    ----------
    model : pyo.ConcreteModel
        The scenario Pyomo model.  Must have an objective or an ``obj_expr``
        attribute (see ``objective_expr`` below).
    first_stage_vars : list | dict
        First-stage Pyomo variable references for this scenario.
        Accepted forms (checked at validation time):
          - ``list`` of Pyomo var-like objects (order must match across scenarios)
          - ``dict[str, var]`` mapping canonical names to var-like objects (preferred)
    probability : float
        Scenario probability. Must be in (0, 1].
    scenario_name : str
        Human-readable scenario identifier (for logging / debug).
    objective_expr : optional
        Pyomo expression for the scenario recourse objective.
        If ``None``, ``ProblemSetup`` will attempt to locate the active
        Objective component on ``model`` or fall back to ``model.obj_expr``.
        *Not used in Patch 1; reserved for Patches 5–6 (EF upper-bound).*
    metadata : dict
        Free-form metadata (solver options overrides, scenario origin, etc.).
        *Not used in Patch 1; reserved for future patches.*
    """

    model: Any  # pyo.ConcreteModel
    first_stage_vars: Union[list, dict]
    probability: float
    scenario_name: str = ""
    objective_expr: Any = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Point <-> Pyomo helpers
# ---------------------------------------------------------------------------

def assign_point_to_vars(first_vars: Sequence, x: np.ndarray) -> None:
    """Set Pyomo variable ``.value`` from a NumPy point array.

    Parameters
    ----------
    first_vars : sequence of Pyomo var-like objects, length *d*
    x : np.ndarray, shape ``(d,)``

    Raises
    ------
    ValueError
        If ``len(first_vars) != len(x)`` or ``x`` is not 1-D.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {x.shape}")
    if len(first_vars) != len(x):
        raise ValueError(
            f"first_vars has {len(first_vars)} entries but x has {len(x)}"
        )
    for var, val in zip(first_vars, x):
        var.value = float(val)


def extract_point_from_vars(first_vars: Sequence) -> np.ndarray:
    """Read current Pyomo variable values into a NumPy array.

    Returns
    -------
    np.ndarray, shape ``(d,)``, dtype float64
    """
    vals = []
    for v in first_vars:
        raw = v.value if hasattr(v, "value") else pyo.value(v)
        vals.append(float(raw) if raw is not None else math.nan)
    return np.array(vals, dtype=float)


# ---------------------------------------------------------------------------
# Canonical naming helpers
# ---------------------------------------------------------------------------

def _get_canonical_name(var) -> str:
    """Best-effort canonical name for a Pyomo variable-like object.

    Priority order (per user requirement D):
      1. ``local_name`` — block-prefix-free component name
      2. ``.name``      — fully qualified (may include block path)
      3. ``str(var)``   — last resort
    """
    # Prefer local_name (no block prefix contamination)
    ln = getattr(var, "local_name", None)
    if ln is not None:
        return str(ln)
    n = getattr(var, "name", None)
    if n is not None:
        return str(n)
    return str(var)


def _extract_canonical_names(first_stage_vars) -> Tuple[List[str], list]:
    """Return (canonical_names, ordered_var_list) from a list or dict.

    If *first_stage_vars* is a ``dict``, the dict keys are the canonical
    names (user requirement D-1).
    Otherwise we derive names via ``_get_canonical_name``.

    Returns
    -------
    names : list[str]
    vars_ordered : list — same length, matching order
    """
    if isinstance(first_stage_vars, dict):
        names = list(first_stage_vars.keys())
        vars_ordered = list(first_stage_vars.values())
    else:
        vars_ordered = list(first_stage_vars)
        names = [_get_canonical_name(v) for v in vars_ordered]
    return names, vars_ordered


# ---------------------------------------------------------------------------
# Bound extraction / validation
# ---------------------------------------------------------------------------

_BOUND_TOL = 1e-12   # tolerance for cross-scenario bound comparison


def _extract_bounds(var) -> Tuple[float, float]:
    """Return (lb, ub) from a Pyomo var-like object, raising on None."""
    lb = getattr(var, "lb", None)
    ub = getattr(var, "ub", None)
    if lb is None or ub is None:
        name = _get_canonical_name(var)
        raise ValueError(
            f"First-stage variable '{name}' has missing bounds "
            f"(lb={lb}, ub={ub}).  All first-stage variables must have "
            f"finite bounds for the simplex algorithm."
        )
    return float(lb), float(ub)


def _validate_bounds_across_scenarios(
    canonical_names: List[str],
    first_vars_list: List[list],
    override_bounds: Optional[np.ndarray],
) -> np.ndarray:
    """Return shape (d, 2) canonical bounds, validated across scenarios.

    Parameters
    ----------
    canonical_names : list[str], length d
    first_vars_list : list of lists (one per scenario), each length d
    override_bounds : (d, 2) array or None
        If provided, used directly (skips extraction from Pyomo vars).

    Returns
    -------
    np.ndarray, shape (d, 2)

    Raises
    ------
    ValueError on missing bounds or cross-scenario mismatch
    """
    d = len(canonical_names)

    if override_bounds is not None:
        ob = np.asarray(override_bounds, dtype=float)
        if ob.shape != (d, 2):
            raise ValueError(
                f"first_stage_bounds override must have shape ({d}, 2), "
                f"got {ob.shape}"
            )
        logger.info("[ProblemSetup] Using user-provided first_stage_bounds override.")
        return ob

    # Extract from scenario 0
    bounds = np.array(
        [_extract_bounds(v) for v in first_vars_list[0]], dtype=float
    )

    # Validate remaining scenarios
    for s in range(1, len(first_vars_list)):
        for i, v in enumerate(first_vars_list[s]):
            lb_s, ub_s = _extract_bounds(v)
            if (abs(lb_s - bounds[i, 0]) > _BOUND_TOL
                    or abs(ub_s - bounds[i, 1]) > _BOUND_TOL):
                raise ValueError(
                    f"Bound mismatch for variable '{canonical_names[i]}' "
                    f"between scenario 0 and scenario {s}:\n"
                    f"  Scenario 0: [{bounds[i,0]}, {bounds[i,1]}]\n"
                    f"  Scenario {s}: [{lb_s}, {ub_s}]"
                )
    return bounds


# ---------------------------------------------------------------------------
# ProblemSetup
# ---------------------------------------------------------------------------

class ProblemSetup:
    """Problem descriptor for a general two-stage stochastic program.

    Responsibility: problem-data / metadata container ONLY.
    Does NOT own mesh state, solver state, iteration logs, or EF caches.

    Attributes
    ----------
    dim : int
        Number of first-stage variables (*d*).
    n_scenarios : int
        Number of scenarios (*S*).
    scenario_names : list[str]
        Human-readable scenario identifiers.
    probabilities : np.ndarray, shape (S,)
        Scenario probabilities summing to 1.
    var_names : list[str]
        Canonical first-stage variable names (length *d*).
    var_bounds : np.ndarray, shape (d, 2)
        ``var_bounds[i] = [lb_i, ub_i]``.
    model_list : list[pyo.ConcreteModel]
        One Pyomo model per scenario.
    first_vars_list : list[list]
        ``first_vars_list[s]`` is a list of *d* Pyomo var-like references
        in canonical ``var_names`` order for scenario *s*.
    subproblems : list[ScenarioSubproblem]
        The raw ScenarioSubproblem objects (retained for future patches).
    """

    def __init__(
        self,
        *,
        dim: int,
        scenario_names: List[str],
        probabilities: np.ndarray,
        var_names: List[str],
        var_bounds: np.ndarray,
        model_list: list,
        first_vars_list: List[list],
        subproblems: List[ScenarioSubproblem],
    ):
        self.dim = dim
        self.n_scenarios = len(model_list)
        self.scenario_names = list(scenario_names)
        self.probabilities = np.asarray(probabilities, dtype=float)
        self.var_names = list(var_names)
        self.var_bounds = np.asarray(var_bounds, dtype=float)
        self.model_list = list(model_list)
        self.first_vars_list = [list(fv) for fv in first_vars_list]
        self.subproblems = list(subproblems)

    # ---- factory: from subproblem_creator ---------------------------------

    @classmethod
    def from_subproblem_creator(
        cls,
        scenario_names: Sequence[str],
        creator_fn: Callable,
        *,
        first_stage_bounds: Optional[np.ndarray] = None,
    ) -> "ProblemSetup":
        """Build a ProblemSetup by calling ``creator_fn`` per scenario.

        Parameters
        ----------
        scenario_names : sequence of str
        creator_fn : callable
            ``creator_fn(scenario_name) -> ScenarioSubproblem``
            OR
            ``creator_fn(scenario_name) -> (model, first_vars, probability)``
            (legacy tuple form, auto-wrapped into ScenarioSubproblem)
        first_stage_bounds : (d, 2) array, optional
            If given, overrides bounds extracted from Pyomo variables.
            Use when variables have None bounds but the user knows the domain.

        Returns
        -------
        ProblemSetup
        """
        subproblems: List[ScenarioSubproblem] = []
        for sname in scenario_names:
            raw = creator_fn(sname)
            if isinstance(raw, ScenarioSubproblem):
                sp = raw
                if not sp.scenario_name:
                    sp.scenario_name = str(sname)
            elif isinstance(raw, (tuple, list)):
                # Legacy return form: (model, first_vars, probability)
                if len(raw) < 3:
                    raise ValueError(
                        f"creator_fn('{sname}') returned {len(raw)}-element "
                        f"tuple; expected at least 3 (model, first_vars, prob)"
                    )
                sp = ScenarioSubproblem(
                    model=raw[0],
                    first_stage_vars=raw[1],
                    probability=float(raw[2]),
                    scenario_name=str(sname),
                )
            else:
                raise TypeError(
                    f"creator_fn('{sname}') returned {type(raw).__name__}; "
                    f"expected ScenarioSubproblem or (model, vars, prob) tuple"
                )
            # Basic probability sanity
            if not (0.0 < sp.probability <= 1.0):
                raise ValueError(
                    f"Scenario '{sname}' probability={sp.probability} "
                    f"is outside (0, 1]"
                )
            subproblems.append(sp)

        if not subproblems:
            raise ValueError("No scenarios provided.")

        # ----- Canonical variable naming (requirement D) -----
        canonical_names, vars_s0 = _extract_canonical_names(
            subproblems[0].first_stage_vars
        )
        dim = len(canonical_names)
        if dim == 0:
            raise ValueError("First-stage variable list is empty.")

        # Validate every subsequent scenario matches
        model_list = [subproblems[0].model]
        first_vars_list = [vars_s0]
        probs = [subproblems[0].probability]

        for idx in range(1, len(subproblems)):
            sp = subproblems[idx]
            names_s, vars_s = _extract_canonical_names(sp.first_stage_vars)
            if names_s != canonical_names:
                missing = set(canonical_names) - set(names_s)
                extra = set(names_s) - set(canonical_names)
                order_ok = set(names_s) == set(canonical_names)
                raise ValueError(
                    f"Scenario '{sp.scenario_name}' first-stage variable "
                    f"mismatch vs scenario 0!\n"
                    f"  Expected (canonical): {canonical_names}\n"
                    f"  Got:                  {names_s}\n"
                    f"  Missing:  {missing or 'none'}\n"
                    f"  Extra:    {extra or 'none'}\n"
                    f"  Order only: {order_ok}"
                )
            model_list.append(sp.model)
            first_vars_list.append(vars_s)
            probs.append(sp.probability)

        # ----- Probabilities -----
        probabilities = np.array(probs, dtype=float)
        psum = probabilities.sum()
        if abs(psum - 1.0) > 1e-6:
            logger.warning(
                "[ProblemSetup] Probabilities sum to %.8f (not 1.0); "
                "normalizing.", psum
            )
            probabilities /= psum

        # ----- Bounds extraction / validation (requirement C) -----
        var_bounds = _validate_bounds_across_scenarios(
            canonical_names, first_vars_list, first_stage_bounds
        )

        # ----- Log canonical setup -----
        sn_list = [sp.scenario_name for sp in subproblems]
        logger.info(
            "[ProblemSetup] dim=%d, scenarios=%d, names=%s",
            dim, len(subproblems), sn_list,
        )
        logger.info(
            "[ProblemSetup] Canonical first-stage variables: %s",
            canonical_names,
        )
        for i, name in enumerate(canonical_names):
            logger.info(
                "[ProblemSetup]   %s : bounds [%g, %g]",
                name, var_bounds[i, 0], var_bounds[i, 1],
            )

        # Also print so it's visible even without logging config
        print(f"[ProblemSetup] dim={dim}, scenarios={len(subproblems)}")
        print(f"[ProblemSetup] Canonical variable order: {canonical_names}")
        for i, name in enumerate(canonical_names):
            print(
                f"[ProblemSetup]   {name} : [{var_bounds[i,0]}, "
                f"{var_bounds[i,1]}]"
            )

        return cls(
            dim=dim,
            scenario_names=sn_list,
            probabilities=probabilities,
            var_names=canonical_names,
            var_bounds=var_bounds,
            model_list=model_list,
            first_vars_list=first_vars_list,
            subproblems=subproblems,
        )

    # ---- factory: backward-compatible PID wrapper -------------------------

    @classmethod
    def from_pid_models(
        cls,
        model_list: list,
        first_vars_list: list,
        *,
        probabilities: Optional[Sequence[float]] = None,
        scenario_names: Optional[Sequence[str]] = None,
        first_stage_bounds: Optional[np.ndarray] = None,
    ) -> "ProblemSetup":
        """Wrap existing PID model_list / first_vars_list into ProblemSetup.

        This is the zero-change integration path for existing PID notebooks.

        Parameters
        ----------
        model_list : list of pyo.ConcreteModel
        first_vars_list : list of lists of Pyomo var-like objects
        probabilities : optional, defaults to uniform 1/S
        scenario_names : optional, defaults to ["s0", "s1", …]
        first_stage_bounds : optional (d,2) override
        """
        S = len(model_list)
        if S == 0:
            raise ValueError("model_list is empty.")
        if len(first_vars_list) != S:
            raise ValueError(
                f"model_list has {S} entries but first_vars_list has "
                f"{len(first_vars_list)}"
            )
        if probabilities is None:
            probabilities = [1.0 / S] * S
        if scenario_names is None:
            scenario_names = [f"s{i}" for i in range(S)]

        # Build ScenarioSubproblem objects
        subproblems = []
        for i in range(S):
            subproblems.append(ScenarioSubproblem(
                model=model_list[i],
                first_stage_vars=list(first_vars_list[i]),
                probability=float(probabilities[i]),
                scenario_name=str(scenario_names[i]),
            ))

        # Delegate to the main factory (gets all validation + logging)
        def _creator(sname):
            idx = list(scenario_names).index(sname)
            return subproblems[idx]

        return cls.from_subproblem_creator(
            scenario_names=scenario_names,
            creator_fn=_creator,
            first_stage_bounds=first_stage_bounds,
        )

    # ---- Convenience ------

    def __repr__(self) -> str:
        return (
            f"ProblemSetup(dim={self.dim}, scenarios={self.n_scenarios}, "
            f"vars={self.var_names})"
        )
