# ef_upper_bounder.py
"""
Extensive Form (EF) Upper Bounder for the simplex-based algorithm.

At each simplex iteration, build (once) and solve an EF model restricted to
the selected simplex region using IPOPT.  The first-stage solution is then
treated as a UB candidate after true-objective evaluation.

Design mirrors SNoGloDe's SolveExtensiveForm / ExtensiveForm, adapted for
the flat scenario-model list used in this codebase.

Generalized for d-dimensional first-stage variables (d <= 10).
"""

import math
import numpy as np
from time import perf_counter

import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.opt import SolverStatus, TerminationCondition

# Suppress Pyomo model-loading warnings for infeasible loads
import logging
logging.getLogger('pyomo.core').setLevel(logging.ERROR)


class SimplexEFUb:
    """
    Builds an Extensive Form once from the scenario model list and solves
    it each iteration with updated simplex-membership constraints.

    Parameters
    ----------
    model_list : list[pyo.ConcreteModel]
        Original per-scenario Pyomo models (NOT modified).
    first_vars_list : list[list[pyo.Var]]
        First-stage variables for each scenario model (d variables per scenario).
    probabilities : Sequence[float] | None
        Scenario probabilities.  Must have length == len(model_list).
        If None, uniform weights 1/S are used.
        Validated: non-negative, sum normalised to 1 (with warning if off).
    time_ub : float
        Maximum CPU time (seconds) for the IPOPT solve.
    solver_name : str
        Pyomo SolverFactory name (default "ipopt").
    solver_opts : dict | None
        Extra solver options forwarded to IPOPT.
    """

    def __init__(self, model_list, first_vars_list, *,
                 probabilities=None,
                 time_ub: float = 60.0,
                 solver_name: str = "ipopt",
                 solver_opts: dict | None = None):

        self.S = len(model_list)
        self.time_ub = time_ub
        self.solver_name = solver_name
        self._extra_opts = solver_opts or {}

        # ---- Validate / normalize probabilities ----
        if probabilities is None:
            probs = [1.0 / self.S] * self.S
        else:
            probs = [float(p) for p in probabilities]
            if len(probs) != self.S:
                raise ValueError(
                    f"probabilities length ({len(probs)}) != "
                    f"number of scenarios ({self.S})")
            if any(p < 0 for p in probs):
                raise ValueError("probabilities must be non-negative")
            psum = sum(probs)
            if abs(psum - 1.0) > 1e-6:
                import warnings
                warnings.warn(
                    f"EF probabilities sum to {psum:.8f}, normalising to 1.0")
                probs = [p / psum for p in probs]

        # Dimension from first scenario's first-stage variables
        self._dim = len(first_vars_list[0])
        d = self._dim

        # ---- Build the EF model ----
        self.model = ef = pyo.ConcreteModel("simplex_ef")

        # Global first-stage variables — copy bounds from scenario 0
        fv0 = first_vars_list[0]
        ef.x = pyo.Var(range(d), domain=pyo.Reals)
        for i in range(d):
            ef.x[i].setlb(fv0[i].lb)
            ef.x[i].setub(fv0[i].ub)

        # Backward-compat aliases for PID (d=3)
        if d >= 1: ef.K_p = ef.x[0]
        if d >= 2: ef.K_i = ef.x[1]
        if d >= 3: ef.K_d = ef.x[2]

        # ---- Per-scenario blocks (clones of original models) ----
        # Store per-scenario first-stage variable references for warm-start
        self._scen_first_vars = {}   # s -> list[pyo.Var]  (d entries)

        ef.scenarios = pyo.Block(range(self.S))
        obj_expr = 0.0

        for s in range(self.S):
            src = model_list[s]
            blk = ef.scenarios[s]
            prob = probs[s]

            # Clone the scenario model into this block
            cloned = src.clone()
            blk.scen = cloned

            # Linking constraints: scenario first-stage == global
            blk.links = pyo.ConstraintList()
            fv_s = first_vars_list[s]
            cloned_fvars = []
            for i in range(d):
                cloned_var = cloned.find_component(fv_s[i].name)
                if cloned_var is None:
                    raise RuntimeError(
                        f"EF: can't find variable '{fv_s[i].name}' in "
                        f"cloned scenario {s}")
                blk.links.add(cloned_var == ef.x[i])
                cloned_fvars.append(cloned_var)

            # Store the canonical first-stage variable references
            self._scen_first_vars[s] = cloned_fvars

            # Accumulate probability-weighted objective
            if hasattr(cloned, 'obj_expr'):
                obj_expr += prob * cloned.obj_expr
            else:
                for obj_comp in cloned.component_objects(pyo.Objective, active=True):
                    obj_expr += prob * obj_comp.expr
                    obj_comp.deactivate()
                    break

            # Deactivate any scenario-level objectives
            for obj_comp in cloned.component_objects(pyo.Objective, active=True):
                obj_comp.deactivate()

        ef.obj = pyo.Objective(expr=obj_expr, sense=pyo.minimize)

        # ---- Simplex membership constraints (d-dimensional) ----
        n_verts = d + 1
        ef.VERTS = pyo.RangeSet(0, d)             # 0..d  (d+1 elements)
        ef.DIMS = pyo.RangeSet(0, d - 1)          # 0..d-1

        # Mutable vertex parameter matrix V[v, dim]
        ef.V = pyo.Param(ef.VERTS, ef.DIMS, default=0.0, mutable=True)

        # Barycentric weights
        ef.lam = pyo.Var(ef.VERTS, domain=pyo.NonNegativeReals, bounds=(0, 1))
        ef.lam_sum = pyo.Constraint(
            expr=sum(ef.lam[i] for i in ef.VERTS) == 1)

        # Linking x to convex combination of vertices
        ef.conv = pyo.ConstraintList()
        for dim_i in range(d):
            ef.conv.add(
                ef.x[dim_i] == sum(ef.lam[v] * ef.V[v, dim_i]
                                    for v in ef.VERTS))

        # ---- Solver ----
        self.opt = pyo.SolverFactory(solver_name)
        if solver_name == "ipopt":
            self.opt.options["max_cpu_time"] = float(time_ub)
        elif solver_name == "gurobi":
            # EXPERIMENTAL: Gurobi for nonconvex NLP
            self.opt.options["TimeLimit"] = float(time_ub)
            self.opt.options["NonConvex"] = 2        # allow non-convex QCP
            self.opt.options["MIPGap"]    = 1e-6
        for k, v in self._extra_opts.items():
            self.opt.options[k] = v

    # -----------------------------------------------------------------
    def set_warm_start(self, c_points_per_scene):
        """
        Set initial variable values for the EF model from per-scenario c_s
        solutions.

        Uses the stored canonical first-stage variable mapping
        (``self._scen_first_vars``) rather than scanning
        ``component_objects`` — guaranteeing correct variable assignment
        even when models contain extra variables.

        Parameters
        ----------
        c_points_per_scene : list of tuple[float, ...] or None
            One per scenario.  None entries are skipped.
        """
        ef = self.model
        d = self._dim

        # Collect valid c_s points for averaging
        valid_pts = [
            pt for pt in c_points_per_scene
            if pt is not None and all(math.isfinite(v) for v in pt)
        ]

        # Set global first-stage variables to the average of valid c_s points
        avg = None
        if valid_pts:
            avg = tuple(np.mean(valid_pts, axis=0))
            for i in range(d):
                ef.x[i].value = float(avg[i])

        # Set each scenario's cloned first-stage variables via stored mapping
        for s in range(self.S):
            pt = c_points_per_scene[s] if s < len(c_points_per_scene) else None
            if pt is None or not all(math.isfinite(v) for v in pt):
                if avg is not None:
                    pt = avg
                else:
                    continue

            # Use the explicit mapping built during __init__
            scen_fvars = self._scen_first_vars[s]
            for i in range(d):
                scen_fvars[i].value = float(pt[i])

    # -----------------------------------------------------------------
    def update_simplex_vertices(self, verts):
        """
        Update the mutable vertex parameters for the current simplex.

        Parameters
        ----------
        verts : list of d+1 tuples, each of length d
        """
        d = self._dim
        n_verts = d + 1
        assert len(verts) == n_verts, f"Expected {n_verts} vertices, got {len(verts)}"
        ef = self.model
        for i in range(n_verts):
            for dim_j in range(d):
                ef.V[i, dim_j] = float(verts[i][dim_j])

    # -----------------------------------------------------------------
    def solve(self):
        """
        Solve the EF restricted to the current simplex.

        Returns
        -------
        ok : bool
            True if solver status==ok and termination==optimal.
        K_ef : tuple[float, ...] | None
            First-stage solution, or None if failed.
        ef_obj : float
            EF objective value, or nan if failed.
        info : dict
            Contains solver_status, termination_condition, time_sec.
        """
        ef = self.model
        d = self._dim
        info = {
            "solver_status": None,
            "termination_condition": None,
            "time_sec": 0.0,
        }

        t0 = perf_counter()
        try:
            results = self.opt.solve(
                ef,
                timelimit=self.time_ub,
                symbolic_solver_labels=True,
                tee=False,
                load_solutions=False,
            )
            dt = perf_counter() - t0
            info["time_sec"] = dt
            info["solver_status"] = str(results.solver.status)
            info["termination_condition"] = str(results.solver.termination_condition)

            # Capture Gurobi lower bound (ObjBound) if available
            try:
                _prob = results.problem
                if hasattr(_prob, '__iter__'):
                    _prob = list(_prob)[0] if list(_prob) else None
                if _prob is not None and hasattr(_prob, 'lower_bound'):
                    info["lower_bound"] = float(_prob.lower_bound)
            except Exception:
                pass

            ok = (results.solver.status == SolverStatus.ok and
                  results.solver.termination_condition == TerminationCondition.optimal)

            if ok:
                ef.solutions.load_from(results)
                K_ef = tuple(float(pyo.value(ef.x[i])) for i in range(d))
                ef_obj = float(pyo.value(ef.obj))
                return True, K_ef, ef_obj, info
            else:
                return False, None, math.nan, info

        except Exception as exc:
            dt = perf_counter() - t0
            info["time_sec"] = dt
            info["solver_status"] = "exception"
            info["termination_condition"] = str(exc)
            return False, None, math.nan, info
