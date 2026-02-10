# ef_upper_bounder.py
"""
Extensive Form (EF) Upper Bounder for the simplex-based PID algorithm.

At each simplex iteration, build (once) and solve an EF model restricted to
the selected simplex region using IPOPT.  The first-stage solution is then
treated as a UB candidate after true-objective evaluation.

Design mirrors SNoGloDe's SolveExtensiveForm / ExtensiveForm, adapted for
the flat scenario-model list used in this codebase.
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
        First-stage variables [K_p, K_i, K_d] for each scenario model.
    time_ub : float
        Maximum CPU time (seconds) for the IPOPT solve.
    solver_name : str
        Pyomo SolverFactory name (default "ipopt").
    solver_opts : dict | None
        Extra solver options forwarded to IPOPT.
    """

    def __init__(self, model_list, first_vars_list, *,
                 time_ub: float = 60.0,
                 solver_name: str = "ipopt",
                 solver_opts: dict | None = None):

        self.S = len(model_list)
        self.time_ub = time_ub
        self.solver_name = solver_name
        self._extra_opts = solver_opts or {}

        # ---- Build the EF model ----
        self.model = ef = pyo.ConcreteModel("simplex_ef")

        # Global first-stage variables — copy bounds from scenario 0
        fv0 = first_vars_list[0]   # [K_p, K_i, K_d]
        ef.K_p = pyo.Var(domain=pyo.Reals,
                         bounds=(fv0[0].lb, fv0[0].ub))
        ef.K_i = pyo.Var(domain=pyo.Reals,
                         bounds=(fv0[1].lb, fv0[1].ub))
        ef.K_d = pyo.Var(domain=pyo.Reals,
                         bounds=(fv0[2].lb, fv0[2].ub))

        # ---- Per-scenario blocks (clones of original models) ----
        ef.scenarios = pyo.Block(range(self.S))
        prob = 1.0 / self.S   # uniform probability
        obj_expr = 0.0

        for s in range(self.S):
            src = model_list[s]
            blk = ef.scenarios[s]

            # Clone the scenario model into this block
            cloned = src.clone()
            blk.scen = cloned

            # Linking constraints: scenario first-stage == global
            blk.link_Kp = pyo.Constraint(expr=cloned.K_p == ef.K_p)
            blk.link_Ki = pyo.Constraint(expr=cloned.K_i == ef.K_i)
            blk.link_Kd = pyo.Constraint(expr=cloned.K_d == ef.K_d)

            # Accumulate weighted objective
            # Use obj_expr attribute if present, else fall back to active Objective
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

        # ---- Simplex membership constraints ----
        ef.VERTS = pyo.RangeSet(0, 3)
        ef.DIMS = pyo.RangeSet(0, 2)   # Kp=0, Ki=1, Kd=2

        # Mutable vertex parameter matrix V[v, d]
        ef.V = pyo.Param(ef.VERTS, ef.DIMS, default=0.0, mutable=True)

        # Barycentric weights
        ef.lam = pyo.Var(ef.VERTS, domain=pyo.NonNegativeReals, bounds=(0, 1))
        ef.lam_sum = pyo.Constraint(
            expr=sum(ef.lam[i] for i in ef.VERTS) == 1)

        # Linking K to convex combination of vertices
        ef.conv_Kp = pyo.Constraint(
            expr=ef.K_p == sum(ef.lam[i] * ef.V[i, 0] for i in ef.VERTS))
        ef.conv_Ki = pyo.Constraint(
            expr=ef.K_i == sum(ef.lam[i] * ef.V[i, 1] for i in ef.VERTS))
        ef.conv_Kd = pyo.Constraint(
            expr=ef.K_d == sum(ef.lam[i] * ef.V[i, 2] for i in ef.VERTS))

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
        solutions, analogous to SNoGloDe loading Gurobi incumbents into
        shared model variables before the EF Ipopt solve.

        Parameters
        ----------
        c_points_per_scene : list of (Kp, Ki, Kd) or None
            One per scenario.  None entries are skipped.
        """
        ef = self.model

        # Collect valid c_s points for averaging
        valid_pts = [
            pt for pt in c_points_per_scene
            if pt is not None and all(math.isfinite(v) for v in pt)
        ]

        # Set global first-stage variables to the average of valid c_s points
        if valid_pts:
            avg = tuple(np.mean(valid_pts, axis=0))
            ef.K_p.value = float(avg[0])
            ef.K_i.value = float(avg[1])
            ef.K_d.value = float(avg[2])

        # Set each scenario's cloned first-stage variables to its own c_s
        for s in range(self.S):
            pt = c_points_per_scene[s] if s < len(c_points_per_scene) else None
            if pt is None or not all(math.isfinite(v) for v in pt):
                # Fall back to global average if this scenario has no c_s
                if valid_pts:
                    pt = avg
                else:
                    continue
            scen = ef.scenarios[s].scen
            scen.K_p.value = float(pt[0])
            scen.K_i.value = float(pt[1])
            scen.K_d.value = float(pt[2])

    # -----------------------------------------------------------------
    def update_simplex_vertices(self, verts4):
        """
        Update the mutable vertex parameters for the current simplex.

        Parameters
        ----------
        verts4 : list of 4 tuples (Kp, Ki, Kd)
        """
        assert len(verts4) == 4, f"Expected 4 vertices, got {len(verts4)}"
        ef = self.model
        for i in range(4):
            for d in range(3):
                ef.V[i, d] = float(verts4[i][d])

    # -----------------------------------------------------------------
    def solve(self):
        """
        Solve the EF restricted to the current simplex.

        Returns
        -------
        ok : bool
            True if solver status==ok and termination==optimal.
        K_ef : tuple(float, float, float) | None
            First-stage solution (Kp, Ki, Kd), or None if failed.
        ef_obj : float
            EF objective value, or nan if failed.
        info : dict
            Contains solver_status, termination_condition, time_sec.
        """
        ef = self.model
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
                K_ef = (float(pyo.value(ef.K_p)),
                        float(pyo.value(ef.K_i)),
                        float(pyo.value(ef.K_d)))
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
