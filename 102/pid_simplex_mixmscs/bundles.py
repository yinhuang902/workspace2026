# bundles.py
import numpy as np
import math
import pyomo.environ as pyo
from pyomo.solvers.plugins.solvers.gurobi_persistent import GurobiPersistent
from pyomo.opt import SolverStatus, TerminationCondition
from time import perf_counter

# if model is infeasible, just set it to Q_max
Q_max = 1e10

class BaseBundle:
    """
    Base bundle for a single scenario: holds a Pyomo model to evaluate the
    true objective (Qs) at given first-stage values. This class does not 
    alter constraints or variable domains, it only installs the objective
    to solve efficiently.

    Parameters
    ----------
    model : pyo.ConcreteModel
        The model will contain `model.obj_expr`. If an `obj` component already 
        exists it will be removed and replaced.
    options : dict | None, optional
        Gurobi parameters to set on the persistent solver:
        - 'MIPGap' (float, default 1e-1)
        - 'NumericFocus' (int {0..3}, default 1)
        - 'Presolve' (int, default 2)
        - 'NonConvex' (int, default 2)
        - 'TimeLimit' (float, seconds)

    Attributes
    ----------
    model : pyo.ConcreteModel
        The Pyomo model.
    gp : GurobiPersistent
        The persistent solver instance bound to `model`.

    Methods
    -------
    eval_at(first_vars, first_vals) -> float
        Fixes the provided first-stage variables to `first_vals`, solves the
        model, reads the objective value at `model.obj_expr`, and then unfixes
        the variables. Returns the scalar objective value.
    """
    def __init__(self, model: pyo.ConcreteModel, options: dict | None = None):
        self.model = model
        self.gp = GurobiPersistent()
        self.gp.set_instance(model)
        if hasattr(model, 'obj'):
            model.del_component('obj')
        model.obj = pyo.Objective(expr=model.obj_expr, sense=pyo.minimize)
        self.gp.set_objective(model.obj)
        if options:
            self.gp.set_gurobi_param('MIPGap', options.get('MIPGap', 1e-1))
            self.gp.set_gurobi_param('NumericFocus', options.get('NumericFocus', 1))
            self.gp.set_gurobi_param('Presolve', options.get('Presolve', 2))
            self.gp.set_gurobi_param('NonConvex', options.get('NonConvex', 2))
            if 'TimeLimit' in options:
                self.gp.set_gurobi_param('TimeLimit', options['TimeLimit'])

    '''
    def eval_at(self, first_vars, first_vals):
        for v, val in zip(first_vars, first_vals):
            v.fix(float(val))
            self.gp.update_var(v)
        self.gp.solve(load_solutions=True,tee=True)
        val = float(pyo.value(self.model.obj_expr))
        for v in first_vars:
            v.unfix()
            self.gp.update_var(v)
        return val
    '''
    
    def eval_at(self, first_vars, first_vals):
        # first_vals is (Kp, Ki, Kd)
        K_tuple = tuple(float(v) for v in first_vals)

        try:
            for v, val in zip(first_vars, first_vals):
                v.fix(float(val))
                self.gp.update_var(v)

            # Use load_solutions=False to prevent Pyomo from raising ValueError on bad status
            res = self.gp.solve(load_solutions=False, tee=False)
            
            status = res.solver.status
            term = res.solver.termination_condition

            # Check for success
            if status == SolverStatus.ok and term in {TerminationCondition.optimal, TerminationCondition.locallyOptimal}:
                # Manually load solution
                self.model.solutions.load_from(res)
                val = float(pyo.value(self.model.obj_expr))
                return val
            else:
                # Infeasible or error
                print(f"[BaseBundle.eval_at] Infeasible/Error for K={K_tuple}: status={status}, term={term}")
                return Q_max

        except Exception as err:
            print(f"\n[BaseBundle.eval_at] Exception when solving Q_s for K={K_tuple}: {err}")
            return Q_max

        finally:
            for v in first_vars:
                v.unfix()
                self.gp.update_var(v)


class MSBundle:
    """
    Single-scenario ms subproblem: solves the single scenario simplex ms
    subproblem using a fixed-structure formulation with barycentric weights.
    We fix the form of the linking constraints. Every time we just adjust the
    coefficients to avoid rebuild.

    Problem sketch
    --------------
    - Barycentric weights `lam[j]` (j=0..3), sum to 1, lam[j] >= 0.
    - Link first-stage variables (Kp, Ki, Kd) to the convex combination of the
      4 vertices of a tetrahedron via linear constraints:
          Kp - sum(x_j * lam[j]) = 0,
          Ki - sum(y_j * lam[j]) = 0,
          Kd - sum(z_j * lam[j]) = 0.
    - Define `As` as the convex combination of per-vertex function values:
          As - sum(f_j * lam[j]) = 0.
    - Objective:
          minimize  model.obj_expr - As

    Parameters
    ----------
    model_base : pyo.ConcreteModel
        A base model that exposes:
          - first-stage variables (Kp, Ki, Kd),
          - an expression `obj_expr` to minimize.
        This model is cloned internally so the ms subproblem can alter
        constraints/objective without touching the original.
    first_vars : Sequence[pyo.Var]
        A 3-tuple/list of the first-stage variables (Kp, Ki, Kd) from
        `model_base`. Their **names** are used to locate the counterparts in
        the cloned model (via `find_component`).

    Attributes
    ----------
    model : pyo.ConcreteModel
        The cloned and augmented model (barycentric vars/constraints installed).
    gp : GurobiPersistent
        The persistent solver instance bound to `model`.
    lam : pyo.Var
        Barycentric weights (index 0..3), enforce convex combination.
    link_kp, link_ki, link_kd : pyo.Constraint
        Linking constraints tying (Kp,Ki,Kd) to the tetra vertices.
    As : pyo.Var
        Convex combination of vertex function values.
    As_def : pyo.Constraint
        Definition constraint for `As`.
    _V_cached : list[tuple[float, float, float]] | None
        Cached coordinates of the current tetrahedron vertices, used to map
        the optimal barycentric weights back to a Cartesian point.

    Methods
    -------
    update_tetra(tet_vertices, fverts_scene) -> None
        Update the coefficients of the linking constraints to reflect
        the current tetrahedron geometry and vertex function values. 
        This avoids rebuilding constraints.
    solve() -> bool
        Solve the current subproblem; returns True if the solve ended with an
        optimal or locally optimal termination condition.
    get_ms_and_point() -> tuple[float, np.ndarray, tuple[float, float, float]]
        Read the scalar ms value, the optimal barycentric weights,
        and the corresponding Cartesian point (Kp,Ki,Kd).
    """
    def __init__(self, model_base: pyo.ConcreteModel, first_vars, options: dict | None = None, scenario_index: int | None = None):
        m = model_base.clone()

        # ---- barycentric weights ----
        m.lam_index = pyo.RangeSet(0, 3)
        m.lam = pyo.Var(m.lam_index, domain=pyo.NonNegativeReals)
        m.lam_sum = pyo.Constraint(expr=sum(m.lam[j] for j in m.lam_index) == 1.0)

        # ---- locate first-stage vars in clone ----
        self.Kp = m.find_component(first_vars[0].name)
        self.Ki = m.find_component(first_vars[1].name)
        self.Kd = m.find_component(first_vars[2].name)
        if any(v is None for v in (self.Kp, self.Ki, self.Kd)):
            raise RuntimeError("Can't find (Kp, Ki, Kd) in clone model")

        # ---- mirrors (mutable Params) for logging----
        m.vx = pyo.Param(m.lam_index, mutable=True, initialize=0.0)
        m.vy = pyo.Param(m.lam_index, mutable=True, initialize=0.0)
        m.vz = pyo.Param(m.lam_index, mutable=True, initialize=0.0)
        m.fv = pyo.Param(m.lam_index, mutable=True, initialize=0.0)

        # ---- fixed structure "link constraints", firstly model them with 0 coefficients; then update them using set_linear_coefficients. ----
        # form: Kp - sum(alpha_j * lam[j]) == 0  (initial alpha_j=0)
        m.link_kp = pyo.Constraint(expr=self.Kp - sum(0.0 * m.lam[j] for j in m.lam_index) == 0.0)
        m.link_ki = pyo.Constraint(expr=self.Ki - sum(0.0 * m.lam[j] for j in m.lam_index) == 0.0)
        m.link_kd = pyo.Constraint(expr=self.Kd - sum(0.0 * m.lam[j] for j in m.lam_index) == 0.0)


        # A_s will be bounded by min/max of vertex values (set in update_tetra)
        m.As = pyo.Var(bounds=(None, None))  # Bounds updated dynamically
        m.As_def = pyo.Constraint(expr=m.As - sum(0.0 * m.lam[j] for j in m.lam_index) == 0.0)

        # ---- objectives ----
        # ms subproblem: min (Qs - As)
        if hasattr(m, 'obj'):
            m.del_component('obj')
        m.obj_ms = pyo.Objective(expr=m.obj_expr - m.As, sense=pyo.minimize)

        # Constant cut subproblem: min Qs (same feasible region + simplex constraint)
        m.obj_const = pyo.Objective(expr=m.obj_expr, sense=pyo.minimize)
        m.obj_const.deactivate()   # Initially use ms objective only

        # ---- Define Quadratic Structures Statically ----
        self._define_quadratic_structures(m)

        # ---- persistent solver ----
        self.model = m
        self.gp = GurobiPersistent()
        self.gp.set_instance(m)
        # Initially use ms objective
        self.gp.set_objective(m.obj_ms)
        
        # Default options + enforced NonConvex
        self.mip_gap = 1e-1 
        self.time_limit = None
        
        if options:
            self.mip_gap = options.get('MIPGap', 1e-1)
            self.gp.set_gurobi_param('MIPGap', self.mip_gap)
            self.gp.set_gurobi_param('NumericFocus', options.get('NumericFocus', 1))
            self.gp.set_gurobi_param('Presolve', options.get('Presolve', 2))
            # Always enable NonConvex for Quadratic support
            self.gp.set_gurobi_param('NonConvex',  2) 
            if 'TimeLimit' in options:
                self.time_limit = float(options['TimeLimit'])
                self.gp.set_gurobi_param('TimeLimit', self.time_limit)
        else:
             self.gp.set_gurobi_param('NonConvex', 2)

        self.lam = m.lam
        self.link_kp = m.link_kp
        self.link_ki = m.link_ki
        self.link_kd = m.link_kd
        self.As     = m.As
        self.As_def = m.As_def
        self.obj_ms    = m.obj_ms
        self.obj_const = m.obj_const
        self._V_cached = None
        self.solve_time_hist: list[float] = []       # MS problem times
        self.solve_const_time_hist: list[float] = []  # c_s problem times
        self.scenario_index = scenario_index  # Track which scenario this bundle is for

    def _define_quadratic_structures(self, m):
        """Define variables and constraints for quadratic interpolation (called from __init__)."""
        # Edge values: 6 edges. Ordering: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        m.edge_idx = pyo.RangeSet(0, 5)
        m.f_edge = pyo.Param(m.edge_idx, mutable=True, initialize=0.0)
        
        # Quadratic As variable
        m.As_quad = pyo.Var(bounds=(None, None))

        # Quadratic As definition:
        # As = sum( f_i * lam_i * (2*lam_i - 1) ) + sum( f_ij * 4 * lam_i * lam_j )
        # Pairs map: 0->(0,1), 1->(0,2), 2->(0,3), 3->(1,2), 4->(1,3), 5->(2,3)
        self._edge_pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]

        def as_quad_rule(m):
            term_vert = sum(m.fv[j] * m.lam[j] * (2*m.lam[j] - 1) for j in range(4))
            term_edge = sum(m.f_edge[k] * 4 * m.lam[u] * m.lam[v] 
                            for k, (u, v) in enumerate(self._edge_pairs))
            return m.As_quad == term_vert + term_edge
        
        m.As_quad_def = pyo.Constraint(rule=as_quad_rule)

        # Objective for quadratic: min (Qs - As_quad)
        m.obj_ms_quad = pyo.Objective(expr=m.obj_expr - m.As_quad, sense=pyo.minimize)
        m.obj_ms_quad.deactivate()

    # ---- Update the LAM coefficient of the "link constraint" in a single operation ----
    def _set_link_coeffs(self, con, coeffs):
        """
        Internal: set the coefficients of `lam[j]` on the LHS of a given linear
        constraint to `coeffs[j]` via direct edits to the Gurobi matrix.

        This path uses the persistent solver's internal mapping
        (`_pyomo_con_to_solver_con_map`, `_pyomo_var_to_solver_var_map`) and
        `chgCoeff` on the underlying Gurobi model to avoid relying on Pyomo
        version-specific helper APIs.

        Parameters
        ----------
        con : pyo.Constraint
            The linear constraint whose LHS lam-coefficients will be updated.
        coeffs : Sequence[float]
            Length-4 sequence specifying the new coefficients for lam[0..3].

        Raises
        ------
        AttributeError
            If the persistent solver does not expose the internal maps needed
            to access the underlying Gurobi objects.
        """
        # Obtain the mapping between the underlying Gurobi model and Pyomo→Gurobi.
        gmodel = getattr(self.gp, "_solver_model", None)
        con_map = getattr(self.gp, "_pyomo_con_to_solver_con_map", None)
        var_map = getattr(self.gp, "_pyomo_var_to_solver_var_map", None)
        if gmodel is None or con_map is None or var_map is None:
            raise AttributeError("The internal mapping of the persistent solver could not be found, and chgCoeff cannot be used.")

        grb_con = con_map[con]
        for j in range(4):
            grb_var = var_map[self.lam[j]]
            gmodel.chgCoeff(grb_con, grb_var, 0.0)
        for j in range(4):
            grb_var = var_map[self.lam[j]]
            gmodel.chgCoeff(grb_con, grb_var, float(coeffs[j]))
        gmodel.update()

    def update_tetra(self, tet_vertices, fverts_scene):
        V = [tuple(map(float, tet_vertices[j])) for j in range(4)]
        F = [float(fverts_scene[j]) for j in range(4)]
        self._V_cached = V

        vx = [V[j][0] for j in range(4)]
        vy = [V[j][1] for j in range(4)]
        vz = [V[j][2] for j in range(4)]

        # Param only for record
        for j in range(4):
            self.model.vx[j] = vx[j]
            self.model.vy[j] = vy[j]
            self.model.vz[j] = vz[j]
            self.model.fv[j] = F[j]

        # LHS: K - Σ(a_j * lam[j]) == 0  ⇒ lam coefficient is -a_j
        self._set_link_coeffs(self.link_kp, [-x for x in vx])
        self._set_link_coeffs(self.link_ki, [-y for y in vy])
        self._set_link_coeffs(self.link_kd, [-z for z in vz])
        self._set_link_coeffs(self.As_def,  [-f for f in F])
        
        # Update A_s bounds: since A_s = sum(lam_j * F_j) and sum(lam_j) = 1, lam_j >= 0
        # A_s must be in [min(F), max(F)]
        f_min = float(min(F))
        f_max = float(max(F))
        self.As.setlb(f_min)
        self.As.setub(f_max)
        self.gp.update_var(self.As)
        
        # Reset solver state to avoid MIP start pollution from previous tetrahedra
        self.gp.reset()

    def solve(self):
        """Original ms subproblem: min(Qs - As). Interface unchanged."""
        t0 = perf_counter()
        # Ensure current objective is obj_ms
        # Ensure current objective is obj_ms
        self.obj_const.deactivate()
        self.obj_ms.activate()
        self.gp.set_objective(self.obj_ms)

        # print(f"[Gurobi] Solving MS (Linear) for scenario {self.scenario_index}...")
        res = self.gp.solve(load_solutions=True, tee=True)
        dt = perf_counter() - t0
        # Here we still only record ms solve time
        self.solve_time_hist.append(dt)

        # Capture dual bound for MS problem
        try:
            lb = res.problem.lower_bound
            if lb is None:
                lb = res.problem[0].lower_bound
            
            if lb is None:
                # Check if infeasible - print debug and stop
                term = res.solver.termination_condition
                if term in {TerminationCondition.infeasible, TerminationCondition.infeasibleOrUnbounded}:
                    print(f"[Subproblem solving issue] MS Problem is {term}! Scenario: {self.scenario_index}. Exiting.")
                    import sys
                    sys.exit(1)
                else:
                    # For other cases, use obj_ms as fallback
                    self._last_ms_val = float(pyo.value(self.model.obj_ms))
            else:
                self._last_ms_val = float(lb)
        except:
            self._last_ms_val = float(pyo.value(self.model.obj_ms))

        # Check if solver has a valid bound
        # For optimal/locally optimal: status should be ok/warning
        # For time/iteration limits: status is aborted but we can still use the bound
        termination = res.solver.termination_condition
        ok = False
        
        if termination in {TerminationCondition.optimal, TerminationCondition.locallyOptimal}:
            ok = res.solver.status in {SolverStatus.ok, SolverStatus.warning}
        elif termination in {TerminationCondition.maxTimeLimit, TerminationCondition.maxIterations}:
            # Accept aborted status when hitting limits - bound may still be valid
            ok = res.solver.status in {SolverStatus.ok, SolverStatus.warning, SolverStatus.aborted}
        
        if not ok:
            print(f"[Subproblem solving issue] MS Problem not optimal. Status: {res.solver.status}, Term: {termination}. Ignoring simplex.")
        return ok


    def solve_const_cut(self):
        """
        Solve min Qs on the current simplex to get c_T,s and corresponding (Kp, Ki, Kd) point.

        Returns:
            ok      : bool, whether optimal/locallyOptimal
            c_val   : float, objective value (or -inf if failed)
            cand_pt : tuple[float,float,float] | None, corresponding (Kp,Ki,Kd)
        """
        # Switch to constant cut objective
        self.obj_ms.deactivate()
        self.obj_const.activate()
        self.gp.set_objective(self.obj_const)

        t0 = perf_counter()
        # print(f"[Gurobi] Solving Constant Cut for scenario {self.scenario_index}...")
        res = self.gp.solve(load_solutions=True, tee=True)
        dt = perf_counter() - t0
        self.solve_const_time_hist.append(dt)  # Record c_s solve time
        
        # Check if solver has a valid bound
        # For optimal/locally optimal: status should be ok/warning
        # For time/iteration limits: status is aborted but we can still use the bound
        termination = res.solver.termination_condition
        has_bound = False
        
        if termination in {TerminationCondition.optimal, TerminationCondition.locallyOptimal}:
            has_bound = res.solver.status in {SolverStatus.ok, SolverStatus.warning}
        elif termination in {TerminationCondition.maxTimeLimit, TerminationCondition.maxIterations}:
            # Accept aborted status when hitting limits - bound may still be valid
            has_bound = res.solver.status in {SolverStatus.ok, SolverStatus.warning, SolverStatus.aborted}
        
        if has_bound:
            # Use actual objective value for c_s instead of dual bound
            c_val = None
            cand_pt = None
            try:
                obj_val = float(pyo.value(self.model.obj_expr))
                dual_bound = float(res.problem.lower_bound)

                if math.isfinite(obj_val):
                    if dual_bound is not None and math.isfinite(dual_bound):
                        c_val = dual_bound
                    else:
                        c_val = -1e2

                    # Also get the candidate point
                    try:
                        kp = float(pyo.value(self.Kp))
                        ki = float(pyo.value(self.Ki))
                        kd = float(pyo.value(self.Kd))
                        cand_pt = (kp, ki, kd)
                    except Exception:
                        cand_pt = None
                else:
                    # Objective value is inf or -inf, treat as failed
                    c_val = None
                    cand_pt = None
            except Exception as e:
                # Cannot access objective value
                print(f"[Subproblem solving issue] Constant Cut (c_s) objective access error: {e}. Using -inf.")
                c_val = None
                cand_pt = None
            
            if c_val is None:
                # Objective value not available - use -inf to maintain lower bound property
                print(f"[Subproblem solving issue] Objective value unavailable for c_s, using -inf (conservative lower bound)")
                c_val = float('-inf')
                cand_pt = None
        else:
            # Solver didn't reach a valid state
            print(f"[Subproblem solving issue] Constant Cut (c_s) failed. Status: {res.solver.status}, Term: {res.solver.termination_condition}. Using -inf.")
            c_val = float('-inf')
            cand_pt = None

        # Switch back to ms objective for future use
        self.obj_const.deactivate()
        self.obj_ms.activate()
        self.gp.set_objective(self.obj_ms)

        return has_bound, c_val, cand_pt


    def get_ms_and_point(self):
        # Note: read from obj_ms, not self.model.obj
        if hasattr(self, '_last_ms_val'):
            ms_val = self._last_ms_val
        else:
            # If we solved quadratic, the active objective might be obj_ms_quad
            # but we want to return the objective value regardless of which one was active
            # However, for API consistency usually we just grab the value.
            # To be safe, we check which obj is active or just return the value of the active one.
            # But simpler: if solve() or solve_quadratic() was called, the result is in the model.
            # We'll just define a helper property or check obj_ms_quad value if obj_ms is deactivated.
            if self.model.obj_ms.active:
                ms_val = float(pyo.value(self.model.obj_ms))
            elif hasattr(self.model, 'obj_ms_quad') and self.model.obj_ms_quad.active:
                ms_val = float(pyo.value(self.model.obj_ms_quad))
            else:
                 ms_val = float(pyo.value(self.model.obj_ms))

        lam_star = np.array([pyo.value(self.lam[j]) for j in range(4)], dtype=float)
        V = np.array(self._V_cached, dtype=float)
        new_pt = lam_star @ V
        return ms_val, lam_star, tuple(map(float, new_pt))

    def solve_quadratic(self, f_edge_midpoints):
        """
        Solve the MS subproblem using Quadratic interpolation.
        
        Args:
            f_edge_midpoints: list of 6 floats, values at midpoints of edges 
                              [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
        
        Returns:
            ok (bool): True if solved successfully
        """
        # Ensure structures are defined (should be from __init__ now)
        if not hasattr(self.model, 'As_quad'):
            self._define_quadratic_structures(self.model)

        m = self.model

        # Update edge values - these are Params, so we can update them in Pyomo
        for k, val in enumerate(f_edge_midpoints):
            m.f_edge[k] = float(val)

        # Update bounds for As_quad? 
        # Quadratic range is harder to bound strictly than linear, 
        # but we can loosely bound or leave unbounded.
        # Ideally [min(all 10 valuations), max(all 10 valuations)] expanded slightly.
        # For now, we reuse the As bounds logic or just set to None to be safe.
        f_all = [pyo.value(m.fv[j]) for j in range(4)] + list(f_edge_midpoints)
        m.As_quad.setlb(min(f_all) - 1e-2)
        m.As_quad.setub(max(f_all) + 1e-2)
        self.gp.update_var(m.As_quad)
        
        # --- FIX: Rebuild the quadratic constraint to propagate Param changes to Gurobi ---
        # Persistent solvers basically compile the model once. If we change Params that are coefficients
        # in a constraint, we must tell the solver. 
        # For simple linear constraints we might use update_constraint if supported, or remove/add.
        # remove_constraint followed by add_constraint is robust.
        if hasattr(self.gp, "remove_constraint") and hasattr(self.gp, "add_constraint"):
            self.gp.remove_constraint(m.As_quad_def)
            self.gp.add_constraint(m.As_quad_def)
        else:
             # Fallback if methods missing (shouldn't happen with GurobiPersistent)
             pass

        # Switch objectives
        self.obj_ms.deactivate()
        self.obj_const.deactivate()
        m.obj_ms_quad.activate()
        self.gp.set_objective(m.obj_ms_quad)

        # Apply Temporary Solver Settings
        self.gp.set_gurobi_param('TimeLimit', 5.0)
        self.gp.set_gurobi_param('MIPGap', 0.1)

        # Solve
        t0 = perf_counter()
        print(f"[Gurobi] Solving MS (Quadratic) for scenario {self.scenario_index}...")
        res = self.gp.solve(load_solutions=True, tee=False) # Quadratic might be noisy
        dt = perf_counter() - t0
        self.solve_time_hist.append(dt) 

        # Restore Settings
        if self.time_limit is not None:
             self.gp.set_gurobi_param('TimeLimit', self.time_limit)
        else:
             # Reset to default infinite if None
             self.gp.set_gurobi_param('TimeLimit', 1e100) # Gurobi default is large
        self.gp.set_gurobi_param('MIPGap', self.mip_gap)

        # Check status
        term = res.solver.termination_condition
        ok = False
        if term in {TerminationCondition.optimal, TerminationCondition.locallyOptimal}:
            ok = res.solver.status in {SolverStatus.ok, SolverStatus.warning}
        elif term in {TerminationCondition.maxTimeLimit, TerminationCondition.maxIterations}:
            ok = res.solver.status in {SolverStatus.ok, SolverStatus.warning, SolverStatus.aborted}

        if ok:
            try:
                lb = res.problem.lower_bound
                if lb is None: lb = res.problem[0].lower_bound
                if lb is not None:
                    self._last_ms_val = float(lb)
                else:
                    self._last_ms_val = float(pyo.value(m.obj_ms_quad))
            except Exception:
                # Fallback if dual bound missing or whatever
                try:
                    self._last_ms_val = float(pyo.value(m.obj_ms_quad))
                except ValueError:
                    # Value error means no solution. Then it's actually NOT ok.
                    ok = False
        
        if not ok:
            # print(f"[Quadratic MS] Failed. Status: {res.solver.status}, Term: {term}")
            self._last_ms_val = float('inf')

        return ok
    

class SurrogateLBBundle:
    """
    Persistent solver for the surrogate LB LP:
        min_{lam in simplex} sum_s t_s
        s.t. t_s >= As_s(lam) + ms_s
             t_s >= c_s

    这里 S（场景数）在构造时固定；每次调用 compute_lb 时只更新系数，不改结构。
    """
    def __init__(self, S: int, options: dict | None = None):
        self.S = int(S)

        m = pyo.ConcreteModel(name="surrogate_lb")

        # index sets
        m.J = pyo.RangeSet(0, 3)
        m.S = pyo.RangeSet(0, self.S - 1)

        # data parameters: F[s,j], ms[s], c[s]
        m.F  = pyo.Param(m.S, m.J, mutable=True, initialize=0.0)
        m.ms = pyo.Param(m.S,        mutable=True, initialize=0.0)
        m.c  = pyo.Param(m.S,        mutable=True, initialize=0.0)

        # variables
        m.lam = pyo.Var(m.J, domain=pyo.NonNegativeReals)
        m.t   = pyo.Var(m.S)

        # sum_j lam_j = 1
        m.lam_sum = pyo.Constraint(expr=sum(m.lam[j] for j in m.J) == 1.0)

        # t_s >= As_s(lam) + ms_s
        def t_ge_aff_rule(m, s):
            return m.t[s] >= sum(m.F[s, j] * m.lam[j] for j in m.J) + m.ms[s]
        m.t_ge_aff = pyo.Constraint(m.S, rule=t_ge_aff_rule)

        # t_s >= c_s
        def t_ge_c_rule(m, s):
            return m.t[s] >= m.c[s]
        m.t_ge_c = pyo.Constraint(m.S, rule=t_ge_c_rule)

        # objective: min sum_s t_s
        m.obj = pyo.Objective(expr=sum(m.t[s] for s in m.S), sense=pyo.minimize)

        self.model = m
        self.gp = GurobiPersistent()
        self.gp.set_instance(m)

        # Set params only for this small LP, affecting neither BaseBundle nor MSBundle
        if options:
            for k, v in options.items():
                self.gp.set_gurobi_param(str(k), v)

        # To avoid external tight Cutoff affecting this,
        # set Cutoff very loose on this solver only
        self.gp.set_gurobi_param('Cutoff', 1e100)

    def _update_data(self, fverts_per_scene, ms_scene, c_scene):
        """
        Write current tetrahedron data into Param.
        fverts_per_scene: list of length S, each element is length-4 As_s at 4 vertices
        """
        import math as _math

        if len(ms_scene) != self.S or len(c_scene) != self.S or len(fverts_per_scene) != self.S:
            raise ValueError("SurrogateLBBundle: data length mismatch with S")

        m = self.model
        for s in range(self.S):
            m.ms[s] = float(ms_scene[s])

            c_val = float(c_scene[s])
            if not _math.isfinite(c_val):
                # c_s = -inf -> make constraint t_s >= c_s basically ineffective
                c_val = -1e20
            m.c[s] = c_val

            rowF = fverts_per_scene[s]
            if len(rowF) != 4:
                raise ValueError("Each fverts_per_scene[s] must have length 4")
            for j in range(4):
                m.F[s, j] = float(rowF[j])

        # After param update, flush model to GurobiPersistent (small scale, rebuild cost negligible)
        self.gp.set_instance(m)

    def compute_lb(self, fverts_per_scene, ms_scene, c_scene, fallback_LB: float) -> float:
        """
        Compute surrogate LB based on current tetrahedron data.
        If small LP fails/not optimal, return fallback_LB.
        """
        self._update_data(fverts_per_scene, ms_scene, c_scene)

        # Like MSBundle/BaseBundle, let solver load solution first
        # print("[Gurobi] Solving Surrogate LB LP...")
        res = self.gp.solve(load_solutions=True, tee=True)
        status = res.solver.status
        term   = res.solver.termination_condition
        msg    = getattr(res.solver, "message", None)

        # Special case: Cutoff/minFunctionValue situation encountered before
        if status == SolverStatus.aborted and term == TerminationCondition.minFunctionValue:
            print(f"[Subproblem solving issue] Surrogate LB aborted (MinFunctionValue). Status: {status}, Term: {term}. Using fallback LB.")
            return float(fallback_LB)

        # Other aborted: still treated as severe error (as per original requirement)
        if status == SolverStatus.aborted:
            raise RuntimeError(
                "[SurrogateLBBundle] Gurobi ABORTED.\n"
                f"  status      = {status}\n"
                f"  termination = {term}\n"
                f"  message     = {msg}"
            )

        # Non-optimal/locallyOptimal case -> warning, use fallback_LB
        ok = (status in (SolverStatus.ok, SolverStatus.warning)) and \
             (term   in (TerminationCondition.optimal,
                         TerminationCondition.locallyOptimal))

        if not ok:
            print(f"[Subproblem solving issue] Surrogate LB not optimal. Status: {status}, Term: {term}. Using fallback LB.")
            return float(fallback_LB)

        # Solve normal -> read obj directly
        val = float(pyo.value(self.model.obj))

        # Theoretically surrogate LB >= fallback_LB, use max for safety
        if val < fallback_LB:
            return float(fallback_LB)
        return val
