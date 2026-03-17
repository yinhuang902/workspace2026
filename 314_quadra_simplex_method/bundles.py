# bundles.py
import numpy as np
import math
import pyomo.environ as pyo
from pyomo.solvers.plugins.solvers.gurobi_persistent import GurobiPersistent
from pyomo.opt import SolverStatus, TerminationCondition
from time import perf_counter

Q_max = 1e10  # default fallback; prefer passing q_max to BaseBundle()

# Set to True to log Pyomo→Gurobi persistent map sizes after each update_tetra call
DEBUG_PERSISTENT_MAPS = False

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
s
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
    def __init__(self, model: pyo.ConcreteModel, options: dict | None = None, q_max: float = 1e10):
        self.model = model
        self.q_max = q_max
        self._options = options or {}
        self.gp = GurobiPersistent()
        self.gp.set_instance(model)
        if hasattr(model, 'obj'):
            model.del_component('obj')
        model.obj = pyo.Objective(expr=model.obj_expr, sense=pyo.minimize)
        self.gp.set_objective(model.obj)
        self._apply_options()

    def _apply_options(self):
        """Apply stored solver options to the persistent solver."""
        options = self._options
        if options:
            if 'MIPGap' in options:
                self.gp.set_gurobi_param('MIPGap', options['MIPGap'])
            self.gp.set_gurobi_param('NumericFocus', options.get('NumericFocus', 1))
            self.gp.set_gurobi_param('Presolve', options.get('Presolve', 2))
            self.gp.set_gurobi_param('NonConvex', options.get('NonConvex', 2))
            if 'TimeLimit' in options:
                self.gp.set_gurobi_param('TimeLimit', options['TimeLimit'])

    def _rebuild_solver(self):
        """Fully recreate the GurobiPersistent solver to recover from C-level corruption.
        
        Called after Gurobi returns error/infeasible/unbounded status which can
        leave the internal model in a corrupted state, causing segfaults on the
        next solve() call.
        """
        try:
            del self.gp
        except Exception:
            pass
        self.gp = GurobiPersistent()
        self.gp.set_instance(self.model)
        self.gp.set_objective(self.model.obj)
        self._apply_options()

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
    
    def eval_at(self, first_vars, first_vals, return_meta=False):
        # first_vals is (Kp, Ki, Kd)
        K_tuple = tuple(float(v) for v in first_vals)
        _meta = {"status": None, "termination_condition": None, "time_sec": 0.0,
                 "obj": None, "K": K_tuple}

        try:
            for v, val in zip(first_vars, first_vals):
                v.fix(float(val))
                self.gp.update_var(v)

            from time import perf_counter as _pc
            _t0 = _pc()
            # Use load_solutions=False to prevent Pyomo from raising ValueError on bad status
            res = self.gp.solve(load_solutions=False, tee=False)
            _meta["time_sec"] = _pc() - _t0
            
            status = res.solver.status
            term = res.solver.termination_condition
            _meta["status"] = str(status)
            _meta["termination_condition"] = str(term)

            # Check for success
            if status == SolverStatus.ok and term in {TerminationCondition.optimal, TerminationCondition.locallyOptimal}:
                # Manually load solution
                self.model.solutions.load_from(res)
                val = float(pyo.value(self.model.obj_expr))
                _meta["obj"] = val
                if return_meta:
                    return val, _meta
                return val

            # --- NEW: Load Gurobi's incumbent on timeout ---
            # When Gurobi times out, the model's variable values remain stale
            # (from the prior warm-start). Loading the incumbent ensures that
            # downstream solvers (e.g. IPOPT on the EF) inherit good variable
            # values. The primal objective is still a valid feasible evaluation.
            elif term in {TerminationCondition.maxTimeLimit, TerminationCondition.maxIterations}:
                if len(res.solution) > 0:
                    self.model.solutions.load_from(res)
                    val = float(pyo.value(self.model.obj_expr))
                    print(f"[BaseBundle.eval_at] Gurobi {term} for K={K_tuple}. "
                          f"Loaded incumbent, obj={val:.6e}")
                    _meta["obj"] = val
                    if return_meta:
                        return val, _meta
                    return val
                else:
                    print(f"[BaseBundle.eval_at] Gurobi {term} for K={K_tuple}, "
                          f"no feasible incumbent found. Returning Q_max.")
                    _meta["obj"] = self.q_max
                    if return_meta:
                        return self.q_max, _meta
                    return self.q_max

            else:
                # Infeasible or error — reset Gurobi model to prevent state
                # corruption that causes segfaults on subsequent solves
                print(f"[BaseBundle.eval_at] Infeasible/Error for K={K_tuple}: status={status}, term={term}")
                self._rebuild_solver()
                _meta["obj"] = self.q_max
                if return_meta:
                    return self.q_max, _meta
                return self.q_max

        except Exception as err:
            print(f"\n[BaseBundle.eval_at] Exception when solving Q_s for K={K_tuple}: {err}")
            self._rebuild_solver()
            _meta["obj"] = self.q_max
            _meta["status"] = "exception"
            _meta["termination_condition"] = str(err)
            if return_meta:
                return self.q_max, _meta
            return self.q_max

        finally:
            try:
                for v in first_vars:
                    v.unfix()
                    self.gp.update_var(v)
            except Exception as _fin_err:
                print(f"[BaseBundle.eval_at] FINALLY block error: {_fin_err}", flush=True)


class MSBundle:
    """
    Single-scenario ms subproblem with **quadratic surrogate**.

    Solves:  min_{x in simplex T}  Q_s(x) - q_s(x)

    where q_s(x) = c + g^T x + 0.5 x^T H x  is a full symmetric quadratic
    surrogate built from interpolation data.

    Design
    ------
    - **Pyomo** owns the fixed structural skeleton: barycentric variables,
      link constraints, simplex membership, and the constant-cut objective.
    - **Gurobi** owns the quadratic ms objective, rebuilt from scratch at
      each update_tetra call via gmodel.setObjective().
    - The Q_s objective terms are captured once at __init__ as explicit
      Gurobi variable/coefficient lists and reused at every rebuild.
    """
    def __init__(self, model_base: pyo.ConcreteModel, first_vars, options: dict | None = None, scenario_index: int | None = None):
        import gurobipy as grb

        m = model_base.clone()
        first_vars = list(first_vars)
        self._dim = len(first_vars)   # d
        d = self._dim
        n_verts = d + 1               # d+1 vertices for a d-simplex

        # ---- barycentric weights (d+1 of them) ----
        m.lam_index = pyo.RangeSet(0, d)    # 0..d  (d+1 elements)
        m.lam = pyo.Var(m.lam_index, domain=pyo.NonNegativeReals)
        m.lam_sum = pyo.Constraint(expr=sum(m.lam[j] for j in m.lam_index) == 1.0)

        # ---- locate first-stage vars in clone (generic d-dim) ----
        self.first_vars_clone = []
        for i, fv in enumerate(first_vars):
            cloned_var = m.find_component(fv.name)
            if cloned_var is None:
                raise RuntimeError(
                    f"Can't find first-stage variable '{fv.name}' in cloned model "
                    f"(index {i}/{d})")
            self.first_vars_clone.append(cloned_var)
        # Backward compat aliases for PID (d=3)
        if d >= 1: self.Kp = self.first_vars_clone[0]
        if d >= 2: self.Ki = self.first_vars_clone[1]
        if d >= 3: self.Kd = self.first_vars_clone[2]

        # ---- mutable Param for vertex coords: V_coord[j, i] (logging only) ----
        m.VDIMS = pyo.RangeSet(0, d - 1)
        m.V_coord = pyo.Param(m.lam_index, m.VDIMS, mutable=True, initialize=0.0)

        # ---- link constraints: x[i] - sum(V[j,i] * lam[j]) == 0 ----
        self.link_cons = []
        for i in range(d):
            con = pyo.Constraint(
                expr=self.first_vars_clone[i] - sum(0.0 * m.lam[j] for j in m.lam_index) == 0.0)
            con_name = f"link_x{i}"
            m.add_component(con_name, con)
            self.link_cons.append(con)
        # Backward compat aliases
        if d >= 1: self.link_kp = self.link_cons[0]
        if d >= 2: self.link_ki = self.link_cons[1]
        if d >= 3: self.link_kd = self.link_cons[2]

        # ---- objectives ----
        # NOTE: No linear As variable/constraint/objective.
        # The ms objective (Q_s - q_s) is set directly in Gurobi.

        # Constant cut subproblem: min Qs (Pyomo-owned, for objective switching)
        if hasattr(m, 'obj'):
            m.del_component('obj')
        m.obj_const = pyo.Objective(expr=m.obj_expr, sense=pyo.minimize)
        m.obj_const.deactivate()

        # ---- persistent solver ----
        self.model = m
        self.gp = GurobiPersistent()
        self.gp.set_instance(m)

        # ---- Capture Q_s objective terms from Gurobi (once) ----
        # Temporarily set obj_expr as objective to extract its Gurobi representation
        m._obj_qs_extract = pyo.Objective(expr=m.obj_expr, sense=pyo.minimize)
        self.gp.set_objective(m._obj_qs_extract)
        gmodel = self.gp._solver_model
        gmodel.update()
        grb_obj = gmodel.getObjective()

        self._qs_obj_constant = 0.0
        self._qs_obj_linear_terms = []   # [(grb_var, coeff)]
        self._qs_obj_quad_terms = []     # [(grb_var1, grb_var2, coeff)]

        if isinstance(grb_obj, grb.QuadExpr):
            le = grb_obj.getLinExpr()
            self._qs_obj_constant = le.getConstant()
            for k in range(le.size()):
                self._qs_obj_linear_terms.append((le.getVar(k), le.getCoeff(k)))
            for k in range(grb_obj.size()):
                self._qs_obj_quad_terms.append(
                    (grb_obj.getVar1(k), grb_obj.getVar2(k), grb_obj.getCoeff(k)))
        elif isinstance(grb_obj, grb.LinExpr):
            self._qs_obj_constant = grb_obj.getConstant()
            for k in range(grb_obj.size()):
                self._qs_obj_linear_terms.append(
                    (grb_obj.getVar(k), grb_obj.getCoeff(k)))

        m._obj_qs_extract.deactivate()

        # Cache Gurobi variable refs for first-stage vars
        var_map = self.gp._pyomo_var_to_solver_var_map
        self._x_grb = [var_map[self.first_vars_clone[i]] for i in range(d)]

        # ---- solver params ----
        if options:
            self.mip_gap = options.get('MIPGap', 1e-1)
            self.gp.set_gurobi_param('MIPGap', self.mip_gap)
            self.gp.set_gurobi_param('NumericFocus', options.get('NumericFocus', 1))
            self.gp.set_gurobi_param('Presolve', options.get('Presolve', 2))
            self.gp.set_gurobi_param('NonConvex', options.get('NonConvex', 2))
            if 'TimeLimit' in options:
                self.gp.set_gurobi_param('TimeLimit', options['TimeLimit'])

        self.lam = m.lam
        self.obj_const = m.obj_const
        self._V_cached = None
        self._quad_coeffs = None  # (c_qs, g_vec, H_mat) or None
        self.solve_time_hist: list[float] = []
        self.solve_const_time_hist: list[float] = []
        self.scenario_index = scenario_index

        # ---- IPOPT solver for warm-starting the CS solve ----
        self._ipopt = None
        try:
            from idaes.core.solvers import get_solver
            self._ipopt = get_solver("ipopt")
        except ImportError:
            try:
                _ipopt_test = pyo.SolverFactory("ipopt")
                if _ipopt_test.available():
                    self._ipopt = _ipopt_test
            except Exception:
                pass
        if self._ipopt is None:
            print(f"[MSBundle scen {scenario_index}] IPOPT not available — CS solves will run without warm start")

    # ---- Rebuild quadratic ms objective from scratch ----
    def _set_ms_objective(self, quad_coeffs):
        """Rebuild min Q_s(x) - q_s(x) from stored Q_s terms + fresh quad_coeffs.

        q_s(x) = c_qs + g^T x + 0.5 x^T H x
        objective = Q_s(x) - q_s(x)
        """
        import gurobipy as grb

        c_qs, g_vec, H_mat = quad_coeffs
        d = self._dim
        gmodel = self.gp._solver_model

        # Build from parts: constant = Q_s_const - c_qs
        obj = grb.QuadExpr(self._qs_obj_constant - float(c_qs))

        # Add Q_s linear terms
        for gvar, coeff in self._qs_obj_linear_terms:
            obj.addTerms(coeff, gvar)

        # Add Q_s quadratic terms
        for gvar1, gvar2, coeff in self._qs_obj_quad_terms:
            obj.add(gvar1 * gvar2, coeff)

        # Subtract q_s linear terms: -g_i * x_i
        for i in range(d):
            obj.addTerms(-float(g_vec[i]), self._x_grb[i])

        # Subtract q_s quadratic terms: -0.5 * x^T H x
        for i in range(d):
            for j in range(i, d):
                h_val = float(H_mat[i, j])
                if abs(h_val) < 1e-30:
                    continue
                if i == j:
                    obj.add(self._x_grb[i] * self._x_grb[i], -0.5 * h_val)
                else:
                    # H symmetric: total coeff of x_i*x_j is -H[i,j]
                    obj.add(self._x_grb[i] * self._x_grb[j], -h_val)

        gmodel.setObjective(obj, grb.GRB.MINIMIZE)
        gmodel.update()

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
            Length-(d+1) sequence specifying the new coefficients for lam[0..d].

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
        n_verts = self._dim + 1
        for j in range(n_verts):
            grb_var = var_map[self.lam[j]]
            gmodel.chgCoeff(grb_con, grb_var, 0.0)
        for j in range(n_verts):
            grb_var = var_map[self.lam[j]]
            gmodel.chgCoeff(grb_con, grb_var, float(coeffs[j]))
        gmodel.update()

    def update_tetra(self, tet_vertices, quad_coeffs):
        """Update simplex geometry and set quadratic surrogate objective.

        Parameters
        ----------
        tet_vertices : array-like, shape (d+1, d)
        quad_coeffs  : (c_qs, g_vec, H_mat) or None.
                       If None, caller must NOT call solve().
        """
        d = self._dim
        n_verts = d + 1
        V = np.array([list(map(float, tet_vertices[j])) for j in range(n_verts)], dtype=float)
        self._V_cached = V   # shape (d+1, d)
        self._quad_coeffs = quad_coeffs

        # Update mutable Params for logging
        for j in range(n_verts):
            for i in range(d):
                self.model.V_coord[j, i] = V[j, i]

        # LHS: x[i] - sum(V[j,i] * lam[j]) == 0  =>  lam coefficient is -V[j,i]
        for i in range(d):
            self._set_link_coeffs(self.link_cons[i], [-V[j, i] for j in range(n_verts)])

        # Tighten first-stage variable bounds to simplex bounding box
        for i in range(d):
            col = V[:, i]
            self.first_vars_clone[i].setlb(float(col.min()))
            self.first_vars_clone[i].setub(float(col.max()))
            self.gp.update_var(self.first_vars_clone[i])

        # Set quadratic ms objective via Gurobi direct API
        if quad_coeffs is not None:
            self._set_ms_objective(quad_coeffs)

        # NOTE: gmodel.reset() is NOT called here.
        # setObjective() and chgCoeff() already invalidate the prior solution.
        # Gurobi automatically discards invalid warm-start incumbents.
        # Calling reset() would only destroy useful warm-start state.

    def solve(self):
        """Quadratic ms subproblem: min(Q_s - q_s). Gurobi objective was set by update_tetra.

        Returns
        -------
        dict with keys:
            bound_ok  : bool — dual bound was extracted and is finite
            point_ok  : bool — primal solution was loaded into model variables
            status    : str  — "optimal", "time_limit", "infeasible", etc.
        """
        t0 = perf_counter()
        # Ensure Pyomo obj_const is deactivated; the Gurobi objective is already
        # the quadratic ms objective set by _set_ms_objective in update_tetra.
        self.obj_const.deactivate()

        # Use load_solutions=False to prevent Pyomo from raising on bad status
        res = self.gp.solve(load_solutions=False, tee=False)
        dt = perf_counter() - t0
        self.solve_time_hist.append(dt)

        # Extract solver status/termination
        termination = res.solver.termination_condition
        solver_status = res.solver.status

        # Map termination condition to status string for logging
        def _term_to_status(term):
            if term in {TerminationCondition.optimal, TerminationCondition.locallyOptimal}:
                return "optimal"
            elif term == TerminationCondition.maxTimeLimit:
                return "time_limit"
            elif term == TerminationCondition.maxIterations:
                return "iter_limit"
            elif term in {TerminationCondition.infeasible, TerminationCondition.infeasibleOrUnbounded}:
                return "infeasible"
            elif solver_status == SolverStatus.aborted:
                return "aborted"
            elif solver_status == SolverStatus.error:
                return "error"
            else:
                return "unknown"

        status_str = _term_to_status(termination)
        used_fallback = False
        fallback_reason = None

        # ---- Determine availability BEFORE accessing any model values ----
        _bound_ok = False
        _point_ok = False

        if solver_status == SolverStatus.ok and termination in {
                TerminationCondition.optimal, TerminationCondition.locallyOptimal}:
            # Fully optimal — safe to load solution
            if len(res.solution) > 0:
                self.model.solutions.load_from(res)
                _point_ok = True
            _bound_ok = True

        elif termination in {TerminationCondition.maxTimeLimit,
                             TerminationCondition.maxIterations}:
            # Timed out — try to load incumbent if available
            if len(res.solution) > 0:
                self.model.solutions.load_from(res)
                _point_ok = True
            _bound_ok = True

        elif solver_status == SolverStatus.warning and termination in {
                TerminationCondition.optimal, TerminationCondition.locallyOptimal}:
            # Warning + optimal/locallyOptimal — accept with caution
            if len(res.solution) > 0:
                self.model.solutions.load_from(res)
                _point_ok = True
            _bound_ok = True

        else:
            # Infeasible / error / unknown — do NOT load, reset Gurobi state
            print(f"[Bundle] MS scen {self.scenario_index}: not optimal. "
                  f"Status={solver_status}, Term={termination}.")
            try:
                self.gp._solver_model.reset(1)
            except Exception:
                pass
            used_fallback = True
            fallback_reason = "infeasible_or_error"

        # ---- Extract dual bound (only when _bound_ok) ----
        _raw_dual_bound = None
        if _bound_ok:
            try:
                lb = res.problem.lower_bound
                if lb is None:
                    lb = res.problem[0].lower_bound
                _raw_dual_bound = float(lb) if lb is not None else None
            except Exception:
                _raw_dual_bound = None

        # ---- Extract primal objective for diagnostics ----
        _raw_primal_obj = None
        if _point_ok:
            try:
                gmodel = self.gp._solver_model
                _raw_primal_obj = float(gmodel.ObjVal)
            except Exception:
                _raw_primal_obj = None

        # ---- Set ms value: dual bound → _last_ms_val ----
        if _raw_dual_bound is not None and math.isfinite(_raw_dual_bound):
            self._last_ms_val = _raw_dual_bound
        else:
            self._last_ms_val = float('inf')  # +inf sentinel
            if _bound_ok:
                used_fallback = True
                fallback_reason = "no_or_nan_dual_bound"

        # Override for infeasible
        if termination in {TerminationCondition.infeasible,
                           TerminationCondition.infeasibleOrUnbounded}:
            print(f"[Bundle] MS scen {self.scenario_index}: {termination}. ms_val=+inf.")
            self._last_ms_val = float('inf')
            _bound_ok = False
            _point_ok = False
            used_fallback = True
            fallback_reason = "infeasible_or_unbounded"

        # Store availability flags for get_ms_and_point
        self._last_solve_bound_ok = _bound_ok and math.isfinite(self._last_ms_val)
        self._last_solve_point_ok = _point_ok

        # Compute dual_gt_primal diagnostic flag (logging only)
        _dual_gt_primal = False
        if (_raw_dual_bound is not None and _raw_primal_obj is not None
                and math.isfinite(_raw_dual_bound) and math.isfinite(_raw_primal_obj)):
            _dual_gt_primal = (_raw_dual_bound > _raw_primal_obj + 1e-8)
            if _dual_gt_primal:
                print(f"[Invariant] MS scen {self.scenario_index}: "
                      f"dual={_raw_dual_bound:.6e} > primal={_raw_primal_obj:.6e}")

        # Store metadata for logging
        self.last_solve_meta = {
            "status": status_str,
            "termination_condition": str(termination) if termination else "None",
            "solver_status": str(solver_status) if solver_status else "None",
            "used_fallback": used_fallback,
            "fallback_reason": fallback_reason,
            "time_sec": dt,
            "ok": self._last_solve_bound_ok,
            "bound_ok": self._last_solve_bound_ok,
            "point_ok": self._last_solve_point_ok,
            "dual_bound": _raw_dual_bound,
            "primal_obj": _raw_primal_obj,
            "dual_gt_primal": _dual_gt_primal,
        }

        return {
            "bound_ok": self._last_solve_bound_ok,
            "point_ok": self._last_solve_point_ok,
            "status": status_str,
        }


    def solve_const_cut(self):
        """Solve min Qs on the current simplex to get c_T,s and the minimizer point.

        Returns
        -------
        dict with keys:
            bound_ok  : bool — dual bound was extracted and is finite
            point_ok  : bool — primal solution was loaded, c_pt is valid
            c_val     : float — dual bound value (-inf if unavailable)
            c_pt      : tuple | None — first-stage minimizer point
            status    : str  — "optimal", "time_limit", "infeasible", etc.
        """
        # Switch to constant-cut objective (Pyomo-owned)
        self.obj_const.activate()
        self.gp.set_objective(self.obj_const)

        # ---- IPOPT warm-start (same as SNoGloDe) ----
        if self._ipopt is not None:
            try:
                self._ipopt.solve(self.model, load_solutions=True, tee=False)
            except Exception:
                pass  # Continue without warm start if IPOPT fails

        try:
            t0 = perf_counter()
            # Use load_solutions=False — same discipline as solve()
            res = self.gp.solve(load_solutions=False, tee=False, warmstart=True)
            dt = perf_counter() - t0
            self.solve_const_time_hist.append(dt)

            # Extract solver status/termination
            termination = res.solver.termination_condition
            solver_status = res.solver.status

            # Map termination condition to status string
            def _term_to_status(term):
                if term in {TerminationCondition.optimal, TerminationCondition.locallyOptimal}:
                    return "optimal"
                elif term == TerminationCondition.maxTimeLimit:
                    return "time_limit"
                elif term == TerminationCondition.maxIterations:
                    return "iter_limit"
                elif term in {TerminationCondition.infeasible, TerminationCondition.infeasibleOrUnbounded}:
                    return "infeasible"
                elif solver_status == SolverStatus.aborted:
                    return "aborted"
                elif solver_status == SolverStatus.error:
                    return "error"
                else:
                    return "unknown"

            status_str = _term_to_status(termination)
            used_fallback = False
            fallback_reason = None

            # ---- Determine availability BEFORE accessing model values ----
            _bound_ok = False
            _point_ok = False

            if solver_status == SolverStatus.ok and termination in {
                    TerminationCondition.optimal, TerminationCondition.locallyOptimal}:
                if len(res.solution) > 0:
                    self.model.solutions.load_from(res)
                    _point_ok = True
                _bound_ok = True

            elif termination in {TerminationCondition.maxTimeLimit,
                                 TerminationCondition.maxIterations}:
                if len(res.solution) > 0:
                    self.model.solutions.load_from(res)
                    _point_ok = True
                _bound_ok = True

            elif solver_status == SolverStatus.warning and termination in {
                    TerminationCondition.optimal, TerminationCondition.locallyOptimal}:
                if len(res.solution) > 0:
                    self.model.solutions.load_from(res)
                    _point_ok = True
                _bound_ok = True

            else:
                print(f"[Bundle] CS scen {self.scenario_index}: failed. "
                      f"Status={solver_status}, Term={termination}.")
                try:
                    self.gp._solver_model.reset(1)
                except Exception:
                    pass
                used_fallback = True
                fallback_reason = "infeasible_or_error"

            # ---- Extract c_val (dual bound) — only when _bound_ok ----
            c_val = float('-inf')
            if _bound_ok:
                try:
                    dual_bound = float(res.problem.lower_bound)
                    if math.isfinite(dual_bound):
                        c_val = dual_bound
                    else:
                        used_fallback = True
                        fallback_reason = "no_or_nan_dual_bound"
                except Exception:
                    used_fallback = True
                    fallback_reason = "no_or_nan_dual_bound"

            # ---- Extract candidate point — only when _point_ok ----
            c_pt = None
            if _point_ok:
                try:
                    c_pt = tuple(float(pyo.value(v)) for v in self.first_vars_clone)
                except Exception:
                    c_pt = None
                    _point_ok = False

            # ---- Capture raw dual/primal for diagnostics ----
            _cs_dual = None
            _cs_primal = None
            if _bound_ok:
                try:
                    _cs_dual = float(res.problem.lower_bound) if res.problem.lower_bound is not None else None
                except Exception:
                    pass
            if _point_ok:
                try:
                    _cs_primal = float(pyo.value(self.model.obj_expr))
                except Exception:
                    pass

            # Compute dual_gt_primal diagnostic flag (logging only)
            _dual_gt_primal = False
            if (_cs_dual is not None and _cs_primal is not None
                    and math.isfinite(_cs_dual) and math.isfinite(_cs_primal)):
                _dual_gt_primal = (_cs_dual > _cs_primal + 1e-8)
                if _dual_gt_primal:
                    print(f"[Invariant] CS scen {self.scenario_index}: "
                          f"dual={_cs_dual:.6e} > primal={_cs_primal:.6e}")

            # Store metadata for logging
            self.last_cs_meta = {
                "status": status_str,
                "termination_condition": str(termination) if termination else "None",
                "solver_status": str(solver_status) if solver_status else "None",
                "used_fallback": used_fallback,
                "fallback_reason": fallback_reason,
                "time_sec": dt,
                "ok": _bound_ok and math.isfinite(c_val),
                "bound_ok": _bound_ok and math.isfinite(c_val),
                "point_ok": _point_ok,
                "dual_bound": _cs_dual,
                "primal_obj": _cs_primal,
                "dual_gt_primal": _dual_gt_primal,
            }

            return {
                "bound_ok": _bound_ok and math.isfinite(c_val),
                "point_ok": _point_ok,
                "c_val": c_val,
                "c_pt": c_pt,
                "status": status_str,
            }

        finally:
            # ---- Restore post-state invariant ----
            # Deactivate Pyomo obj_const; rebuild Gurobi quadratic ms objective
            self.obj_const.deactivate()
            if self._quad_coeffs is not None:
                self._set_ms_objective(self._quad_coeffs)
            # If _quad_coeffs is None, the Gurobi objective is stale but caller
            # must call update_tetra(verts, quad_coeffs) next, which rebuilds it.


    def get_ms_and_point(self):
        """Read ms value and optimal point after a successful solve().

        Returns
        -------
        ms_val  : float — ms bound (+inf if unavailable)
        lam_star: np.ndarray or None — barycentric weights (None if point unavailable)
        new_pt  : tuple or None — first-stage point (None if point unavailable)
        """
        ms_val = getattr(self, '_last_ms_val', float('inf'))

        # Only read variable values if a primal solution was loaded
        if not getattr(self, '_last_solve_point_ok', False):
            return ms_val, None, None

        n_verts = self._dim + 1
        try:
            lam_star = np.array([pyo.value(self.lam[j]) for j in range(n_verts)], dtype=float)
            V = np.array(self._V_cached, dtype=float)  # (d+1, d)
            new_pt = lam_star @ V                       # (d,)
            return ms_val, lam_star, tuple(map(float, new_pt))
        except Exception:
            return ms_val, None, None
    

class SurrogateLBBundle:
    """
    Surrogate LB model with **quadratic surrogates**.

    For each scenario s:
      - If quad-available:  t_s >= q_s(V*lam) + ms_s    (quadratic in lam)
      - Always:             t_s >= c_s

    where q_s(V*lam) = c_qs + a_s^T lam + 0.5 lam^T P_s lam,
    with a_s = V^T g_s, P_s = V^T H_s V.

    This is potentially nonconvex. Rebuilt from scratch at each compute_lb call.
    """
    def __init__(self, S: int, n_verts: int = 4, options: dict | None = None):
        self.S = int(S)
        self._n_verts = int(n_verts)
        self._options = options or {}

    def compute_lb(self, tet_vertices, quad_coeffs_per_scene, ms_scene,
                   c_scene, quad_available, fverts_per_scene) -> float:
        """Compute surrogate LB for one simplex.

        Parameters
        ----------
        tet_vertices       : (d+1, d) simplex vertices
        quad_coeffs_per_scene : list of (c_qs, g, H) or None, length S
        ms_scene           : list of float (meaningful only where quad_available)
        c_scene            : list of float
        quad_available     : list of bool, length S
        fverts_per_scene   : list of list of float (vertex Q vals, for fallback)

        Returns
        -------
        float : the surrogate lower bound
        """
        import math as _math
        from quadratic_surrogate import eval_quad_at

        V = np.asarray(tet_vertices, dtype=float)
        n_verts = self._n_verts
        d = n_verts - 1
        S = self.S

        # ---- Try QCP solve ----
        try:
            lb_qcp = self._solve_qcp(V, d, quad_coeffs_per_scene, ms_scene,
                                      c_scene, quad_available)
            if lb_qcp is not None and _math.isfinite(lb_qcp):
                # Also compute fallback for safety max
                lb_fb = self._compute_fallback(V, d, quad_coeffs_per_scene,
                                                ms_scene, c_scene,
                                                quad_available, fverts_per_scene)
                return max(lb_qcp, lb_fb)
        except Exception as e:
            print(f"[SurrogateLBBundle] QCP solve failed: {e}")

        # ---- Fallback ----
        return self._compute_fallback(V, d, quad_coeffs_per_scene, ms_scene,
                                       c_scene, quad_available, fverts_per_scene)

    def _solve_qcp(self, V, d, quad_coeffs_per_scene, ms_scene, c_scene,
                   quad_available):
        """Build and solve the quadratic surrogate LB model. Returns LB or None."""
        import gurobipy as grb

        n_verts = d + 1
        S = self.S

        gm = grb.Model("surrogate_lb_qcp")
        gm.setParam("OutputFlag", 0)
        gm.setParam("NonConvex", 2)
        gm.setParam("Cutoff", 1e100)
        for k, v in self._options.items():
            gm.setParam(str(k), v)

        # Barycentric variables
        lam = [gm.addVar(lb=0.0, name=f"lam_{j}") for j in range(n_verts)]
        # Simplex constraint
        gm.addConstr(grb.quicksum(lam) == 1.0, name="lam_sum")

        # Per-scenario slack variables
        t = [gm.addVar(lb=-grb.GRB.INFINITY, name=f"t_{s}") for s in range(S)]

        gm.update()

        for s in range(S):
            # Always: t_s >= c_s
            c_val = float(c_scene[s])
            if math.isfinite(c_val):
                gm.addConstr(t[s] >= c_val, name=f"t_ge_c_{s}")

            # If quad-available: t_s >= q_s(V*lam) + ms_s
            if quad_available[s] and quad_coeffs_per_scene[s] is not None:
                c_qs, g_vec, H_mat = quad_coeffs_per_scene[s]
                ms_val = float(ms_scene[s])
                g_vec = np.asarray(g_vec, dtype=float)
                H_mat = np.asarray(H_mat, dtype=float)

                # a_s = V^T @ g  (shape n_verts)
                a_s = V @ g_vec  # (n_verts,)
                # P_s = V^T @ H @ V  (shape n_verts x n_verts)
                P_s = V @ H_mat @ V.T  # (n_verts, n_verts)

                # q_s(V*lam) = c_qs + a_s^T lam + 0.5 lam^T P_s lam
                expr = grb.QuadExpr(float(c_qs) + ms_val)
                for j in range(n_verts):
                    expr.addTerms(float(a_s[j]), lam[j])
                for i in range(n_verts):
                    for j in range(i, n_verts):
                        p_val = float(P_s[i, j])
                        if abs(p_val) < 1e-30:
                            continue
                        if i == j:
                            expr.add(lam[i] * lam[i], 0.5 * p_val)
                        else:
                            # P_s symmetric: P[i,j] appears as coeff of lam_i*lam_j
                            expr.add(lam[i] * lam[j], p_val)

                gm.addQConstr(t[s] >= expr, name=f"t_ge_quad_{s}")

        # Objective: min sum t_s
        gm.setObjective(grb.quicksum(t), grb.GRB.MINIMIZE)
        gm.optimize()

        if gm.Status in (grb.GRB.OPTIMAL, grb.GRB.SUBOPTIMAL):
            return float(gm.ObjVal)
        return None

    def _compute_fallback(self, V, d, quad_coeffs_per_scene, ms_scene,
                          c_scene, quad_available, fverts_per_scene):
        """Auditable two-candidate fallback LB."""
        import math as _math
        from quadratic_surrogate import eval_quad_at

        n_verts = d + 1
        S = self.S

        # ---- Candidate 1: vertex-surrogate fallback ----
        valid_vertex_lbs = []
        for j in range(n_verts):
            vertex_lb_j = 0.0
            vertex_valid = True
            for s in range(S):
                if quad_available[s] and quad_coeffs_per_scene[s] is not None:
                    c_qs, g_vec, H_mat = quad_coeffs_per_scene[s]
                    qs_at_vj = eval_quad_at(c_qs, g_vec, H_mat, V[j])
                    vertex_lb_j += qs_at_vj + float(ms_scene[s])
                elif _math.isfinite(float(c_scene[s])):
                    vertex_lb_j += float(c_scene[s])
                else:
                    vertex_valid = False
                    break
            if vertex_valid:
                valid_vertex_lbs.append(vertex_lb_j)

        LB_vertex = min(valid_vertex_lbs) if valid_vertex_lbs else float('-inf')

        # ---- Candidate 2: pure constant-cut fallback ----
        finite_cs = [float(c_scene[s]) for s in range(S) if _math.isfinite(float(c_scene[s]))]
        LB_const = sum(finite_cs) if finite_cs else float('-inf')

        # ---- Final ----
        LB_fallback = max(LB_vertex, LB_const)
        if LB_fallback == float('-inf'):
            LB_fallback = float('inf')  # unusable simplex
        return LB_fallback
