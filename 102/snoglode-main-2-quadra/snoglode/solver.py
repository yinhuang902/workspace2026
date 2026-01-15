"""
Wraps everything together into one solver
"""
from typing import Tuple
import traceback
from snoglode.components.parameters import SolverParameters
from snoglode.components.tree import Tree
from snoglode.components.subproblems import Subproblems
from snoglode.components.node import Node
from snoglode.bounders.upper_bounders import UpperBounder
from snoglode.utils.ef import ExtensiveForm

from snoglode.utils.solve_stats import SNoGloDeSolutionInformation
from snoglode.utils.iter_logging import IterLogger, MockIterLogger
from snoglode.utils.quadratic_bound import compute_quadratic_surrogate_bound, compute_random_pid_bound
from snoglode.utils.wls_quadratic_bound import compute_wls_quadratic_surrogate_bound
from snoglode.utils.supported import SupportedVars
from snoglode.utils.plotter_3d import WLSQ3DVisualizer

WLSQ_GLOBAL_SEED = 17

# --- Display Configuration Flags ---
ENABLE_QUAD_LB = False
ENABLE_SPEC_LB = False
ENABLE_SEPA_LB = False
ENABLE_WLSQ_LB = False
ENABLE_OBBT_UNI = False

ENABLE_WLSQ_UNIFORM = True
ENABLE_WLSQ_A = False
ENABLE_WLSQ_B = False
ENABLE_WLSQ_C = False
ENABLE_WLSQ_D1 = True
ENABLE_WLSQ_D2 = True
ENABLE_WLSQ_E = True   # NEW: Anchor-mixed sampling method

ENABLE_UB_WLSQ_UNIFORM = True
ENABLE_UB_WLSQ_A = False
ENABLE_UB_WLSQ_B = False
ENABLE_UB_WLSQ_C = False
ENABLE_UB_WLSQ_D1 = True
ENABLE_UB_WLSQ_D2 = True
ENABLE_UB_WLSQ_E = True   # NEW: UB from anchor-mixed method

# --- Summary Plot Configuration ---
# Y-axis range for the final comparison plots (modify as needed)
SUMMARY_PLOT_LB_YMIN = 0.9
SUMMARY_PLOT_LB_YMAX = 1.0
SUMMARY_PLOT_UB_YMIN = 0.9
SUMMARY_PLOT_UB_YMAX = 1.0




class PlotScraper:
    def __init__(self): pass
    def iter_update(self, lb, ub): pass


import snoglode.utils.MPI as MPI
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

import time, math
import pyomo.environ as pyo

def _compute_box_volume(state: dict) -> float:
    """
    Computes the volume of the box defined by the continuous first-stage variables.
    Volume = product of (ub - lb) for all SupportedVars.reals in the state.
    """
    vol = 1.0
    if SupportedVars.reals in state:
        for lifted_var in state[SupportedVars.reals].values():
            lb, ub = float(lifted_var.lb), float(lifted_var.ub)
            width = max(0.0, ub - lb)
            vol *= width
    return vol

class Solver():
    """
    This class executes the SNoGloDe solver.
    """
    
    def __init__(self, 
                 params: SolverParameters) -> None:
        """
        Initializes the Tree / Subproblems / UB / LB problems.

        Parameters
        ----------
        params : SolverParameters
            Class that holds all of the information the solver needs
            to set all defaults.
        """
        self._params = params

        # if we want to produce a log - only do so on rank0
        if self._params._log: 
            self.logger = IterLogger(log_name = self._params._logname,
                                     log_level = self._params._loglevel)
        else: self.logger = MockIterLogger()
        
        # init the subproblem manager
        self.logger.init_start()
        self.subproblems = Subproblems(subproblem_names = self._params._subproblem_names,
                                       subproblem_creator = self._params._subproblem_creator,
                                       subset_subproblem_names = self._params._rank_subproblem_names,
                                       use_fbbt = self._params._fbbt,
                                       use_obbt = self._params._obbt,
                                       obbt_solver_name = self._params._obbt_solver_name,
                                       obbt_solver_opts = self._params._obbt_solver_opts,
                                       relax_binaries = self._params._relax_binaries,
                                       relax_integers = self._params._relax_integers,
                                       verbose = self._params._verbose)

        # init the tree
        self.tree = Tree(params = params,
                         subproblems = self.subproblems)
        
        # Compute root box volume for reporting
        self._root_box_volume = _compute_box_volume(self.subproblems.root_node_state)

        # init bounders (lower, upper, candidate generator)
        self.lower_bounder = self._params._lower_bounder(self._params._lb_solver)
        self.upper_bounder = UpperBounder(candidate_solution_finder = self._params._candidate_solution_finder,
                                          subproblems = self.subproblems,
                                          ub_solver = self._params._ub_solver,
                                          candidate_solution_solver = self._params._cg_solver)
        self.lower_bounder.perform_fbbt = self._params._fbbt
        self.lower_bounder.inhert_solutions = self._params._inhert_solutions
        self.upper_bounder.perform_fbbt = self._params._fbbt

        # check if we are solving the ub or not
        if ((not self._params._global_guarantee) and self.upper_bounder.candidate_solution_finder.ub_required):
            print("The user specified that we did not NEED to solve for the UB problem.\n"
                  f"However, the candidate generator, {self._params._candidate_solution_finder.__name__}, requires an UB solve so it will be done anyways.")
        self.solve_ub = self._params._global_guarantee or self.upper_bounder.candidate_solution_finder.ub_required

        # make sure the user specified check_node_feasibility is a func
        self.perform_node_feasibility_check = False
        if (self._params._node_feasibility_check != None):
            self.user_specified_feasibility_check = self._params._node_feasibility_check
            self.perform_node_feasibility_check = True

        # track solution
        self.solution = SNoGloDeSolutionInformation()
        self.logger.init_stop()
    

    def solve(self, 
              max_iter: int = 100, 
              rel_tolerance: float = 1e-2,
              abs_tolerance: float = 1e-3,
              time_limit: float = math.inf, 
              collect_plot_info: bool = False):
        """
        Performs a custom prioritized spatial branch and bound algorithm.

        Parameters
        ----------
        max_iter : int
            maximum number of iterations.
        rel_tolerance : float, optional
            Tolerance level for relative gap
            Default, 1e-2
        abs_tolerance : float, optional
            Tolerance level for absolute gap
            Default, 1e-3
        time_limit: float, optional
            max time allowed for algorithm to progress.
            Default, inf
        collect_plot_info : bool
            init and collect per iteration information for plotting after.
        """
        # sets up all of the tolerances and plotters, if needed
        self.dispatch_setup(max_iter = max_iter, 
                            rel_tolerance = rel_tolerance,
                            abs_tolerance = abs_tolerance,
                            time_limit = time_limit, 
                            collect_plot_info = collect_plot_info)
        
        # while we have not met any termination conditions, proceed through tree.
        while (not self.tree.converged(self.runtime)) and (self.iteration <= max_iter):

            # grab a node & perform any preprocessing
            current_node, current_node_feasible = self.dispatch_node_selection()

            # solve LB / UB problem if node is not determined infeasible
            if current_node_feasible: self.dispatch_node_solver(current_node)
 
            # branch & bound tree according to new results
            bnb_result = self.dispatch_bnb(current_node)
            
            # update terminal, logs, plotter, etc.
            self.dispatch_updates(bnb_result, current_node)
        
        # Generate WLSQ method comparison plots at end
        if self.collect_plot_info and rank == 0:
            try:
                import matplotlib
                matplotlib.use('Agg')  # Non-interactive backend
                import matplotlib.pyplot as plt
                import numpy as np
                import os
                
                # Enabled method flags
                enabled_lb_methods = {
                    'uniform': ENABLE_WLSQ_UNIFORM,
                    'A': ENABLE_WLSQ_A,
                    'B': ENABLE_WLSQ_B,
                    'C': ENABLE_WLSQ_C,
                    'D1': ENABLE_WLSQ_D1,
                    'D2': ENABLE_WLSQ_D2,
                    'E': ENABLE_WLSQ_E
                }
                enabled_ub_methods = {
                    'uniform': ENABLE_UB_WLSQ_UNIFORM,
                    'A': ENABLE_UB_WLSQ_A,
                    'B': ENABLE_UB_WLSQ_B,
                    'C': ENABLE_UB_WLSQ_C,
                    'D1': ENABLE_UB_WLSQ_D1,
                    'D2': ENABLE_UB_WLSQ_D2,
                    'E': ENABLE_UB_WLSQ_E
                }
                
                iterations = self.wlsq_hist["iter"]
                if len(iterations) > 0:
                    # Plot 1: Lower Bounds
                    fig, ax = plt.subplots(figsize=(10, 6))
                    # Plot original SNoGloDe LB
                    ax.plot(iterations, self.wlsq_hist["original_lb"], 
                            marker='D', markersize=4, label='Original SNoGloDe', 
                            linewidth=2.5, linestyle='--', color='black', zorder=10)
                    # Plot WLSQ methods
                    for method in ['uniform', 'A', 'B', 'C', 'D1', 'D2', 'E']:
                        if enabled_lb_methods.get(method, False):
                            lb_series = self.wlsq_hist["lb"][method]
                            ax.plot(iterations, lb_series, marker='o', markersize=3, label=f'WLSQ_{method}', linewidth=2)
                    ax.set_xlabel('Iteration', fontsize=12)
                    ax.set_ylabel('Lower Bound', fontsize=12)
                    ax.set_title('Lower Bound Comparison', fontsize=14, fontweight='bold')
                    ax.set_ylim(SUMMARY_PLOT_LB_YMIN, SUMMARY_PLOT_LB_YMAX)
                    ax.legend(loc='best', fontsize=9)
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig('plots_3d/lb_compare.png', dpi=200, bbox_inches='tight')
                    plt.close(fig)
                    
                    # Plot 2: Upper Bounds
                    fig, ax = plt.subplots(figsize=(10, 6))
                    # Plot original SNoGloDe UB
                    ax.plot(iterations, self.wlsq_hist["original_ub"], 
                            marker='D', markersize=4, label='Original SNoGloDe', 
                            linewidth=2.5, linestyle='--', color='black', zorder=10)
                    # Plot WLSQ methods
                    for method in ['uniform', 'A', 'B', 'C', 'D1', 'D2', 'E']:
                        if enabled_ub_methods.get(method, False):
                            ub_series = self.wlsq_hist["ub"][method]
                            ax.plot(iterations, ub_series, marker='s', markersize=3, label=f'WLSQ_{method}', linewidth=2)
                    ax.set_xlabel('Iteration', fontsize=12)
                    ax.set_ylabel('Upper Bound', fontsize=12)
                    ax.set_title('Upper Bound Comparison', fontsize=14, fontweight='bold')
                    ax.set_ylim(SUMMARY_PLOT_UB_YMIN, SUMMARY_PLOT_UB_YMAX)
                    ax.legend(loc='best', fontsize=9)
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig('plots_3d/ub_compare.png', dpi=200, bbox_inches='tight')
                    plt.close(fig)
                    
                    print(f"[SUMMARY] Generated WLSQ comparison plots: lb_compare.png, ub_compare.png")
                else:
                    print("[SUMMARY] No iteration data to plot")
            except Exception as e:
                import traceback
                print(f"[SUMMARY PLOT ERROR] Failed to generate comparison plots: {e}")
                traceback.print_exc()
        
        # write log summary when termination conditions met
        self.logger.complete()


    def dispatch_setup(self,
                       max_iter: int, 
                       rel_tolerance: float,
                       abs_tolerance: float,
                       time_limit: float, 
                       collect_plot_info: bool) -> None:
        """
        performs any of the necessary setup for the algorithm.

        Parameters
        ----------
        max_iter : int
            maximum number of iterations.
        rel_tolerance : float
            Tolerance level for relative gap
        abs_tolerance : float
            Tolerance level for absolute gap
        time_limit: float
            max time allowed for algorithm to progress.
        collect_plot_info : bool
            init and collect per iteration information for plotting after.
        """
        assert type(max_iter)==int or type(max_iter)==float
        assert type(rel_tolerance)==int or type(rel_tolerance)==float
        assert type(abs_tolerance)==int or type(abs_tolerance)==float
        assert type(time_limit)==int or type(time_limit)==float
        self.tree.max_iter = max_iter
        self.tree.rel_tolerance = rel_tolerance
        self.tree.abs_tolerance = abs_tolerance
        self.tree.time_limit = time_limit

        assert type(collect_plot_info)==bool
        self.collect_plot_info = collect_plot_info
        if collect_plot_info: 
            self.plotter = WLSQ3DVisualizer(base_dir="plots_3d")
            
            # Initialize WLSQ history tracker (rank 0 only)
            if rank == 0:
                self.wlsq_hist = {
                    "iter": [],
                    "lb": {},
                    "ub": {},
                    "original_lb": [],
                    "original_ub": []
                }
                # Initialize series for each method
                for method in ['uniform', 'A', 'B', 'C', 'D1', 'D2', 'E']:
                    self.wlsq_hist["lb"][method] = []
                    self.wlsq_hist["ub"][method] = []

        self.runtime = 0
        self.iteration = 0

        MPI.COMM_WORLD.barrier() # ensures timing is synced
        self.start_time = time.perf_counter() # start time marker
        self.logger.alg_start(self.start_time)


    def dispatch_node_selection(self) -> Tuple[Node, bool]:
        """
        Everything involving selecting a node and pre-processing.
        Selects node and then preprocesses for feasibility.

        Returns the node and if it was deemed feasible.
        """

        # grab a node, check if it passes basic feasibility checks
        node = self.tree.get_node()
        node_feasible = self.node_feasibility_checks(node)
        if not node_feasible: return node, False

        # set state - only set second stage if we are performing bounds tightening
        # (otherwise, the bounds are never touched so they should be valid)
        bounds_tightening = (self._params._obbt or self._params._fbbt)
        self.subproblems.set_all_states(node.state,
                                        set_second_stage = bounds_tightening)
        
        # tighten bounds & sync across all existing subproblems
        # Compute pre-tightening box volume
        pre_vol = _compute_box_volume(node.state)
        pre_pct = 0.0
        if self._root_box_volume > 0:
            pre_pct = 100.0 * pre_vol / self._root_box_volume
        node.lb_problem.box_vol_pct = pre_pct

        bounds_feasible = self.subproblems.tighten_and_sync_bounds(node)

        # Compute post-tightening box volume
        if bounds_tightening:
            post_vol = _compute_box_volume(node.state)
            post_pct = 0.0
            if self._root_box_volume > 0:
                post_pct = 100.0 * post_vol / self._root_box_volume
            node.lb_problem.box_vol_pct_tight = post_pct
        else:
            node.lb_problem.box_vol_pct_tight = pre_pct
        return node, (node_feasible and bounds_feasible)

    
    def dispatch_node_solver(self, 
                             current_node: Node) -> None:
        """
        Solves the designated lowerbounding / canddiate generation / upperbounding
        problem for the give node.
        https://mpitutorial.com/tutorials/

        Parameters
        ----------
        current_node : snoglode.components.node.Node
            Current node of the BnB tree.
        """
        # current_node.display()
        # try to solve a LB problem and then an CG/UB problem
        self.dispatch_lb_solve(current_node)
        self.dispatch_ub_solve(current_node)

    
    def dispatch_bnb(self,
                     current_node: Node) -> str:
        """
        Reponsible for branching & bounding results of the tree.

        Parameters
        ----------
        current_node : snoglode.components.node.Node
            Current node of the BnB tree.
        """
        # evaluate solution, bound
        self.logger.bounding_start()
        bnb_result = self.tree.bound(current_node)
        self.logger.bounding_stop()
        
        # if we did not prune by bound / infeasibility -> branch on this node
        if bnb_result != "pruned by bound" and bnb_result != "pruned by infeasibility":
            self.logger.branching_start()
            # bnb_result += self.tree.branch(current_node, self.subproblems)
            self.tree.branch(current_node, self.subproblems)
            self.logger.branching_stop()
        bnb_result += self.tree.update_lb()
        
        # if we updated the UB -> update best solution information
        if "ub" in bnb_result:
            self.solution.update_best_solution(objective = current_node.ub_problem.objective,
                                                full_solution = self.subproblems.save_results_to_dict(),
                                                iteration = self.iteration,
                                                relative_gap = self.tree.metrics.relative_gap)
        return bnb_result
    

    def dispatch_updates(self,
                         bnb_result: str,
                         current_node: Node) -> None:
        """
        Collecting any information and printing information to the
        terminal, logger, plotter, etc.
        """
        # if we are collecting plot info
        # if self.collect_plot_info:
        #     self.plotter.iter_update(lb = self.tree.metrics.lb,
        #                                 ub = self.tree.metrics.ub)

        # update gap, iter, runtime
        self.tree.update_gap()
        self.iteration += 1
        self.runtime = time.perf_counter() - self.start_time

        # print metrics to terminal
        if (rank==0): self.display_status(bnb_result, current_node)

        # write to log
        self.logger.update()

        # Plotting
        if self.collect_plot_info and rank==0:
            try:
                self.plotter.save_iteration(iteration=self.iteration, node=current_node, subproblems=self.subproblems)
            except Exception as e:
                import traceback
                print(f"[PLOT ERROR] iter={self.iteration}: {e}")
                traceback.print_exc()
            
            # Record WLSQ history for summary plots
            import numpy as np
            self.wlsq_hist["iter"].append(self.iteration)
            
            # Record original SNoGloDe LB/UB
            orig_lb = self.tree.metrics.lb if hasattr(self.tree.metrics, 'lb') else np.nan
            orig_ub = self.tree.metrics.ub if hasattr(self.tree.metrics, 'ub') else np.nan
            self.wlsq_hist["original_lb"].append(float(orig_lb) if math.isfinite(orig_lb) else np.nan)
            self.wlsq_hist["original_ub"].append(float(orig_ub) if math.isfinite(orig_ub) else np.nan)
            
            for method in ['uniform', 'A', 'B', 'C', 'D1', 'D2', 'E']:
                # LB
                lb_val = getattr(current_node.lb_problem, f'wlsq_{method}_bound', None)
                self.wlsq_hist["lb"][method].append(float(lb_val) if lb_val is not None and math.isfinite(lb_val) else np.nan)
                # UB
                ub_val = getattr(current_node.lb_problem, f'wlsq_{method}_ub', None)
                self.wlsq_hist["ub"][method].append(float(ub_val) if ub_val is not None and math.isfinite(ub_val) else np.nan)



    def dispatch_lb_solve(self,
                          current_node: Node) -> None:
        """
        Everything related to solving the LB problem of a current node.

        Nothing is returned because all information is appended
        to the current objects in place.
        """
        # here, we solve all of the models associated with this subproblem.
        self.logger.lb_start()
        self.lower_bounder.solve(subproblems = self.subproblems,
                                 node = current_node)

        # Experimental: Compute quadratic surrogate bound
        # Gate by flag
        if ENABLE_QUAD_LB:
            try:
                # We use the solver from lower_bounder.opt (which is the Pyomo solver factory)
                q_lb, spec_lb = compute_quadratic_surrogate_bound(current_node, self.subproblems, self.lower_bounder.opt)
                current_node.lb_problem.quadratic_bound = q_lb
                current_node.lb_problem.spectral_quadratic_bound = spec_lb
            except BaseException as e:
                # If it fails, just ignore or log
                print(f"DEBUG: Quadratic bound failed: {e}")
                current_node.lb_problem.quadratic_bound = None
                current_node.lb_problem.spectral_quadratic_bound = None
        else:
            current_node.lb_problem.quadratic_bound = None
            current_node.lb_problem.spectral_quadratic_bound = None

        # Experimental: Compute Random PID bound
        # Gate by flag
        if ENABLE_SEPA_LB:
            try:
                rand_lb = compute_random_pid_bound(current_node, self.subproblems, self.lower_bounder.opt)
                current_node.lb_problem.random_quad_bound = rand_lb
            except BaseException as e:
                # If it fails, just ignore or log
                print(f"DEBUG: Random PID bound failed: {e}")
                traceback.print_exc()
                current_node.lb_problem.random_quad_bound = None
        else:
             current_node.lb_problem.random_quad_bound = None

        # Experimental: Compute WLS Quadratic Surrogate bound
        # Construct config dicts
        wlsq_methods = {
            'uniform': ENABLE_WLSQ_UNIFORM,
            'A': ENABLE_WLSQ_A,
            'B': ENABLE_WLSQ_B,
            'C': ENABLE_WLSQ_C,
            'D1': ENABLE_WLSQ_D1,
            'D2': ENABLE_WLSQ_D2,
            'E': ENABLE_WLSQ_E
        }
        wlsq_ub_methods = {
            'uniform': ENABLE_UB_WLSQ_UNIFORM,
            'A': ENABLE_UB_WLSQ_A,
            'B': ENABLE_UB_WLSQ_B,
            'C': ENABLE_UB_WLSQ_C,
            'D1': ENABLE_UB_WLSQ_D1,
            'D2': ENABLE_UB_WLSQ_D2,
            'E': ENABLE_UB_WLSQ_E
        }
        
        # Only run if AT LEAST ONE method is enabled
        # Note: ENABLE_WLSQ_LB is used for display purposes (aggregate max), 
        # but we compute if any variant is enabled.
        if any(wlsq_methods.values()):
            try:
                wlsq_lb = compute_wls_quadratic_surrogate_bound(
                    current_node, 
                    self.subproblems, 
                    self.lower_bounder.opt, 
                    seed=WLSQ_GLOBAL_SEED,
                    enabled_methods=wlsq_methods,
                    enabled_ub_methods=wlsq_ub_methods
                )
                current_node.lb_problem.wlsq_bound = wlsq_lb
            except BaseException as e:
                # If it fails, just ignore or log
                print(f"DEBUG: WLSQ bound failed: {e}")
                traceback.print_exc()
                current_node.lb_problem.wlsq_bound = None
                current_node.lb_problem.wlsq_uniform_bound = None
                current_node.lb_problem.wlsq_A_bound = None
        else:
            current_node.lb_problem.wlsq_bound = None
            current_node.lb_problem.wlsq_uniform_bound = None
            # ... others are None by default or handled by display logic check

        # Experimental: Run WLSQ Uniform OBBT Diagnostic
        if ENABLE_OBBT_UNI:
            try:
                from snoglode.utils.wls_quadratic_bound import run_surrogate_obbt_uniform
                if math.isfinite(self.tree.metrics.ub):
                    # Only run if uniform surrogate was actually computed
                    if getattr(current_node.lb_problem, 'wlsq_uniform_beta', None) is not None:
                        vol_ratio = run_surrogate_obbt_uniform(current_node, self.tree.metrics.ub)
                        current_node.lb_problem.wlsq_uniform_obbt_vol_ratio = vol_ratio
                    else:
                        current_node.lb_problem.wlsq_uniform_obbt_vol_ratio = None
                else:
                    current_node.lb_problem.wlsq_uniform_obbt_vol_ratio = None
            except BaseException as e:
                # print(f"DEBUG: WLSQ OBBT failed: {e}")
                current_node.lb_problem.wlsq_uniform_obbt_vol_ratio = None
        else:
            current_node.lb_problem.wlsq_uniform_obbt_vol_ratio = None

        # Update global UB if any WLSQ method produces a better UB
        wlsq_ub_candidates = [
            getattr(current_node.lb_problem, 'wlsq_uniform_ub', float('nan')),
            getattr(current_node.lb_problem, 'wlsq_A_ub', float('nan')),
            getattr(current_node.lb_problem, 'wlsq_B_ub', float('nan')),
            getattr(current_node.lb_problem, 'wlsq_C_ub', float('nan')),
            getattr(current_node.lb_problem, 'wlsq_D1_ub', float('nan')),
            getattr(current_node.lb_problem, 'wlsq_D2_ub', float('nan')),
            getattr(current_node.lb_problem, 'wlsq_E_ub', float('nan'))
        ]
        
        # Find the best (minimum) WLSQ UB
        best_wlsq_ub = float('inf')
        for ub_val in wlsq_ub_candidates:
            if math.isfinite(ub_val) and ub_val < best_wlsq_ub:
                best_wlsq_ub = ub_val
        
        # Update global UB if WLSQ found a better one
        if math.isfinite(best_wlsq_ub) and best_wlsq_ub < self.tree.metrics.ub:
            self.tree.metrics.ub = best_wlsq_ub





















        self.logger.lb_stop()
        
        # we need to wait for all LB problems at the other subproblems to find obj / feasibility
        MPI.COMM_WORLD.barrier()
        
        # update current node stats
        current_node.lb_problem.feasible  = \
            MPI.COMM_WORLD.allreduce(current_node.lb_problem.feasible, op=MPI.PROD)
        current_node.lb_problem.objective = \
            MPI.COMM_WORLD.allreduce(current_node.lb_problem.objective, op=MPI.SUM)
        
        # Save original objective for reporting
        current_node.lb_problem.original_objective = current_node.lb_problem.objective

        # Integrate Quadratic Bound if valid and tighter (taking MAX of all valid bounds)
        
        # Helper to safely get value or -inf
        def get_val(val):
            return val if val is not None else float('-inf')

        # Current LB candidates:
        # 1. current_node.lb_problem.objective (Original DropNonants)
        # 2. current_node.lb_problem.quadratic_bound
        # 3. current_node.lb_problem.spectral_quadratic_bound
        
        candidates = [
            current_node.lb_problem.objective,
            get_val(current_node.lb_problem.quadratic_bound),
            get_val(current_node.lb_problem.spectral_quadratic_bound)
        ]
        
        max_other = max(candidates)
        
        # 4. Random Quad LB (Only if strictly dominates)
        rand_val = get_val(current_node.lb_problem.random_quad_bound)
        
        # 5. WLSQ LB
        wlsq_val = get_val(current_node.lb_problem.wlsq_bound)
        
        # Update objective to the max of all valid bounds
        current_node.lb_problem.objective = max(max_other, rand_val, wlsq_val)
        
    
    def dispatch_ub_solve(self,
                          current_node: Node) -> None:
        """
        Everything related to solving the CG & UB problem of a current node.

        Nothing is returned because all information is appended
        to the current objects in place.
        """
        # if feasible & LB < UB -> proceed to UB problem
        if current_node.lb_problem.feasible and \
            (current_node.lb_problem.objective <= self.tree.metrics.ub):
                    
                # generate a candidate
                self.logger.cg_start()
                candidate_found, candidate_solution, candidate_objective = \
                    self.upper_bounder.candidate_solution_finder.generate(node = current_node,
                                                                        subproblems = self.subproblems)
                self.logger.cg_stop()

                # if we found a candidate & we need to solve for globally optimal subproblem specific vars
                if (candidate_found):

                    # if we solve ub, fix to candidate and solve subproblems again.
                    if (self.solve_ub):
                        self.logger.ub_start()
                        self.upper_bounder.solve(subproblems = self.subproblems,
                                                    node = current_node,
                                                    candidate_solution = candidate_solution)
                        self.logger.ub_stop()
                    
                    # otw, the candidate solution is representative of our ub feasible solution
                    else:
                        current_node.ub_problem.is_feasible(candidate_objective)
                
                # if we did not find a candidate, set to infeasible and move on
                else:
                    assert not candidate_found
                    current_node.ub_problem.is_infeasible()
                    
                # wait for all of the parallel upper bound problems to solve
                MPI.COMM_WORLD.barrier()

                # if we solved upper bound again - agregated objective & determine feasibility
                if (self.solve_ub):

                    ub_feasible = MPI.COMM_WORLD.allreduce(current_node.ub_problem.feasible, op=MPI.PROD)
                    ub_obj = MPI.COMM_WORLD.allreduce(current_node.ub_problem.objective, op=MPI.SUM)
                    
                    # update node metrics
                    if (ub_feasible): current_node.ub_problem.is_feasible(ub_obj)
                    else: current_node.ub_problem.is_infeasible()
        
        else: current_node.ub_problem.is_infeasible()


    def node_feasibility_checks(self,
                                node: Node) -> bool:
        """
        Indicates if we have a feasible node or not.

        If we have a parental LB > current UB, then
        we will prune by bound anyways, so return false.

        Also perform any user specified feasibility checks.
        
        Parameters
        ----------
        node : Node
            current node in the spatial branch and bound tree
        """
        self.logger.node_feas_start()

        # if the space's LB is already greater than the UB, prune by bound
        # just return False - this will be caught in the tree.bound call.
        if (node.lb_problem.objective > self.tree.metrics.ub):
            node.ub_problem.is_infeasible()
            return False

        # otw, check for user specified feasiblity and return
        node_feasible = True        
        if (self.perform_node_feasibility_check): 
            node_feasible = self.user_specified_feasibility_check(node)

        self.logger.node_feas_stop()

        # if the user specifies infeasible, mark and move on
        if not node_feasible:
            node.lb_problem.is_infeasible()
            node.ub_problem.is_infeasible()
            return False
    
        # otw all feasibility have passed, return True
        else: return True

    
    def get_ef(self) -> pyo.ConcreteModel:
        """
        Returns the extensive form.

        Returns
        --------
        pyo.ConcreteModel
            EF build (ie., all scenarios are build, 1 per block,
            and there is a set of nonants that link them together)
        """
        # if we have not built the EF already, do so now
        if not hasattr(self.subproblems, "ef"):
            ef = ExtensiveForm(self.subproblems)
            ef.activate()
            return ef.model
        else: 
            self.subproblems.ef.activate()
            return self.subproblems.ef.model


    def display_status(self, 
                       bnb_result: str,
                       current_node: Node) -> None:
        """
        Printing the current state of the algorithm

        Parameters
        ----------
        bnb_result : str
            string corresponding to the result of bounding
        """
        # did we prune? for what reason?
        pruned = " "
        if "pruned by bound" in bnb_result:
            pruned = "Bound"
        elif "pruned by infeasibility" in bnb_result:
            pruned = "Infeas."
        
        # were the bounds updated? 
        bound_update = " "
        if "ublb" in bnb_result:
            bound_update = "* L U"
        elif "ub" in bnb_result:
            bound_update = "* U  "
        elif "lb" in bnb_result:
            bound_update = "* L  "

        # Determine which method would be used for pruning
        method_str = "Orig"
        
        # Check if Quad or Spec dominated
        # We need to re-evaluate max logic to see which one won
        orig_obj = getattr(current_node.lb_problem, 'original_objective', current_node.lb_problem.objective)
        quad_obj = getattr(current_node.lb_problem, 'quadratic_bound', None)
        spec_obj = getattr(current_node.lb_problem, 'spectral_quadratic_bound', None)
        rand_obj = getattr(current_node.lb_problem, 'random_quad_bound', None)
        wlsq_obj = getattr(current_node.lb_problem, 'wlsq_bound', None)

        # Basic comparison (treating None as -inf)
        def val(v): return v if v is not None else float('-inf')
        
        v_orig = val(orig_obj)
        v_quad = val(quad_obj)
        v_spec = val(spec_obj)
        
        v_rand = val(rand_obj)
        v_wlsq = val(wlsq_obj)
        
        max_val = max(v_orig, v_quad, v_spec, v_rand, v_wlsq)
        
        # Prioritize labeling: WLSQ > Sepa > Spec > Quad > Orig
        # Only consider method if it is ENABLED
        if max_val == v_wlsq and v_wlsq > v_orig and v_wlsq > v_quad and v_wlsq > v_spec and v_wlsq > v_rand:
            method_str = "WLSQ"
        elif ENABLE_SEPA_LB and max_val == v_rand and v_rand > v_orig and v_rand > v_quad and v_rand > v_spec:
            method_str = "Sepa"
        elif ENABLE_SPEC_LB and max_val == v_spec and v_spec > v_orig and v_spec > v_quad:
            method_str = "Spec"
        elif ENABLE_QUAD_LB and max_val == v_quad and v_quad > v_orig:
            method_str = "Quad"

        # formatting for display
        display_node_lb = orig_obj
        node_lb_str = f"{display_node_lb:.8g}" if display_node_lb is not None else "-"
        # quad_lb_str, spec_lb_str, etc. are now handled below with flags

        main_outputs = [round(self.runtime,3),
                   self.tree.metrics.nodes.explored,
                   pruned,
                   bound_update,
                   f"{self.tree.metrics.lb:.8g}",
                   f"{self.tree.metrics.ub:.8g}",
                   f"{round(self.tree.metrics.relative_gap*100, 4)}%",
                   round(self.tree.metrics.absolute_gap, 6),
                   self.tree.n_nodes(),
                   method_str]

        # if this is first iter, print header
        header = ["Time (s)", 
                  "Nodes Explored", 
                  "Pruned by", 
                  " ", 
                  "LB", 
                  "UB", 
                  "Rel. Gap", 
                  "Abs. Gap",
                  "# Nodes",
                  "Prune Method"]
        if self.iteration == 1:
            header_print = ""
            header_line = ""
            for i, title in enumerate(header):
                header_print += f"  {title:>14}"
                if i != 0:
                    header_line += "-"*(15+len(title))
            print(header_print)
            print(header_line)

        # Print main line
        line_print = ""
        for output in main_outputs:
            line_print += f"  {output:>14}"
        print(line_print)
        
        # Helper for conditional formatting
        def fmt_method(enabled, value, fmt="{:.8g}"):
            if not enabled: return "-"
            if value is None or not math.isfinite(value): return "-"
            return fmt.format(value)

        # Print indented detail line with all lower bounds
        quad_lb_str = fmt_method(ENABLE_QUAD_LB, quad_obj)
        spec_lb_str = fmt_method(ENABLE_SPEC_LB, spec_obj)
        rand_lb_str = fmt_method(ENABLE_SEPA_LB, rand_obj)
        wlsq_lb_str = fmt_method(ENABLE_WLSQ_LB, wlsq_obj)

        detail_line = f"    LBs: Orig={node_lb_str}, Quad={quad_lb_str}, Spec={spec_lb_str}, Sepa={rand_lb_str}, WLSQ={wlsq_lb_str}"
        # Get box volume info
        box_pct = getattr(current_node.lb_problem, 'box_vol_pct', 0.0)
        tight_pct = getattr(current_node.lb_problem, 'box_vol_pct_tight', 0.0)
        obbt_ratio = getattr(current_node.lb_problem, 'wlsq_uniform_obbt_vol_ratio', None)
        obbt_str = fmt_method(ENABLE_OBBT_UNI, obbt_ratio, "{:.2%}")
        detail_line += f" | Box={box_pct:.2f}%, Tight={tight_pct:.2f}%, OBBT_uni={obbt_str}"
        print(detail_line)
        
        # Extra WLSQ details
        wlsq_uni = getattr(current_node.lb_problem, 'wlsq_uniform_bound', None)
        wlsq_A = getattr(current_node.lb_problem, 'wlsq_A_bound', None)
        wlsq_B = getattr(current_node.lb_problem, 'wlsq_B_bound', None)
        wlsq_C = getattr(current_node.lb_problem, 'wlsq_C_bound', None)
        wlsq_D1 = getattr(current_node.lb_problem, 'wlsq_D1_bound', None)
        wlsq_D2 = getattr(current_node.lb_problem, 'wlsq_D2_bound', None)
        wlsq_E = getattr(current_node.lb_problem, 'wlsq_E_bound', None)
        
        v_uni_str = fmt_method(ENABLE_WLSQ_UNIFORM, wlsq_uni)
        v_A_str = fmt_method(ENABLE_WLSQ_A, wlsq_A)
        v_B_str = fmt_method(ENABLE_WLSQ_B, wlsq_B)
        v_C_str = fmt_method(ENABLE_WLSQ_C, wlsq_C)
        v_D1_str = fmt_method(ENABLE_WLSQ_D1, wlsq_D1)
        v_D2_str = fmt_method(ENABLE_WLSQ_D2, wlsq_D2)
        v_E_str = fmt_method(ENABLE_WLSQ_E, wlsq_E)
        
        wlsq_line = f"    WLSQ_uniform={v_uni_str}, WLSQ_A={v_A_str}, WLSQ_B={v_B_str}, WLSQ_C={v_C_str}, WLSQ_D1={v_D1_str}, WLSQ_D2={v_D2_str}, WLSQ_E={v_E_str}"
        print(wlsq_line)
        
        # UB_WLSQ line
        # Retrieve UBs
        ub_uni = getattr(current_node.lb_problem, 'wlsq_uniform_ub', float('nan'))
        ub_A = getattr(current_node.lb_problem, 'wlsq_A_ub', float('nan'))
        ub_B = getattr(current_node.lb_problem, 'wlsq_B_ub', float('nan'))
        ub_C = getattr(current_node.lb_problem, 'wlsq_C_ub', float('nan'))
        ub_D1 = getattr(current_node.lb_problem, 'wlsq_D1_ub', float('nan'))
        ub_D2 = getattr(current_node.lb_problem, 'wlsq_D2_ub', float('nan'))
        ub_E = getattr(current_node.lb_problem, 'wlsq_E_ub', float('nan'))
        
        # List of all, including disabled (for iteration order)
        # Format: (name, val, enabled_flag)
        all_ubs = [
            ('uniform', ub_uni, ENABLE_UB_WLSQ_UNIFORM),
            ('A',       ub_A,   ENABLE_UB_WLSQ_A),
            ('B',       ub_B,   ENABLE_UB_WLSQ_B),
            ('C',       ub_C,   ENABLE_UB_WLSQ_C),
            ('D1',      ub_D1,  ENABLE_UB_WLSQ_D1),
            ('D2',      ub_D2,  ENABLE_UB_WLSQ_D2),
            ('E',       ub_E,   ENABLE_UB_WLSQ_E),
        ]
        
        # Find best UB (min) among implicitly enabled & valid ones
        # Users didn't ask to change logic for determining "best" star, but conventionally if it's hidden it shouldn't be starred?
        # Sticking to "OFF -> don't compute" logic implies it shouldn't be considered "best" if effectively off.
        
        best_ub = float('inf')
        for _, val, enabled in all_ubs:
            # We treat disabled as invalid for best calculation
            if enabled and math.isfinite(val) and val < best_ub:
                best_ub = val
                
        # Format parts
        parts = []
        for name, val, enabled in all_ubs:
            if not enabled:
                parts.append(f"{name}=-")
            elif not math.isfinite(val):
                parts.append(f"{name}=nan")
            else:
                diff = val - self.tree.metrics.ub
                diff_str = f"{diff:+.4f}"
                star = "*" if val == best_ub and math.isfinite(best_ub) else ""
                parts.append(f"{name}={val:.4f}({diff_str}){star}")
                
        ub_line = "    UB_WLSQ: " + " | ".join(parts)
        print(ub_line)
