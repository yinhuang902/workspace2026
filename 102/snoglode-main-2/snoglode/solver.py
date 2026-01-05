"""
Wraps everything together into one solver
"""
from typing import Tuple
from snoglode.components.parameters import SolverParameters
from snoglode.components.tree import Tree
from snoglode.components.subproblems import Subproblems
from snoglode.components.node import Node
from snoglode.bounders.upper_bounders import UpperBounder
from snoglode.utils.ef import ExtensiveForm

from .utils.plotter import PlotScraper

from snoglode.utils.solve_stats import SNoGloDeSolutionInformation
from snoglode.utils.logging import IterLogger, MockIterLogger

import snoglode.utils.MPI as MPI
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

import time, math
import pyomo.environ as pyo

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
        -----------
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
        -----------
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
        if collect_plot_info: self.plotter = PlotScraper()

        self.runtime = 0
        self.iteration = 0

        # --- QUADRATIC SURROGATE SETUP ---
        if self._params._use_quadratic_surrogate:
            from snoglode.utils.quadratic_surrogate import QuadraticSurrogate
            from snoglode.utils.supported import SupportedVars
            
            # Count lifted reals
            self.lifted_reals_ids = sorted(self.subproblems.root_node_state[SupportedVars.reals].keys())
            dim = len(self.lifted_reals_ids)
            
            if rank == 0:
                print(f"Initialized Quadratic Surrogate with dim={dim}")

            # Only rank 0 maintains the "master" surrogate for fitting
            # But all ranks need an instance to receive broadcasts and evaluate
            self.q_surrogate = QuadraticSurrogate(dim=dim, ridge=self._params._quad_ridge)
            self.q_surrogate.var_ids = self.lifted_reals_ids # Store IDs for mapping creation later
            
            # WARM STARTING
            # 8 Corners + Center + 1 Seeded Random
            # Generate points on Rank 0
            if rank == 0: print("Start Warm Start Generation")
            warm_start_points = []
            if rank == 0:
                try:
                    # 1. Get Global Bounds
                    print("Accessing root node state...")
                    root_reals = self.subproblems.root_node_state[SupportedVars.reals]
                    lbs = []
                    ubs = []
                    for vid in self.lifted_reals_ids:
                        lbs.append(root_reals[vid].lb)
                        ubs.append(root_reals[vid].ub)
                    
                    import itertools
                    import numpy as np
                    
                    print(f"Bounds gathered. LBs: {lbs}")

                    # Corners
                    corners = list(itertools.product(*zip(lbs, ubs)))
                    if len(corners) > 256: 
                        corners = corners[:256]
                    warm_start_points.extend([list(c) for c in corners])
                    
                    # Center
                    center = [(l+u)/2.0 for l, u in zip(lbs, ubs)]
                    warm_start_points.append(center)
                    
                    # Random Point
                    print("Generating random point...")
                    best_rand = None
                    max_min_dist = -1.0
                    if len(warm_start_points) > 0:
                        existing = np.array(warm_start_points)
                        for _ in range(10):
                            cand = [np.random.uniform(l, u) for l, u in zip(lbs, ubs)]
                            dists = np.linalg.norm(existing - np.array(cand), axis=1)
                            min_d = np.min(dists)
                            if min_d > max_min_dist:
                                max_min_dist = min_d
                                best_rand = cand
                    else:
                        best_rand = [np.random.uniform(l, u) for l, u in zip(lbs, ubs)]
                    
                    if best_rand is not None:
                        warm_start_points.append(best_rand)
                    
                    print(f"Generated {len(warm_start_points)} warm start points.")
                except Exception as e:
                    print(f"Error in Warm Start Generation: {e}")
                    import traceback
                    traceback.print_exc()

            # Broadcast points to all ranks to evaluate
            if rank == 0: print("Broadcasting warm start points...")
            warm_start_points = MPI.COMM_WORLD.bcast(warm_start_points, root=0)
            if rank == 0: print("Broadcast complete.")
            
            # Evaluate points
            # Use root node as context
            root_node = self.tree.root
            self.subproblems.set_all_states(root_node.state)
            
            idx_pt = 0
            for pt in warm_start_points:
                idx_pt += 1
                if rank == 0: print(f"Evaluating point {idx_pt}/{len(warm_start_points)}...")
                # Fix variables
                fix_dict = {}
                for idx, vid in enumerate(self.lifted_reals_ids):
                    fix_dict[vid] = pt[idx]
                
                # Use UpperBounder checks/logic
                # It expects a flat dict {var_id: val}
                
                # Solve using UpperBounder mechanism locally
                try:
                    local_feasible = self.upper_bounder.solve(root_node, self.subproblems, fix_dict)
                except Exception as e:
                    print(f"Error in upper_bounder.solve at rank {rank}: {e}")
                    import traceback
                    traceback.print_exc()
                    local_feasible = False
                
                local_obj = root_node.ub_problem.objective # This is local sum
                
                # Aggregate
                feasible = MPI.COMM_WORLD.allreduce(local_feasible, op=MPI.PROD)
                # If feasible, sum objective
                obj = 0.0
                if feasible:
                    obj = MPI.COMM_WORLD.allreduce(local_obj, op=MPI.SUM)
                    # We could update global UB here but let's keep it simple for surrogate only
                else:
                    obj = 1e8 # Infeasible penalty
                
                if rank == 0:
                    self.q_surrogate.add_point(pt, obj)

            # Fit initial model
            global_m = 0.0
            check_fitted_pack = None
            if rank == 0:
                print("Fitting initial surrogate...")
                try:
                    self.q_surrogate.fit()
                    print(f"Surrogate fitted. Fitted={self.q_surrogate.fitted}")
                    if self.q_surrogate.fitted:
                        check_fitted_pack = (self.q_surrogate.fitted, self.q_surrogate.c, self.q_surrogate.b, self.q_surrogate.H)
                    else:
                        print("Surrogate not fitted (fit() returned False or failed).")
                        check_fitted_pack = (False, None, None, None)
                except Exception as e:
                    print(f"Error during surrogate fitting: {e}")
                    check_fitted_pack = (False, None, None, None)
            
            print(f"Rank {rank} waiting for broadcast...")
            check_fitted_pack = MPI.COMM_WORLD.bcast(check_fitted_pack, root=0)
            print(f"Rank {rank} received broadcast.")
            
            self.q_surrogate.fitted = check_fitted_pack[0]
            if self.q_surrogate.fitted:
                self.q_surrogate.c = check_fitted_pack[1]
                self.q_surrogate.b = check_fitted_pack[2]
                self.q_surrogate.H = check_fitted_pack[3]
            
            # Compute initial global m
            if self.q_surrogate.fitted:
                print(f"Rank {rank} computing global shift...")
                try:
                    global_m_local = self.subproblems.compute_global_shift(self.q_surrogate, self.lower_bounder)
                    print(f"Rank {rank} local m = {global_m_local}")
                    global_m = MPI.COMM_WORLD.allreduce(global_m_local, op=MPI.SUM) 
                    if rank == 0: print(f"Global m computed: {global_m}")
                except Exception as e:
                    print(f"Error computing global shift on rank {rank}: {e}")
                    import traceback
                    traceback.print_exc()
            
            self.q_surrogate.global_m = global_m

        # --- PRUNE LOG SETUP ---
        if rank == 0 and self._params._prune_csv_path:
             with open(self._params._prune_csv_path, 'w') as f:
                 f.write("iter,time_sec,node_id,prune_type,global_ub_at_prune,lb_drop,lb_quad,lb_final,prune_trigger\n")

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
        bounds_feasible = self.subproblems.tighten_and_sync_bounds(node)
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
            
            '''
            # print full solution
            print('here is full solution')
            d = self.subproblems.save_results_to_dict()
            for k, v in d.items():
                print(f"{k}: {v}")
            '''
        return bnb_result
    

    def dispatch_updates(self,
                         bnb_result: str,
                         current_node: Node) -> None:
        """
        Collecting any information and printing information to the
        terminal, logger, plotter, etc.
        """
        # --- PRUNE LOGGING ---
        if rank == 0 and self._params._prune_csv_path:
            prune_type = "none"
            if "pruned by bound" in bnb_result:
                prune_type = "bound"
            elif "pruned by infeasibility" in bnb_result:
                prune_type = "infeasibility"
            
            if prune_type != "none":
                # Determine trigger
                trigger = "n/a"
                lb_drop = getattr(current_node.lb_problem, 'objective_drop', current_node.lb_problem.objective)
                lb_quad = getattr(current_node.lb_problem, 'objective_quad', -math.inf)
                lb_final = current_node.lb_problem.objective
                global_ub = self.tree.metrics.ub
                
                if prune_type == "bound":
                    if lb_drop > global_ub:
                        trigger = "drop"
                    elif (lb_drop <= global_ub) and (lb_final > global_ub):
                        trigger = "quad"
                    else:
                        # Should not happen if pruned by bound
                        trigger = "unknown"
                
                # Write to CSV
                with open(self._params._prune_csv_path, 'a') as f:
                    # iter,time_sec,node_id,prune_type,global_ub_at_prune,lb_drop,lb_quad,lb_final,prune_trigger
                    # node_id? use node address or unique id if available. Use hex(id(node))
                    nid = hex(id(current_node))
                    f.write(f"{self.iteration},{self.runtime:.4f},{nid},{prune_type},{global_ub:.6e},{lb_drop:.6e},{lb_quad:.6e},{lb_final:.6e},{trigger}\n")

        # if we are collecting plot info
        if self.collect_plot_info:
            self.plotter.iter_update(lb = self.tree.metrics.lb,
                                        ub = self.tree.metrics.ub)

        # update gap, iter, runtime
        self.tree.update_gap()
        self.iteration += 1
        self.runtime = time.perf_counter() - self.start_time

        # print metrics to terminal
        if (rank==0): self.display_status(bnb_result)

        # write to log
        self.logger.update()


    def dispatch_lb_solve(self,
                          current_node: Node) -> None:
        """
        Everything related to solving the LB problem of a current node.
        """
        # here, we solve all of the models associated with this subproblem.
        self.logger.lb_start()
        self.lower_bounder.solve(subproblems = self.subproblems,
                                 node = current_node)
        self.logger.lb_stop()
        
        # we need to wait for all LB problems at the other subproblems to find obj / feasibility
        MPI.COMM_WORLD.barrier()
        
        # update current node stats
        current_node.lb_problem.feasible  = \
            MPI.COMM_WORLD.allreduce(current_node.lb_problem.feasible, op=MPI.PROD)
        
        if current_node.lb_problem.feasible:
            # Aggregate standard LB (drop)
            lb_drop = MPI.COMM_WORLD.allreduce(current_node.lb_problem.objective, op=MPI.SUM)
            current_node.lb_problem.objective_drop = lb_drop
            # Initialize finalized objective
            current_node.lb_problem.objective = lb_drop
            current_node.lb_problem.quad_computed = False
            
            # --- QUADRATIC LOWER BOUND COMPUTATION ---
            # Gate: enabled? surrogate fitted?
            if (self._params._use_quadratic_surrogate and 
                self._params._use_quadratic_lb and 
                self.q_surrogate.fitted):
                
                # Check gating conditions
                # 1. Box width
                max_width = 0

                # reals_state is retrieved inside the check below
                
                from snoglode.utils.supported import SupportedVars
                
                if SupportedVars.reals in current_node.state:
                    reals_state = current_node.state[SupportedVars.reals]
                    lbs = []
                    ubs = []
                    for vid in self.lifted_reals_ids:
                        l = reals_state[vid].lb
                        u = reals_state[vid].ub
                        lbs.append(l)
                        ubs.append(u)
                        if (u - l) > max_width: max_width = (u - l)
                    
                    gate_width = (max_width <= self._params._quad_box_width_threshold)
                    
                    # 2. Gap
                    current_gap = self.tree.metrics.ub - lb_drop
                    gate_gap = (current_gap <= self._params._quad_gap_trigger_threshold)
                    
                    # 3. Always flag
                    gate_always = self._params._always_compute_quad_lb
                    
                    if gate_width or gate_gap or gate_always:
                        
                        # Compute min_Q over box (Exact for small dim)
                        if self.q_surrogate.dim <= self._params._quad_max_dim_for_exact_min:
                            min_Q_box = self.q_surrogate.min_over_box(lbs, ubs)
                            
                            # lb_quad = min_Q_box + global_m
                            # We use the 'global_m' stored on the surrogate object
                            # If not set yet (no refit), assume 0 or handle
                            global_m = getattr(self.q_surrogate, 'global_m', 0.0)
                            
                            if global_m != float('inf'):
                                lb_quad = min_Q_box + global_m
                                current_node.lb_problem.objective_quad = lb_quad
                                current_node.lb_problem.quad_computed = True
                                
                                # Update final
                                current_node.lb_problem.objective = max(lb_drop, lb_quad)
        else:
             current_node.lb_problem.objective = float('inf')

    def dispatch_ub_solve(self,
                          current_node: Node) -> None:
        """
        Everything related to solving the CG & UB problem of a current node.
        """
        from snoglode.utils.supported import SupportedVars
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
                
                # --- UPDATE QUADRATIC SURROGATE ---
                if self._params._use_quadratic_surrogate:
                    
                    # Determine X and Val
                    final_x = None
                    final_val = None
                    valid_data = False
                    
                    # Extract X from candidate_solution or subproblems
                    # Assuming candidate_solution is valid "X" even if infeasible?
                    # Yes, candidate_solution_finder makes a candidate X usually.
                    if candidate_found:
                         final_x = []
                         # Extract in canonical order
                         # Should be reliable if candidate was found.
                         for vid in self.lifted_reals_ids:
                             if SupportedVars.reals in candidate_solution:
                                 final_x.append(candidate_solution[SupportedVars.reals][vid])
                    
                    # Determine Value
                    # if ub_feasible -> ub_obj
                    # else -> 1e8
                    if self.solve_ub:
                        if ub_feasible:
                             final_val = ub_obj
                             valid_data = True
                        else:
                             final_val = 1e8 # Infeasible Penalty
                             valid_data = True # We still add it!
                    elif candidate_found:
                         final_val = candidate_objective
                         valid_data = True
                    
                    if valid_data and final_x:
                        # Rank 0 adds point and fits
                        check_fitted_pack = None # (fitted, c, b, H, global_m)
                        
                        if rank == 0:
                            self.q_surrogate.add_point(final_x, final_val)
                            # Always Refit? "After each new point..."
                            # Yes, assuming not extremely expensive.
                            self.q_surrogate.fit()
                            
                            check_fitted_pack = (self.q_surrogate.fitted, self.q_surrogate.c, self.q_surrogate.b, self.q_surrogate.H)
                        
                        # Broadcast updated model state
                        check_fitted_pack = MPI.COMM_WORLD.bcast(check_fitted_pack, root=0)
                        
                        # Update local surrogates
                        self.q_surrogate.fitted = check_fitted_pack[0]
                        if self.q_surrogate.fitted:
                            self.q_surrogate.c = check_fitted_pack[1]
                            self.q_surrogate.b = check_fitted_pack[2]
                            self.q_surrogate.H = check_fitted_pack[3]
                            
                            # Update Global Shift m
                            # This needs to happen on ALL ranks locally then reduce
                            global_m_local = self.subproblems.compute_global_shift(self.q_surrogate, self.lower_bounder)
                            global_m = MPI.COMM_WORLD.allreduce(global_m_local, op=MPI.SUM)
                            self.q_surrogate.global_m = global_m
         
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
                       bnb_result: str) -> None:
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

        outputs = [round(self.runtime,3),
                   self.tree.metrics.nodes.explored,
                   pruned,
                   bound_update,
                   f"{self.tree.metrics.lb:.8}",
                   f"{self.tree.metrics.ub:.8}",
                   f"{round(self.tree.metrics.relative_gap*100, 4)}%",
                   round(self.tree.metrics.absolute_gap, 6),
                   self.tree.n_nodes()]

        # if this is first iter, print header
        header = ["Time (s)", 
                  "Nodes Explored", 
                  "Pruned by", 
                  " ", 
                  "LB", 
                  "UB", 
                  "Rel. Gap", 
                  "Abs. Gap",
                  "# Nodes"]
        if self.iteration == 1:
            header_print = ""
            header_line = ""
            for i, title in enumerate(header):
                header_print += f"  {title:>14}"
                if i != 0:
                    header_line += "-"*(15+len(title))
            print(header_print)
            print(header_line)

        # if nothing changed, print nothing (saving output space)
        # if bnb_result != "":
        line_print = ""
        for output in outputs:
            line_print += f"  {output:>14}"
        print(line_print)