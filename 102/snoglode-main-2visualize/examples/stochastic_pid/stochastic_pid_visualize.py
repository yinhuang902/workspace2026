
import sys
import os
import math
import copy as cp
import numpy as np
import pyomo.environ as pyo
import plotly.graph_objects as go
from IPython.display import display, clear_output

# Add parent directory to path to import snoglode if not installed as package
# (Assuming running from examples/stochastic_pid/)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))

import snoglode as sno
import snoglode.utils.MPI as MPI
from snoglode.components.node import Node
from snoglode.utils.supported import SupportedVars

# Import the model builder from existing file
import stochastic_pid as sp_orig

# Check rank - we only plot on rank 0
rank = MPI.COMM_WORLD.Get_rank()

def extract_first_stage_bounds(state):
    """
    Extracts (lb, ub) for K_p, K_i, K_d from the node state.
    Returns lists [kp_lb, ki_lb, kd_lb], [kp_ub, ki_ub, kd_ub]
    """
    reals = state[SupportedVars.reals]
    
    # We assume variable names are 'K_p', 'K_i', 'K_d' as seen in stochastic_pid.py
    # Order: K_p (x), K_i (y), K_d (z)
    var_names = ['K_p', 'K_i', 'K_d']
    
    lbs = []
    ubs = []
    
    for vname in var_names:
        if vname in reals:
            # Check if it's a fixed value or bounds
            vdata = reals[vname]
            # snoglode.components.node.VariableData might use 'is_fixed', 'value', 'lb', 'ub'
            # Based on node.display() output in node.py:
            # print(f"\t  lb = {var.lb}")
            # print(f"\t  ub = {var.ub}")
            # if var.is_fixed: ...
            
            if hasattr(vdata, 'is_fixed') and vdata.is_fixed:
                val = vdata.value
                lbs.append(val)
                ubs.append(val)
            else:
                lbs.append(vdata.lb)
                ubs.append(vdata.ub)
        else:
            # Should not happen for this problem, but fallback
            lbs.append(-np.inf)
            ubs.append(np.inf)
            
    return lbs, ubs

def make_box_traces(lbs, ubs, color, name, opacity=0.1, line_style="solid", line_width=2):
    """
    Creates Plotly traces for a 3D box defined by lbs and ubs.
    lbs: [x_min, y_min, z_min]
    ubs: [x_max, y_max, z_max]
    """
    x_min, y_min, z_min = lbs
    x_max, y_max, z_max = ubs
    
    # 8 corners of the box
    x = [x_min, x_min, x_max, x_max, x_min, x_min, x_max, x_max]
    y = [y_min, y_max, y_max, y_min, y_min, y_max, y_max, y_min]
    z = [z_min, z_min, z_min, z_min, z_max, z_max, z_max, z_max]
    
    # Define the 12 edges for wireframe
    # Bottom face: (0,1), (1,2), (2,3), (3,0)
    # Top face: (4,5), (5,6), (6,7), (7,4)
    # Vertical: (0,4), (1,5), (2,6), (3,7)
    
    lines_x = []
    lines_y = []
    lines_z = []
    
    pairs = [(0,1), (1,2), (2,3), (3,0),
             (4,5), (5,6), (6,7), (7,4),
             (0,4), (1,5), (2,6), (3,7)]
             
    for i, j in pairs:
        lines_x.extend([x[i], x[j], None])
        lines_y.extend([y[i], y[j], None])
        lines_z.extend([z[i], z[j], None])
        
    # Wireframe trace
    wireframe = go.Scatter3d(
        x=lines_x, y=lines_y, z=lines_z,
        mode='lines',
        name=name + " (outline)",
        line=dict(color=color, width=line_width, dash=line_style),
        showlegend=False
    )
    
    # Mesh3d for semi-transparent volume
    # Vertices are already x, y, z
    # We need to define triangles (i, j, k)
    # A cube has 6 faces, 2 triangles per face = 12 triangles
    # Indices for vertices 0-7
    # 0: 000, 1: 010, 2: 110, 3: 100
    # 4: 001, 5: 011, 6: 111, 7: 101
    
    i = [0, 0, 4, 4, 0, 0, 3, 3, 1, 1, 0, 0] # Example indices
    # Actually, simpler to just rely on Mesh3d's alphahull or define manually.
    # Manual definition is safer for exact box.
    
    # Face 1 (Bottom): 0-1-2-3 -> 0-1-2, 0-2-3
    # Face 2 (Top): 4-5-6-7 -> 4-5-6, 4-6-7
    # Face 3 (Front): 0-1-5-4 -> 0-1-5, 0-5-4
    # Face 4 (Right): 1-2-6-5 -> 1-2-6, 1-6-5
    # Face 5 (Back): 2-3-7-6 -> 2-3-7, 2-7-6
    # Face 6 (Left): 3-0-4-7 -> 3-0-4, 3-4-7
    
    I = [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3] # This is hard to get right manually quickly without lookup
    # Let's just use the wireframe for clarity as primary, and a light Mesh3d using intensity if needed.
    # Actually, for "semi-transparent solid", Mesh3d is best.
    
    # Correct indices for a box
    # 0: min min min, 7: max max max
    I = [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3] # Still guessing
    
    # Let's use `go.Mesh3d` with `alphahull=0` (convex hull) of the 8 points
    mesh = go.Mesh3d(
        x=x, y=y, z=z,
        color=color,
        opacity=opacity,
        alphahull=0, # Convex hull of vertices = box
        name=name,
        flatshading=True,
        showlegend=True
    )
    
    return [mesh, wireframe]


class VisualizingSolver(sno.Solver):
    """
    Subclass of snoglode.Solver that adds 3D visualization of the search process.
    """
    def __init__(self, params):
        super().__init__(params)
        self.interp_points_acc = [] # List of (kp, ki, kd) tuples
        self.iter_pruned_nodes = [] # List of (bounds, reason) for current iter
        
        # Create directory for plots if it doesn't exist
        if rank == 0:
            os.makedirs("plots_3d", exist_ok=True)
        
    def solve(self, 
              max_iter: int = 100, 
              rel_tolerance: float = 1e-2,
              abs_tolerance: float = 1e-3,
              time_limit: float = math.inf, 
              collect_plot_info: bool = False):
        
        # --- Copied setup from snoglode.Solver.solve ---
        self.dispatch_setup(max_iter = max_iter, 
                            rel_tolerance = rel_tolerance,
                            abs_tolerance = abs_tolerance,
                            time_limit = time_limit, 
                            collect_plot_info = collect_plot_info)
        
        # --- Main Loop with Hooks ---
        while (not self.tree.converged(self.runtime)) and (self.iteration <= max_iter):
            
            # Reset per-iteration visualization data
            self.iter_pruned_nodes = []
            current_node_before_tighten_bounds = None
            current_node_after_tighten_bounds = None
            
            # 1. Node Selection
            # self.tree.get_node() happens in dispatch_node_selection
            # We need to intersect dispatch_node_selection to capture bounds BEFORE tightening
            
            # We can't easily hook into dispatch_node_selection without copying it or modifying calls.
            # So we will copy the logic of dispatch_node_selection here.
            
            # --- Start dispatch_node_selection logic ---
            node = self.tree.get_node()
            
            # Hook: Capture bounds BEFORE tightening
            current_node_before_tighten_bounds = extract_first_stage_bounds(node.state)
            
            node_feasible = self.node_feasibility_checks(node)
            
            current_node_feasible = False
            if not node_feasible:
                # Node infeasible by basic checks
                current_node_feasible = False
                # Track pruning? node_feasibility_checks might prune by bound.
                # If so, it might not be in our list yet.
                # We can check node status or bnb_result later.
            else:
                 # set state - only set second stage if we are performing bounds tightening
                bounds_tightening = (self._params._obbt or self._params._fbbt)
                self.subproblems.set_all_states(node.state,
                                                set_second_stage = bounds_tightening)
                
                # tighten bounds & sync across all existing subproblems
                bounds_feasible = self.subproblems.tighten_and_sync_bounds(node)
                current_node_feasible = (node_feasible and bounds_feasible)
                
            # Hook: Capture bounds AFTER tightening
            current_node_after_tighten_bounds = extract_first_stage_bounds(node.state)
            
            current_node = node
            # --- End dispatch_node_selection logic ---
            
            
            # 2. Solve Node
            if current_node_feasible: 
                self.dispatch_node_solver(current_node)
                # Hook: Check for candidate solutions generated
                if current_node.ub_problem.feasible:
                    # Where is the candidate solution? 
                    # dispatch_ub_solve solves it.
                    # We can look at the subproblems state or reconstruction.
                    # For simplicity, let's look at the node state or model values if available.
                    # Or better, we can extract it from self.subproblems if a candidate was just found.
                    # As a proxy, we use the center of the current node or the LB solution if available.
                    # Ideally dispatch_ub_solve would return this, but it doesn't.
                    pass

            
            # 3. Branch & Bound
            bnb_result = self.dispatch_bnb(current_node)
            
            # Hook: Track pruning
            if "pruned by" in bnb_result:  # "pruned by bound" or "pruned by infeasibility"
                 # Only current node is pruned here
                 self.iter_pruned_nodes.append((current_node_after_tighten_bounds, bnb_result))
            
            
            # 4. Updates
            self.dispatch_updates(bnb_result)


            # --- VISUALIZATION HOOK ---
            if rank == 0:
                self.visualize_iteration(
                    current_node_before_tighten_bounds,
                    current_node_after_tighten_bounds,
                    current_node
                )
                
        # End loop
        self.logger.complete()


    def visualize_iteration(self, bounds_pre, bounds_post, current_node):
        """
        Generates and displays the 3D plot for the current iteration.
        """
        fig = go.Figure()
        
        # 1. Alive Regions (Queue)
        # Cap at top 200 to avoid performance hit
        # Access internal heap list '_q' (not _queue)
        # The heap stores tuples: (priority, node)
        alive_items = list(self.tree.node_queue._q) 
        
        num_alive = len(alive_items)
        
        for i, item in enumerate(reversed(alive_items)):
            if i > 200: break
            
            # Unpack the node from the heap item
            # SnogLode queues usually store (metric, node)
            if isinstance(item, tuple) and len(item) >= 2:
                node = item[1]
            else:
                node = item # Fallback
                
            lbs, ubs = extract_first_stage_bounds(node.state)
            # User request: Darken light gray active search area.
            # Changed color to 'gray' and opacity to 0.1
            traces = make_box_traces(lbs, ubs, color='gray', name=f'Node {node.id}', opacity=0.1)
            # Only add wireframe to reduce heaviness? Or just mesh? 
            # Let's add just mesh for alive nodes to keep it clean, or just wireframe.
            # User asked for "semi-transparent LIGHT GRAY boxes".
            fig.add_trace(traces[0]) # Mesh
            # fig.add_trace(traces[1]) # Wireframe (skip for background nodes)
            
        
        # 2. Pruned Nodes (This iteration)
        for bounds, reason in self.iter_pruned_nodes:
            lbs, ubs = bounds
            color = 'red' if 'infeasibility' in reason else 'orange'
            name = reason
            traces = make_box_traces(lbs, ubs, color=color, name=name, opacity=0.4)
            fig.add_traces(traces) # Add both mesh and wireframe
            
        
        # 3. Current Node (Tightening effect)
        # Before Tightening (Dashed)
        if bounds_pre:
            # User request: Thicken the dashed line
            traces_pre = make_box_traces(bounds_pre[0], bounds_pre[1], color='blue', name='Current (Pre)', opacity=0.0, line_style='dash', line_width=5)
            fig.add_trace(traces_pre[1]) # Only wireframe
            
        # After Tightening (Solid)
        if bounds_post:
             traces_post = make_box_traces(bounds_post[0], bounds_post[1], color='blue', name='Current (Post)', opacity=0.2, line_style='solid')
             fig.add_traces(traces_post)

        
        # 4. Interpolation Points (Accumulated)
        # We need to harvest these from somewhere.
        # Ideally, anytime `candidate_solution_finder.generate` is called successfully.
        # We can try to assume center of processed nodes or extract from `current_node.lb_problem.subproblem_solutions`
        pass # Skip for now if we can't easily access exact points without more invasiveness
        
        # 5. Best Incumbent
        if self.solution and self.solution.subproblem_solutions is not None:
             # Extract Kp, Ki, Kd from solution dictionary
             sol = self.solution.subproblem_solutions
             # Format in snoglode solution dict is: subproblem_name -> var_name -> value
             # But variables are per scenario? First stage should be same.
             # Just pick first subproblem
             try:
                 first_sub = list(sol.keys())[0]
                 kp = sol[first_sub].get('K_p', None)
                 ki = sol[first_sub].get('K_i', None)
                 kd = sol[first_sub].get('K_d', None)
                 
                 if kp is not None:
                     fig.add_trace(go.Scatter3d(
                         x=[kp], y=[ki], z=[kd],
                         mode='markers',
                         marker=dict(size=5, color='red'),
                         name='Best Incumbent'
                     ))
             except:
                 pass

        # Update Layout
        fig.update_layout(
            title=f"Iteration {self.iteration} | Nodes: {num_alive} | LB: {self.tree.metrics.lb:.4f} | UB: {self.tree.metrics.ub:.4f}",
            scene=dict(
                xaxis_title='Kp',
                yaxis_title='Ki',
                zaxis_title='Kd'
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        # Save to HTML (draggable 3D format)
        filename = f"plots_3d/iter_{self.iteration:04d}.html"
        fig.write_html(filename)
        
        # Display
        # clear_output(wait=True) # Removed to preserve history as requested
        # fig.show() method is safer for ensuring correct renderer is used in various notebook environments
        # instead of printing the object repr.
        fig.show()
        # display(fig) # Caused raw text output in some contexts


if __name__ == '__main__':
    # --- Reusing setup from stochastic_pid.py ---
    
    # Define solvers
    nonconvex_gurobi = pyo.SolverFactory("gurobi")
    nonconvex_gurobi.options["NonConvex"] = 2
    
    nonconvex_gurobi_lb = pyo.SolverFactory("gurobi")
    nonconvex_gurobi_lb.options["NonConvex"] = 2
    nonconvex_gurobi_lb.options["MIPGap"] = 0.2
    nonconvex_gurobi_lb.options["TimeLimit"] = 15
    
    ipopt = sp_orig.ipopt # Use same ipopt as module
    
    num_scenarios = sp_orig.num_scenarios
    scenarios = [f"scen_{i}" for i in range(1, num_scenarios+1)]

    obbt_solver_opts = {
        "NonConvex": 2,
        "MIPGap": 1,
        "TimeLimit": 5
    }

    # Setup Parameters
    params = sno.SolverParameters(subproblem_names = scenarios,
                                  subproblem_creator = sp_orig.build_pid_model,
                                  lb_solver = nonconvex_gurobi_lb,
                                  cg_solver = ipopt,
                                  ub_solver = nonconvex_gurobi)
    
    params.set_bounders(candidate_solution_finder = sno.SolveExtensiveForm,
                        lower_bounder = sp_orig.GurobiLBLowerBounder)
                        
    params.set_bounds_tightening(fbbt=True, 
                                 obbt=True,
                                 obbt_solver_opt=obbt_solver_opts)
                                 
    params.set_branching(selection_strategy = sno.HybridBranching,
                         partition_strategy = sno.ExpectedValue)
    
    params.activate_verbose()
    if (rank==0): params.display()

    # --- Instantiate Visualizing Solver ---
    solver = VisualizingSolver(params)

    # --- Solve ---
    # Reducing time limit/iter for demo purposes in notebook context usually, 
    # but keeping original settings as requested.
    solver.solve(max_iter=1000,
                 rel_tolerance = 1e-3,
                 time_limit = 600*6)

    # --- Final Output (Same as original) ---
    if (rank==0):
        print("\n====================================================================")
        print("SOLUTION")
        # Reuse solver.solution information which is populated by base class
        for n in solver.subproblems.names:
            print(f"subproblem = {n}")
            x, u = {}, {}
            if n in solver.solution.subproblem_solutions:
                 sol_dict = solver.solution.subproblem_solutions[n]
                 for vn in sol_dict:
                    # display first stage
                    if vn in ["K_p", "K_i", "K_d"]:
                        print(f"  var name = {vn}, value = {sol_dict[vn]}")
            print()
        print("====================================================================")

