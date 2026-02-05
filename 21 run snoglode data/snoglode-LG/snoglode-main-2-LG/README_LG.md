# SNoGloDe-LG: Branch-and-Bound with Lagrangian Lower Bounding

## What This Solver Does

This codebase implements a **Branch-and-Bound (B&B) solver** for **two-stage stochastic programs** with nonconvex (or difficult) recourse. The target problem class involves:

- **First-stage (linking) variables** `y` that must be determined before uncertainty is revealed
- **Multiple scenario subproblems** ω ∈ Ω, each defining a second-stage recourse cost `f_ω(y, x_ω)`
- **Expected objective**: minimize `Σ p_ω · f_ω(y)` subject to non-anticipativity (all scenarios agree on `y`)

### High-Level Algorithm

```
┌─────────────────────────────────────────────────────────────────┐
│  solver.py : Main Loop                                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ 1. Setup → Initialize tree (root node), queues, bounders  │  │
│  │ 2. While not converged:                                   │  │
│  │    ├─ Select node from queue                              │  │
│  │    ├─ Solve LB problem (LG method or DropNonants)         │  │
│  │    ├─ Generate candidate → Solve UB problem               │  │
│  │    ├─ Bound (prune by infeasibility or by bound)          │  │
│  │    ├─ Branch (create children nodes) → Enqueue            │  │
│  │    └─ Update global LB/UB, check convergence              │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

The **LG (Lagrangian) lower bounding** method:
1. Dualizes non-anticipativity constraints using multipliers μ
2. Solves Lagrangian subproblems per scenario (in parallel via MPI)
3. Generates **Lagrangian cuts** from each solve
4. Stores cuts in a **global cut pool** for reuse across nodes
5. Solves a **Relaxed Master Problem (RMP)** to compute the node lower bound
6. Updates multipliers via **projected subgradient** method

---

## Repository / Code Structure

### Core Solver
| File | Description |
|------|-------------|
| `snoglode/solver.py` | Top-level solve loop. Orchestrates tree, queues, and bounders. Entry point: `Solver.solve()` |

### Components (`snoglode/components/`)
| File | Description |
|------|-------------|
| `tree.py` | B&B tree. Manages node spawning, global LB/UB tracking, convergence checks, branching dispatch, and **global cut pool** |
| `node.py` | Node representation. Stores state (variable bounds), `to_branch` sets, `lb_problem`/`ub_problem` metrics, and Lagrangian multipliers |
| `queues.py` | Node selection queues: FIFO, LIFO, worst-bound. Implements `NodeQueue.push/pop` |
| `branching.py` | Variable selection (Random, MostInfeasibleBinary, MaximumDisagreement, Pseudocost, Hybrid, StrongBranching) and split-point strategies |
| `subproblems.py` | Scenario subproblem container. Builds Pyomo models, manages complicating vars, applies node state to models |
| `parameters.py` | Configuration: tolerances, timelimits, queue strategy, branching strategy, solver options |

### Bounders (`snoglode/bounders/`)
| File | Description |
|------|-------------|
| `lower_bounders.py` | Lower bounding procedures. Key classes: `DropNonants` (solve each scenario independently) and `LGLowerBounder` (Lagrangian method) |
| `upper_bounders.py` | Upper bounding / incumbent generation. `UpperBounder` fixes to candidate solution and evaluates feasibility |

### Utilities (`snoglode/utils/`)
| File | Description |
|------|-------------|
| `cut_pool.py` | **Global cut pool** for Lagrangian branch-and-cut. Contains `LagrangeanCut` dataclass and `CutPool` class |
| `MPI.py` | MPI wrapper for parallel execution |
| `logging.py` | Solve statistics and logging |

---

## LG Method in This Code

The `LGLowerBounder` class in `lower_bounders.py` implements the Lagrangian relaxation lower bounding.

### Multipliers
- **Structure**: `current_mu[scenario_name][var_id] → float`
- One multiplier per scenario per linking variable dimension
- Initialized to zero; inherited from parent node if available

### Lagrangian Cut
Each cut (stored in `cut_pool.py`) contains:
| Field | Description |
|-------|-------------|
| `scenario_name` | Which scenario this cut came from |
| `mu_vector` | Snapshot of μ at time of generation |
| `v_val` | Optimal Lagrangian subproblem value: `v(μ) = min_y [f_ω(y) - μᵀy]` |
| `y_bounds` | Node bounds at generation time (for validity checking) |
| `node_id` | Node ID where cut was generated |

### Global Cut Pool
Cuts are stored in `tree.cut_pool` and can be reused across nodes:
- A cut from node N is **valid** at node M if M's domain ⊆ N's domain
- `CutPool.get_valid_cuts()` filters cuts by domain containment
- Pool is wired to `LGLowerBounder` via `solver.py`

### Relaxed Master Problem (RMP)
The RMP solves:
```
minimize   Σ p_ω · η_ω
subject to η_ω ≥ v_val + μᵀ y   ∀ cuts for scenario ω
           y ∈ [node bounds]
```
- **Output**: node lower bound = RMP objective
- Uses **true scenario probabilities** (gathered from all ranks)
- Uses **configured solver** (`self.opt`) instead of hardcoded solver
- If some scenarios have no cuts, a conservative LB (parent's LB or -∞) is used

### Multiplier Update
After each inner iteration:
1. Compute consensus `y_bar = (1/|Ω'|) Σ y*_ω` over successful solves
2. Subgradient: `g_ω = y*_ω - y_bar`
3. Step: `μ_ω ← μ_ω + t_k · g_ω` with `t_k = t₀ / √(k+1)`
4. **Projection**: enforce `Σ_ω μ_ω = 0` by subtracting the mean

### Safety Checks
- **Solver validation**: Warns if Gurobi `NonConvex≠2` (required for global optimality)
- **Finite bound check**: Asserts complicating variables have finite bounds
- **Probability validation**: Asserts probabilities sum to 1

### MPI Handling
- **Probability gathering**: All ranks participate in `gather()` before rank-0 block
- **Objective aggregation**: Only rank 0 contributes objective; others contribute 0 for SUM allreduce
- **Broadcast**: RMP result and multipliers broadcast from rank 0

---

## How to Run / Where to Start Reading

### Minimal Usage

```python
from snoglode.solver import Solver
from snoglode.components.parameters import SolverParameters

# Define your subproblem creator function
def my_subproblem_creator(scenario_name):
    model = ...  # build Pyomo model
    complicating_vars = {...}  # dict of linking var name → Pyomo Var
    return model, complicating_vars

# Configure
params = SolverParameters(
    subproblem_names=['s1', 's2', ...],
    subproblem_creator=my_subproblem_creator,
    # ... other options
)

solver = Solver(params)
solver.solve(max_iter=100, rel_tolerance=1e-3, time_limit=3600)
```

See `examples/` for complete working examples.

### Start Reading Here (Recommended Order)

1. **`solver.py`** – Main loop, understand `Solver.solve()` and dispatch methods
2. **`components/tree.py`** – Node lifecycle, branching, cut pool initialization
3. **`components/node.py`** – Node data structure, state, metrics
4. **`bounders/lower_bounders.py`** – Focus on `LGLowerBounder.solve()` for LG internals
5. **`utils/cut_pool.py`** – Global cut pool for Lagrangian cuts
6. **`components/subproblems.py`** – How scenario models are built and managed

---

## Key Files Reference

| Purpose | File |
|---------|------|
| Entry point | `snoglode/solver.py` |
| LG lower bounding | `snoglode/bounders/lower_bounders.py : LGLowerBounder` |
| Drop-nonants baseline | `snoglode/bounders/lower_bounders.py : DropNonants` |
| Global cut pool | `snoglode/utils/cut_pool.py : CutPool, LagrangeanCut` |
| Branching strategies | `snoglode/components/branching.py` |
| Node queues | `snoglode/components/queues.py` |
| Configuration | `snoglode/components/parameters.py` |
