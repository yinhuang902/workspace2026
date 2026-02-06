# Quickstart

This guide walks you through running your first example using **SNoGloDe**.

## Prerequisites

Before continuing, ensure that:

- The package is installed (see **Installation** page)
- Python 3.9 or newer
- Gurobi 11.0 or newer

## Example

Consider solving the simple problem: 

$ \text{min}$ $ x + y_1^2 + y_2^2$

&nbsp; $\text{s.t.} $ $x + y_1 \geq 10$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$x + y_2 \geq 7$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$ 0 \leq x \leq 100$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$x \in \mathbb{R}^1$, $y_1 \in \mathbb{R}^1$, $y_2 \in \mathbb{R}^1$

Which can be broken down into subproblem $1$:

$ \text{min}$ $ \frac{1}{2} x + y_1^2$

&nbsp;$\text{s.t.} $ $x + y_1 \geq 10$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $ 0 \leq x \leq 100$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $x \in \mathbb{R}^1$, $y_1 \in \mathbb{R}^1$

and subproblem $2$:

$ \text{min}$ $ \frac{1}{2} x + y_2^2$

&nbsp;$\text{s.t.} $ $x + y_2 \geq 7$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $ 0 \leq x \leq 100$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $x \in \mathbb{R}^1$, $y_2 \in \mathbb{R}^1$

```python
import snoglode as sno
import pyomo.environ as pyo
gurobi = pyo.SolverFactory("gurobi")

def subproblem_creator(name):
    m = pyo.ConcreteModel()
    m.x = pyo.Var(domain=pyo.Reals,
                  bounds = (0, 100))
    
    if name == "1":
        m.y_1 = pyo.Var(domain=pyo.Reals)
        m.obj = pyo.Objective( expr = m.x + m.y_1**2 )
        @m.Constraint()
        def c1(model):
            return ( m.x + m.y_1 >= 10 )
    if name == "2":
        m.y_2 = pyo.Var(domain=pyo.Reals)
        m.obj = pyo.Objective( expr = m.x + m.y_2**2 )
        @m.Constraint()
        def c1(model):
            return ( m.x + m.y_2 >= 7 )

    # model, complicating vars mapping, probability weight    
    return m, {"x": m.x}, 1

subproblem_names = ["1", "2"]
params = sno.SolverParameters(subproblem_names = subproblem_names,
                              subproblem_creator = subproblem_creator,
                              lb_solver = gurobi,
                              cg_solver = gurobi,
                              ub_solver = gurobi)

solver = sno.Solver(params)
solver.solve(max_iter=100)
print(f"The best primal/feasible solution has a value of: {solver.tree.metrics.ub}")
print(f"The best lower bound solution has a value of: {solver.tree.metrics.lb}")
```

This problem can also be run in parallel (with max. 2 ranks). Note, `mpi4py` must be installed properly.

```bash
mpiexec -np 2 python <FILE_NAME>.py
```

For a more in depth walk through, see examples/quickstart/quickstart.ipynb.
