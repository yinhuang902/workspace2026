import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def plot_3d(x: float, y1: float, y2: float, opt_obj: float) -> None:

    # Create grid ranges for variables
    x_vals = np.linspace(0, 20, 60)
    y1_vals = np.linspace(-5, 20, 60)
    y2_vals = np.linspace(-5, 20, 60)

    # Build 3D mesh
    X, Y1, Y2 = np.meshgrid(x_vals, y1_vals, y2_vals, indexing="xy")

    # Feasibility masks for constraints:
    feasible = (X + Y1 >= 10) & (X + Y2 >= 7)

    # Extract feasible points
    Xf = X[feasible]
    Y1f = Y1[feasible]
    Y2f = Y2[feasible]

    # Objective: x + y1^2 + y2^2
    obj = Xf + Y1f**2 + Y2f**2

    # Optimal point (user-provided)
    x_opt = x
    y1_opt = y1
    y2_opt = y2
    obj_opt = opt_obj

    # --- Build plot ---
    fig = go.Figure()

    # Feasible region points (interactive, transparent)
    fig.add_trace(go.Scatter3d(
        x=Xf,
        y=Y1f,
        z=Y2f,
        mode="markers",
        marker=dict(
            size=3,
            opacity=0.1,        # transparency like matplotlib alpha=0.1
            color=obj,          # color by objective
            colorscale="Viridis",
            colorbar=dict(title="Objective Value")
        ),
        name="Feasible Region"
    ))

    # Optimal point
    fig.add_trace(go.Scatter3d(
        x=[x_opt],
        y=[y1_opt],
        z=[y2_opt],
        mode="markers",
        marker=dict(
            size=8,
            color="red",
            symbol="cross",     # 3D Scatter does not support "*"
            line=dict(width=2, color="black")
        ),
        name=f"Optimal Point (obj={round(obj_opt,2)})"
    ))

    fig.update_layout(
        title="Interactive 3D Feasible Region + Objective + Optimal Point",
        scene=dict(
            xaxis_title="x",
            yaxis_title="y₁",
            zaxis_title="y₂"
        ),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="black",
            borderwidth=1
        ),
        width=900,
        height=700
    )

    fig.show()


def plot_subproblem_1_feasible_region():

    # Decision variable ranges
    x_vals = np.linspace(0, 100, 400)
    y_vals = np.linspace(-20, 100, 400)

    # Mesh grid
    X, Y = np.meshgrid(x_vals, y_vals, indexing="xy")

    # Feasible region
    feasible = (X + Y >= 10)

    Xf = X[feasible]
    Yf = Y[feasible]

    # Objective
    obj = 0.5 * Xf + Yf**2

    # Point to highlight
    x_star = 9.5
    y_star = 0.5

    # --- Plot ---
    plt.figure(figsize=(9, 7))

    # Feasible region shading
    plt.scatter(
        Xf, Yf,
        c=obj,
        cmap="viridis",
        s=6,
        alpha=0.25,
        edgecolors="none",
        label="Feasible region"
    )

    # Boundary line
    plt.plot(
        x_vals, 10 - x_vals,
        color="black",
        linewidth=2
    )

    plt.xlim(0, 100)
    plt.ylim(-20, 100)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$y_1$")
    plt.title("Subproblem $1$: Feasible Region)")

    # Red X for the given point
    plt.scatter(
        [x_star], [y_star],
        color="red",
        marker="x",
        s=120,
        linewidths=3,
        zorder=10,              # <<< ensures it is drawn on top
        label=r"Full Model Optimum: $(9.5,\;0.5)$"
    )

    plt.colorbar(label=r"Objective: $\frac{1}{2}x_1 + y_1^2$")
    plt.legend()

    plt.show()



def plot_feasible_regions():

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ------------------------------------------------
    # Subproblem 1
    # ------------------------------------------------
    ax = axes[0]

    # Decision variable ranges
    x_vals = np.linspace(0, 100, 400)
    y_vals = np.linspace(-20, 100, 400)
    X, Y = np.meshgrid(x_vals, y_vals, indexing="xy")

    # Feasible region
    feasible = (X + Y >= 10)
    Xf = X[feasible]
    Yf = Y[feasible]

    # Objective
    obj = 0.5 * Xf + Yf**2

    # Highlighted point
    x_star = 9.5
    y_star = 0.5

    # Feasible region scatter
    ax.scatter(
        Xf, Yf,
        c=obj,
        cmap="viridis",
        s=6,
        alpha=0.25,
        edgecolors="none",
        label="Feasible region"
    )

    # Constraint boundary
    ax.plot(
        x_vals, 10 - x_vals,
        color="black", linewidth=2
    )

    ax.set_xlim(0, 100)
    ax.set_ylim(-20, 100)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$y_1$")
    ax.set_title("Subproblem $1$: Feasible Region")

    # Red X on top
    ax.scatter(
        [x_star], [y_star],
        color="red", marker="x",
        s=120, linewidths=3,
        zorder=10,
        label=r"Full Model Optimum: $(9.5,\;0.5)$"
    )

    local_obj, x_local_star, y1_local_star = subproblem_opt("1")
    ax.scatter(
        [x_local_star], [y1_local_star],
        color="green", marker="x",
        s=120, linewidths=3,
        zorder=10,
        label=f"Local Model Optimum: ({x_local_star:.2f}, {y1_local_star:.2f})"
    )

    cbar = ax.figure.colorbar(ax.collections[0], ax=ax)
    cbar.set_label(r"Objective: $\frac{1}{2}x_1 + y_1^2$")

    ax.legend(loc="upper left")


    # ------------------------------------------------
    # Subproblem 2
    # ------------------------------------------------
    ax = axes[1]

    # Decision variable ranges
    x_vals2 = np.linspace(0, 100, 400)
    y_vals2 = np.linspace(-20, 100, 400)
    X2, Y2 = np.meshgrid(x_vals2, y_vals2, indexing="xy")

    # Feasible region: x2 + y2 >= 7
    feasible2 = (X2 + Y2 >= 7)
    X2f = X2[feasible2]
    Y2f = Y2[feasible2]

    # Objective
    obj2 = 0.5 * X2f + Y2f**2

    # Feasible region scatter
    ax.scatter(
        X2f, Y2f,
        c=obj2,
        cmap="viridis",
        s=6,
        alpha=0.25,
        edgecolors="none",
        label="Feasible region"
    )

    # Constraint boundary
    ax.plot(
        x_vals2, 7 - x_vals2,
        color="black", linewidth=2
    )

    # Highlighted point
    x_star = 9.5
    y_star = 0.0

    # Red X on top
    ax.scatter(
        [x_star], [y_star],
        color="red", marker="x",
        s=120, linewidths=3,
        zorder=10,
        label=r"Full Model Optimum: $(9.5,0.0)$"
    )

    local_obj, x_local_star, y2_local_star = subproblem_opt("2")
    ax.scatter(
        [x_local_star], [y2_local_star],
        color="green", marker="x",
        s=120, linewidths=3,
        zorder=10,
        label=f"Local Model Optimum: ({x_local_star:.2f}, {y2_local_star:.2f})"
    )

    ax.set_xlim(0, 100)
    ax.set_ylim(-20, 100)
    ax.set_xlabel(r"$x_2$")
    ax.set_ylabel(r"$y_2$")
    ax.set_title("Subproblem $2$: Feasible Region")

    cbar2 = ax.figure.colorbar(ax.collections[-1], ax=ax)
    cbar2.set_label(r"Objective: $\frac{1}{2}x_2 + y_2^2$")

    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.show()

def subproblem_opt(name: str):
    import pyomo.environ as pyo
    from pyomo.opt import SolverFactory

    gurobi = SolverFactory('gurobi')

    m = pyo.ConcreteModel()
    m.x = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0,100))

    if name == "1":
        m.y_1 = pyo.Var(domain=pyo.Reals)
        m.obj = pyo.Objective( expr = (1/2)*m.x + m.y_1**2 )
        m.c = pyo.Constraint( expr = m.x + m.y_1 >= 10 )
    if name == "2":
        m.y_2 = pyo.Var(domain=pyo.Reals)
        m.obj = pyo.Objective( expr = (1/2)*m.x + m.y_2**2 )
        m.c = pyo.Constraint( expr = m.x + m.y_2 >= 7 )

    results = gurobi.solve(m)
    return pyo.value(m.obj), pyo.value(m.x), pyo.value(m.y_1) if name=="1" else pyo.value(m.y_2)


def plot_results(solver) -> None:
    import matplotlib.pyplot as plt

    plotter = solver.plotter

    plt.figure(figsize=(10, 6))

    # Continuous lines (no markers)
    plt.plot(plotter.iter_lb, label="Lower Bound")
    plt.plot(plotter.iter_ub, label="Upper Bound")

    plt.xlabel("Iteration")
    plt.ylabel("Objective Value")
    plt.title("SNoGloDe Solver Progression")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
