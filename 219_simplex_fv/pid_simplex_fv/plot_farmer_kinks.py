"""
plot_farmer_kinks.py — Compute and visualize all kink (inflection) points
of the farmer stochastic programming problem in 3D.

Kink planes arise where scenario-dependent constraints switch between
active/inactive. For each scenario s with yield multiplier ym:
  - Wheat: x₁ = 200 / (2.5 * ym)   (minimum wheat requirement binding)
  - Corn:  x₂ = 240 / (3.0 * ym)   (minimum corn  requirement binding)
  - Beets: x₃ = 6000 / (20 * ym)   (beet quota binding)

With 3 scenarios (ym ∈ {1.2, 1.0, 0.8}), we get 3 kink values per axis = 9 planes total.
The kink points are intersections of these planes within the feasible region.

Usage:
    python plot_farmer_kinks.py
"""
import itertools
import webbrowser
import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

# ── Problem parameters (same as run_farmer_case.py) ──
SCENARIOS = {"good": 1.2, "fair": 1.0, "bad": 0.8}
PLANTING_COST  = {"wheat": 150, "corn": 230, "beets": 260}
SELLING_PRICE  = {"wheat": 170, "corn": 150, "beets_fav": 36, "beets_unfav": 10}
PURCHASE_PRICE = {"wheat": 238, "corn": 210}
MIN_REQ        = {"wheat": 200, "corn": 240}
BEETS_QUOTA    = 6000
TOTAL_ACRES    = 500
YIELD_BASE     = {"wheat": 2.5, "corn": 3.0, "beets": 20.0}


def compute_Q_scenario(xw, xc, xb, ym):
    """Second-stage recourse cost for one scenario."""
    wp = YIELD_BASE["wheat"] * ym * xw
    cp = YIELD_BASE["corn"]  * ym * xc
    bp = YIELD_BASE["beets"] * ym * xb
    cost = 0.0
    # Wheat requirement
    if wp >= MIN_REQ["wheat"]:
        cost -= (wp - MIN_REQ["wheat"]) * SELLING_PRICE["wheat"]
    else:
        cost += (MIN_REQ["wheat"] - wp) * PURCHASE_PRICE["wheat"]
    # Corn requirement
    if cp >= MIN_REQ["corn"]:
        cost -= (cp - MIN_REQ["corn"]) * SELLING_PRICE["corn"]
    else:
        cost += (MIN_REQ["corn"] - cp) * PURCHASE_PRICE["corn"]
    # Beets: favorable up to quota, unfavorable above
    if bp <= BEETS_QUOTA:
        cost -= bp * SELLING_PRICE["beets_fav"]
    else:
        cost -= BEETS_QUOTA * SELLING_PRICE["beets_fav"]
        cost -= (bp - BEETS_QUOTA) * SELLING_PRICE["beets_unfav"]
    return cost


def compute_EQ(xw, xc, xb):
    """Expected total cost = planting + (1/3) * sum(recourse)."""
    plant = PLANTING_COST["wheat"]*xw + PLANTING_COST["corn"]*xc + PLANTING_COST["beets"]*xb
    rec = sum(compute_Q_scenario(xw, xc, xb, ym) for ym in SCENARIOS.values())
    return plant + rec / len(SCENARIOS)


def compute_kink_values():
    """Return dict of kink values per axis."""
    kinks = {"wheat": set(), "corn": set(), "beets": set()}
    for ym in SCENARIOS.values():
        kinks["wheat"].add(round(MIN_REQ["wheat"] / (YIELD_BASE["wheat"] * ym), 10))
        kinks["corn"].add(round(MIN_REQ["corn"] / (YIELD_BASE["corn"] * ym), 10))
        kinks["beets"].add(round(BEETS_QUOTA / (YIELD_BASE["beets"] * ym), 10))
    return {k: sorted(v) for k, v in kinks.items()}


def is_feasible(xw, xc, xb, tol=1e-8):
    return xw >= -tol and xc >= -tol and xb >= -tol and (xw + xc + xb) <= TOTAL_ACRES + tol


def compute_kink_points(kinks):
    """All grid intersection points of kink planes within feasible region."""
    points = []
    for xw in kinks["wheat"]:
        for xc in kinks["corn"]:
            for xb in kinks["beets"]:
                if is_feasible(xw, xc, xb):
                    points.append((xw, xc, xb))
    return points


def find_optimal():
    """
    Brute-force search over a fine grid + kink intersections
    to find the optimal point.
    """
    kinks = compute_kink_values()

    # Candidate points: all kink grid intersections
    candidates = compute_kink_points(kinks)

    # Also add a fine grid search within the feasible region
    grid_vals = {"wheat": list(kinks["wheat"]) + list(np.linspace(0, TOTAL_ACRES, 50)),
                 "corn":  list(kinks["corn"])  + list(np.linspace(0, TOTAL_ACRES, 50)),
                 "beets": list(kinks["beets"]) + list(np.linspace(0, TOTAL_ACRES, 50))}

    for xw in grid_vals["wheat"]:
        for xc in grid_vals["corn"]:
            for xb in grid_vals["beets"]:
                if is_feasible(xw, xc, xb):
                    candidates.append((xw, xc, xb))

    # Evaluate all candidates
    best_cost = float("inf")
    best_pt = None
    for pt in candidates:
        cost = compute_EQ(*pt)
        if cost < best_cost:
            best_cost = cost
            best_pt = pt

    return best_pt, best_cost


def main():
    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir / "results" / "farmer_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute kink planes
    kinks = compute_kink_values()
    print("=== Kink Planes ===")
    for axis, vals in kinks.items():
        for v in vals:
            print(f"  {axis:6s} = {v:.4f} acres")

    # Compute kink intersection points
    kink_pts = compute_kink_points(kinks)
    print(f"\n=== Kink Intersection Points ({len(kink_pts)} feasible) ===")

    # Evaluate EQ at each
    kink_costs = []
    for pt in kink_pts:
        cost = compute_EQ(*pt)
        kink_costs.append(cost)
        print(f"  ({pt[0]:8.2f}, {pt[1]:8.2f}, {pt[2]:8.2f})  EQ = {cost:12.2f}")

    # Find optimal
    opt_pt, opt_cost = find_optimal()
    print(f"\n=== Optimal Point ===")
    print(f"  x* = ({opt_pt[0]:.4f}, {opt_pt[1]:.4f}, {opt_pt[2]:.4f})")
    print(f"  EQ(x*) = {opt_cost:.4f}")
    print(f"  EQ(x*) per scenario = {opt_cost / len(SCENARIOS):.4f}")

    # ── Build 3D Plotly figure ──
    fig = go.Figure()

    kk = np.array(kink_pts)
    kc = np.array(kink_costs)

    # Color kink points by EQ value
    fig.add_trace(go.Scatter3d(
        x=kk[:, 0], y=kk[:, 1], z=kk[:, 2],
        mode="markers",
        marker=dict(
            size=6, color=kc, colorscale="RdYlGn_r",
            colorbar=dict(title="EQ cost", x=1.05),
            line=dict(width=1, color="black"),
        ),
        customdata=kc,
        hovertemplate=(
            "wheat: %{x:.2f}<br>corn: %{y:.2f}<br>beets: %{z:.2f}<br>"
            "EQ: %{customdata:.2f}<extra>kink point</extra>"
        ),
        name="Kink intersection points",
    ))

    # Mark optimal point
    fig.add_trace(go.Scatter3d(
        x=[opt_pt[0]], y=[opt_pt[1]], z=[opt_pt[2]],
        mode="markers+text",
        marker=dict(size=12, color="gold", symbol="diamond",
                    line=dict(width=2, color="black")),
        text=[f"Optimal: EQ={opt_cost:.0f}"],
        textposition="top center",
        textfont=dict(size=11, color="gold"),
        hovertemplate=(
            f"wheat: {opt_pt[0]:.2f}<br>corn: {opt_pt[1]:.2f}<br>"
            f"beets: {opt_pt[2]:.2f}<br>EQ: {opt_cost:.2f}"
            "<extra>OPTIMAL</extra>"
        ),
        name="Optimal solution",
    ))

    # Draw kink planes as semi-transparent surfaces
    plane_colors = {"wheat": "rgba(255,100,100,0.1)",
                    "corn":  "rgba(100,255,100,0.1)",
                    "beets": "rgba(100,100,255,0.1)"}
    plane_edge = {"wheat": "red", "corn": "green", "beets": "blue"}

    for axis, vals in kinks.items():
        for v in vals:
            # Draw a square representing the plane within [0, 500]²
            if axis == "wheat":
                xs = [v, v, v, v]
                ys = [0, TOTAL_ACRES, TOTAL_ACRES, 0]
                zs = [0, 0, TOTAL_ACRES, TOTAL_ACRES]
            elif axis == "corn":
                xs = [0, TOTAL_ACRES, TOTAL_ACRES, 0]
                ys = [v, v, v, v]
                zs = [0, 0, TOTAL_ACRES, TOTAL_ACRES]
            else:  # beets
                xs = [0, TOTAL_ACRES, TOTAL_ACRES, 0]
                ys = [0, 0, TOTAL_ACRES, TOTAL_ACRES]
                zs = [v, v, v, v]

            fig.add_trace(go.Mesh3d(
                x=xs, y=ys, z=zs,
                i=[0, 0], j=[1, 2], k=[2, 3],
                color=plane_colors[axis],
                opacity=0.15,
                name=f"{axis}={v:.1f}",
                showlegend=True,
                hoverinfo="name",
            ))

    # Draw feasible region boundary (tetrahedron: x+y+z ≤ 500)
    fig.add_trace(go.Mesh3d(
        x=[0, TOTAL_ACRES, 0, 0],
        y=[0, 0, TOTAL_ACRES, 0],
        z=[0, 0, 0, TOTAL_ACRES],
        i=[0, 0, 0, 1], j=[1, 1, 2, 2], k=[2, 3, 3, 3],
        color="lightgray", opacity=0.08,
        name="Feasible region (x₁+x₂+x₃≤500)",
        showlegend=True,
    ))

    fig.update_layout(
        title=dict(text="Farmer Problem — Kink Points & Optimal",
                   font=dict(size=18)),
        scene=dict(
            xaxis_title="wheat (acres)",
            yaxis_title="corn (acres)",
            zaxis_title="beets (acres)",
            aspectmode="cube",
            bgcolor="#1a1a2e",
            xaxis=dict(gridcolor="gray", zerolinecolor="gray"),
            yaxis=dict(gridcolor="gray", zerolinecolor="gray"),
            zaxis=dict(gridcolor="gray", zerolinecolor="gray"),
        ),
        width=1100, height=800,
        paper_bgcolor="#16213e",
        font=dict(color="#eee"),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)"),
    )

    # Annotation with summary
    fig.add_annotation(
        x=0.02, y=0.02, xref="paper", yref="paper",
        text=(
            f"Kink planes: wheat={kinks['wheat']}, corn={kinks['corn']}, beets={kinks['beets']}<br>"
            f"Feasible kink intersections: {len(kink_pts)}<br>"
            f"Optimal: ({opt_pt[0]:.2f}, {opt_pt[1]:.2f}, {opt_pt[2]:.2f}), EQ={opt_cost:.2f}"
        ),
        showarrow=False, align="left",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black", borderwidth=1,
        font=dict(size=10, color="black"),
    )

    # Save as HTML and open
    html_path = str(out_dir / "farmer_kink_points_3d.html")
    fig.write_html(html_path)
    print(f"\nSaved: {html_path}")
    webbrowser.open(f"file:///{os.path.abspath(html_path)}")


if __name__ == "__main__":
    main()
