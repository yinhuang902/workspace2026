"""
plot_farmer_Q.py — Visualize the farmer problem's expected Q(x1, x2, x3).

Since Q lives in 3D, we plot 2D heatmap slices by fixing one variable.
Also plots the convergence history from simplex_result.csv.

Usage:
    python plot_farmer_Q.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path

# ── Farmer problem parameters ──
SCENARIOS = {
    "good": {"yield": 1.2},
    "fair": {"yield": 1.0},
    "bad":  {"yield": 0.8},
}

# Costs
PLANTING_COST  = {"wheat": 150, "corn": 230, "beets": 260}
SELLING_PRICE  = {"wheat": 170, "corn": 150, "beets_fav": 36, "beets_unfav": 10}
PURCHASE_PRICE = {"wheat": 238, "corn": 210}
MIN_REQ        = {"wheat": 200, "corn": 240}
BEETS_QUOTA    = 6000
TOTAL_ACRES    = 500

def compute_Q_scenario(x_wheat, x_corn, x_beets, yield_mult):
    """
    Compute optimal recourse cost for one scenario given first-stage x.
    Returns the second-stage cost (selling revenue - purchase cost).
    
    For each scenario, given planted acres x and yield_mult:
      - wheat produced = 2.5 * yield_mult * x_wheat
      - corn  produced = 3.0 * yield_mult * x_corn
      - beets produced = 20  * yield_mult * x_beets
    Then optimally buy/sell to meet requirements and maximize revenue.
    """
    wheat_prod = 2.5 * yield_mult * x_wheat
    corn_prod  = 3.0 * yield_mult * x_corn
    beets_prod = 20.0 * yield_mult * x_beets
    
    cost = 0.0
    
    # --- Wheat: must have >= 200 tons ---
    if wheat_prod >= MIN_REQ["wheat"]:
        # Sell excess
        cost -= (wheat_prod - MIN_REQ["wheat"]) * SELLING_PRICE["wheat"]
    else:
        # Buy deficit
        cost += (MIN_REQ["wheat"] - wheat_prod) * PURCHASE_PRICE["wheat"]
    
    # --- Corn: must have >= 240 tons ---
    if corn_prod >= MIN_REQ["corn"]:
        cost -= (corn_prod - MIN_REQ["corn"]) * SELLING_PRICE["corn"]
    else:
        cost += (MIN_REQ["corn"] - corn_prod) * PURCHASE_PRICE["corn"]
    
    # --- Beets: sell at favorable price up to quota, unfavorable above ---
    if beets_prod <= BEETS_QUOTA:
        cost -= beets_prod * SELLING_PRICE["beets_fav"]
    else:
        cost -= BEETS_QUOTA * SELLING_PRICE["beets_fav"]
        cost -= (beets_prod - BEETS_QUOTA) * SELLING_PRICE["beets_unfav"]
    
    return cost


def compute_EQ(x_wheat, x_corn, x_beets):
    """
    Compute E[Q] = (1/3) * sum over scenarios of:
      planting_cost + Q_scenario(x)
    
    Total cost = planting_cost + (1/3) * sum(recourse costs)
    """
    # First-stage planting cost
    planting = (PLANTING_COST["wheat"] * x_wheat +
                PLANTING_COST["corn"]  * x_corn +
                PLANTING_COST["beets"] * x_beets)
    
    # Expected recourse cost (equal probability = 1/3 each)
    recourse = 0.0
    for sname, sinfo in SCENARIOS.items():
        recourse += compute_Q_scenario(x_wheat, x_corn, x_beets, sinfo["yield"])
    
    # Total expected cost (per-scenario average)
    return planting + recourse / 3.0


def plot_2d_slices(fixed_vals=None, N=200):
    """
    Plot three 2D heatmaps, each fixing one variable.
    fixed_vals: dict with keys 'wheat', 'corn', 'beets' for the fixed values.
                Defaults to the known optimal point.
    """
    if fixed_vals is None:
        # Known near-optimal: x_wheat≈120, x_corn≈80, x_beets≈300
        fixed_vals = {"wheat": 120.0, "corn": 80.0, "beets": 300.0}
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    
    configs = [
        ("wheat", "corn",  "beets", fixed_vals["beets"]),
        ("wheat", "beets", "corn",  fixed_vals["corn"]),
        ("corn",  "beets", "wheat", fixed_vals["wheat"]),
    ]
    
    for ax, (xname, yname, fixed_name, fixed_val) in zip(axes, configs):
        xs = np.linspace(0, TOTAL_ACRES, N)
        ys = np.linspace(0, TOTAL_ACRES, N)
        X, Y = np.meshgrid(xs, ys)
        Z = np.full_like(X, np.nan)
        
        for i in range(N):
            for j in range(N):
                xv, yv = X[i, j], Y[i, j]
                # Check feasibility: sum <= 500
                vals = {xname: xv, yname: yv, fixed_name: fixed_val}
                if vals["wheat"] + vals["corn"] + vals["beets"] > TOTAL_ACRES + 1e-8:
                    continue
                if any(v < 0 for v in vals.values()):
                    continue
                Z[i, j] = compute_EQ(vals["wheat"], vals["corn"], vals["beets"])
        
        # Plot heatmap
        im = ax.contourf(X, Y, Z, levels=50, cmap='RdYlGn_r')
        plt.colorbar(im, ax=ax, shrink=0.8, label='E[cost]')
        
        # Add contour lines
        valid = ~np.isnan(Z)
        if valid.any():
            ax.contour(X, Y, np.where(valid, Z, np.nanmean(Z)), 
                      levels=20, colors='k', linewidths=0.3, alpha=0.5)
        
        # Mark the known optimal
        opt = {"wheat": 120.0, "corn": 80.0, "beets": 300.0}
        ax.plot(opt[xname], opt[yname], 'w*', markersize=15, 
                markeredgecolor='black', markeredgewidth=1.5, label='Optimal')
        
        ax.set_xlabel(f'{xname} (acres)', fontsize=11)
        ax.set_ylabel(f'{yname} (acres)', fontsize=11)
        ax.set_title(f'E[cost] | {fixed_name}={fixed_val:.0f} acres', fontsize=12)
        ax.legend(loc='upper right', fontsize=9)
    
    fig.suptitle('Farmer Problem: Expected Cost E[Q(x)] — 2D Slices', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_1d_slices():
    """
    Plot Q along each axis individually (fixing the other two at the optimal).
    This clearly shows the kink points.
    """
    opt = {"wheat": 120.0, "corn": 80.0, "beets": 300.0}
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, (vary_name, other_names) in zip(axes, [
        ("wheat", ["corn", "beets"]),
        ("corn",  ["wheat", "beets"]),
        ("beets", ["wheat", "corn"]),
    ]):
        xs = np.linspace(0, TOTAL_ACRES, 2000)
        total_q = []
        per_scen = {s: [] for s in SCENARIOS}
        
        for x in xs:
            vals = {vary_name: x}
            for on in other_names:
                vals[on] = opt[on]
            
            if vals["wheat"] + vals["corn"] + vals["beets"] > TOTAL_ACRES + 1e-8:
                total_q.append(np.nan)
                for s in SCENARIOS:
                    per_scen[s].append(np.nan)
                continue
            
            eq = compute_EQ(vals["wheat"], vals["corn"], vals["beets"])
            total_q.append(eq)
            
            planting = (PLANTING_COST["wheat"] * vals["wheat"] +
                       PLANTING_COST["corn"]  * vals["corn"] +
                       PLANTING_COST["beets"] * vals["beets"])
            for s, sinfo in SCENARIOS.items():
                q_s = planting + compute_Q_scenario(
                    vals["wheat"], vals["corn"], vals["beets"], sinfo["yield"])
                per_scen[s].append(q_s)
        
        xs_arr = np.array(xs)
        
        # Plot per-scenario costs
        colors = {"good": "#2ecc71", "fair": "#3498db", "bad": "#e74c3c"}
        for s in SCENARIOS:
            ax.plot(xs_arr, per_scen[s], '-', color=colors[s], 
                   alpha=0.4, linewidth=1, label=f'{s} scenario')
        
        # Plot expected cost (bold)
        ax.plot(xs_arr, total_q, 'k-', linewidth=2.5, label='E[cost]')
        
        # Mark kink points
        yield_mults = [1.2, 1.0, 0.8]
        crop_yield_base = {"wheat": 2.5, "corn": 3.0, "beets": 20.0}
        req = {"wheat": 200, "corn": 240, "beets": BEETS_QUOTA}
        
        kink_xs = []
        if vary_name in ["wheat", "corn"]:
            for ym in yield_mults:
                kink = req[vary_name] / (crop_yield_base[vary_name] * ym)
                if 0 < kink < TOTAL_ACRES:
                    kink_xs.append(kink)
        elif vary_name == "beets":
            for ym in yield_mults:
                kink = BEETS_QUOTA / (crop_yield_base["beets"] * ym)
                if 0 < kink < TOTAL_ACRES:
                    kink_xs.append(kink)
        
        for kx in sorted(set(round(k, 2) for k in kink_xs)):
            ax.axvline(kx, color='red', linestyle='--', alpha=0.6, linewidth=1)
            ax.text(kx, ax.get_ylim()[0], f'{kx:.1f}', 
                   color='red', fontsize=8, ha='center', va='bottom')
        
        ax.set_xlabel(f'{vary_name} (acres)', fontsize=11)
        ax.set_ylabel('Cost ($)', fontsize=11)
        others_str = ', '.join(f'{n}={opt[n]:.0f}' for n in other_names)
        ax.set_title(f'Cost vs {vary_name} ({others_str})', fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Farmer Problem: 1D Slices — Kink Points Shown in Red', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_convergence(csv_path):
    """Plot LB/UB convergence from simplex_result.csv."""
    import csv
    
    times, lbs, ubs = [], [], []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) < 4:
                continue
            times.append(float(row[0]))
            lbs.append(float(row[2]))
            ubs.append(float(row[3]))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    iters = range(len(lbs))
    
    # Plot 1: LB/UB vs iteration
    ax1.plot(iters, lbs, 'b-', linewidth=1.5, label='LB (surrogate)')
    ax1.plot(iters, ubs, 'r-', linewidth=1.5, label='UB (incumbent)')
    ax1.axhline(-108390, color='green', linestyle=':', linewidth=1, label='Known optimal')
    ax1.fill_between(iters, lbs, ubs, alpha=0.1, color='gray')
    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('Cost (per scenario)', fontsize=11)
    ax1.set_title('Convergence: LB and UB', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Gap vs iteration
    gaps = [ub - lb for lb, ub in zip(lbs, ubs)]
    ax2.semilogy(iters, gaps, 'k-', linewidth=1.5)
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('Gap (UB - LB)', fontsize=11)
    ax2.set_title('Gap Convergence (log scale)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle('Simplex Method Convergence on Farmer Problem', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir / "results" / "farmer_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 1D slices (best for seeing kink structure) ---
    print("Generating 1D slice plots...")
    fig1 = plot_1d_slices()
    fig1.savefig(out_dir / "farmer_Q_1d_slices.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {out_dir / 'farmer_Q_1d_slices.png'}")
    
    # --- 2D heatmap slices ---
    print("Generating 2D heatmap slices...")
    fig2 = plot_2d_slices()
    fig2.savefig(out_dir / "farmer_Q_2d_slices.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {out_dir / 'farmer_Q_2d_slices.png'}")
    
    # --- Convergence plot ---
    csv_path = out_dir / "simplex_result.csv"
    if csv_path.exists():
        print("Generating convergence plot...")
        fig3 = plot_convergence(csv_path)
        fig3.savefig(out_dir / "farmer_convergence.png", dpi=150, bbox_inches='tight')
        print(f"  Saved: {out_dir / 'farmer_convergence.png'}")
    else:
        print(f"  CSV not found: {csv_path}")
    
    print("\nDone! Open the PNG files to view.")
    plt.show()
