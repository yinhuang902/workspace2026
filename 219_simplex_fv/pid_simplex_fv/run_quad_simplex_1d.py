#!/usr/bin/env python
"""
run_quad_simplex_1d.py
======================
1D simplex-style interval refinement for the Quad stochastic program
from the SNoGloDe examples.

Problem (matching snoglode/examples/quad/quad.py):

    min  E[Q(x)] = sum_s p_s * (x - c_s)^2
    s.t. -3 <= x <= 3

Scenarios:
    s=0 ('scenario1'): c=0, p=0.1
    s=1 ('scenario2'): c=1, p=0.1
    s=2 ('scenario3'): c=2, p=0.8

Known optimum:
    x* = sum p_s*c_s = 1.7
    F*  = sum p_s*(x*-c_s)^2 = 0.41

In 1D the "simplex" is an interval [a, b].
  - Affine surrogate A_s(x): linear interpolant of Q_s through (a, Q_s(a)) and (b, Q_s(b)).
  - ms_s = min_{x in [a,b]} [ Q_s(x) - A_s(x) ]   (<= 0 for convex Q_s)
  - Interval LB = min_{x in [a,b]} sum_s p_s * [A_s(x) + ms_s]

Per-iteration plots show Q_s(x), A_s(x), A_s(x)+ms for each scenario.
"""

import io
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Windows console fix
# ---------------------------------------------------------------------------
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding=sys.stdout.encoding, errors="replace")

# ---------------------------------------------------------------------------
# Problem parameters  (must match snoglode quad.py exactly)
# ---------------------------------------------------------------------------
SCENARIO_NAMES = ['scenario1', 'scenario2', 'scenario3']
S = len(SCENARIO_NAMES)

C_VALUES = [0, 1, 2]                # per-scenario centre
PROBS    = [0.1, 0.1, 0.8]          # per-scenario probability

X_LB, X_UB = -3.0, 3.0              # first-stage bounds

# Analytical optimum
X_OPT = sum(p * c for p, c in zip(PROBS, C_VALUES))       # 1.7
F_OPT = sum(p * (X_OPT - c)**2 for p, c in zip(PROBS, C_VALUES))  # 0.41


# ---------------------------------------------------------------------------
# Q_s(x)  and  E[Q(x)]
# ---------------------------------------------------------------------------
def Q_s(x, s):
    """Q_s(x) = (x - c_s)^2."""
    return (x - C_VALUES[s]) ** 2


def F_exp(x):
    """E[Q(x)] = sum_s p_s * Q_s(x)."""
    return sum(PROBS[s] * Q_s(x, s) for s in range(S))


# ---------------------------------------------------------------------------
# Affine surrogate on interval [xa, xb]
# ---------------------------------------------------------------------------
def build_surrogate(xa, xb, s):
    """Linear interpolant A_s(x) = intercept + slope * x  on [xa, xb].

    A_s(xa) = Q_s(xa),  A_s(xb) = Q_s(xb).
    """
    qa, qb = Q_s(xa, s), Q_s(xb, s)
    slope = (qb - qa) / (xb - xa)
    intercept = qa - slope * xa
    return intercept, slope


def eval_surrogate(x, intercept, slope):
    return intercept + slope * x


# ---------------------------------------------------------------------------
# ms_s on interval [xa, xb]
# ---------------------------------------------------------------------------
def solve_ms(xa, xb, s):
    """Compute ms_s = min_{x in [xa,xb]} [Q_s(x) - A_s(x)].

    For Q_s(x) = (x-c)^2 (quadratic, leading coeff = +1):
        Q_s(x) - A_s(x) = (x - xa)(x - xb)   (independent of c!)
        argmin = (xa + xb)/2   (midpoint)
        ms = -(xb - xa)^2 / 4

    Returns (ms_value, argmin_x).
    """
    intercept, slope = build_surrogate(xa, xb, s)

    # Analytical for any quadratic with leading coeff = 1
    x_star = 0.5 * (xa + xb)  # midpoint
    x_star = max(xa, min(xb, x_star))
    ms_val = Q_s(x_star, s) - eval_surrogate(x_star, intercept, slope)

    return ms_val, x_star


# ---------------------------------------------------------------------------
# Interval LB
# ---------------------------------------------------------------------------
def interval_lb(xa, xb, surrogates, ms_vals):
    """LB = min_{x in [xa,xb]} sum_s p_s * (A_s(x) + ms_s).

    Since sum_s p_s * A_s(x) is affine, the min is at an endpoint.
    """
    val_a = sum(PROBS[s] * (eval_surrogate(xa, *surrogates[s]) + ms_vals[s])
                for s in range(S))
    val_b = sum(PROBS[s] * (eval_surrogate(xb, *surrogates[s]) + ms_vals[s])
                for s in range(S))
    return min(val_a, val_b)


# ---------------------------------------------------------------------------
# Per-iteration plot
# ---------------------------------------------------------------------------
_COLORS_Q = ['#e74c3c', '#3498db', '#2ecc71']     # curve
_COLORS_A = ['#c0392b', '#2980b9', '#27ae60']     # surrogate
_COLORS_M = ['#f39c12', '#8e44ad', '#16a085']     # A+ms


def _plot_iteration(it, nodes, ivl_data, ub, lb, best_x, sel_idx, plot_dir):
    """Plot Q_s, A_s, A_s+ms for each scenario + expected value."""

    x_fine = np.linspace(X_LB - 0.3, X_UB + 0.3, 600)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()   # [s0, s1, s2, expected]

    for s in range(S):
        ax = axes[s]

        # --- Ground truth ---
        y_q = np.array([Q_s(x, s) for x in x_fine])
        ax.plot(x_fine, y_q, color=_COLORS_Q[s], lw=2.5,
                label=f'$Q_{{{s+1}}}(x) = (x-{C_VALUES[s]})^2$', zorder=3)

        # --- Per-interval surrogate & underestimator ---
        for idx, d in enumerate(ivl_data):
            xa, xb = d['xa'], d['xb']
            x_seg = np.array([xa, xb])
            interc, slp = d['surrogates'][s]
            y_a   = eval_surrogate(x_seg, interc, slp)
            y_ams = y_a + d['ms_values'][s]

            lbl_a = '$A_s(x)$' if idx == 0 else None
            lbl_m = '$A_s+m_s$' if idx == 0 else None

            ax.plot(x_seg, y_a, '--', color=_COLORS_A[s], lw=1.8,
                    label=lbl_a, zorder=4)
            ax.plot(x_seg, y_ams, ':', color=_COLORS_M[s], lw=1.8,
                    label=lbl_m, zorder=4)
            # fill between A_s and A_s+ms
            ax.fill_between(x_seg, y_ams, y_a,
                            color=_COLORS_M[s], alpha=0.10, zorder=2)

        # --- Highlight selected interval ---
        if sel_idx is not None:
            sd = ivl_data[sel_idx]
            ax.axvspan(sd['xa'], sd['xb'],
                       color='yellow', alpha=0.18, zorder=1)

        # --- Nodes ---
        for n in nodes:
            ax.plot(n, Q_s(n, s), 'ko', ms=5, zorder=6)
        ax.plot(best_x, Q_s(best_x, s), 'r*', ms=14, zorder=7)

        ax.set_title(f'Scenario {s+1}:  $c={C_VALUES[s]}$,  $p={PROBS[s]}$',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('$x$')
        ax.set_ylabel(f'$Q_{{{s+1}}}(x)$')
        ax.legend(fontsize=8, loc='upper center')
        ax.grid(alpha=0.25)
        ax.set_xlim(X_LB - 0.3, X_UB + 0.3)

    # ---- Expected value subplot ----
    ax = axes[S]
    y_f = np.array([F_exp(x) for x in x_fine])
    ax.plot(x_fine, y_f, 'k-', lw=2.5, label='$E[Q(x)]$', zorder=3)

    for idx, d in enumerate(ivl_data):
        xa, xb = d['xa'], d['xb']
        x_seg = np.array([xa, xb])
        y_ea = np.zeros(2)
        y_eams = np.zeros(2)
        for s in range(S):
            interc, slp = d['surrogates'][s]
            y_ea += PROBS[s] * eval_surrogate(x_seg, interc, slp)
            y_eams += PROBS[s] * (eval_surrogate(x_seg, interc, slp)
                                  + d['ms_values'][s])
        ax.plot(x_seg, y_ea, '--', color='#e67e22', lw=1.8,
                label='$E[A(x)]$' if idx == 0 else None, zorder=4)
        ax.plot(x_seg, y_eams, ':', color='#8e44ad', lw=1.8,
                label='$E[A+m_s]$' if idx == 0 else None, zorder=4)
        ax.fill_between(x_seg, y_eams, y_ea,
                        color='#8e44ad', alpha=0.08, zorder=2)

    if sel_idx is not None:
        sd = ivl_data[sel_idx]
        ax.axvspan(sd['xa'], sd['xb'], color='yellow', alpha=0.18, zorder=1)

    for n in nodes:
        ax.plot(n, F_exp(n), 'ko', ms=5, zorder=6)
    ax.plot(best_x, F_exp(best_x), 'r*', ms=14, zorder=7)
    ax.axvline(X_OPT, color='green', ls='--', lw=1.2, alpha=0.7,
               label=f'$x^*={X_OPT}$')

    gap = ub - lb
    ax.set_title(f'Expected:  UB={ub:.4f}   LB={lb:.4f}   gap={gap:.4f}',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$E[Q(x)]$')
    ax.legend(fontsize=8, loc='upper center')
    ax.grid(alpha=0.25)
    ax.set_xlim(X_LB - 0.3, X_UB + 0.3)

    fig.suptitle(f'Iteration {it}  —  {len(nodes)} nodes',
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(plot_dir / f'iter_{it:03d}.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Convergence plot
# ---------------------------------------------------------------------------
def _plot_convergence(ub_hist, lb_hist, node_hist, plot_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax1.plot(node_hist, ub_hist, 'ro-', ms=5, label='UB (best F)')
    ax1.plot(node_hist, lb_hist, 'bs-', ms=5, label='Simplex LB')
    ax1.axhline(F_OPT, color='green', ls='--', lw=1.5,
                label=f'$F^*={F_OPT:.4f}$')
    ax1.set_xlabel('# Nodes')
    ax1.set_ylabel('Objective')
    ax1.set_title('Convergence')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    gaps = [u - l for u, l in zip(ub_hist, lb_hist)]
    ax2.semilogy(node_hist, [max(g, 1e-16) for g in gaps], 'mo-', ms=5)
    ax2.set_xlabel('# Nodes')
    ax2.set_ylabel('UB – LB')
    ax2.set_title('Gap')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(plot_dir / 'convergence.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Core driver
# ---------------------------------------------------------------------------
def run_1d_simplex(max_iters=20, gap_tol=1e-6, plot_dir=None):
    """Run 1D simplex refinement on the Quad problem."""

    if plot_dir is None:
        plot_dir = Path.cwd() / "quad_simplex_plots"
    else:
        plot_dir = Path(plot_dir)
    if plot_dir.exists():
        shutil.rmtree(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ---------- initial mesh ----------
    nodes = [X_LB, X_UB]
    q_at = {}           # x -> {s: Q_s(x)}
    f_at = {}           # x -> F(x)

    for x in nodes:
        q_at[x] = {s: Q_s(x, s) for s in range(S)}
        f_at[x] = F_exp(x)

    print("=" * 62)
    print("  Quad 1D Simplex Refinement")
    print(f"  x in [{X_LB}, {X_UB}],  {S} scenarios")
    print(f"  Known:  x* = {X_OPT:.4f},  F* = {F_OPT:.6f}")
    print("=" * 62)

    ub_hist, lb_hist, node_hist = [], [], []

    # ---------- main loop ----------
    for it in range(max_iters):
        nodes_sorted = sorted(nodes)
        n_nodes = len(nodes_sorted)
        n_ivl   = n_nodes - 1

        # ---- UB ----
        ub      = min(f_at.values())
        best_x  = min(f_at, key=f_at.get)

        # ---- per-interval data ----
        ivl_data = []
        for i in range(n_ivl):
            xa, xb = nodes_sorted[i], nodes_sorted[i + 1]
            surrogates, ms_vals, argmins = [], [], []
            for s in range(S):
                surrogates.append(build_surrogate(xa, xb, s))
                ms_v, xm = solve_ms(xa, xb, s)
                ms_vals.append(ms_v)
                argmins.append(xm)
            lb_ivl = interval_lb(xa, xb, surrogates, ms_vals)
            ivl_data.append(dict(xa=xa, xb=xb,
                                 surrogates=surrogates,
                                 ms_values=ms_vals,
                                 argmins=argmins,
                                 lb=lb_ivl))

        # ---- global LB ----
        global_lb = min(d['lb'] for d in ivl_data)
        gap       = ub - global_lb

        ub_hist.append(ub)
        lb_hist.append(global_lb)
        node_hist.append(n_nodes)

        # ---- select interval with min LB ----
        sel_idx = min(range(n_ivl), key=lambda i: ivl_data[i]['lb'])
        sel = ivl_data[sel_idx]

        print(f"[Iter {it:3d}] nodes={n_nodes}  "
              f"UB={ub:.6f}  LB={global_lb:.6f}  gap={gap:.6f}  "
              f"sel=[{sel['xa']:.4f},{sel['xb']:.4f}]")

        # ---- plot ----
        _plot_iteration(it, nodes_sorted, ivl_data, ub, global_lb,
                        best_x, sel_idx, plot_dir)

        # ---- converged? ----
        if gap <= gap_tol:
            print(f"  CONVERGED (gap={gap:.2e} <= {gap_tol})")
            break

        # ---- insertion point: ms argmin (midpoint for quadratic) ----
        worst_s = min(range(S), key=lambda s: sel['ms_values'][s])
        ins_x = sel['argmins'][worst_s]

        if any(abs(ins_x - n) < 1e-12 for n in nodes_sorted):
            print("  [WARN] insertion point at existing node; stop")
            break

        print(f"    -> insert x = {ins_x:.6f}   "
              f"ms = {sel['ms_values'][worst_s]:.6f}")

        nodes.append(ins_x)
        q_at[ins_x] = {s: Q_s(ins_x, s) for s in range(S)}
        f_at[ins_x] = F_exp(ins_x)

    # ---------- final results ----------
    final_ub   = min(f_at.values())
    final_best = min(f_at, key=f_at.get)

    print(f"\n{'=' * 62}")
    print(f"  RESULTS  ({len(nodes)} nodes)")
    print(f"  Best x  = {final_best:.6f}   F(x) = {final_ub:.6f}")
    print(f"  Known x*= {X_OPT:.4f}        F*   = {F_OPT:.6f}")
    print(f"  |x-x*|  = {abs(final_best - X_OPT):.6f}")
    print(f"  Plots:   {plot_dir}")
    print(f"{'=' * 62}")

    _plot_convergence(ub_hist, lb_hist, node_hist, plot_dir)

    return {
        'nodes': sorted(nodes),
        'f_at': dict(f_at),
        'best_x': final_best,
        'best_f': final_ub,
        'ub_hist': ub_hist,
        'lb_hist': lb_hist,
    }


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def _in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


if __name__ == "__main__":
    if _in_notebook():
        results = run_1d_simplex(max_iters=15, gap_tol=1e-4)
    else:
        matplotlib.use("Agg")
        results = run_1d_simplex(max_iters=15, gap_tol=1e-4)
