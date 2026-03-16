import nbformat as nbf
import os

CORE_ALGO = """from dataclasses import dataclass

@dataclass
class Segment:
    Y: Tuple[float, float]
    # Local models for scene 1 and 2
    a1: float
    b1: float
    c1: float
    ms1: float
    y_ms1: float
    
    a2: float
    b2: float
    c2: float
    ms2: float
    y_ms2: float
    
    lb: float # the local expected lower bound of this segment

def fit_quadratic_interpolant(v_func, Y: Tuple[float, float], n_samples: int = 10) -> Tuple[float, float, float]:
    \"\"\"Fits A_s(y) = a*y^2 + b*y + c using a uniform grid of n_samples on interval Y.\"\"\"
    lo, hi = Y
    ys = np.linspace(lo, hi, n_samples)
    fs = np.array([v_func(y) for y in ys])
    
    # Simple Least Squares to fit a quadratic strictly to these evaluate points
    X = np.vstack((ys**2, ys, np.ones_like(ys))).T
    coefs, _, _, _ = np.linalg.lstsq(X, fs, rcond=None)
    a, b, c = coefs
    return a, b, c

def compute_ms_shift(v_func, a: float, b: float, c: float, Y: Tuple[float, float], n_grid: int = 5000) -> Tuple[float, float]:
    \"\"\"Computes ms = min(v_s - A_s) and the point y_ms where this minimum (worst overestimate) occurs.\"\"\"
    lo, hi = Y
    ys = np.linspace(lo, hi, n_grid)
    
    A_s = a*ys**2 + b*ys + c
    v_s = np.array([v_func(y) for y in ys])
    
    diff = v_s - A_s
    ms = np.min(diff)
    y_ms = ys[np.argmin(diff)]
    
    return float(ms), float(y_ms)

def min_of_quadratic_on_interval(a: float, b: float, c: float, Y: Tuple[float, float]) -> float:
    \"\"\"Finds the mathematical minimum of the quadratic a*y^2 + b*y + c over interval Y.\"\"\"
    lo, hi = Y
    vals = [a*lo**2 + b*lo + c, a*hi**2 + b*hi + c]
    if a > 0:
        ystar = -b / (2.0*a)
        if lo <= ystar <= hi:
            vals.append(a*ystar**2 + b*ystar + c)
    return min(vals)

def compute_interval_lower_bound(a1, b1, c1, ms1, a2, b2, c2, ms2, Y) -> float:
    \"\"\"Computes the relaxed lower bound of the expected objective over interval Y.\"\"\"
    expected_a = P1_weight * a1 + P2_weight * a2
    expected_b = P1_weight * b1 + P2_weight * b2
    expected_c = P1_weight * (c1 + ms1) + P2_weight * (c2 + ms2)
    return min_of_quadratic_on_interval(expected_a, expected_b, expected_c, Y)

def create_segment(Y: Tuple[float, float]) -> Segment:
    a1, b1, c1 = fit_quadratic_interpolant(v1, Y, n_samples=10)
    a2, b2, c2 = fit_quadratic_interpolant(v2, Y, n_samples=10)
    
    ms1, y_ms1 = compute_ms_shift(v1, a1, b1, c1, Y)
    ms2, y_ms2 = compute_ms_shift(v2, a2, b2, c2, Y)
    
    lb = compute_interval_lower_bound(a1, b1, c1, ms1, a2, b2, c2, ms2, Y)
    
    return Segment(Y, a1, b1, c1, ms1, y_ms1, a2, b2, c2, ms2, y_ms2, lb)
"""

ITERATIVE_LOOP = """def run_iterative_splitting(Y0: Tuple[float, float], num_iterations: int = 5):
    # Maintain a list of active segments forming a partition of Y0. 
    active_segments = [create_segment(Y0)]
    records = []
    
    # Pre-calculate global upper bound on initial interval Y0
    dense_y = np.linspace(Y0[0], Y0[1], 1000)
    global_ub = np.min([expected_v(y) for y in dense_y])
    
    for k in range(num_iterations):
        # 1. Select the segment with the smallest (worst) lower bound globally
        active_segments.sort(key=lambda s: s.lb)
        worst_seg = active_segments.pop(0)  # Remove it to split it
        
        # 2. Form record and find split point
        global_lb = worst_seg.lb
        y_split = worst_seg.y_ms1 if worst_seg.ms1 <= worst_seg.ms2 else worst_seg.y_ms2
        
        records.append({
            'Iteration': k,
            'Worst_Interval_L': worst_seg.Y[0],
            'Worst_Interval_U': worst_seg.Y[1],
            'ms1': worst_seg.ms1,
            'ms2': worst_seg.ms2,
            'split_point_y': y_split,
            'Global Lower Bound': global_lb,
            'Global Upper Bound': global_ub,
            'Num Segments': len(active_segments) + 2
        })
        
        # 3. Split logic: Create 2 new segments replacing the worst one
        Y = worst_seg.Y
        for new_Y in [(Y[0], y_split), (y_split, Y[1])]:
            if new_Y[1] - new_Y[0] > 1e-6:
                active_segments.append(create_segment(new_Y))
                
        # IMPORTANT: Sort active segments by their domain for plotting left-to-right nicely
        active_segments.sort(key=lambda s: s.Y[0])
                
        # 4. Plotting
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Plot True functions globally
        ys_global = np.linspace(Y0[0], Y0[1], 1000)
        axes[0].plot(ys_global, [v1(y) for y in ys_global], 'k-', lw=4, alpha=0.3, label='True v1')
        axes[1].plot(ys_global, [v2(y) for y in ys_global], 'k-', lw=4, alpha=0.3, label='True v2')
        axes[2].plot(ys_global, [expected_v(y) for y in ys_global], 'k-', lw=3, label='True Expected Total')
        
        # Plot Piecewise Underestimators
        for i, seg in enumerate(active_segments):
            ys_seg = np.linspace(seg.Y[0], seg.Y[1], 100)
            ys_samples = np.linspace(seg.Y[0], seg.Y[1], 10)  # The 10 sample points for WLS
            
            # Scene 1
            lbl_A = 'A1 (PW Interpolant)' if i == 0 else None
            lbl_U = 'U1 (PW Underestimator)' if i == 0 else None
            lbl_S = 'Samples (10 pts)' if i == 0 else None
            
            p1_interp = seg.a1*ys_seg**2 + seg.b1*ys_seg + seg.c1
            p1_under = p1_interp + seg.ms1
            axes[0].plot(ys_seg, p1_interp, '--', lw=2, color='orange', label=lbl_A)
            axes[0].plot(ys_seg, p1_under, 'b-', lw=2, label=lbl_U)
            axes[0].plot(ys_samples, [v1(y) for y in ys_samples], 'ro', markersize=3, label=lbl_S)
            axes[0].axvline(seg.Y[0], color='gray', linestyle=':', alpha=0.5)
            
            # Scene 2
            lbl_A = 'A2 (PW Interpolant)' if i == 0 else None
            lbl_U = 'U2 (PW Underestimator)' if i == 0 else None
            lbl_S = 'Samples (10 pts)' if i == 0 else None
            
            p2_interp = seg.a2*ys_seg**2 + seg.b2*ys_seg + seg.c2
            p2_under = p2_interp + seg.ms2
            axes[1].plot(ys_seg, p2_interp, '--', lw=2, color='orange', label=lbl_A)
            axes[1].plot(ys_seg, p2_under, 'b-', lw=2, label=lbl_U)
            axes[1].plot(ys_samples, [v2(y) for y in ys_samples], 'ro', markersize=3, label=lbl_S)
            axes[1].axvline(seg.Y[0], color='gray', linestyle=':', alpha=0.5)
            
            # Average
            lbl_U = 'Total PW Underestimator' if i == 0 else None
            expected_under = P1_weight * p1_under + P2_weight * p2_under
            axes[2].plot(ys_seg, expected_under, 'b-', lw=2, label=lbl_U)
            axes[2].axvline(seg.Y[0], color='gray', linestyle=':', alpha=0.5)
            
        # Highlight the exact ms point chosen for splitting in this iteration
        axes[2].axvline(y_split, color='purple', linestyle='--', lw=2, label=f'Chosen Split Point ({y_split:.2f})')
        axes[2].plot(y_split, expected_v(y_split), 'm*', markersize=15, label='Worst ms Location')
            
        # Common formatting
        axes[0].set_title(f'Scene 1 (Iter {k})')
        axes[0].legend()
        axes[1].set_title(f'Scene 2 (Iter {k})')
        axes[1].legend()
        axes[2].set_title(f'Average Scenario (Iter {k})')
        axes[2].legend()
        
        plt.tight_layout()
        plt.show()
        
    # Output Summary Table
    df = pd.DataFrame(records)
    print("\\n=== Iteration Summary Table ===")
    print(df.to_string(index=False))
    return df
"""

def create_notebook(name, title, defs):
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell(f"# Iterative Scenario Splitting ({title}) - Piecewise Quadratic\n\nThis notebook implements an iterative algorithm to construct **true piecewise quadratic underestimators** over first-stage variable intervals."))
    nb.cells.append(nbf.v4.new_code_cell("import numpy as np\nimport matplotlib.pyplot as plt\nfrom typing import Tuple, List, Dict\nimport pandas as pd\n\nplt.rcParams['figure.figsize'] = [15, 4]\nplt.rcParams['figure.dpi'] = 100"))
    nb.cells.append(nbf.v4.new_code_cell(defs))
    nb.cells.append(nbf.v4.new_code_cell(CORE_ALGO))
    nb.cells.append(nbf.v4.new_code_cell(ITERATIVE_LOOP))
    nb.cells.append(nbf.v4.new_code_cell("df_summary = run_iterative_splitting((-10.0, 10.0), num_iterations=5)"))

    with open(f'{name}_iterative_split.ipynb', 'w') as f:
        nbf.write(nb, f)

# For P1 and P2
P1_P2_DEFS = """# Scenario 1
def v1(y: float) -> float:
    return -np.abs(y) + 0.5

# Scenario 2
def v2(y: float) -> float:
    return np.abs(y) - 1.0

P1_weight = 0.5
P2_weight = 0.5

# Average expected value function
def expected_v(y: float) -> float:
    return P1_weight * v1(y) + P2_weight * v2(y)
"""

# For P3
P3_DEFS = """# Scenario 1
def v1(y: float) -> float:
    return -np.abs(y) + 0.5

# Scenario 2
def v2(y: float) -> float:
    return np.abs(y) - 1.0

# In P3, weights are P1=1.0 and we just sum them (total v = v1+v2)
P1_weight = 1.0
P2_weight = 1.0

# Average expected value function (or total in this case)
def expected_v(y: float) -> float:
    return P1_weight * v1(y) + P2_weight * v2(y)
"""

create_notebook('P1', 'P1', P1_P2_DEFS)
create_notebook('P2', 'P2', P1_P2_DEFS)
create_notebook('P3', 'P3', P3_DEFS)
print("P1, P2, P3 Piecewise Notebooks saved!")
