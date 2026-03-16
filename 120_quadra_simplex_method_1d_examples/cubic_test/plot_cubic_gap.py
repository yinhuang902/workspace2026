import re
import os
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def parse_summary(filepath):
    """Parse iteration summary txt file and extract iteration numbers, abs gaps, and rel gaps."""
    iterations = []
    abs_gaps = []
    rel_gaps = []

    with open(filepath, 'r') as f:
        content = f.read()

    pattern = r"Iteration (\d+).*?Abs Gap = ([\d.e+-]+).*?Rel Gap = ([\d.e+-]+)"
    matches = re.findall(pattern, content, re.DOTALL)

    for iter_num, agap, rgap in matches:
        iterations.append(int(iter_num))
        abs_gaps.append(float(agap))
        rel_gaps.append(float(rgap))

    return iterations, abs_gaps, rel_gaps

# Parse both files
iters_quad, gaps_quad, rgaps_quad = parse_summary(os.path.join(SCRIPT_DIR, "cubic_quad_iteration_summary.txt"))
iters_lin, gaps_lin, rgaps_lin = parse_summary(os.path.join(SCRIPT_DIR, "cubic_linear_iteration_summary.txt"))

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Absolute gap
ax1.plot(iters_quad, gaps_quad, 'o-', lw=2, markersize=6, color='#2196F3', label='Quadratic Surrogate')
ax1.plot(iters_lin, gaps_lin, 's--', lw=2, markersize=6, color='#FF5722', label='Linear Surrogate')
ax1.set_xlabel('Iteration', fontsize=13)
ax1.set_ylabel('Absolute Gap', fontsize=13)
ax1.set_title('Absolute Gap vs Iteration', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# Relative gap
ax2.plot(iters_quad, rgaps_quad, 'o-', lw=2, markersize=6, color='#2196F3', label='Quadratic Surrogate')
ax2.plot(iters_lin, rgaps_lin, 's--', lw=2, markersize=6, color='#FF5722', label='Linear Surrogate')
ax2.set_xlabel('Iteration', fontsize=13)
ax2.set_ylabel('Relative Gap', fontsize=13)
ax2.set_title('Relative Gap vs Iteration', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

plt.suptitle('Cubic Problem: Convergence Comparison', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'cubic_gap_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"Saved cubic_gap_comparison.png")
print(f"  Quadratic: {list(zip(iters_quad, gaps_quad))}")
print(f"  Linear:    {list(zip(iters_lin, gaps_lin))}")
