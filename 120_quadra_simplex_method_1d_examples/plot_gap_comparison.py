import re
import matplotlib.pyplot as plt

def parse_summary(filepath):
    """Parse iteration summary txt file and extract iteration numbers and abs gaps."""
    iterations = []
    abs_gaps = []
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find all "Abs Gap = X.XXXX" entries in [4] Bounds sections
    pattern = r"Iteration (\d+).*?Abs Gap = ([\d.]+)"
    matches = re.findall(pattern, content, re.DOTALL)
    
    for iter_num, gap in matches:
        iterations.append(int(iter_num))
        abs_gaps.append(float(gap))
    
    return iterations, abs_gaps

# Parse both files
iters_quad, gaps_quad = parse_summary("p4_iteration_summary.txt")
iters_lin, gaps_lin = parse_summary("p4_linear_iteration_summary.txt")

# Plot
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(iters_quad, gaps_quad, 'o-', lw=2, markersize=8, color='#2196F3', label='Quadratic Surrogate')
ax.plot(iters_lin, gaps_lin, 's--', lw=2, markersize=8, color='#FF5722', label='Linear Surrogate')

ax.set_xlabel('Iteration', fontsize=13)
ax.set_ylabel('Absolute Gap', fontsize=13)
ax.set_title('P4: Absolute Gap vs Iteration', fontsize=14, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xticks(range(max(len(iters_quad), len(iters_lin))))

plt.tight_layout()
plt.savefig('p4_gap_comparison.png', dpi=150)
plt.close()

print(f"Saved p4_gap_comparison.png")
print(f"  Quadratic: {list(zip(iters_quad, gaps_quad))}")
print(f"  Linear:    {list(zip(iters_lin, gaps_lin))}")
