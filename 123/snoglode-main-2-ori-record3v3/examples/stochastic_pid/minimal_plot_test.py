
# Force backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

print("--- Minimal Test Script ---")

try:
    # Minimal data
    x = [0.1, 0.2, 0.3, 0.4, 0.5]
    y = [1, 2, 3, 4, 5]
    print(f"Plotting minimal data: {x}, {y}")
    
    plt.figure()
    plt.plot(x, y)
    plt.title("Test Plot")
    
    # Try alternate location
    alt_out = r'c:\Users\pc\.gemini\test_plot.png'
    print(f"Trying save to: {alt_out}")
    plt.savefig(alt_out)
    print("Clean save executed.")
    
    if os.path.exists(alt_out):
        print(f"SUCCESS: Created test file at {alt_out}")
    else:
        print("FAIL: File not found.")
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("--- END TEST ---")
