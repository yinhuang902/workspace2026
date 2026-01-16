import sys
import os
import traceback

print("Checking environment...")
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    print(f"Added {current_dir} to path")
    
    import snoglode.utils.wls_quadratic_bound as wls
    print("Import successful!")
except Exception:
    print("Import failed with exception:")
    traceback.print_exc()
except SystemExit as e:
    print(f"SystemExit caught: {e}")
