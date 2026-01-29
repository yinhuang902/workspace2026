import sys
import os

# Add the project root to the python path
project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.append(project_root)

print(f"Checking import of snoglode.utils.wls_quadratic_bound from {project_root}...")

try:
    import snoglode.utils.wls_quadratic_bound
    print("\nSUCCESS: snoglode.utils.wls_quadratic_bound imported successfully.")
    print("The file syntax is correct and the module is accessible.")
except ImportError as e:
    print(f"\nFAILURE: Could not import module. Error: {e}")
except Exception as e:
    print(f"\nFAILURE: An error occurred during import. Error: {e}")
