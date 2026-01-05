
try:
    print("Starting import test...")
    import sys
    sys.path.append("../../")
    print(sys.executable)
    import snoglode
    print("snoglode imported")
    from snoglode.solver import Solver
    print("Solver imported")
    from snoglode.utils.quadratic_surrogate import QuadraticSurrogate
    print("QuadraticSurrogate imported")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
