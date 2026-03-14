import sys
import gurobipy as gp

v = gp.gurobi.version()
print("Python executable:", sys.executable)
print("Gurobi version:", f"{v[0]}.{v[1]}.{v[2]}")
print("gurobipy module:", gp.__file__)