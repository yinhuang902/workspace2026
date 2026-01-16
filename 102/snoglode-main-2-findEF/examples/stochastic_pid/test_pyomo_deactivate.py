import pyomo.environ as pyo

# Create a model
m = pyo.ConcreteModel()
m.x = pyo.Var(initialize=5)
m.obj = pyo.Objective(expr=m.x)

# Create a parent model
parent = pyo.ConcreteModel()
parent.add_component("sub", m)

# Solve parent (mock solve by setting value)
parent.sub.x.value = 10

# Deactivate sub
parent.sub.deactivate()

# Try to access value
try:
    val = pyo.value(parent.sub.x)
    print(f"Value of deactivated var: {val}")
except Exception as e:
    print(f"Error accessing deactivated var: {e}")

# Try to access via component_objects
print("Iterating component_objects:")
for var in m.component_objects(pyo.Var):
    try:
        print(f"{var.name}: {pyo.value(var)}")
    except Exception as e:
        print(f"{var.name}: Error {e}")
