# Fix script for stochastic_pid.py
# This script reads the file and fixes the K_p, K_i, K_d extraction logic

with open('stochastic_pid.py', 'r') as f:
    content = f.read()

# Replace the problematic section
old_code = """        # Print first-stage variables (they are identical across all scenarios)
        first_scen = solver.subproblems.names[0]
        if first_scen in solver.solution.subproblem_solutions:
            sol = solver.solution.subproblem_solutions[first_scen]
            print(f"Final Point: K_p = {sol.get('K_p', 'N/A')}, K_i = {sol.get('K_i', 'N/A')}, K_d = {sol.get('K_d', 'N/A')}")"""

new_code = """        # Print first-stage variables (they are identical across all scenarios)
        first_scen = solver.subproblems.names[0]
        K_p_val, K_i_val, K_d_val = None, None, None
        for vn in solver.solution.subproblem_solutions[first_scen]:
            if vn == "K_p":
                K_p_val = solver.solution.subproblem_solutions[first_scen][vn]
            elif vn == "K_i":
                K_i_val = solver.solution.subproblem_solutions[first_scen][vn]
            elif vn == "K_d":
                K_d_val = solver.solution.subproblem_solutions[first_scen][vn]
        print(f"Final Point: K_p = {K_p_val}, K_i = {K_i_val}, K_d = {K_d_val}")"""

content = content.replace(old_code, new_code)

with open('stochastic_pid.py', 'w') as f:
    f.write(content)

print("Fixed!")
