try:
    with open('stochastic_pid.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Print lines 285-290 for debugging
    print("Current lines 285-290:")
    for i in range(284, 290):
        print(f"{i+1}: {repr(lines[i])}")
    
    # Create new lines
    new_lines = lines[:285]  # Keep everything up to line 286
    new_lines.append("        first_scen = solver.subproblems.names[0]\r\n")
    new_lines.append("        K_p_val, K_i_val, K_d_val = None, None, None\r\n")
    new_lines.append("        for vn in solver.solution.subproblem_solutions[first_scen]:\r\n")
    new_lines.append("            if vn == \"K_p\":\r\n")
    new_lines.append("                K_p_val = solver.solution.subproblem_solutions[first_scen][vn]\r\n")
    new_lines.append("            elif vn == \"K_i\":\r\n")
    new_lines.append("                K_i_val = solver.solution.subproblem_solutions[first_scen][vn]\r\n")
    new_lines.append("            elif vn == \"K_d\":\r\n")
    new_lines.append("                K_d_val = solver.solution.subproblem_solutions[first_scen][vn]\r\n")
    new_lines.append("        print(f\"Final Point: K_p = {K_p_val}, K_i = {K_i_val}, K_d = {K_d_val}\")\r\n")
    new_lines.extend(lines[289:])  # Add everything from line 290 onwards
    
    #Write back
    with open('stochastic_pid.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print("\nSuccess! File has been updated.")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
