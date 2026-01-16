import sys

# Read the file
with open('stochastic_pid.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find and replace the problematic section (lines 285-289, 0-indexed as 284-288)
# We need to replace lines 287-288 (0-indexed 286-287)
new_section = [
    "        first_scen = solver.subproblems.names[0]\n",
    "        K_p_val, K_i_val, K_d_val = None, None, None\n",
    "        for vn in solver.solution.subproblem_solutions[first_scen]:\n",
    "            if vn == \"K_p\":\n",
    "                K_p_val = solver.solution.subproblem_solutions[first_scen][vn]\n",
    "            elif vn == \"K_i\":\n",
    "                K_i_val = solver.solution.subproblem_solutions[first_scen][vn]\n",
    "            elif vn == \"K_d\":\n",
    "                K_d_val = solver.solution.subproblem_solutions[first_scen][vn]\n",
    "        print(f\"Final Point: K_p = {K_p_val}, K_i = {K_i_val}, K_d = {K_d_val}\")\n",
]

# Replace lines 286-288 (idx 285-287) with new section
# Line 285 (idx 284) is the comment, keep it
# Line 286 (idx 285) is first_scen =, replace with new section
# Lines 287-289 (idx 286-288) are the if/sol/print, remove them

new_lines = lines[:285] + new_section + lines[289:]

# Write back
with open('stochastic_pid.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Fixed successfully!")
