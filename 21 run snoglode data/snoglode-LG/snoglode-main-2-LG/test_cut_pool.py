#!/usr/bin/env python
"""Test script to verify cut_pool module syntax and imports."""
import sys
print(f"Python: {sys.version}")

try:
    import ast
    with open('snoglode/utils/cut_pool.py', 'r', encoding='utf-8') as f:
        code = f.read()
    ast.parse(code)
    print("cut_pool.py: SYNTAX OK")
except SyntaxError as e:
    print(f"cut_pool.py: SYNTAX ERROR - {e}")
    sys.exit(1)

try:
    from snoglode.utils.cut_pool import CutPool, LagrangeanCut
    print("cut_pool imports: OK")
except Exception as e:
    print(f"cut_pool imports: FAILED - {e}")
    sys.exit(1)

try:
    from snoglode.components.tree import Tree
    print("tree imports: OK")
except Exception as e:
    print(f"tree imports: FAILED - {e}")
    sys.exit(1)

print("ALL CHECKS PASSED")
