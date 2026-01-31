#!/usr/bin/env python
import sys
with open('_test_result.txt', 'w', encoding='utf-8') as f:
    f.write("Script started\n")
    try:
        # Write immediately before any imports
        f.write("Attempting import...\n")
        f.flush()
        
        import simplex_specialstart
        f.write("Import succeeded!\n")
        
        if hasattr(simplex_specialstart, 'evaluate_one_tet'):
            f.write("evaluate_one_tet found!\n")
        else:
            f.write("evaluate_one_tet NOT found!\n")
            
        sys.exit(0)
    except SyntaxError as e:
        f.write(f"SyntaxError: {e}\n")
        f.write(f"File: {e.filename}, Line: {e.lineno}\n")
        f.write(f"Text: {e.text}\n")
        sys.exit(1)
    except Exception as e:
        import traceback
        f.write(f"Exception: {type(e).__name__}: {e}\n")
        f.write(traceback.format_exc())
        sys.exit(1)
