<!-- ![Tests](https://github.com/gcstinchfield/snoglode/actions/workflows/run_tests.yml/badge.svg)
![Coverage](https://codecov.io/gh/gcstinchfield/snoglode/branch/main/graph/badge.svg)
![Python](https://img.shields.io/pypi/pyversions/snoglode)
![License](https://img.shields.io/github/license/gcstinchfield/snoglode) -->

# SNoGloDe: Structured Nonlinear Global Decomposition

## Overview

This package provides a framework for solving block-angular decomposable (typically identifiable by complicating variables across subproblems) optimization problems using a *prioritized* spatial branch and bound tree.

There are four major algorithmic elements that form this framework:
- **Node selection & processing** (which open node should be solved next / bounds tightening)
- **Node processing** (generate a relaxation & determine a lower bound / generate a candidate & determine an upper bound)
- **Branching & Bounding** (prune by bound or infeasibility / spawn new children)
- **Termination evaluation** (check maximum time / epsilon gap / maximum iterations)

which can be summarized (generally) in the following diagram:
<p align="center">
  <img src="docs/_static/snoglode-basic-logic-map.png" width="500">
</p>

The elements outlined in green have been implemented such that custom callbacks can be used. Users can update the logic of (1) node selection (e.g., queuing), (2) lower bound solution process (e.g., relaxations, warm-starting), (3) candidate generation, and/or (4) branching (e.g. child generation) by selecting in one of the available choices *or* by injecting their own logic.

If both the lower-bound relaxation *and* the upper-bound with a fixed candidate solution are solved to global optimality, there are provable convergance guarantees to the globally optimal solution. Otherwise, we can still search for feasible solutions and compute gaps, but there is no guarantee that the gap will close (though, it still might). 
__________________________________________


## Installation

To install the package in editable/development mode, download the code locally. Navigate to the root directory via terminal and run:

```bash
pip install -e .
```

This installs the *experimental* version of the code into your current environment (Conda/venv). Any changes made to your local code will be reflected when you run the package in the properly activated environment. 

## License

[MIT](https://choosealicense.com/licenses/mit/)