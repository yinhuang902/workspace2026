# Installation

This page describes how to install **SNoGloDe**, including dependencies for optional solver and parallel-computing support.

---

## 1. Prerequisites

Before installing SNoGloDe, ensure the following requirements are met:

- **Python 3.9 or newer**
- (Optional) **Gurobi 11.0 or newer** for commercial MIP/LP solving
- (Optional) **MPI** for distributed execution  
  - Linux / Mac: `mpich` or `openmpi`

We *strongly* recommend using a dedicated Python environment (Conda or `venv`) to avoid dependency conflicts.

---

## 2. Environment 

```bash
conda create -n snoglode python=3.10
conda activate snoglode
```

## 3.  Package Install

Once this repo is made public, we will have a simple pip installable version. For now, the following two options are available:

### Option A — Using SSH (recommended)
Requires that your GitHub SSH key is configured and that you have been granted access to the repository.

```bash
pip install git+ssh://git@github.com/gcstinchfield/snoglode.git
```

### Option B — Developer installation (editable mode)
If you plan to modify or contribute to the package, clone the repo and install in editable mode:

```bash
git clone git@github.com:gcstinchfield/snoglode.git
cd snoglode
pip install -e .
```

## 4.  Verification

After installation, you can confirm that the package is available in your environment.

Run the following in a terminal:

```bash
python -c "import snoglode; print('success!')"
```