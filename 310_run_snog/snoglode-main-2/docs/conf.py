# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SNoGloDe'
copyright = '2025, Georgia Stinchfield'
author = 'Georgia Stinchfield'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["myst_parser",
              "sphinxcontrib.bibtex",
              "sphinx.ext.mathjax",
              "sphinx.ext.autodoc",        # core autodoc support
              "sphinx.ext.napoleon",       # supports Google/NumPy-style docstrings
              "sphinx.ext.autosummary",    # auto-summary tables (optional)
              "sphinx_autodoc_typehints"]  # show type hints in docs
myst_enable_extensions = ["dollarmath"]
bibtex_bibfiles = ["references.bib"]

# Include your project path so autodoc can import it
import os
import sys
sys.path.insert(0, os.path.abspath(".."))  # assumes docs/ is one level below the package

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = "furo"
html_static_path = ['_static']
