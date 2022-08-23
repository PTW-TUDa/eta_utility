""" Configuration file for the Sphinx documentation builder.
This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""
from __future__ import annotations

import os
import sys

try:
    from importlib.metadata import version as get_version
except ImportError:
    from importlib_metadata import version as get_version

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath("../eta_utility"))  # Insert path to eta_utility


# -- Project information -----------------------------------------------------
project = "eta_utility"
release = get_version(project)  # The full version, including alpha/beta/rc tags
version = ".".join(release.split(".")[:2])  # Top level version
copyright = "Technical University of Darmstadt, Institute for Production Management, Technology and Machine Tools (PTW)"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

autodoc_mock_imports = ["opcua", "numpy.random", "pandas"]
autodoc_default_options = {"undoc-members": True, "member-order": "bysource"}

intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    "stable_baselines": ("https://stable-baselines3.readthedocs.io/en/master/", None),
    "numpy": ("https://numpy.org/doc/1.22/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

autosummary_generate = True
