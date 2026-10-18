"""Sphinx configuration for the rocksteady documentation (https://rocksteady.readthedocs.io)."""

import rocksteady

project = "rocksteady"
author = "UT Computational NE"
copyright = "2026, UT Computational NE"
release = rocksteady.__version__
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",   # NumPy-style docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "myst_parser",           # the guide pages include sections of the README (Markdown)
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = []
templates_path = []

# The guide pages include README sections under their own title, so README "###" headings follow an H1.
suppress_warnings = ["myst.header"]
myst_enable_extensions = ["dollarmath"]
myst_heading_anchors = 3

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {"members": True, "show-inheritance": True}
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_rtype = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

html_theme = "sphinx_rtd_theme"
html_theme_options = {"navigation_depth": 3}
html_static_path = []
