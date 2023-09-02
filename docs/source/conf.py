# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../"))

project = "Tabular Ensemble"
copyright = "2023, Tabular Ensemble developers"
author = "Tabular Ensemble developers"
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "nbsphinx",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "numpydoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx_paramlinks",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

myst_enable_extensions = [
    "tasklist",
    "deflist",
    "dollarmath",
]

autodoc_default_options = {
    "member-order": "bysource",
    "special-members": True,
    "undoc-members": False,
    "exclude-members": "__weakref__, __dict__,__module__",
    "private-members": True,
}

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for numpydoc -------------------------------------------------

# Manually generate members in _templates/autosummary/class.rst
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_show_inherited_class_members = False

# -- Options for autosummary -------------------------------------------------

autosummary_generate = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_css_files = [
    "css/nbsphinx_dataframe.css",
]

html_theme_options = {
    "use_edit_page_button": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/LuoXueling/tabular_ensemble",
            "icon": "fa-brands fa-github",
        },
    ],
}

html_context = {
    "github_user": "LuoXueling",
    "github_repo": "tabular_ensemble",
    "github_version": "main/docs/source",
}
