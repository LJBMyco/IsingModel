"""Sphinx configuration."""
project = "Isingmodel"
author = "Lawrence Bradley"
copyright = "2024, Lawrence Bradley"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
