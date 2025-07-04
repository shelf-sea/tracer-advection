# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

import tomlkit

sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------


def _get_project_meta():
    with open("../pyproject.toml") as pyproject:
        file_contents = pyproject.read()

    return tomlkit.parse(file_contents)["tool"]["poetry"]


pkg_meta = _get_project_meta()
project = str(pkg_meta["name"])
copyright = '2022, shelf-seas'  # noqa: WPS125
author = "shelf-seas"

# The short X.Y version
version = str(pkg_meta["version"])
# The full version, including alpha/beta/rc tags
release = version


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = "3.3"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    # Used to write beautiful docstrings:
    "sphinx.ext.napoleon",
    # Used to include .md files:
    "m2r2",
    # Used to insert typehints into the final docs:
    "sphinx_autodoc_typehints",
]

autoclass_content = "class"
autodoc_member_order = "bysource"

autodoc_member_order = "bysource"
autodoc_default_flags = {
    "members": "",
    "undoc-members": "code,error_template",
    "exclude-members": "__dict__,__weakref__",
}

# Set `typing.TYPE_CHECKING` to `True`:
# https://pypi.org/project/sphinx-autodoc-typehints/
set_type_checking_flag = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:

source_suffix = [".rst", ".md"]

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

add_module_names = False

autodoc_default_options = {
    "show-inheritance": True,
}


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "alabaster"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "sidebar_collapse": False,
    "show_powered_by": False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# This is required for the alabaster theme
# refs: http://alabaster.readthedocs.io/en/latest/installation.html#sidebars
html_sidebars = {
    "**": [
        "about.html",
        "badges.html",
        "navigation.html",
        "moreinfo.html",
        "github.html",
        "searchbox.html",
    ],
}

# -- Extension configuration -------------------------------------------------

napoleon_numpy_docstring = False

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True
