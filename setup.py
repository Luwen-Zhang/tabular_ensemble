from setuptools import setup, find_packages
import os


def requirements(fname):
    return [
        line.strip() for line in open(os.path.join(os.path.dirname(__file__), fname))
    ]


req_all = requirements("requirements.txt")

setup(
    name="tabensemb",
    version="0.1",
    author="xueling luo",
    description="A framework to ensemble model bases and evaluate various models for tabular predictions.",
    packages=find_packages(),
    url="https://github.com/LuoXueling/tabular_ensemble",
    python_requires=">=3.8.0",
    install_requires=req_all,
    extras_require={
        "torch": ["torch>=1.12.0"],
        "test": [
            "torch>=1.12.0",
            "pytest",
            "pytest-cov",
            "pytest-order",
            "pytest-mock",
            "black",
        ],
        "doc": [
            "sphinx",
            "sphinx_rtd_theme",
            "nbsphinx",
            "pandoc",
            "myst-parser",
            "sphinx_copybutton",
            "sphinx_paramlinks",
            "numpydoc",
        ],
        "notebook": ["jupyter", "notebook<7.0.0"],
    },
)
