from setuptools import setup, find_packages
import os


def requirements(fname):
    return [
        line.strip() for line in open(os.path.join(os.path.dirname(__file__), fname))
    ]


req_all = requirements("requirements.txt")

setup(
    name="tabensemb",
    version="0.2",
    author="xueling luo",
    description="A framework to ensemble model bases and evaluate various models for tabular predictions.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
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
            "sphinx==7.2.5",
            "sphinx_rtd_theme==1.3.0",
            "nbsphinx==0.9.6",
            "pandoc==2.3",
            "myst-parser==2.0.0",
            "sphinx_copybutton==0.5.2",
            "sphinx_paramlinks==0.6.0",
            "numpydoc==1.5.0",
            "pydata_sphinx_theme==0.13.3",
        ],
        "notebook": ["jupyter", "notebook<7.0.0"],
    },
)
