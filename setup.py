from setuptools import setup, find_packages
import os


def requirements(fname):
    return [
        line.strip() for line in open(os.path.join(os.path.dirname(__file__), fname))
    ]


req_lite = requirements("requirements_lite.txt")
req_all = [x for x in requirements("requirements.txt") if x not in req_lite]

setup(
    name="tabensemb",
    version="0.1",
    author="xueling luo",
    description="A framework to ensemble model bases and evaluate various models for tabular predictions.",
    packages=find_packages(),
    url="https://github.com/LuoXueling/tabular_ensemble",
    python_requires=">=3.8.0",
    install_requires=req_lite,
    extras_require={
        "torch": ["torch>=1.12.0"],
        "all": req_all,
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
            "myst-parser",
            "sphinx_copybutton",
            "sphinx-apidoc",
            "sphinx_paramlinks",
        ],
    },
)
