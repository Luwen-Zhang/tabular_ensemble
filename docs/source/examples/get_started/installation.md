# Installation

## Ubuntu

Before installing `tabular-ensemble`, `torch>=1.12.0` should be installed first following its [documentation](https://pytorch.org/get-started/locally/).

### From PyPI

```shell
pip install tabensemb
```

Use `pip install tabensemb[test]` instead if you want to run unit tests.

### From source

Clone the repository from GitHub

```shell
git clone https://github.com/Luwen-Zhang/tabular_ensemble.git
cd tabular_ensemble
```

Then install the package

```shell
pip install -e .
```

Use `pip install -e .[test]` instead if you want to run unit tests.

### Unit test

To run unit tests:

```shell
cd test
pytest .
```

## Other OS

The installation on other OSs is not tested.