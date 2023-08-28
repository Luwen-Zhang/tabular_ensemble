# Installation

## Ubuntu

Before installing `tabular-ensemble`, `PyTorch` should be installed first following its [documentation](https://pytorch.org/get-started/locally/).

Clone the repository from GitHub

```shell
git clone https://github.com/LuoXueling/tabular_ensemble.git
cd tabular_ensemble
```

Then install the package

```shell
pip install -e .
```

To test the functionality of the package

```shell
pip install -e .[test]
cd test
pytest .
```

## Other OS

The installation on other OSs is not tested.