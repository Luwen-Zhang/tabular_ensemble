# Quick start

Place your `.csv` or `.xlsx` file in a `data` subfolder (e.g., `data/sample.csv`), and generate a configuration file in a `configs` subfolder (e.g., `configs/sample.py`), containing the following content
```python
cfg = {
    "database": "sample",
    "continuous_feature_names": ["cont_0", "cont_1", "cont_2", "cont_3", "cont_4"],
    "categorical_feature_names": ["cat_0", "cat_1", "cat_2"],
    "label_name": ["target"],
}
```

Run the experiment using the configuration and the data using
```python
python main.py --base sample --epoch 10
```
where `--base` refers to the configuration file, and additional arguments (such as `--epoch` here) refer to those in `config/default.py`.

Here is a typical usage of the repository

```python
python main.py --base sample --epoch 200 --batch_size 128 --bayes_opt --split_ratio 0.6 0.2 0.2
```
