import os
import sys

current_work_dir = os.path.abspath(".")
parent_dir = os.path.split(current_work_dir)[-1]

if parent_dir == "test":
    sys.path.append("../")
try:
    import tabensemb
except:
    raise Exception(
        f"Test units should be placed in a folder named `test` that is in the same parent folder as `tabensemb`."
    )

tabensemb.setting["default_data_path"] = "../data"
tabensemb.setting["default_config_path"] = "../configs"
tabensemb.setting["default_output_path"] = "../output/unittest"

iris_columns = [
    "sepal length",
    "sepal width",
    "petal length",
    "petal width",
    "class",
]
mpg_columns = [
    "mpg",
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
    "origin",
    "car_name",
]
adult_columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]
