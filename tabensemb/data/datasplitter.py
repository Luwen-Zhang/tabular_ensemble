import numpy as np
from tabensemb.utils import *
from tabensemb.data import AbstractSplitter
import inspect
from sklearn.model_selection import train_test_split
from typing import Type, List, Tuple


class RandomSplitter(AbstractSplitter):
    """
    Randomly split the dataset.
    """

    def _split(self, df, cont_feature_names, cat_feature_names, label_name):
        length = len(df)
        train_indices, test_indices = train_test_split(
            np.arange(length), test_size=self.train_val_test[2], shuffle=True
        )
        train_indices, val_indices = train_test_split(
            train_indices,
            test_size=self.train_val_test[1] / np.sum(self.train_val_test[0:2]),
            shuffle=True,
        )

        return train_indices, val_indices, test_indices

    @property
    def support_cv(self):
        return True

    def _next_cv(
        self,
        df: pd.DataFrame,
        cont_feature_names: List[str],
        cat_feature_names: List[str],
        label_name: List[str],
        cv: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._sklearn_k_fold(np.arange(len(df)), cv)


splitter_mapping = {}
clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
for name, cls in clsmembers:
    if issubclass(cls, AbstractSplitter):
        splitter_mapping[name] = cls


def get_data_splitter(name: str) -> Type[AbstractSplitter]:
    if name not in splitter_mapping.keys():
        raise Exception(f"Data splitter {name} not implemented.")
    elif not issubclass(splitter_mapping[name], AbstractSplitter):
        raise Exception(f"{name} is not the subclass of AbstractSplitter.")
    else:
        return splitter_mapping[name]
