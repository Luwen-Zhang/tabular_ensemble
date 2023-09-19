import numpy as np
from typing import List, Union, Dict, Type
import pandas as pd


def get_corr_sets(where_corr: np.ndarray, names: List):
    where_corr = [[names[x] for x in y] for y in where_corr]
    corr_chain = {}

    def add_edge(x, y):
        if x not in corr_chain.keys():
            corr_chain[x] = [y]
        elif y not in corr_chain[x]:
            corr_chain[x].append(y)

    for x, y in zip(*where_corr):
        if x != y:
            add_edge(x, y)
            add_edge(y, x)
    corr_feature = list(corr_chain.keys())
    for x in np.setdiff1d(names, corr_feature):
        corr_chain[x] = []

    def dfs(visited, graph, node, ls):
        if node not in visited:
            ls.append(node)
            visited.add(node)
            for neighbour in graph[node]:
                ls = dfs(visited, graph, neighbour, ls)
        return ls

    corr_sets = []
    for x in corr_feature[::-1]:
        if len(corr_sets) != 0:
            for sets in corr_sets:
                if x in sets:
                    break
            else:
                corr_sets.append(dfs(set(), corr_chain, x, []))
        else:
            corr_sets.append(dfs(set(), corr_chain, x, []))

    corr_sets = [[x for x in y] for y in corr_sets]
    return corr_feature, corr_sets


object_unknown_value = "UNK"
number_unknown_value = -1


def fill_cat_nan(df: pd.DataFrame, cat_dtypes: Dict[str, np.dtype]) -> pd.DataFrame:
    """
    Imputation of categorical features.

    Parameters
    ----------
    df
        The dataframe to be imputed.
    cat_dtypes
        The dtype of each categorical feature. If it is a numerical type, ``number_unknown_value`` (default to -1) is
        used for imputation, otherwise ``object_unknown_value`` (default to "UNK") is used. Change these two values if
        you want other values for missing or unknown values.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    for feature, dtype in cat_dtypes.items():
        dtype = get_imputed_dtype(dtype)
        unknown_val = get_unknown_value(dtype)
        if feature in df.columns:
            values = df[feature].fillna(unknown_val).values
            if dtype == int and not np.all(np.mod(values, 1) == 0):
                raise Exception(
                    f"The numerical categorical feature {feature} is not integeral but {values.dtype}."
                )
            df[feature] = values.astype(dtype)
    return df


def get_imputed_dtype(dtype: np.dtype) -> Union[Type[int], Type[str]]:
    """
    Numerical columns will be transformed to "int", and others will be transformed to "str".

    Parameters
    ----------
    dtype
        The dtype of a column.

    Returns
    -------
    Type[int] or Type[str]
    """
    if np.issubdtype(dtype, np.number):
        return int
    else:
        return str


def get_unknown_value(dtype: Union[Type[int], Type[str]]) -> Union[int, str]:
    """
    Select the unknown value for the dtype judged by :func:`get_imputed_dtype`.

    Parameters
    ----------
    dtype
        int or str from :func:`get_imputed_dtype`.

    Returns
    -------
    int or str
    """
    if dtype == int:
        return number_unknown_value
    else:
        return object_unknown_value


class _OrdinalEncodingWrongDirException(Exception):
    """
    The exception might be raised by :class:`OrdinalEncoder` under the circumstance that
    :meth:`OrdinalEncoder.transform` is called for transformed data or :meth:`OrdinalEncoder.inverse_transform` is
    called for inverse-transformed data. If it is caught, the other method will be called to check whether it is now
    really the case.
    """

    pass


class OrdinalEncoder:
    """
    An ordinal encoder for categorical features that better supports ``pd.DataFrame`` even with missing columns. It
    supports ``np.ndarray`` when calling :meth:`transform` or :meth:`inverse_transform`, but does not support fitting
    on a ``np.ndarray`` because it is designed for dataframes.
    It can also identify a miss-calling of :meth:`transform` and :meth:`inverse_transform` (calling transform on
    transformed dataframe, and vice versa), and return the input dataframe directly. But the functionality won't work
    if the dataframe to be transformed/inverse-transformed only contains categorical features whose categories before
    encoding are all integers.
    """

    def __init__(self):
        self.mapping = {}
        self.num_unique = {}
        self.features = []
        self.dtypes = {}
        self.dtypes_samples = {}
        self.fitted = False

    def fit(self, df: pd.DataFrame):
        """
        Fit the ordinal encoder.

        Parameters
        ----------
        df
            A dataframe that only contains categorical features.
        """
        df = df.copy()
        self.features = list(df.columns)
        for feature, col_type in zip(df.columns, df.dtypes):
            # The imputation procedure is the same as that in AbstractImputer
            dtype = get_imputed_dtype(col_type)
            unknown_value = get_unknown_value(dtype)
            values = fill_cat_nan(df[[feature]], {feature: dtype}).values.flatten()
            self.dtypes[feature] = dtype
            unique_values = list(set(values))
            if unknown_value not in unique_values:
                unique_values += [unknown_value]
            self.mapping[feature] = unique_values
            self.num_unique[feature] = len(unique_values)
            self.dtypes_samples[feature] = unknown_value
        self.fitted = True
        return self

    def _transform_or_inverse_transform(
        self, df: Union[pd.DataFrame, np.ndarray], transform: bool
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Automatically distinguish transform/inverse-transform and ``pd.DataFrame``/``np.ndarray``.

        Parameters
        ----------
        df
            A pd.DataFrame or a np.ndarray
        transform
            True for transform and False for inverse-transform.

        Returns
        -------
        A pd.DataFrame or a np.ndarray
            depending on the type of the input.
        """
        if isinstance(df, pd.DataFrame):
            input_type = "dataframe"
            df = df.copy()
        else:
            input_type = str(type(df))
            df = pd.DataFrame(data=df, columns=self.features, index=np.arange(len(df)))
        try:
            if transform:
                trans_df = self._transform(df.copy())
            else:
                trans_df = self._inverse_transform(df.copy())
        except _OrdinalEncodingWrongDirException as e_forw:
            try:
                if transform:
                    _ = self._inverse_transform(df.copy())
                else:
                    _ = self._transform(df.copy())
                trans_df = df
            except Exception as e_inv:
                raise Exception(
                    f"The dataframe can be neither transformed nor inverse transformed by the ordinal encoder.\n"
                    f"Exception when calling {'transform' if transform else 'inverse_transform'}: {e_forw}\n"
                    f"Exception when calling {'inverse_transform' if transform else 'transform'}: {e_inv}"
                )
        return trans_df if input_type == "dataframe" else trans_df.values

    def transform(self, df: Union[pd.DataFrame, np.ndarray]):
        """
        Ordinal-encoding categorical features. If the input is a ``np.ndarray``, the columns should match the recorded
        categorical features (:attr:`features`).
        """
        return self._transform_or_inverse_transform(df, transform=True)

    def inverse_transform(self, df: Union[pd.DataFrame, np.ndarray]):
        """
        Inverse ordinal-encoding categorical features. If the input is a ``np.ndarray``, the columns should match the
        recorded categorical features (:attr:`features`).
        """
        return self._transform_or_inverse_transform(df, transform=False)

    def _transform(self, df: pd.DataFrame):
        for idx, feature in enumerate(self.features):
            if feature not in df.columns:
                continue
            # The imputation procedure is the same as that in fit.
            unknown_val = get_unknown_value(self.dtypes[feature])
            values = fill_cat_nan(
                df[[feature]], {feature: self.dtypes[feature]}
            ).values.flatten()

            unique_values = list(set(values))
            encoded_values = list(range(self.num_unique[feature]))
            unknown_values = [
                val for val in unique_values if val not in self.mapping[feature]
            ]
            known_values = [
                val
                for val in unique_values
                if val not in unknown_values and val != unknown_val
            ]

            is_int = (
                lambda x: str(x).replace(".", "").isdigit() and float(x).is_integer()
            )
            # If the input is transformed, the unique values will be strings of integers because of fill_cat_nan.
            # Otherwise, they will be at least non-digits. One exception is that all categories are integers.
            str_int_in_encoded = lambda x: str(x).isdigit() and int(x) in encoded_values
            if (
                any([str_int_in_encoded(val) for val in unknown_values])
                and all([str_int_in_encoded(val) for val in known_values])
                and not (
                    all([is_int(val) for val in unknown_values + known_values])
                    and self.dtypes[feature] == int
                )
            ):
                # The input is already transformed.
                raise _OrdinalEncodingWrongDirException

            transformed_values = np.zeros_like(values, dtype=int)
            for val in unique_values:
                transformed_values[values == val] = self.mapping[feature].index(
                    unknown_val if val in unknown_values else val
                )
            df[feature] = transformed_values.astype(int)
        return df

    def _inverse_transform(self, df: pd.DataFrame):
        for idx, feature in enumerate(self.features):
            if feature not in df.columns:
                continue

            unknown_val = get_unknown_value(self.dtypes[feature])
            encoded_unknown_val = self.mapping[feature].index(unknown_val)
            values = df[feature].fillna(encoded_unknown_val).values

            unique_values = list(set(values))
            encoded_values = list(range(self.num_unique[feature]))
            unknown_values = [val for val in unique_values if val not in encoded_values]
            known_values = [
                val
                for val in unique_values
                if val not in unknown_values and val != encoded_unknown_val
            ]

            dtype = self.dtypes[feature]
            is_int = (
                lambda x: str(x).replace(".", "").isdigit() and float(x).is_integer()
            )
            if dtype == int:
                # Do not let floats pass the following check and return None.
                dtype = lambda x: int(x) if is_int(x) else x
            # In fit or _transform, the values are all translated to a consistent dtype (str or int) because of
            # fill_cat_nan. If the input here is an integer, it can also be a category before transform when other
            # categories are strings.
            if (
                any([dtype(val) in self.mapping[feature] for val in unknown_values])
                and all([dtype(val) in self.mapping[feature] for val in known_values])
                and self.dtypes[feature] != int
            ):
                raise _OrdinalEncodingWrongDirException

            if not all([is_int(x) for x in unique_values]):
                raise _OrdinalEncodingWrongDirException(
                    f"The feature {feature} is not integeral ({unique_values}), therefore can not be "
                    f"inverse-transformed."
                )

            transformed_values = np.empty_like(values).astype(self.dtypes[feature])
            for val in unique_values:
                transformed_values[values == val] = (
                    unknown_val
                    if val in unknown_values
                    else self.mapping[feature][int(val)]
                )
            df[feature] = transformed_values.astype(self.dtypes[feature])
        return df
