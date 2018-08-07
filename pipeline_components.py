"""
    Lev.G

    Scikit-Learn Pipeline components developed during the study of
    'Hands-On Machine Learning With Scikit-Learn & TensorFlow' O'Reilly - Aurelien Geron
"""

import pandas as pd
import numpy as np

from typing import List, Dict, Union
from sklearn.base import BaseEstimator, TransformerMixin

from ml_utills import multi_cat_series_to_dummies


class DataFrameCopySelector(BaseEstimator, TransformerMixin):
    """
        Return a dataframe copy, of the column subset specified by attribute_names
    """


    def __init__(self, attribute_names):
        self.attribute_names = attribute_names


    def fit(self, X, y=None):
        return self


    def transform(self, X) -> pd.DataFrame:
        return X[self.attribute_names].copy()


class ColumnBasedNanFilter(BaseEstimator, TransformerMixin):
    """
        Returns the data frame, without the rows including NaN at the specified columns
    """


    def __init__(self, columns_to_consider: List[str]):
        self.columns_to_consider = columns_to_consider


    def fit(self, X, y=None):
        return self


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        notna_indicies = df[self.columns_to_consider[0]].notna()
        for col in self.columns_to_consider[1:]:
            notna_indicies &= df[col].notna()

        return df[notna_indicies]


class ColumnValueFilter(BaseEstimator, TransformerMixin):
    """
        Returns the data frame, with the rows that match/don't match the specified values for columns
    """


    def __init__(self, columns_to_consider: List[str], values: List[str], to_match: List[bool]):
        """

        :param columns_to_consider: columns names where values will matched/anti-matched
        :param values: the value to match/anti-match in the columns
        :param to_match: for each value:
                        if True - only rows not including the value will be excluded,
                        if False - only rows including the values will be excluded
        """
        assert len(columns_to_consider) == len(values)
        assert len(columns_to_consider) == len(to_match)

        self.columns_to_consider = columns_to_consider
        self.values = values
        self.to_match = to_match


    def fit(self, X, y=None):
        return self


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        num_lines = df.shape[0]
        include_indices = np.full(num_lines, True)
        for col, val, is_match in zip(self.columns_to_consider, self.values, self.to_match):

            if is_match:
                include_indices &= df[col] == val
            else:
                include_indices &= ~(df[col] == val)

        return df[include_indices]


class DataFrameColumnOmiter(BaseEstimator, TransformerMixin):
    """
        Return a dataframe without the mentioned columns
    """


    def __init__(self, columns_to_omit: List[str]):
        self.columns_to_omit = columns_to_omit


    def fit(self, X, y=None):
        return self


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=self.columns_to_omit)


class DataFrameReplaceNonMatching(BaseEstimator, TransformerMixin):
    """
        Return a dataframe replacing all not matching cells with replace_val
    """


    def __init__(self, match_patters: List[str], replace_val: str):
        self.replace_val = replace_val
        self.match_patters = match_patters


    def fit(self, X, y=None):
        return self


    def transform(self, df: Union[pd.DataFrame, pd.Series]) -> np.ndarray:

        if isinstance(df, pd.Series):
            df = df.to_frame()

        mask = df != self.match_patters[0]
        for pattern in self.match_patters[1:]:
            mask &= (df != pattern)
        df[mask] = self.replace_val
        return df


class DataFrameReplaceInColsByDict(BaseEstimator, TransformerMixin):
    """
        Return a dataframe replacing for each column, using the dict replacing[column]
    """


    def __init__(self, columns: List[str], replace_dict: Dict[str, Dict[str, str]]):
        self.replace_dict = replace_dict
        self.columns = columns


    def fit(self, X, y=None):
        return self


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.columns:
            df[col].replace(self.replace_dict[col], inplace=True)

        return df


class DataFrameID(BaseEstimator, TransformerMixin):
    """
        Return the dataframe
    """

    def fit(self, X, y=None):
        return self


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


class OneHotEncoderWithNaNs(BaseEstimator, TransformerMixin):
    """
        Similar to future_encoders.OneHotEncoder but for multi categorical dataframes
        can handle NaN values (will get 0 at each category)
    """

    last_categories = []


    def fit(self, X, y=None):
        return self


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """

        :param df: dataframe of series needed to be converted to dummies data frames
        :return:
        """
        dummy_frames = []
        for col in df:
            dummy_frame = pd.DataFrame(data=pd.get_dummies(df[col], prefix=col))
            dummy_frames.append(dummy_frame)

        full_df = pd.concat(dummy_frames, axis=1)
        OneHotEncoderWithNaNs.last_categories = list(full_df.columns)
        return full_df


class OneHotEncoderWithNaNsMulti(BaseEstimator, TransformerMixin):
    """
        Similar to future_encoders.OneHotEncoder but for multi categorical dataframes
        can handle NaN values (will get 0 at each category)
    """

    last_categories = []


    def __init__(self, delim: str):
        self.delim = delim


    def fit(self, X, y=None):
        return self


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        dummy_frames = []
        for col in df:
            dummy_frames.append(multi_cat_series_to_dummies(df[col], self.delim))

        full_df = pd.concat(dummy_frames, axis=1)
        OneHotEncoderWithNaNsMulti.last_categories = list(full_df.columns)
        return full_df


class ConvertStringToBool(BaseEstimator, TransformerMixin):
    """
        For series/dataframe of some string representation of bool values, for example ['Yes', 'No]
        convert to series/dataframe with bool values (Doesnt support NaN values!)
    """


    def __init__(self, true_repr: str):
        self.true_repr = true_repr


    def fit(self, X, y=None):
        return self


    def transform(self, df: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        return df == self.true_repr
