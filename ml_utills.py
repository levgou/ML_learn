"""
    Lev.G

    Utils developed during the study of
    'Hands-On Machine Learning With Scikit-Learn & TensorFlow' O'Reilly - Aurelien Geron
"""

import pandas as pd
import numpy as np

from collections import Counter
from typing import List, Union, Iterable, Set


def load_data_from_csv(csv_path: str, index_col: Union[bool, int] = False) -> pd.DataFrame:
    """
    :param csv_path:  abs/relative path to the csv to load
    :param index_col: if specified - this col in the csv will be used
                      as indexing for the data

    :return: data frame containing the data from the csv
    """
    print('Loading {}'.format(csv_path))
    data_frame = pd.read_csv(csv_path, index_col=index_col, low_memory=False)
    print('Finished!')
    return data_frame


def nan_stats(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    provide a data frame with statistics for NaN value for input data_frame
    """
    # check which columns have NaN:
    is_nan_matrix = data_frame.isna().values
    headers = data_frame.columns.values
    row_count = data_frame.shape[0]
    nan_dict = {}

    # count NaNs
    for i, col in enumerate(headers):
        nan_dict[col] = Counter(is_nan_matrix[:, i])[True]

    df = pd.DataFrame()
    df['category'] = nan_dict.keys()
    df['count'] = nan_dict.values()
    df['%'] = [(count / row_count * 100) for count in nan_dict.values()]

    return df


def nan_stats_with_summary(data_frame: pd.DataFrame) -> pd.DataFrame:
    """

    :param data_frame: Data frame to analyse
    :return: data fram with NaN stats for each column from input dataframe

    very similar to nan_stats, but also prints a short summary about the whole dataframe
    """
    nan_df = nan_stats(data_frame)
    cells_df = data_frame.shape[0] * data_frame.shape[1]
    cells_nan = nan_df['count'].sum()
    nan_percentage = cells_nan / cells_df * 100

    print('NaN values in DataFrame: [{}/{}] - [{:.2f}%]'.format(cells_nan, cells_df, nan_percentage))
    return nan_df


def pl() -> None:
    """
        Print separation line
    """
    print("=" * 80)


def print_header(header: str) -> None:
    """
    :param header: string to be part of the separation line
    prints a separation line with the header in the middle
    """
    print('=' * 20, header, '=' * 20)


def print_categories_value_counts(data_frame: pd.DataFrame, categories: Iterable[str]) -> None:
    """
    :param data_frame: some pandas data frame
    :param categories: sub-set of the data_frame's column names

    for each column in categories - print value_counts statistics
    """
    for c in categories:
        print_header(c)
        print(data_frame[c].value_counts(dropna=False))
        pl()


def get_uniques_for_delimited_categories(series: pd.Series, delim: str)-> Set[str]:

    uniques_set = set()
    for data_p in series[series.notna()]:
        types = set(data_p.strip().split(delim))
        uniques_set |= types

    return uniques_set


def categories_stats_for_delimited_categories(series: pd.Series, delim: str, verbose: bool = True) -> Counter:
    """

    :param verbose: if True - print the counts and the number of categories
    :param series: panda series
    :param delim: delimiter of valus in each cell for example for 6;7;8 -> delim == ;
    prints statistics about count of categories and num of categories

    > use this when a pandas table include a column,
    > where in each cell multiple values could be found
    """

    cnt = Counter()
    for data_p in series[series.notna()]:
        dev_types = data_p.strip().split(delim)
        cnt.update(dev_types)

    if verbose:
        print(pd.Series(cnt))  # convert to series for pretty print
        pl()
        print('Num of categories: {}'.format(len(cnt)))

    return cnt


def convert_multi_value_in_cell_series_to_dataframe(series: pd.Series, headers: List[str],
                                                    delim: str, prefix: str='') -> pd.DataFrame:
    frame_entries = []
    for entry in series.str.split(delim):
        if entry is np.nan:
            entry = []
        frame_entries.append([int(h in entry) for h in headers])

    if len(prefix):
        prefix += '_'

    headers = [(prefix + h) for h in headers]
    return pd.DataFrame(data=frame_entries, columns=headers)


def multi_cat_series_to_dummies(series: pd.Series, delim: str) -> pd.DataFrame:
    categories = list(get_uniques_for_delimited_categories(series, delim))
    return convert_multi_value_in_cell_series_to_dataframe(series, categories, delim, str(series.name))


def add_missing_columns(df: pd.DataFrame, needed_columns: List[str], fill_val: object=0) -> pd.DataFrame:
    return df.loc[:, needed_columns].fillna(fill_val)
