import pandas as pd
from packaging import version
from typing import Union, List


is_deprecated_lexsorted_pandas = version.parse(pd.__version__) > version.parse("1.3.0")


def get_level_index(df: pd.DataFrame, level=Union[str, int]) -> int:
    if isinstance(level, str):
        try:
            return df.index.names.index(level)
        except (AttributeError, ValueError):
            return ("datetime", "instrument").index(level)
    elif isinstance(level, int):
        return level
    else:
        raise NotImplementedError(f"This type of input is not supported")


def lazy_sort_index(df: pd.DataFrame, axis=0) -> pd.DataFrame:
    idx = df.index if axis == 0 else df.columns
    if (
        not idx.is_monotonic_increasing
        or not is_deprecated_lexsorted_pandas
        and isinstance(idx, pd.MultiIndex)
        and not idx.is_lexsorted()
    ):  # this case is for the old version
        return df.sort_index(axis=axis)
    else:
        return df


def fetch_df_by_index(
    df: pd.DataFrame,
    selector: Union[pd.Timestamp, slice, str, list, pd.Index],
    level: Union[str, int],
    fetch_orig=True,
) -> pd.DataFrame:
    # level = None -> use selector directly
    if level is None or isinstance(selector, pd.MultiIndex):
        return df.loc(axis=0)[selector]
    # Try to get the right index
    idx_slc = (selector, slice(None, None))
    if get_level_index(df, level) == 1:
        idx_slc = idx_slc[1], idx_slc[0]
    if fetch_orig:
        for slc in idx_slc:
            if slc != slice(None, None):
                return df.loc[
                    pd.IndexSlice[idx_slc],
                ]
        else:  # pylint: disable=W0120
            return df
    else:
        return df.loc[
            pd.IndexSlice[idx_slc],
        ]


def fetch_df_by_col(df: pd.DataFrame, col_set: Union[str, List[str]]) -> pd.DataFrame:

    if not isinstance(df.columns, pd.MultiIndex) or col_set == '__raw':
        return df
    elif col_set =='__all':
        return df.droplevel(axis=1, level=0)
    else:
        return df.loc(axis=1)[col_set]