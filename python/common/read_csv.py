import pandas as pd
from typing import Union, Dict


def read_csv(filepath: str, dtype: Union[None, Dict[str, str]] = None) -> pd.DataFrame:
    return pd.read_csv(filepath, sep=',', decimal='.', encoding='utf-8',
                       index_col=None, dtype=dtype)
