import pandas as pd
import numpy as np


def read_xls(filename: str, input_row: int = 2):
    df = pd.read_excel(filename, sheet_name="Sheet1", header=None, skiprows=input_row-1,nrows=1)
    _x = df.dropna(axis=1)
    return list(_x.values[0])
