import pandas as pd
import numpy as np


def read_xls(filename: str):
    df = pd.read_excel(filename, sheet_name="Sheet1", header=None, skiprows=1)
    return list(df.values[0])
