

import os
import pandas as pd
from ast import literal_eval


def concat_results(path, sheet_name='Overall_Best_Solution'):

    files = [f for f in os.listdir(path) if f.endswith('.xlsx')]
    dfs = [
        pd.read_excel(os.path.join(path, file_), sheet_name)
        for file_ in files
    ]
    df_best = pd.concat(dfs)
    df_best['Representation'] = df_best['Representation'].apply(literal_eval)
    return df_best
