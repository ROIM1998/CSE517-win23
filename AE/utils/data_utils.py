import pandas as pd

def read_tsv(fn):
    return pd.read_csv(fn, sep='\t', header=None)


def data_split(df, train_ratio=0.8):
    df = df.sample(frac=1)
    train_df, eval_df = df[:int(len(df) * train_ratio)], df[int(len(df) * train_ratio):]
    return train_df, eval_df

def gen_eval_data(df):
    eval_data = []
    for i in range(len(df)):
        eval_data.append((df[0][i], df[1][i]))
    return eval_data


