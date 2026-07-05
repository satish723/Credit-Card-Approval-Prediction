import pandas as pd
import numpy as np

def load_data(path):
    """Load CSV into DataFrame."""
    df = pd.read_csv(path)
    return df

def clean_dataframe(df):
    # drop exact duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # simple missing value strategy: numeric -> median, categorical -> mode
    for col in df.columns:
        if df[col].dtype.kind in 'biufc':
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        else:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mode().iloc[0])

    return df

def split_features_target(df, target_col=None):
    """Return X, y. If target_col is None, assume last column is target."""
    if target_col is None:
        target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def prepare_for_model(df, target_col=None, drop_columns=None):
    df = clean_dataframe(df)
    if drop_columns:
        df = df.drop(columns=drop_columns)
    X, y = split_features_target(df, target_col=target_col)

    # One-hot encode categorical columns
    X = pd.get_dummies(X, drop_first=True)

    return X, y
