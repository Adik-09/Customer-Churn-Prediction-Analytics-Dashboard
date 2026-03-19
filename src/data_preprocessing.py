import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.drop("customerID", axis=1, inplace=True)
    return df

def encode_data(df):
    return pd.get_dummies(df, drop_first=True)

def split_features_target(df):
    X = df.drop("Churn_Yes", axis=1)
    y = df["Churn_Yes"]
    return X, y