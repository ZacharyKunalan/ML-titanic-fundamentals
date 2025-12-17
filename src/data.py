import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path="data/train.csv"):
    df = pd.read_csv(path)
    return df


def split_features_target(df, target="Survived"):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def train_test_split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
