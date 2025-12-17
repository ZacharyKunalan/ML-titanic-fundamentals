import pandas as pd

def preprocess(df):
    df = df.copy()

    features = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked"
    ]

    X = df[features].copy()

    # Missing values
    X["Age"] = X["Age"].fillna(X["Age"].median())
    X["Embarked"] = X["Embarked"].fillna(X["Embarked"].mode()[0])

    # Encoding
    X["Sex"] = X["Sex"].map({"male": 0, "female": 1})
    X = pd.get_dummies(X, columns=["Embarked"], drop_first=True, dtype=int)

    return X
