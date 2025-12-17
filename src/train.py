import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from preprocessing import preprocess
from scratch.logistic_regression_from_scratch import LogisticRegressionScratch

# Load data
df = pd.read_csv("data/train.csv")

# Preprocess
X = preprocess(df)
y = df["Survived"]

# Train / validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegressionScratch(learning_rate=0.01, epochs=1000)
model.fit(X_train.values, y_train.values)

# Evaluate
y_pred = model.predict(X_val.values)
accuracy = accuracy_score(y_val, y_pred)

print(f"Validation accuracy (from scratch): {accuracy:.3f}")
