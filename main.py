from src.data import load_data, split_features_target, train_test_split_data


def main():
    df = load_data()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    print("Data loaded successfully")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")


if __name__ == "__main__":
    main()
