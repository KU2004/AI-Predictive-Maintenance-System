import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    """
    Load and process NASA turbofan dataset
    """

    # Column names
    columns = ["unit", "time"] + \
              [f"op{i}" for i in range(1, 4)] + \
              [f"s{i}" for i in range(1, 22)]

    # Load dataset
    data = pd.read_csv(path, sep=" ", header=None)

    # Remove extra empty columns
    data = data.dropna(axis=1)

    # Assign column names
    data.columns = columns

    # Calculate Remaining Useful Life (RUL)
    rul = data.groupby("unit")["time"].max().reset_index()
    rul.columns = ["unit", "max_time"]

    # Merge RUL with original data
    data = data.merge(rul, on="unit")

    # Calculate RUL
    data["RUL"] = data["max_time"] - data["time"]

    # Convert into classification (Failure / No Failure)
    data["failure"] = data["RUL"].apply(lambda x: 1 if x <= 30 else 0)

    return data


def preprocess_data(data):
    """
    Prepare training and testing data
    """

    # Select only sensor columns
    feature_columns = [f"s{i}" for i in range(1, 22)]

    X = data[feature_columns]
    y = data["failure"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test