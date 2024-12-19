import os
import pandas as pd
from sklearn.model_selection import train_test_split


# Load numerical dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


# Preprocess and split data
def prepare_data(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Define file paths
    file_path = r"C:\Users\anura\Desktop\Project 4- Climate Change Modeling\data\processed\regression_data.csv"
    save_dir = r"C:\Users\anura\Desktop\Project 4- Climate Change Modeling\data\data splits\regression data"

    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Load and prepare data
    df = load_data(file_path)
    X_train, X_test, y_train, y_test = prepare_data(df, target_column="commentsCount")

    # Save prepared data to the specified directory
    X_train.to_csv(os.path.join(save_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(save_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(save_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(save_dir, "y_test.csv"), index=False)
