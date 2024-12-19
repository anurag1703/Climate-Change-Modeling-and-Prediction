import os
import pandas as pd

# Load and prepare time-series data
def prepare_time_series_data(file_path, target_column, date_column):
    df = pd.read_csv(file_path)
    df[date_column] = pd.to_datetime(df[date_column])
    df.set_index(date_column, inplace=True)
    return df[[target_column]]

if __name__ == "__main__":
    # File paths
    file_path = r"C:\Users\anura\Desktop\Project 4- Climate Change Modeling\data\processed\trend_data.csv"
    save_dir = r"C:\Users\anura\Desktop\Project 4- Climate Change Modeling\data\data splits\trend data"

    # Ensure the target directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Prepare the trend data
    target_column = "likesCount"
    date_column = "date"
    ts_data = prepare_time_series_data(file_path, target_column, date_column)

    # Save the prepared data
    save_path = os.path.join(save_dir, "prepared_trend_data.csv")
    ts_data.to_csv(save_path)
    print(f"Prepared data saved at {save_path}")
