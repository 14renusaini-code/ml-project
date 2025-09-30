import pandas as pd
from eda_scripts.act_eda import (
    load_data, basic_info, summary_statistics, missing_values,
    duplicate_values, correlation_matrix, data_distribution,
    detect_outliers, pairplot
)
from model_scripts.model import train_model

# Load Dataset (Replace with your actual file name)
file_path = r"E:\MyProject\boston_data.csv"
df = load_data(file_path)

if df is not None:
    # Perform EDA
    basic_info(df)
    summary_statistics(df)
    missing_values(df)
    duplicate_values(df)
    correlation_matrix(df)
    data_distribution(df)
    detect_outliers(df)
    pairplot(df)

    # Train Model (Specify Target Column)
    target_column = "PRICE"  # Change based on your dataset
    train_model(df, target_column)
