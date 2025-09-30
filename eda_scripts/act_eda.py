import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path):
    """Loads dataset from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        print(" Dataset loaded successfully!\n")
        return data
    except Exception as e:
        print(f" Error loading dataset: {e}")
        return None

def basic_info(data):
    """Displays basic information about the dataset."""
    print("\n Basic Info:")
    print(data.info())

def summary_statistics(data):
    """Displays summary statistics of the dataset."""
    print("\n Summary Statistics:")
    print(data.describe())

def missing_values(data):
    """Checks for missing values in the dataset."""
    print("\n Missing Values:")
    print(data.isnull().sum())

def duplicate_values(data):
    """Checks for duplicate rows in the dataset."""
    print("\n Duplicate Values:", data.duplicated().sum())

def correlation_matrix(data):
    """Plots a correlation heatmap."""
    print("\n Correlation Matrix:")
    plt.figure(figsize=(8,6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

def data_distribution(data):
    """Plots the distribution of numerical columns."""
    print("\n Data Distribution:")
    data.hist(figsize=(10, 6), bins=30)
    plt.suptitle("Feature Distributions")
    plt.show()

def detect_outliers(data):
    """Plots boxplots to detect outliers."""
    print("\n Outlier Detection:")
    plt.figure(figsize=(10,6))
    sns.boxplot(data=data)
    plt.title("Boxplots for Outlier Detection")
    plt.xticks(rotation=45)
    plt.show()

def pairplot(data):
    """Creates a pairplot to visualize relationships between numerical variables."""
    print("\n Pairplot:")
    sns.pairplot(data)
    plt.show()
