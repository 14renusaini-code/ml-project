import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def basic_info(data):
    """Displays basic information about the DataFrame."""
    print(" Basic Info:")
    print(data.info())

def summary_statistics(data):
    """Displays summary statistics of the DataFrame."""
    print("\n Summary Statistics:")
    print(data.describe())

def missing_values(data):
    """Displays missing values in the DataFrame."""
    print("\n Missing Values:")
    print(data.isnull().sum())

def duplicate_values(data):
    """Displays duplicate values in the DataFrame."""
    print("\n Duplicate Values:", data.duplicated().sum())

def correlation_matrix(data):
    """Displays the correlation matrix as a heatmap."""
    print("\n Correlation Matrix:")
    plt.figure(figsize=(8,6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

def data_distribution(data):
    """Displays the distribution of numerical columns."""
    print("\n Data Distribution:")
    data.hist(figsize=(10, 6), bins=30)
    plt.suptitle("Feature Distributions")
    plt.show()

def detect_outliers(data):
    """Displays boxplots to detect outliers."""
    print("\n Outlier Detection:")
    plt.figure(figsize=(10,6))
    sns.boxplot(data=data)
    plt.title("Boxplots for Outlier Detection")
    plt.xticks(rotation=45)
    plt.show()

def pairplot(data):
    """Displays a pairplot for numerical columns."""
    print("\n Pairplot:")
    sns.pairplot(data)
    plt.show()

# If you want to test the functions within this file, you can add:
if __name__ == "__main__":
    df = pd.DataFrame({
        "A": [1, 2, 3, 4, 5, 100],  # Outlier in A
        "B": [10, 20, 30, 40, 50, 60],
        "C": [5, 15, 25, 35, None, 55]  # Missing Value in C
    })
    basic_info(df)
    summary_statistics(df)
    missing_values(df)
    duplicate_values(df)
    correlation_matrix(df)
    data_distribution(df)
    detect_outliers(df)
    pairplot(df)
