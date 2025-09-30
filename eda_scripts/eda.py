import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def perform_eda(data):
    """Function to perform basic EDA on a given DataFrame"""
    print("Basic Info:")
    print(data.info())

    print("\nSummary Statistics:")
    print(data.describe())

    print("\nMissing Values:")
    print(data.isnull().sum())

    # Pairplot Example (Uncomment for visualization)
    # sns.pairplot(data)
    # plt.show()

# Testing EDA (Optional, can be removed in production)
if __name__ == "__main__":
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, None]})  # Sample Data
    perform_eda(df)
