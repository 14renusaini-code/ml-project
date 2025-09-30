import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_model(data, target_column):
    """Trains a Linear Regression model and evaluates performance."""
    try:
        # Splitting features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Splitting into train & test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Model evaluation
        print("\n Model Evaluation:")
        print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred):.4f}")
        print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.4f}")
        print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

        return model

    except Exception as e:
        print(f" Error in model training: {e}")
        return None

