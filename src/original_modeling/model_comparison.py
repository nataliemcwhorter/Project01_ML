# Create model_comparison.py in src/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


def compare_models(model_data):
    """
    Compare multiple regression models
    """
    X_train, X_test = model_data['X_train'], model_data['X_test']
    y_train, y_test = model_data['y_train'], model_data['y_test']

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Support Vector Regression': SVR(kernel='rbf'),
        'XGBoost': XGBRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }

    # Store results
    results = {}

    # Evaluate each model
    for name, model in models.items():
        # Fit the model
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred = model.predict(X_test_scaled)

        # Evaluate
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[name] = {
            'MAE': mae,
            'R2': r2
        }

        print(f"{name}:")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"R-squared Score: {r2:.4f}\n")

    # Visualize results
    plt.figure(figsize=(10, 6))

    # MAE Plot
    plt.subplot(1, 2, 1)
    mae_values = [results[model]['MAE'] for model in models]
    plt.bar(models.keys(), mae_values)
    plt.title('Mean Absolute Error')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # R2 Plot
    plt.subplot(1, 2, 2)
    r2_values = [results[model]['R2'] for model in models]
    plt.bar(models.keys(), r2_values)
    plt.title('R-squared Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.savefig('../notebooks/model_comparison.png')
    plt.close()

    return results


# If running as a script
if __name__ == "__main__":
    from feature_engineering import load_and_merge_data, create_driver_features
    from model_prep import prepare_model_dataset
    from sklearn.ensemble import GradientBoostingRegressor

    # Load and prepare data
    df = load_and_merge_data()
    df_with_features = create_driver_features(df)
    model_data = prepare_model_dataset(df_with_features)

    # Compare models
    comparison_results = compare_models(model_data)