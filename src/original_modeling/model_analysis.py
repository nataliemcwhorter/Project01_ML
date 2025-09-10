import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor


def perform_cross_validation(X, y):
    """
    Perform cross-validation to get a more robust performance estimate
    """
    gb_model = GradientBoostingRegressor(random_state=42)

    # Perform 5-fold cross-validation
    cv_scores = cross_val_score(
        gb_model,
        X, y,
        cv=5,
        scoring='neg_mean_absolute_error'
    )

    # Convert to positive MAE
    cv_scores = -cv_scores

    print("Cross-Validation Results:")
    print(f"Mean MAE: {cv_scores.mean():.4f}")
    print(f"Standard Deviation: {cv_scores.std():.4f}")

    return cv_scores


def hyperparameter_tuning(X_train, X_test, y_train, y_test):
    """
    Perform grid search for hyperparameter tuning
    """
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.3],
        'max_depth': [3, 4, 5]
    }

    # Create GridSearchCV object
    gb_model = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=gb_model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1  # Use all available cores
    )

    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)

    # Best parameters and score
    print("\nBest Parameters:")
    print(grid_search.best_params_)

    # Evaluate best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nBest Model Performance:")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R-squared Score: {r2:.4f}")

    return best_model


# If running as a script
if __name__ == "__main__":
    from model_prep import prepare_model_dataset
    from feature_engineering import load_and_merge_data, create_driver_features

    # Load and prepare data
    df = load_and_merge_data()
    df_with_features = create_driver_features(df)
    model_data = prepare_model_dataset(df_with_features)

    # Perform cross-validation
    cv_scores = perform_cross_validation(
        model_data['X_train'],
        model_data['y_train']
    )

    # Hyperparameter tuning
    best_model = hyperparameter_tuning(
        model_data['X_train'],
        model_data['X_test'],
        model_data['y_train'],
        model_data['y_test']
    )