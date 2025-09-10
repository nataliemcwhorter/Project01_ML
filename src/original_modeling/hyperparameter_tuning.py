import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, make_scorer


def advanced_hyperparameter_tuning(model_data):
    """
    Perform advanced hyperparameter tuning for Gradient Boosting
    """
    X_train, X_test = model_data['X_train'], model_data['X_test']
    y_train, y_test = model_data['y_train'], model_data['y_test']

    # Define the parameter space
    param_distributions = {
        'n_estimators': [100, 200, 300, 400, 500],
        'learning_rate': [0.01, 0.1, 0.3, 0.5],
        'max_depth': [3, 4, 5, 6, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2'],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0]
    }

    # Create the model
    gb_model = GradientBoostingRegressor(random_state=42)

    # Create a custom scorer (MAE)
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    # Perform RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=gb_model,
        param_distributions=param_distributions,
        n_iter=100,  # Number of parameter settings that are sampled
        cv=5,  # 5-fold cross-validation
        scoring=mae_scorer,
        n_jobs=-1,  # Use all available cores
        random_state=42
    )

    # Fit the random search
    random_search.fit(X_train, y_train)

    # Best model
    best_model = random_search.best_estimator_

    # Evaluate best model
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = best_model.score(X_test, y_test)

    print("Best Hyperparameters:")
    for param, value in random_search.best_params_.items():
        print(f"{param}: {value}")

    print("\nBest Model Performance:")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R-squared Score: {r2:.4f}")

    return best_model, random_search.best_params_


# If running as a script
if __name__ == "__main__":
    from feature_engineering import load_and_merge_data, create_driver_features
    from model_prep import prepare_model_dataset

    # Load and prepare data
    df = load_and_merge_data()
    df_with_features = create_driver_features(df)
    model_data = prepare_model_dataset(df_with_features)

    # Perform hyperparameter tuning
    best_model, best_params = advanced_hyperparameter_tuning(model_data)