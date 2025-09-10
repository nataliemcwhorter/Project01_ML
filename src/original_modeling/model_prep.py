import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


def prepare_model_dataset(df):
    """
    Prepare dataset for gradient boosting model

    Steps:
    1. Handle missing values
    2. Encode categorical variables
    3. Split into features and target
    4. Create train/test split
    """

    # Select features
    feature_columns = [
        'driverId',
        'constructorId',
        'circuitId',
        'is_raining',
        'driver_avg_position',
        'driver_circuit_avg',
        'driver_wet_avg',
        'driver_dry_avg',
        'driver_recent_avg'
    ]

    # Create a copy to avoid modifying original dataframe
    df_model = df[feature_columns + ['position']].copy()

    # Handle missing values - fill with median
    for col in feature_columns:
        if df_model[col].isnull().sum() > 0:
            df_model[col] = df_model[col].fillna(df_model[col].median())

    # Encode categorical variables
    categorical_cols = ['driverId', 'constructorId', 'circuitId']
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        label_encoders[col] = le

    # Separate features and target
    X = df_model[feature_columns]
    y = df_model['position']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,  # 80% train, 20% test
        random_state=42  # for reproducibility
    )

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'label_encoders': label_encoders
    }


def train_gradient_boosting_model(model_data):
    """
    Train Gradient Boosting Regressor and evaluate performance
    """
    # Extract data
    X_train = model_data['X_train']
    X_test = model_data['X_test']
    y_train = model_data['y_train']
    y_test = model_data['y_test']

    # Initialize and train the model
    gb_model = GradientBoostingRegressor(
        n_estimators=100,  # Number of boosting stages
        learning_rate=0.1,  # Shrinks the contribution of each tree
        max_depth=3,  # Maximum depth of individual trees
        random_state=42
    )

    gb_model.fit(X_train, y_train)

    # Make predictions
    y_pred = gb_model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Model Performance:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R-squared Score: {r2:.4f}")

    # Feature importance
    feature_importance = gb_model.feature_importances_
    feature_names = X_train.columns

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0])
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, [feature_names[i] for i in sorted_idx])
    plt.title('Feature Importance in Gradient Boosting Model')
    plt.tight_layout()
    plt.savefig('../notebooks/feature_importance.png')
    plt.close()

    # Scatter plot of predicted vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Qualifying Position')
    plt.ylabel('Predicted Qualifying Position')
    plt.title('Actual vs Predicted Qualifying Positions')
    plt.tight_layout()
    plt.savefig('../notebooks/actual_vs_predicted.png')
    plt.close()

    return gb_model


# If running as a script
if __name__ == "__main__":
    from model_prep import prepare_model_dataset
    from feature_engineering import load_and_merge_data, create_driver_features

    # Load and prepare data
    df = load_and_merge_data()
    df_with_features = create_driver_features(df)
    model_data = prepare_model_dataset(df_with_features)

    # Train and evaluate model
    model = train_gradient_boosting_model(model_data)

