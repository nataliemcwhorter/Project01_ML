import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer  # Add this import
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline  # Add this import


def create_advanced_features(df):
    """
    Create sophisticated features for F1 qualifying prediction
    """
    # Existing feature engineering
    df = create_driver_features(df)

    # Temporal Performance Trends
    df['driver_performance_trend'] = df.groupby('driverId')['position'].diff().fillna(0)

    # Constructor Performance Interaction
    df['driver_constructor_interaction'] = df['driver_avg_position'] * df.groupby('constructorId')[
        'position'].transform('mean').fillna(df['driver_avg_position'])

    # Weighted Recent Performance
    df['weighted_recent_performance'] = (
            df['driver_recent_avg'].fillna(df['driver_recent_avg'].mean()) * 0.6 +
            df.groupby('driverId')['position'].transform(lambda x: x.rolling(window=3, min_periods=1).mean()).fillna(
                df['position'].mean()) * 0.4
    )

    # Wet/Dry Performance Ratio
    df['wet_dry_performance_ratio'] = (df['driver_wet_avg'].fillna(df['driver_wet_avg'].mean()) /
                                       (df['driver_dry_avg'].fillna(df['driver_dry_avg'].mean()) + 1))

    # Qualifying Time Complexity
    for col in ['q1', 'q2', 'q3']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['qualifying_time_variance'] = df[['q1', 'q2', 'q3']].var(axis=1).fillna(0)

    # Fill any remaining NaNs
    df = df.fillna(df.mean())

    return df


def prepare_chronological_dataset(df, test_ratio=0.2):
    """
    Prepare dataset with chronological train/test split
    """
    # Sort by date to ensure chronological order
    df_sorted = df.sort_values('date')

    # Calculate split point
    split_index = int(len(df_sorted) * (1 - test_ratio))

    # Split datasets
    train_data = df_sorted.iloc[:split_index].copy()
    test_data = df_sorted.iloc[split_index:].copy()

    # Feature columns
    feature_columns = [
        'driverId',
        'constructorId',
        'circuitId',
        'is_raining',
        'driver_avg_position',
        'driver_circuit_avg',
        'driver_wet_avg',
        'driver_dry_avg',
        'driver_recent_avg',
        'driver_performance_trend',
        'driver_constructor_interaction',
        'weighted_recent_performance',
        'wet_dry_performance_ratio',
        'qualifying_time_variance'
    ]

    # Encode categorical variables
    label_encoders = {}
    for col in ['driverId', 'constructorId', 'circuitId']:
        # Get unique values from training data
        unique_train_values = train_data[col].unique()

        # Create label encoder for training data
        le = LabelEncoder()
        le.fit(unique_train_values)

        # Transform training data
        train_data[col] = le.transform(train_data[col])

        # For test data, replace unseen labels with most common label
        test_data[col] = test_data[col].map(
            lambda x: le.transform([x])[0] if x in unique_train_values else le.transform([unique_train_values[0]])[0])

        label_encoders[col] = le

    # Prepare features and target
    X_train = train_data[feature_columns]
    y_train = train_data['position']
    X_test = test_data[feature_columns]
    y_test = test_data['position']

    # Create pipeline with imputation and scaling
    preprocessing_pipeline = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler()
    )

    # Fit and transform
    X_train_processed = preprocessing_pipeline.fit_transform(X_train)
    X_test_processed = preprocessing_pipeline.transform(X_test)

    return {
        'X_train': X_train_processed,
        'X_test': X_test_processed,
        'y_train': y_train,
        'y_test': y_test,
        'preprocessor': preprocessing_pipeline,
        'label_encoders': label_encoders
    }


def stacked_ensemble_model(model_data):
    """
    Create a stacked ensemble model with advanced features
    """
    X_train, X_test = model_data['X_train'], model_data['X_test']
    y_train, y_test = model_data['y_train'], model_data['y_test']

    # Base models
    base_models = [
        ('gb', GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=7,
            random_state=42
        )),
        ('rf', RandomForestRegressor(
            n_estimators=500,
            max_depth=7,
            random_state=42
        ))
    ]

    # Prepare meta-features
    meta_features = np.column_stack([
        cross_val_predict(model, X_train, y_train, cv=5)
        for name, model in base_models
    ])

    # Meta-model
    meta_model = LinearRegression()
    meta_model.fit(meta_features, y_train)

    # Train base models on full data
    base_predictions = np.column_stack([
        model.fit(X_train, y_train).predict(X_test)
        for name, model in base_models
    ])

    # Final predictions
    final_predictions = meta_model.predict(base_predictions)

    # Evaluate
    mae = mean_absolute_error(y_test, final_predictions)
    r2 = r2_score(y_test, final_predictions)

    print("Stacked Ensemble Results:")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R-squared Score: {r2:.4f}")

    return final_predictions


# Main execution
if __name__ == "__main__":
    from feature_engineering import load_and_merge_data, create_driver_features

    # Load and prepare data
    df = load_and_merge_data()
    df_with_features = create_driver_features(df)
    df_with_advanced_features = create_advanced_features(df_with_features)

    # Prepare chronological dataset
    model_data = prepare_chronological_dataset(df_with_advanced_features)

    # Run stacked ensemble
    stacked_ensemble_model(model_data)