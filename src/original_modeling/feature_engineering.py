import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_merge_data():
    """Load and merge all datasets"""
    df_races = pd.read_csv('../../data/races.csv')
    df_qualifying = pd.read_csv('../../data/qualifying.csv')
    df_weather = pd.read_csv('../../data/weather.csv')

    # Basic merge
    df_work = df_qualifying.merge(df_races[['raceId', 'circuitId', 'year', 'date']], on='raceId')
    df_work = df_work.merge(df_weather[['raceId', 'race_prcp', 'quali_prcp']], on='raceId')

    # Filter and clean
    df_work = df_work[df_work['year'] >= 2000]
    df_work['is_raining'] = (pd.to_numeric(df_work['race_prcp'], errors='coerce') > 0).astype(int)

    return df_work


def create_driver_features(df):
    """Create comprehensive driver performance features"""

    # Sort by date so we can calculate rolling averages
    df = df.sort_values(['driverId', 'date']).copy()
    df = df.reset_index(drop=True)  # Reset index to avoid conflicts

    # Initialize new columns
    feature_columns = [
        'driver_avg_position',
        'driver_circuit_avg',
        'driver_wet_avg',
        'driver_dry_avg',
        'driver_recent_avg'
    ]
    for col in feature_columns:
        df[col] = np.nan

    # Process each row
    for i in range(len(df)):
        current_driver = df.loc[i, 'driverId']
        current_circuit = df.loc[i, 'circuitId']
        current_date = df.loc[i, 'date']

        # Get all previous races for this driver
        prev_races = df[(df['driverId'] == current_driver) & (df['date'] < current_date)]

        if len(prev_races) > 0:
            # 1. Overall driver average position
            df.loc[i, 'driver_avg_position'] = prev_races['position'].mean()

            # 2. Driver average at this specific circuit
            prev_circuit_races = prev_races[prev_races['circuitId'] == current_circuit]
            if len(prev_circuit_races) > 0:
                df.loc[i, 'driver_circuit_avg'] = prev_circuit_races['position'].mean()

            # 3. Wet weather performance
            wet_races = prev_races[prev_races['is_raining'] == 1]
            if len(wet_races) > 0:
                df.loc[i, 'driver_wet_avg'] = wet_races['position'].mean()

            # 4. Dry weather performance
            dry_races = prev_races[prev_races['is_raining'] == 0]
            if len(dry_races) > 0:
                df.loc[i, 'driver_dry_avg'] = dry_races['position'].mean()

            # 5. Recent form (last 5 races)
            recent_races = prev_races.tail(5)
            if len(recent_races) > 0:
                df.loc[i, 'driver_recent_avg'] = recent_races['position'].mean()
    return df


def analyze_features(df):
    # Select the feature columns we created
    feature_columns = [
        'driver_avg_position',
        'driver_circuit_avg',
        'driver_wet_avg',
        'driver_dry_avg',
        'driver_recent_avg',
        'position'  # Include target variable
    ]

    # Create a correlation matrix
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[feature_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('../notebooks/feature_correlation.png')
    plt.close()

    # Distribution of features
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(feature_columns, 1):
        plt.subplot(2, 3, i)
        df[col].hist(bins=30)
        plt.title(f'Distribution of {col}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('../notebooks/feature_distributions.png')
    plt.close()

    # Print out some basic statistics
    print("Feature Statistics:")
    print(df[feature_columns].describe())

    # Check for missing values
    print("\nMissing Values:")
    print(df[feature_columns].isnull().sum())

if __name__ == "__main__":
    # Test the functions
    df = load_and_merge_data()
    #print(f"Dataset loaded: {df.shape}")
    # NOW ACTUALLY CALL THE FEATURE FUNCTION
    df_with_features = create_driver_features(df)
    #print(f"Features created: {df_with_features.shape}")
    #print(f"New columns: {df_with_features.columns.tolist()}")
    # Show a sample
    #print("\nSample with features:")
    #print(df_with_features[['driverId', 'circuitId', 'position', 'driver_avg_position', 'driver_circuit_avg']].head(10))
    analyze_features(df_with_features)