# Configuration file for F1 Qualifying Speed Prediction Model

# Data paths
DATA_PATHS = {
    'lap_times': '../data/lap_times.csv',
    'weather': '../data/weather.csv',
    'drivers': '../data/drivers.csv',
    'circuits': '../data/circuits.csv',
    'qualifying_old': '../data/qualifying.csv',
    'results': '../data/results.csv',
    'races': '../data/races.csv',
    'qualifying': '../data/qualifying_enhanced.csv',
}

# Model parameters
MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    'target_r2': 0.9
}

# Feature engineering parameters
FEATURE_CONFIG = {
    'min_races_for_circuit_history': 1,  # Minimum races at circuit for historical analysis
    'min_races_for_general_history': 5,  # Minimum races overall for historical analysis
    'lookback_races': 5,  # Number of recent races to consider for form
    'weather_threshold': 0.01,  # Rain threshold (>0 = raining)
}

# Model types to try (for finding best model)
MODELS_TO_TEST = [
    'linear_regression',
    'random_forest',
    'gradient_boosting',
    'xgboost',
    'neural_network',
    'ensemble'
]

# Output paths
OUTPUT_PATHS = {
    'models': '../models/',
    'results': '../results/',
    'plots': '../results/plots/'
}