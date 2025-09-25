"""
F1 Qualifying Speed Prediction Model
Main execution script with user interface
"""
__author__ = 'Natalie McWhorter'
__version__ = '09.25.2025'

#imports!!!
import os
import sys
from typing import Dict, List
import pandas as pd
from data_preprocessing import DataPreprocessor
from model_training import ModelTrainer
from prediction import F1QualifyingPredictor
from config import MODEL_CONFIG, OUTPUT_PATHS
from utils import create_directories, get_user_input, validate_weather_input
predictor = F1QualifyingPredictor()
from model_analysis import analyze_model_results


'''train_model()
uses the training algorithm from model_training.py to train a variety of models including:
    linear regression, random forest, gradient boosting, xgboost, neural network, and ensemble
'''
def train_model():
    print("\n🏎️  F1 QUALIFYING SPEED PREDICTION MODEL - TRAINING MODE")
    print("=" * 60)

    # Create necessary directories
    create_directories()

    # Initialize preprocessor
    preprocessor = DataPreprocessor()

    # Load and clean data
    preprocessor.load_data()
    preprocessor.clean_data()

    # Create features
    features_df = preprocessor.create_features()

    if len(features_df) == 0:
        print("❌ No features created. Please check your data files.")
        return False

    #train models
    trainer = ModelTrainer()
    X, y = trainer.prepare_data(features_df)
    results = trainer.train_models(X, y)
    trainer.print_results_summary()
    # Add this to the END of your existing training script
    # (after trainer.print_results_summary())

    # Get the test data from your trainer
    X_test, y_test = trainer.get_test_data()

    # Run the complete analysis
    print("\n" + "=" * 50)
    print("STARTING VISUALIZATION ANALYSIS")
    print("=" * 50)

    analyzer = analyze_model_results(
        models_path="../models/",
        output_path="../results/model_analysis/",
        X_test=X_test,
        y_test=y_test
    )

    print("\n🎉 Analysis complete! Check the 'analysis_charts' folder for your PNG files.")
    trainer.save_models()

    # Check if target R² achieved
    if trainer.best_score >= MODEL_CONFIG['target_r2']:
        print(f"\n🎉 SUCCESS! Target R² of {MODEL_CONFIG['target_r2']} achieved!")
        print(f"Best model R²: {trainer.best_score:.4f}")
        return True
    else:
        print(f"\n⚠️  Target R² of {MODEL_CONFIG['target_r2']} not achieved.")
        print(f"Best model R²: {trainer.best_score:.4f}")
        print("Consider collecting more data or adjusting features.")
        return False

def predict_qualifying():
    """Interactive prediction mode"""
    print("\n🏎️  F1 QUALIFYING SPEED PREDICTION MODEL - PREDICTION MODE")
    print("=" * 60)

    # Initialize predictor
    predictor = F1QualifyingPredictor()

    try:
        # Load trained model
        predictor.load_model()
        predictor.initialize_preprocessor()
    except FileNotFoundError:
        print("❌ No trained model found. Please run training mode first.")
        return

    while True:
        print("\nSelect prediction mode:")
        print("1. Predict for all drivers at a circuit")
        print("2. Predict for a single driver")
        print("3. Return to main menu")

        choice = input("\nEnter your choice (1-3): ").strip()

        if choice == '1':
            predict_all_drivers(predictor)
        elif choice == '2':
            predict_single_driver(predictor)
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


def predict_all_drivers(predictor: F1QualifyingPredictor):
    """Predict qualifying times for all drivers"""
    print("\n--- PREDICT ALL DRIVERS ---")

    # Get circuit
    circuit = get_user_input(
        "Enter circuit name (e.g., Monaco, Silverstone, Spa): ",
        str, required=True
    )

    # Get weather conditions
    weather = get_weather_conditions()

    # Get custom driver list (optional)
    custom_drivers = input(
        "\nEnter specific drivers (comma-separated) or press Enter for all current drivers: ").strip()
    drivers = None
    if custom_drivers:
        drivers = [driver.strip() for driver in custom_drivers.split(',')]

    try:
        # Make predictions
        predictions = predictor.predict_qualifying_times(circuit, weather, drivers)

        # Display results
        predictor.print_predictions(predictions)

        # Save results
        save_results = input("\nSave results to CSV? (y/n): ").strip().lower()
        if save_results == 'y':
            filename = f"{OUTPUT_PATHS['results']}{circuit.lower()}_predictions.csv"
            predictions.to_csv(filename, index=False)
            print(f"✓ Results saved to {filename}")

    except Exception as e:
        print(f"❌ Error making predictions: {e}")


def predict_single_driver(predictor: F1QualifyingPredictor):
    """Predict qualifying time for a single driver"""
    print("\n--- PREDICT SINGLE DRIVER ---")

    # Get driver name
    driver = get_user_input(
        "Enter driver name (e.g., Max Verstappen): ",
        str, required=True
    )

    # Get circuit
    circuit = get_user_input(
        "Enter circuit name (e.g., Monaco): ",
        str, required=True
    )

    # Get weather conditions
    weather = get_weather_conditions()

    try:
        # Make prediction
        prediction = predictor.predict_single_driver(driver, circuit, weather)
        #TODO
        # Display result
        print("\n" + "=" * 50)
        print("SINGLE DRIVER PREDICTION")
        print("=" * 50)
        print(f"Driver: {prediction['driver']}")
        print(f"Circuit: {prediction['circuit']}")
        print(f"Predicted Time: {prediction['predicted_time_formatted']}")
        print(f"Weather: {'Wet' if weather['is_raining'] else 'Dry'}")
        print("=" * 50)

    except Exception as e:
        print(f"❌ Error making prediction: {e}")


def get_weather_conditions() -> Dict:
    """Get weather conditions from user input"""
    print("\n--- WEATHER CONDITIONS ---")

    # Rain condition
    rain_input = input("Is it raining? (y/n): ").strip().lower()
    is_raining = 1 if rain_input == 'y' else 0
    # Precipitation for different sessions
    prcp_columns = ['race', 'quali', 'fp1', 'fp2', 'fp3', 'sprint']
    weather_conditions = {'is_raining': is_raining}

    for col in prcp_columns:
        prcp_input = input(f"Precipitation for {col.upper()} session (mm, 0 if none): ").strip()
        try:
            prcp_value = float(prcp_input)
            weather_conditions[f'{col}_prcp_binary'] = int(prcp_value > 0)
        except ValueError:
            weather_conditions[f'{col}_prcp_binary'] = 0

    return weather_conditions


def main():
    predictor.load_model()
    """Main program loop"""
    print("🏎️  Welcome to F1 Qualifying Speed Prediction Model!")

    while True:
        print("\n" + "=" * 60)
        print("MAIN MENU")
        print("=" * 60)
        print("1. Train Model")
        print("2. Make Predictions")
        print("3. Check Model Status")
        print("4. Exit")

        choice = input("\nEnter your choice (1-4): ").strip()

        if choice == '1':
            success = train_model()
            if success:
                input("\nPress Enter to continue...")

        elif choice == '2':
            predict_qualifying()

        elif choice == '3':
            check_model_status()
            input("\nPress Enter to continue...")

        elif choice == '4':
            print("\n👋 Thanks for using F1 Qualifying Prediction Model!")
            break

        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


def check_model_status():
    """Check if trained model exists and show info"""
    print("\n--- MODEL STATUS ---")

    model_path = f"{OUTPUT_PATHS['models']}best_model.pkl"

    if os.path.exists(model_path):
        print("✅ Trained model found!")

        # Try to load and show basic info
        try:
            import joblib
            model = joblib.load(model_path)
            print(f"Model type: {type(model).__name__}")

            # Check for other files
            files_to_check = ['scaler.pkl', 'label_encoders.pkl', 'feature_columns.pkl']
            for file in files_to_check:
                file_path = f"{OUTPUT_PATHS['models']}{file}"
                status = "✅" if os.path.exists(file_path) else "❌"
                print(f"{status} {file}")

        except Exception as e:
            print(f"⚠️  Model file exists but couldn't load details: {e}")
    else:
        print("❌ No trained model found.")
        print("Please run 'Train Model' first.")

    # Check data files
    print("\n--- DATA FILES STATUS ---")
    from config import DATA_PATHS

    for name, path in DATA_PATHS.items():
        status = "✅" if os.path.exists(path) else "❌"
        print(f"{status} {name}: {path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Program interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check your data files and try again.")