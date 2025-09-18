import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import joblib
import os
from typing import Dict, List, Tuple, Any
from config import MODEL_CONFIG, MODELS_TO_TEST, OUTPUT_PATHS


class ModelTrainer:
    """Handles model training, evaluation, and selection"""

    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None

    def prepare_data(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training with proper scaling and encoding"""
        print("Preparing data for training...")

        # ALWAYS use qualifying time as target
        target_col = 'q1_time'  # Predicting fastest qualifying time

        if target_col not in features_df.columns:
            raise ValueError(
                f"Target column '{target_col}' not found in features. Available columns: {list(features_df.columns)}")

        print(f"Using '{target_col}' as target (qualifying time prediction)")

        # Remove non-feature columns - ALWAYS exclude position since we're predicting time
        exclude_cols = ['raceId', 'driverId', 'circuitId', 'constructorId', 'position', 'year']

        # Separate features and target
        feature_cols = [col for col in features_df.columns if col not in exclude_cols + [target_col]]

        X = features_df[feature_cols].copy()
        y = features_df[target_col].copy()

        print(f"Features selected: {len(feature_cols)} columns")
        print(f"Target column: {target_col}")
        print(f"Target sample values: {y.head().values}")
        print(f"Target range: {y.min():.3f} - {y.max():.3f} seconds")

        # Handle missing values in target first
        print(f"Missing values in target before cleaning: {y.isna().sum()}")
        y = y.dropna()
        X = X.loc[y.index]  # Align X with y after dropping NaN values
        print(f"Samples after removing missing targets: {len(y)}")

        # Handle missing values in features
        # For numeric columns, use median
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

        # For categorical columns, use mode or 'unknown'
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X[col] = X[col].fillna('unknown')

        # Handle categorical encoding
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            else:
                # Handle unseen categories during transform
                try:
                    X[col] = self.label_encoders[col].transform(X[col].astype(str))
                except ValueError:
                    # If there are unseen categories, refit the encoder
                    print(f"Warning: Refitting encoder for column {col} due to new categories")
                    X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))

        # Convert boolean columns to int
        bool_cols = X.select_dtypes(include=['bool']).columns
        if len(bool_cols) > 0:
            X[bool_cols] = X[bool_cols].astype(int)

        # Ensure all columns are numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                print(f"Warning: Column {col} still has object dtype, converting to numeric")
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

        # Store feature columns for later use
        self.feature_columns = X.columns.tolist()
        print(f"Final feature columns: {self.feature_columns}")

        # Initialize scaler if not already done
        if self.scaler is None:
            self.scaler = StandardScaler()

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        print(f"✓ Data prepared: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
        print(f"Final target range: {y.min():.3f} - {y.max():.3f} seconds")
        print(f"✓ Scaler fitted and ready for saving")
        print(f"✓ {len(self.label_encoders)} label encoders fitted")

        return X_scaled, y.values

    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict]:
        """Train multiple models and compare performance"""
        print("Training models...")

        # Add training diagnostics
        self._print_training_diagnostics(X, y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=MODEL_CONFIG['test_size'],
            random_state=MODEL_CONFIG['random_state']
        )

        results = {}

        for model_name in MODELS_TO_TEST:
            print(f"\nTraining {model_name}...")

            try:
                model, params = self._get_model_and_params(model_name)

                if params:
                    # Use GridSearchCV for hyperparameter tuning
                    grid_search = GridSearchCV(
                        model, params, cv=MODEL_CONFIG['cv_folds'],
                        scoring='r2', n_jobs=-1
                    )
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                    print(f"Best params: {grid_search.best_params_}")
                else:
                    # Train with default parameters
                    best_model = model
                    best_model.fit(X_train, y_train)

                # Evaluate model
                train_pred = best_model.predict(X_train)
                test_pred = best_model.predict(X_test)

                train_r2 = r2_score(y_train, train_pred)
                test_r2 = r2_score(y_test, test_pred)
                test_mae = mean_absolute_error(y_test, test_pred)
                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

                # Cross-validation score
                cv_scores = cross_val_score(
                    best_model, X_train, y_train,
                    cv=MODEL_CONFIG['cv_folds'], scoring='r2'
                )

                results[model_name] = {
                    'model': best_model,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'cv_r2_mean': cv_scores.mean(),
                    'cv_r2_std': cv_scores.std(),
                    'test_mae': test_mae,
                    'test_rmse': test_rmse
                }

                print(f"✓ {model_name} - Test R²: {test_r2:.4f}, CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

                # Update best model if this one is better
                if test_r2 > self.best_score:
                    self.best_score = test_r2
                    self.best_model = best_model
                    print(f"🏆 New best model: {model_name}")

            except Exception as e:
                print(f"❌ Error training {model_name}: {e}")
                continue

        # Train ensemble if multiple models succeeded
        if len(results) > 1:
            ensemble_result = self._train_ensemble(results, X_train, X_test, y_train, y_test)
            if ensemble_result:
                results['ensemble'] = ensemble_result

        self.models = results
        return results

    def _print_training_diagnostics(self, X: np.ndarray, y: np.ndarray):
        """Print comprehensive diagnostics on training data"""
        print("\n=== MODEL TRAINING DIAGNOSTICS ===")

        # Print input data characteristics
        print(f"Input Features Shape: {X.shape}")
        print(f"Target Values Shape: {y.shape}")
        print(f"Target Value Range: {y.min():.3f} - {y.max():.3f}")
        print(f"Target Mean: {np.mean(y):.3f}")
        print(f"Target Median: {np.median(y):.3f}")

        # Feature distribution
        print("\n=== FEATURE DISTRIBUTION ===")
        feature_means = np.mean(X, axis=0)
        feature_stds = np.std(X, axis=0)

        print(f"Feature means range: {feature_means.min():.3f} - {feature_means.max():.3f}")
        print(f"Feature stds range: {feature_stds.min():.3f} - {feature_stds.max():.3f}")

        # Check for potential issues
        zero_std_features = np.sum(feature_stds == 0)
        if zero_std_features > 0:
            print(f"⚠️  Warning: {zero_std_features} features have zero standard deviation")

        high_std_features = np.sum(feature_stds > 10)
        if high_std_features > 0:
            print(f"⚠️  Warning: {high_std_features} features have very high standard deviation (>10)")

    def analyze_feature_importance(self):
        """Analyze and print feature importance for the best model"""
        print("\n=== FEATURE IMPORTANCE ANALYSIS ===")

        if self.best_model is None:
            print("No best model available for feature importance analysis")
            return

        if self.feature_columns is None:
            print("No feature columns available for analysis")
            return

        try:
            # For tree-based models (Random Forest, Gradient Boosting, XGBoost)
            if hasattr(self.best_model, 'feature_importances_'):
                importances = self.best_model.feature_importances_
                indices = np.argsort(importances)[::-1]

                print("Top 10 Most Important Features:")
                for f in range(min(10, len(indices))):
                    print(f"{self.feature_columns[indices[f]]}: {importances[indices[f]]:.4f}")

            # For linear models (Linear Regression)
            elif hasattr(self.best_model, 'coef_'):
                coef_importance = np.abs(self.best_model.coef_)
                indices = np.argsort(coef_importance)[::-1]

                print("Top 10 Most Influential Features:")
                for f in range(min(10, len(indices))):
                    print(f"{self.feature_columns[indices[f]]}: {coef_importance[indices[f]]:.4f}")
            else:
                print("Current model does not support feature importance analysis")

        except Exception as e:
            print(f"Could not analyze feature importance: {e}")

    def _get_model_and_params(self, model_name: str) -> Tuple[Any, Dict]:
        """Get model instance and hyperparameter grid"""
        if model_name == 'linear_regression':
            return LinearRegression(), {}

        elif model_name == 'random_forest':
            model = RandomForestRegressor(random_state=MODEL_CONFIG['random_state'])
            params = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
            return model, params

        elif model_name == 'gradient_boosting':
            model = GradientBoostingRegressor(random_state=MODEL_CONFIG['random_state'])
            params = {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
            return model, params

        elif model_name == 'xgboost':
            model = xgb.XGBRegressor(random_state=MODEL_CONFIG['random_state'])
            params = {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
            return model, params

        elif model_name == 'neural_network':
            model = MLPRegressor(random_state=MODEL_CONFIG['random_state'], max_iter=1000)
            params = {
                'hidden_layer_sizes': [(100,), (100, 50), (200, 100)],
                'alpha': [0.0001, 0.001, 0.01]
            }
            return model, params

        else:
            raise ValueError(f"Unknown model: {model_name}")

    def _train_ensemble(self, models_results: Dict, X_train: np.ndarray,
                        X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> Dict:
        """Train ensemble model using best performing models"""
        print("\nTraining ensemble model...")

        try:
            # Select top 3 models based on test R²
            top_models = sorted(models_results.items(),
                                key=lambda x: x[1]['test_r2'], reverse=True)[:3]

            # Create ensemble predictions
            train_preds = []
            test_preds = []

            for model_name, result in top_models:
                model = result['model']
                train_preds.append(model.predict(X_train))
                test_preds.append(model.predict(X_test))

            # Average predictions
            ensemble_train_pred = np.mean(train_preds, axis=0)
            ensemble_test_pred = np.mean(test_preds, axis=0)

            # Evaluate ensemble
            train_r2 = r2_score(y_train, ensemble_train_pred)
            test_r2 = r2_score(y_test, ensemble_test_pred)
            test_mae = mean_absolute_error(y_test, ensemble_test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, ensemble_test_pred))

            print(f"✓ Ensemble - Test R²: {test_r2:.4f}")

            if test_r2 > self.best_score:
                self.best_score = test_r2
                print(f"🏆 New best model: ensemble")

            return {
                'models': [result['model'] for _, result in top_models],
                'model_names': [name for name, _ in top_models],
                'train_r2': train_r2,
                'test_r2': test_r2,
                'cv_r2_mean': test_r2,  # Approximation
                'cv_r2_std': 0,
                'test_mae': test_mae,
                'test_rmse': test_rmse
            }

        except Exception as e:
            print(f"❌ Error training ensemble: {e}")
            return None

    def save_models(self) -> None:
        """Save trained models and preprocessing objects"""
        print("Saving models...")

        os.makedirs(OUTPUT_PATHS['models'], exist_ok=True)

        # Save best model
        if self.best_model is not None:
            joblib.dump(self.best_model, f"{OUTPUT_PATHS['models']}best_model.pkl")

        # Save preprocessing objects
        joblib.dump(self.scaler, f"{OUTPUT_PATHS['models']}scaler.pkl")
        joblib.dump(self.label_encoders, f"{OUTPUT_PATHS['models']}label_encoders.pkl")
        joblib.dump(self.feature_columns, f"{OUTPUT_PATHS['models']}feature_columns.pkl")

        # Save all models
        for name, result in self.models.items():
            if name != 'ensemble':  # Ensemble is handled separately
                joblib.dump(result['model'], f"{OUTPUT_PATHS['models']}{name}.pkl")

        print(f"✓ Models saved to {OUTPUT_PATHS['models']}")

    def print_results_summary(self) -> None:
        """Print summary of all model results"""
        print("\n" + "=" * 60)
        print("MODEL TRAINING RESULTS SUMMARY")
        print("=" * 60)

        if not self.models:
            print("No models trained successfully.")
            return

        # Sort by test R²
        sorted_models = sorted(self.models.items(),
                               key=lambda x: x[1]['test_r2'], reverse=True)

        print(f"{'Model':<20} {'Test R²':<10} {'CV R²':<15} {'MAE':<10} {'RMSE':<10}")
        print("-" * 70)

        for name, result in sorted_models:
            cv_score = f"{result['cv_r2_mean']:.4f} ± {result['cv_r2_std']:.4f}"
            print(f"{name:<20} {result['test_r2']:<10.4f} {cv_score:<15} "
                  f"{result['test_mae']:<10.4f} {result['test_rmse']:<10.4f}")

        print("-" * 70)
        best_name = sorted_models[0][0]
        best_r2 = sorted_models[0][1]['test_r2']

        print(f"🏆 Best Model: {best_name} (R² = {best_r2:.4f})")

        if best_r2 >= MODEL_CONFIG['target_r2']:
            print(f"✅ Target R² of {MODEL_CONFIG['target_r2']} achieved!")
        else:
            print(f"⚠️  Target R² of {MODEL_CONFIG['target_r2']} not yet achieved.")
            print("   Consider: more data, feature engineering, or different models.")

        # Add feature importance analysis
        self.analyze_feature_importance()
