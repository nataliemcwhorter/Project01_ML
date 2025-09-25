import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from config import DATA_PATHS, FEATURE_CONFIG
from feature_engineering import create_driver_features, DriverPerformanceAnalyzer


class DataPreprocessor:
    """Handles data loading, cleaning, and preprocessing"""

    def __init__(self):
        self.data = {}
        self.analyzer = None

    def load_data(self) -> None:
        """Load all required datasets"""
        print("Loading datasets...")

        for key, path in DATA_PATHS.items():
            try:
                self.data[key] = pd.read_csv(path)
                print(f"✓ Loaded {key}: {len(self.data[key])} rows")
            except FileNotFoundError:
                print(f"⚠ Warning: {path} not found. Skipping {key}.")
                self.data[key] = pd.DataFrame()

    def clean_data(self) -> None:
        """Clean and prepare data for analysis"""
        print("Cleaning data...")

        # Clean qualifying data
        if not self.data['qualifying'].empty:
            # Create a copy to avoid SettingWithCopyWarning
            qualifying_df = self.data['qualifying'].copy()

            # Convert time strings to seconds
            qualifying_df = self._convert_times_to_seconds(qualifying_df)

            print(f"Total rows before cleaning: {len(qualifying_df)}")

            # Drop rows where all Q1, Q2, Q3 are NaN
            qualifying_df.dropna(subset=['q1', 'q2', 'q3'], how='all', inplace=True)

            # Handle NaN times more safely
            for col in ['q1', 'q2', 'q3']:
                if col in qualifying_df.columns:
                    # Calculate median, ignoring NaN values
                    median_time = qualifying_df[col].median()

                    # Fill NaN values with median
                    qualifying_df[col] = qualifying_df[col].fillna(median_time)

                    # Print diagnostic information
                    print(f"{col} column:")
                    print(f"  Median time: {median_time:.3f}")
                    print(f"  Missing values after filling: {qualifying_df[col].isna().sum()}")

            print(f"Total rows after cleaning: {len(qualifying_df)}")
            print("✓ Qualifying data cleaned successfully")

            # Update the original data with the cleaned DataFrame
            self.data['qualifying'] = qualifying_df

    def create_features(self) -> pd.DataFrame:
        """Create feature matrix for training"""
        print("Creating features...")

        if self.data['qualifying'].empty:
            raise ValueError("No qualifying data available for feature creation")

        # Verify required columns
        required_columns = [
            'raceId', 'driverId', 'constructorId', 'position',
            'q1', 'q2', 'q3', 'year', 'circuitId', 'date'
        ]

        # Check if all required columns are present
        missing_columns = [col for col in required_columns if col not in self.data['qualifying'].columns]
        if missing_columns:
            print("Diagnostic information:")
            print("Qualifying columns:", self.data['qualifying'].columns)
            print("Missing columns:", missing_columns)
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Initialize analyzer for feature engineering
        self.analyzer = DriverPerformanceAnalyzer(
            self.data.get('lap_times', pd.DataFrame()),
            self.data['qualifying']
        )

        # Process each qualifying session
        features_list = []
        for idx, row in self.data['qualifying'].iterrows():
            try:
                # Basic features
                features = {
                    'raceId': row['raceId'],
                    'driverId': row['driverId'],
                    'circuitId': row['circuitId'],
                    'constructorId': row['constructorId'],
                    'year': row['year'],
                    'position': row['position'],  # Target variable
                    'q1_time': row['q1'],
                    'q2_time': row.get('q2') if not pd.isna(row.get('q2')) else None,
                    'q3_time': row.get('q3') if not pd.isna(row.get('q3')) else None
                }

                # Add weather features
                try:
                    weather_features = self._get_weather_features(row['raceId'])
                    features.update(weather_features)
                except Exception as weather_error:
                    print(f"Warning: Could not get weather features for race {row['raceId']}: {weather_error}")
                    # Add default weather features
                    features.update({
                        'is_raining': 0,
                        'race_prcp_binary': 0,
                        'quali_prcp_binary': 0,
                        'fp1_prcp_binary': 0,
                        'fp2_prcp_binary': 0,
                        'fp3_prcp_binary': 0,
                        'sprint_prcp_binary': 0
                    })

                try:
                    driver_features = create_driver_features(
                        row['driverId'], row['circuitId'], row['raceId'], self.analyzer
                    )
                    features.update(driver_features)
                except Exception as driver_error:
                    print(f"Warning: Could not get driver features for driver {row['driverId']}: {driver_error}")

                # Add circuit characteristics
                try:
                    circuit_features = self._get_circuit_features(row['circuitId'])
                    features.update(circuit_features)
                except Exception as circuit_error:
                    print(f"Warning: Could not get circuit features for circuit {row['circuitId']}: {circuit_error}")

                features_list.append(features)

            except Exception as e:
                print(f"Warning: Error processing row {idx}: {e}")
                continue

        features_df = pd.DataFrame(features_list)
        print(f"✓ Created features: {len(features_df)} samples, {len(features_df.columns)} features")

        return features_df

    def prepare_prediction_features(self, driver_id: str, circuit_id: str,
                                    weather_conditions: Dict) -> Dict:
        """Prepare features for a single prediction with comprehensive feature handling"""
        if self.analyzer is None:
            raise ValueError("Must call create_features() first to initialize analyzer")

        # Create dummy race_id for current prediction
        current_race_id = f"prediction_{circuit_id}_{driver_id}"

        features = {
            'driverId': driver_id,
            'circuitId': circuit_id,
            'raceId': current_race_id
        }

        # Add weather features
        features.update(weather_conditions)

        # Add driver performance features
        driver_features = create_driver_features(
            driver_id, circuit_id, current_race_id, self.analyzer
        )
        features.update(driver_features)

        # Add circuit features
        circuit_features = self._get_circuit_features(circuit_id)
        features.update(circuit_features)

        # Explicitly handle missing features with intelligent defaults
        missing_features = {
            'q2_time': self._estimate_q2_time(driver_id, circuit_id),
            'q3_time': self._estimate_q3_time(driver_id, circuit_id),
            'teammate_teammate_gap': self._estimate_teammate_gap(driver_id, circuit_id),
            'teammate_beats_teammate': self._estimate_teammate_performance(driver_id, circuit_id)
        }
        features.update(missing_features)

        return features

    def _estimate_q2_time(self, driver_id: str, circuit_id: str) -> float:
        """Estimate Q2 time based on historical data or circuit averages"""
        try:
            # First, try to find driver's historical Q2 time at this circuit
            historical_q2 = self.analyzer.get_driver_circuit_q2_time(driver_id, circuit_id)
            if historical_q2 is not None:
                return historical_q2
        except Exception:
            pass

        # Fallback to circuit average Q2 time
        try:
            circuit_avg_q2 = self.analyzer.get_circuit_avg_q2_time(circuit_id)
            return circuit_avg_q2
        except Exception:
            # Ultimate fallback: global Q2 time average
            return 85.0  # Typical Q2 time across circuits

    def _estimate_q3_time(self, driver_id: str, circuit_id: str) -> float:
        """Estimate Q3 time based on historical data or circuit averages"""
        try:
            # First, try to find driver's historical Q3 time at this circuit
            historical_q3 = self.analyzer.get_driver_circuit_q3_time(driver_id, circuit_id)
            if historical_q3 is not None:
                return historical_q3
        except Exception:
            pass

        # Fallback to circuit average Q3 time
        try:
            circuit_avg_q3 = self.analyzer.get_circuit_avg_q3_time(circuit_id)
            return circuit_avg_q3
        except Exception:
            # Ultimate fallback: global Q3 time average
            return 84.5  # Typical Q3 time across circuits

    def _estimate_teammate_gap(self, driver_id: str, circuit_id: str) -> float:
        """Estimate teammate gap based on historical performance"""
        try:
            # Try to find actual teammate gap
            teammate_gap = self.analyzer.get_driver_teammate_gap(driver_id, circuit_id)
            if teammate_gap is not None:
                return teammate_gap
        except Exception:
            pass

        # Fallback to driver's general performance variation
        try:
            performance_variation = self.analyzer.get_driver_performance_variation(driver_id)
            return performance_variation
        except Exception:
            # Ultimate fallback: moderate performance variation
            return 0.5  # Seconds

    def _estimate_teammate_performance(self, driver_id: str, circuit_id: str) -> float:
        """Estimate how often driver beats teammate"""
        try:
            # Try to find actual teammate beat rate
            beat_rate = self.analyzer.get_driver_teammate_beat_rate(driver_id, circuit_id)
            if beat_rate is not None:
                return beat_rate
        except Exception:
            pass

        # Fallback to driver's general teammate performance
        try:
            general_beat_rate = self.analyzer.get_driver_general_teammate_beat_rate(driver_id)
            return general_beat_rate
        except Exception:
            # Ultimate fallback: 50% beat rate
            return 0.5

    def _convert_times_to_seconds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert qualifying time strings to seconds"""
        time_columns = ['q1', 'q2', 'q3']

        for col in time_columns:
            if col in df.columns:
                df[col] = df[col].apply(self._time_string_to_seconds)

        return df

    def _time_string_to_seconds(self, time_str) -> float:
        """Convert time string to total seconds with robust handling"""
        # Handle various null/missing value representations
        if (pd.isna(time_str) or
                time_str is None or
                time_str == '' or
                time_str == '\\N' or
                str(time_str).strip().lower() in ['nan', 'none']):
            return np.nan

        try:
            # If it's already a numeric type, return as is
            if isinstance(time_str, (int, float)):
                return float(time_str)

            # Convert string time format like '1:26.572'
            if isinstance(time_str, str) and ':' in time_str:
                minutes, seconds = time_str.split(':')
                total_seconds = float(minutes) * 60 + float(seconds)
                return total_seconds

            # Last resort: try converting to float
            return float(time_str)

        except (ValueError, TypeError):
            print(f"Warning: Could not convert time {time_str}")
            return np.nan

    def _convert_lap_times(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert lap time data"""
        if 'time' in df.columns:
            df['time_seconds'] = df['time'].apply(self._time_string_to_seconds)
        return df

    def _get_weather_features(self, race_id: str) -> Dict:
        """Get weather features for a race"""
        if self.data['weather'].empty:
            return {
                'is_raining': 0,
                'race_prcp_binary': 0,
                'quali_prcp_binary': 0,
                'fp1_prcp_binary': 0,
                'fp2_prcp_binary': 0,
                'fp3_prcp_binary': 0,
                'sprint_prcp_binary': 0
            }

        weather_data = self.data['weather'][self.data['weather']['raceId'] == race_id]

        if weather_data.empty:
            return {
                'is_raining': 0,
                'race_prcp_binary': 0,
                'quali_prcp_binary': 0,
                'fp1_prcp_binary': 0,
                'fp2_prcp_binary': 0,
                'fp3_prcp_binary': 0,
                'sprint_prcp_binary': 0
            }

        weather_row = weather_data.iloc[0]

        # Precipitation columns
        prcp_columns = ['race_prcp', 'quali_prcp', 'fp1_prcp', 'fp2_prcp', 'fp3_prcp', 'sprint_prcp']

        # Create features
        weather_features = {}

        # Binary precipitation features
        is_raining = 0
        for col in prcp_columns:
            # Convert to float and handle potential string/NaN values
            try:
                prcp_value = float(weather_row.get(col, 0))
            except (ValueError, TypeError):
                prcp_value = 0

            # Create binary feature
            binary_value = int(prcp_value > FEATURE_CONFIG['weather_threshold'])
            weather_features[f'{col}_binary'] = binary_value

            # Update is_raining if any precipitation is detected
            is_raining = max(is_raining, binary_value)

        # Other weather features with safe defaults
        weather_features.update({
            'is_raining': is_raining,
        })

        return weather_features

    def _get_circuit_features(self, circuit_id: str) -> Dict:
        """Get circuit characteristics"""
        if self.data['circuits'].empty:
            return {'circuit_length': 5000, 'circuit_type': 'unknown'}

        circuit_data = self.data['circuits'][self.data['circuits']['circuitId'] == circuit_id]

        if circuit_data.empty:
            return {'circuit_length': 5000, 'circuit_type': 'unknown'}

        circuit_row = circuit_data.iloc[0]

        return {
            'circuit_length': circuit_row.get('length', 5000),
            'circuit_type': circuit_row.get('type', 'unknown'),
            'altitude': circuit_row.get('alt', 0)
        }