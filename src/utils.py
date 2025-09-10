#!/usr/bin/env python3
"""
Utility functions for F1 Qualifying Speed Prediction Model
"""

import os
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Union, Optional
from config import OUTPUT_PATHS, DATA_PATHS


def create_directories():
    """Create necessary directories for the project"""
    directories = [
        'data/',
        OUTPUT_PATHS['models'],
        OUTPUT_PATHS['results'],
        OUTPUT_PATHS['plots']
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    print("✓ Project directories created/verified")


def get_user_input(prompt: str, input_type: type, required: bool = True,
                   default: Any = None, min_val: float = None, max_val: float = None) -> Any:
    """Get validated user input"""
    while True:
        try:
            user_input = input(prompt).strip()

            # Handle empty input
            if not user_input:
                if not required and default is not None:
                    return default
                elif not required:
                    return None
                else:
                    print("This field is required. Please enter a value.")
                    continue

            # Convert to requested type
            if input_type == str:
                value = user_input
            elif input_type == int:
                value = int(user_input)
            elif input_type == float:
                value = float(user_input)
            elif input_type == bool:
                value = user_input.lower() in ['true', 't', 'yes', 'y', '1']
            else:
                raise ValueError(f"Unsupported input type: {input_type}")

            # Validate range for numeric inputs
            if input_type in [int, float] and (min_val is not None or max_val is not None):
                if min_val is not None and value < min_val:
                    print(f"Value must be at least {min_val}")
                    continue
                if max_val is not None and value > max_val:
                    print(f"Value must be at most {max_val}")
                    continue

            return value

        except ValueError as e:
            print(f"Invalid input. Please enter a valid {input_type.__name__}.")
            continue
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            return None


def validate_weather_input(weather: Dict) -> Dict:
    """Validate and clean weather input data"""
    # Ensure rain is binary
    weather['is_raining'] = int(weather.get('is_raining', 0))

    return weather


def load_sample_data() -> Dict[str, pd.DataFrame]:
    """Load sample data for testing (when real data is not available)"""
    print("Loading sample data for testing...")

    # Create sample qualifying data
    drivers = ['hamilton', 'verstappen', 'leclerc', 'norris', 'russell']
    circuits = ['monaco', 'silverstone', 'spa', 'monza', 'suzuka']

    sample_data = {
        'qualifying': [],
        'weather': [],
        'drivers': [],
        'circuits': []
    }

    # Generate sample qualifying data
    race_id = 1
    for year in [2022, 2023, 2024]:
        for circuit in circuits:
            for driver in drivers:
                # Generate realistic qualifying times (80-90 seconds base)
                base_time = np.random.uniform(80, 90)
                q1_time = base_time + np.random.uniform(-2, 2)
                q2_time = q1_time - np.random.uniform(0, 1) if np.random.random() > 0.2 else np.nan
                q3_time = q2_time - np.random.uniform(0, 0.5) if not pd.isna(
                    q2_time) and np.random.random() > 0.3 else np.nan

                sample_data['qualifying'].append({
                    'raceId': race_id,
                    'driverId': driver,
                    'circuitId': circuit,
                    'constructorId': f"team_{driver}",
                    'year': year,
                    'position': np.random.randint(1, 21),
                    'q1': q1_time,
                    'q2': q2_time,
                    'q3': q3_time
                })

            # Generate weather data for this race
            sample_data['weather'].append({
                'raceId': race_id,
                'rain': np.random.choice([0, 1], p=[0.8, 0.2])
            })

            race_id += 1

    # Generate driver data
    for driver in drivers:
        sample_data['drivers'].append({
            'driverId': driver,
            'name': driver.title(),
            'nationality': 'Unknown'
        })

    # Generate circuit data
    for circuit in circuits:
        sample_data['circuits'].append({
            'circuitId': circuit,
            'name': circuit.title(),
            'length': np.random.uniform(3000, 7000),
            'type': 'road',
            'alt': np.random.uniform(0, 1000)
        })

    # Convert to DataFrames
    for key in sample_data:
        sample_data[key] = pd.DataFrame(sample_data[key])

    print("✓ Sample data generated")
    return sample_data


def save_sample_data():
    """Save sample data to CSV files for testing"""
    sample_data = load_sample_data()

    for key, df in sample_data.items():
        filepath = DATA_PATHS[key]
        df.to_csv(filepath, index=False)
        print(f"✓ Saved {key} to {filepath}")

    print("\n✓ All sample data files created!")
    print("You can now run the training mode.")


def check_data_quality(data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Check data quality and return summary statistics"""
    quality_report = {}

    for name, df in data.items():
        if df.empty:
            quality_report[name] = {'status': 'empty', 'rows': 0, 'columns': 0}
            continue

        # Basic stats
        stats = {
            'status': 'ok',
            'rows': len(df),
            'columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        }

        # Specific checks for qualifying data
        if name == 'qualifying':
            stats['unique_drivers'] = df['driverId'].nunique() if 'driverId' in df.columns else 0
            stats['unique_circuits'] = df['circuitId'].nunique() if 'circuitId' in df.columns else 0
            stats['unique_races'] = df['raceId'].nunique() if 'raceId' in df.columns else 0

            if 'q1' in df.columns:
                stats['q1_missing'] = df['q1'].isnull().sum()
                stats['avg_q1_time'] = df['q1'].mean()

        quality_report[name] = stats

    return quality_report


def print_data_quality_report(quality_report: Dict[str, Any]):
    """Print formatted data quality report"""
    print("\n" + "=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)

    for name, stats in quality_report.items():
        print(f"\n{name.upper()}:")
        print("-" * 20)

        if stats['status'] == 'empty':
            print("❌ No data found")
            continue

        print(f"✅ Rows: {stats['rows']:,}")
        print(f"✅ Columns: {stats['columns']}")
        print(f"Missing values: {stats['missing_values']:,} ({stats['missing_percentage']:.1f}%)")

        # Additional stats for qualifying data
        if name == 'qualifying' and 'unique_drivers' in stats:
            print(f"Unique drivers: {stats['unique_drivers']}")
            print(f"Unique circuits: {stats['unique_circuits']}")
            print(f"Unique races: {stats['unique_races']}")
            if 'avg_q1_time' in stats and not pd.isna(stats['avg_q1_time']):
                print(f"Average Q1 time: {stats['avg_q1_time']:.3f}s")


def format_time_delta(seconds: float) -> str:
    """Format time difference in a readable way"""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    else:
        return f"{seconds:.3f}s"


def calculate_feature_importance(model, feature_names: List[str], top_n: int = 10) -> pd.DataFrame:
    """Calculate and return feature importance for tree-based models"""
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            print("Model doesn't support feature importance calculation")
            return pd.DataFrame()

        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })

        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)

        return importance_df.head(top_n)

    except Exception as e:
        print(f"Error calculating feature importance: {e}")
        return pd.DataFrame()


def print_feature_importance(importance_df: pd.DataFrame):
    """Print formatted feature importance"""
    if importance_df.empty:
        return

    print("\n" + "=" * 50)
    print("TOP FEATURE IMPORTANCE")
    print("=" * 50)

    for _, row in importance_df.iterrows():
        bar_length = int(row['importance'] * 50 / importance_df['importance'].max())
        bar = "█" * bar_length
        print(f"{row['feature']:<30} {bar} {row['importance']:.4f}")


def export_predictions_to_excel(predictions_df: pd.DataFrame, filename: str):
    """Export predictions to Excel with formatting"""
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            predictions_df.to_excel(writer, sheet_name='Predictions', index=False)

            # Get the workbook and worksheet
            workbook = writer.book
            worksheet = writer.sheets['Predictions']

            # Add some basic formatting
            from openpyxl.styles import Font, PatternFill

            # Header formatting
            header_font = Font(bold=True)
            header_fill = PatternFill(start_color="CCCCCC", end_color="CCCCCC", fill_type="solid")

            for cell in worksheet[1]:
                cell.font = header_font
                cell.fill = header_fill

            # Adjust column widths
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width

        print(f"✓ Predictions exported to {filename}")

    except ImportError:
        print("⚠️  openpyxl not installed. Saving as CSV instead.")
        csv_filename = filename.replace('.xlsx', '.csv')
        predictions_df.to_csv(csv_filename, index=False)
        print(f"✓ Predictions saved as CSV: {csv_filename}")
    except Exception as e:
        print(f"❌ Error exporting to Excel: {e}")