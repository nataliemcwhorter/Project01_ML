import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from config import FEATURE_CONFIG


class DriverPerformanceAnalyzer:
    """Analyzes driver performance for feature engineering"""

    def __init__(self, lap_data: pd.DataFrame, qualifying_data: pd.DataFrame):
        self.lap_data = lap_data
        self.qualifying_data = qualifying_data

    '''def get_driver_circuit_history(self, driver_id: str, circuit_id: str) -> Dict:
        """Get driver's historical performance at specific circuit"""
        circuit_data = self.qualifying_data[
            (self.qualifying_data['driverId'] == driver_id) &
            (self.qualifying_data['circuitId'] == circuit_id)
            ]

        if len(circuit_data) < FEATURE_CONFIG['min_races_for_circuit_history']:
            return {'has_circuit_history': False}

        return {
            'has_circuit_history': True,
            'circuit_races_count': len(circuit_data),
            'circuit_avg_position': circuit_data['position'].mean(),
            'circuit_best_position': circuit_data['position'].min(),
            'circuit_avg_q1_time': circuit_data['q1'].mean(),
            'circuit_avg_q2_time': circuit_data['q2'].mean(),
            'circuit_avg_q3_time': circuit_data['q3'].mean(),
            'circuit_recent_form': circuit_data.tail(5)['position'].mean(),
            'circuit_improvement_trend': self._calculate_trend(circuit_data['position'].values)
        }'''

    def get_driver_circuit_history(self, driver_id: str, circuit_id: str) -> Dict:
        """Get driver's historical performance at specific circuit"""

        circuit_data = self.qualifying_data[
            (self.qualifying_data['driverId'] == driver_id) &
            (self.qualifying_data['circuitId'] == circuit_id)
            ]


        if len(circuit_data) < FEATURE_CONFIG['min_races_for_circuit_history']:
            print(f"WARNING: Not enough races for driver {driver_id} at circuit {circuit_id}")
            return {'has_circuit_history': False}

        return {
            'has_circuit_history': True,
            'circuit_races_count': len(circuit_data),
            'circuit_avg_position': circuit_data['position'].mean(),
            'circuit_best_position': circuit_data['position'].min(),
            'circuit_avg_q1_time': circuit_data['q1'].mean(),
            'circuit_avg_q2_time': circuit_data['q2'].mean(),
            'circuit_avg_q3_time': circuit_data['q3'].mean(),
            'circuit_recent_form': circuit_data.tail(5)['position'].mean(),
            'circuit_improvement_trend': self._calculate_trend(circuit_data['position'].values)
        }

    def get_driver_general_history(self, driver_id: str, exclude_circuit: str = None) -> Dict:
        """Get driver's general historical performance across all circuits"""
        general_data = self.qualifying_data[self.qualifying_data['driverId'] == driver_id]

        if exclude_circuit:
            general_data = general_data[general_data['circuitId'] != exclude_circuit]

        if len(general_data) < FEATURE_CONFIG['min_races_for_general_history']:
            return {'has_general_history': False}

        recent_data = general_data.tail(FEATURE_CONFIG['lookback_races'])

        return {
            'has_general_history': True,
            'total_races': len(general_data),
            'general_avg_position': general_data['position'].mean(),
            'general_best_position': general_data['position'].min(),
            'recent_form_position': recent_data['position'].mean(),
            'recent_form_trend': self._calculate_trend(recent_data['position'].values),
            'consistency_score': 1 / (general_data['position'].std() + 1),
            'q3_reach_rate': len(general_data[general_data['q3'].notna()]) / len(general_data),
            'top5_rate': len(general_data[general_data['position'] <= 5]) / len(general_data),
            'pole_rate': len(general_data[general_data['position'] == 1]) / len(general_data)
        }

    def get_driver_vs_field_stats(self, driver_id: str, circuit_id: str = None) -> Dict:
        """Compare driver performance against the field"""
        if circuit_id:
            field_data = self.qualifying_data[self.qualifying_data['circuitId'] == circuit_id]
            driver_data = field_data[field_data['driverId'] == driver_id]
        else:
            field_data = self.qualifying_data
            driver_data = field_data[field_data['driverId'] == driver_id]

        if len(driver_data) == 0:
            return {'has_comparison_data': False}

        field_avg_position = field_data.groupby('raceId')['position'].mean().mean()
        driver_avg_position = driver_data['position'].mean()

        return {
            'has_comparison_data': True,
            'position_vs_field': driver_avg_position - field_avg_position,
            'percentile_rank': self._calculate_percentile_rank(driver_data['position'], field_data['position']),
            'outperformance_rate': len(driver_data[driver_data['position'] < driver_avg_position]) / len(driver_data)
        }

    def get_teammate_comparison(self, driver_id: str, race_id: str) -> Dict:
        """Compare driver to teammate in same race"""
        race_data = self.qualifying_data[self.qualifying_data['raceId'] == race_id]
        driver_row = race_data[race_data['driverId'] == driver_id]

        if len(driver_row) == 0:
            return {'has_teammate_data': False}

        constructor_id = driver_row.iloc[0]['constructorId']
        teammate_data = race_data[
            (race_data['constructorId'] == constructor_id) &
            (race_data['driverId'] != driver_id)
            ]

        if len(teammate_data) == 0:
            return {'has_teammate_data': False}

        driver_position = driver_row.iloc[0]['position']
        teammate_position = teammate_data.iloc[0]['position']

        return {
            'has_teammate_data': True,
            'teammate_gap': driver_position - teammate_position,
            'beats_teammate': driver_position < teammate_position
        }

    def _calculate_trend(self, values: np.array) -> float:
        """Calculate trend in performance (negative = improving)"""
        if len(values) < 2:
            return 0
        x = np.arange(len(values))
        return np.polyfit(x, values, 1)[0]

    def _calculate_percentile_rank(self, driver_positions: pd.Series, all_positions: pd.Series) -> float:
        """Calculate driver's percentile rank"""
        driver_avg = driver_positions.mean()
        return (all_positions < driver_avg).mean() * 100


def create_driver_features(driver_id: str, circuit_id: str, race_id: str,
                           analyzer: DriverPerformanceAnalyzer) -> Dict:
    """Create all driver-specific features for prediction"""
    features = {}


    # Circuit-specific history
    circuit_history = analyzer.get_driver_circuit_history(driver_id, circuit_id)

    if not circuit_history:
        print(f"WARNING: No circuit history found for driver {driver_id} at circuit {circuit_id}")
        # Add some default/fallback features
        circuit_history = {
            'has_circuit_history': False,
            'circuit_races_count': 0,
            'circuit_avg_position': 10,  # middle of the grid
            'circuit_best_position': 20,  # worst possible
            'circuit_avg_q1_time': 90.0,  # average qualifying time
            'circuit_recent_form': 0.5,  # neutral performance
            'circuit_improvement_trend': 0
        }

    features.update({f'circuit_{k}': v for k, v in circuit_history.items()})

    # General history (excluding current circuit for fair comparison)
    general_history = analyzer.get_driver_general_history(driver_id, exclude_circuit=circuit_id)

    if not general_history:
        print(f"WARNING: No general history found for driver {driver_id}")
        # Add some default/fallback features
        general_history = {
            'has_general_history': False,
            'total_races': 0,
            'general_avg_position': 10,
            'general_best_position': 20,
            'recent_form_position': 10,
            'recent_form_trend': 0,
            'consistency_score': 0.5,
            'q3_reach_rate': 0.5,
            'top5_rate': 0.3,
            'pole_rate': 0.1
        }

    features.update({f'general_{k}': v for k, v in general_history.items()})

    # Field comparison
    field_comparison = analyzer.get_driver_vs_field_stats(driver_id, circuit_id)


    if not field_comparison:
        print(f"WARNING: No field comparison found for driver {driver_id}")
        # Add some default/fallback features
        field_comparison = {
            'has_comparison_data': False,
            'position_vs_field': 0,
            'percentile_rank': 0.5,
            'outperformance_rate': 0.5
        }

    features.update({f'field_{k}': v for k, v in field_comparison.items()})

    # Teammate comparison
    teammate_comparison = analyzer.get_teammate_comparison(driver_id, race_id)

    if not teammate_comparison:
        print(f"WARNING: No teammate comparison found for driver {driver_id}")
        # Add some default/fallback features
        teammate_comparison = {
            'has_teammate_data': False,
            'teammate_gap': 0,
            'beats_teammate': 0.5
        }

    features.update({f'teammate_{k}': v for k, v in teammate_comparison.items()})

    return features