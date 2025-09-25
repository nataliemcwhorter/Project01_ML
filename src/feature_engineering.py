import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from config import FEATURE_CONFIG


class DriverPerformanceAnalyzer:
    """Analyzes driver performance for feature engineering"""

    def __init__(self, lap_data: pd.DataFrame, qualifying_data: pd.DataFrame):
        self.lap_data = lap_data
        self.qualifying_data = qualifying_data
        self._precompute_statistics()

    def _precompute_statistics(self):
        """Precompute various statistical metrics to speed up feature generation"""
        # Circuit-level statistics
        self.circuit_stats = self._compute_circuit_level_stats()
        #print("circuit level stats:", self.circuit_stats)

        # Driver-level statistics
        self.driver_stats = self._compute_driver_level_stats()
        #print("driver level stats:", self.driver_stats)

        # Teammate comparison statistics
        self.teammate_stats = self._compute_teammate_stats()
        #print("teammate level stats:", self.teammate_stats)

    def _compute_circuit_level_stats(self) -> Dict:
        """Compute circuit-level qualifying time statistics"""
        circuit_stats = {}

        for circuit_id in self.qualifying_data['circuitId'].unique():
            circuit_data = self.qualifying_data[self.qualifying_data['circuitId'] == circuit_id]

            circuit_stats[circuit_id] = {
                'avg_q1_time': circuit_data['q1'].mean(),
                'avg_q2_time': circuit_data['q2'].mean(),
                'avg_q3_time': circuit_data['q3'].mean(),
                'q1_std': circuit_data['q1'].std(),
                'q2_std': circuit_data['q2'].std(),
                'q3_std': circuit_data['q3'].std()
            }

        return circuit_stats

    def _compute_driver_level_stats(self) -> Dict:
        """Compute driver-level qualifying performance statistics"""
        driver_stats = {}

        for driver_id in self.qualifying_data['driverId'].unique():
            driver_data = self.qualifying_data[self.qualifying_data['driverId'] == driver_id]

            driver_stats[driver_id] = {
                'avg_q1_time': driver_data['q1'].mean(),
                'avg_q2_time': driver_data['q2'].mean(),
                'avg_q3_time': driver_data['q3'].mean(),
                'performance_variation': driver_data['q1'].std(),
                'total_races': len(driver_data),
                'avg_position': driver_data['position'].mean()
            }

        return driver_stats

    def _compute_teammate_stats(self) -> Dict:
        """Compute teammate comparison statistics"""
        teammate_stats = {}

        # Group by constructor to approximate teammates
        for constructor_id in self.qualifying_data['constructorId'].unique():
            constructor_data = self.qualifying_data[self.qualifying_data['constructorId'] == constructor_id]

            # Get drivers in this constructor
            constructor_drivers = constructor_data['driverId'].unique()

            for driver_id in constructor_drivers:
                driver_data = constructor_data[constructor_data['driverId'] == driver_id]
                other_drivers_data = constructor_data[constructor_data['driverId'] != driver_id]

                # Compute teammate gap and beat rate
                if not other_drivers_data.empty:
                    teammate_gap = abs(driver_data['q1'].mean() - other_drivers_data['q1'].mean())
                    beat_rate = (driver_data['position'] < other_drivers_data['position'].mean()).mean()

                    teammate_stats[driver_id] = {
                        'teammate_gap': teammate_gap,
                        'beat_teammate_rate': beat_rate
                    }
                else:
                    teammate_stats[driver_id] = {
                        'teammate_gap': 0.5,  # Default
                        'beat_teammate_rate': 0.5  # Default
                    }

        return teammate_stats

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

    # New methods for feature estimation
    def get_driver_circuit_q2_time(self, driver_id: str, circuit_id: str) -> Optional[float]:
        """Get driver's historical Q2 time for a specific circuit"""
        driver_circuit_data = self.qualifying_data[
            (self.qualifying_data['driverId'] == driver_id) &
            (self.qualifying_data['circuitId'] == circuit_id)
            ]

        return driver_circuit_data['q2'].mean() if not driver_circuit_data.empty else None

    def get_driver_circuit_q3_time(self, driver_id: str, circuit_id: str) -> Optional[float]:
        """Get driver's historical Q3 time for a specific circuit"""
        driver_circuit_data = self.qualifying_data[
            (self.qualifying_data['driverId'] == driver_id) &
            (self.qualifying_data['circuitId'] == circuit_id)
            ]

        return driver_circuit_data['q3'].mean() if not driver_circuit_data.empty else None

    def get_circuit_avg_q2_time(self, circuit_id: str) -> float:
        """Get average Q2 time for a specific circuit"""
        return self.circuit_stats.get(circuit_id, {}).get('avg_q2_time', 85.0)

    def get_circuit_avg_q3_time(self, circuit_id: str) -> float:
        """Get average Q3 time for a specific circuit"""
        return self.circuit_stats.get(circuit_id, {}).get('avg_q3_time', 84.5)

    def get_driver_teammate_gap(self, driver_id: str, circuit_id: str) -> Optional[float]:
        """Estimate teammate performance gap"""
        return self.teammate_stats.get(driver_id, {}).get('teammate_gap', 0.5)

    def get_driver_performance_variation(self, driver_id: str) -> float:
        """Get driver's overall performance variation"""
        return self.driver_stats.get(driver_id, {}).get('performance_variation', 0.5)

    def get_driver_teammate_beat_rate(self, driver_id: str, circuit_id: str) -> float:
        """Get driver's rate of beating teammate at a specific circuit"""
        return self.teammate_stats.get(driver_id, {}).get('beat_teammate_rate', 0.5)

    def get_driver_general_teammate_beat_rate(self, driver_id: str) -> float:
        """Get driver's overall rate of beating teammate"""
        return self.teammate_stats.get(driver_id, {}).get('beat_teammate_rate', 0.5)

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
    circuit_data = analyzer.qualifying_data[analyzer.qualifying_data['circuitId'] == circuit_id]

    if not circuit_history:
        print(f"WARNING: No circuit history found for driver {driver_id} at circuit {circuit_id}")
        # Add some default/fallback features
        circuit_history = {
            'has_circuit_history': False,
            'circuit_races_count': 0,
            'circuit_avg_position': 10,  # middle of the grid
            'circuit_best_position': 20,  # worst possible
            #TODO: NATALIE EDIT THIS AND SEE WHAT IT DOES
            'circuit_avg_q1_time': circuit_data['q1'].mean() + 0.5,  # average qualifying time
            'circuit_recent_form': 0.25,  # neutral performance
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
            'consistency_score': 0.4,
            'q3_reach_rate': 0.2,
            'top5_rate': 0.1,
            'pole_rate': 0.02
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
            #TODO SEE HOW THESE AFFECT IT
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
