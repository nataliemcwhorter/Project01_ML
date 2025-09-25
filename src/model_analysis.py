import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from typing import Dict, List, Any, Optional
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import learning_curve, validation_curve
import warnings

warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ModelAnalysisVisualizer:
    """Comprehensive analysis and visualization of model training results"""

    def __init__(self, models_path: str = "../models/", output_path: str = "../results/model_analysis/"):
        self.models_path = models_path
        self.output_path = output_path
        self.models = {}
        self.results = {}
        self.feature_columns = None
        self.scaler = None
        self.label_encoders = None

        # Create output directory
        os.makedirs(self.output_path, exist_ok=True)

        # Load saved models and preprocessing objects
        self._load_models_and_preprocessors()

    def _load_models_and_preprocessors(self):
        """Load all saved models and preprocessing objects"""
        print("Loading models and preprocessors...")

        try:
            # Load preprocessing objects
            if os.path.exists(f"{self.models_path}scaler.pkl"):
                self.scaler = joblib.load(f"{self.models_path}scaler.pkl")
                print("✓ Scaler loaded")

            if os.path.exists(f"{self.models_path}label_encoders.pkl"):
                self.label_encoders = joblib.load(f"{self.models_path}label_encoders.pkl")
                print("✓ Label encoders loaded")

            if os.path.exists(f"{self.models_path}feature_columns.pkl"):
                self.feature_columns = joblib.load(f"{self.models_path}feature_columns.pkl")
                print(f"✓ Feature columns loaded ({len(self.feature_columns)} features)")

            # Load individual models
            model_files = {
                'linear_regression': 'linear_regression.pkl',
                'random_forest': 'random_forest.pkl',
                'gradient_boosting': 'gradient_boosting.pkl',
                'xgboost': 'xgboost.pkl',
                'neural_network': 'neural_network.pkl',
                'best_model': 'best_model.pkl'
            }

            for model_name, filename in model_files.items():
                filepath = f"{self.models_path}{filename}"
                if os.path.exists(filepath):
                    self.models[model_name] = joblib.load(filepath)
                    print(f"✓ {model_name} loaded")

            print(f"Total models loaded: {len(self.models)}")

        except Exception as e:
            print(f"Error loading models: {e}")

    def evaluate_models_on_data(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate all loaded models on test data"""
        print("Evaluating models on test data...")

        results = {}

        for model_name, model in self.models.items():
            if model_name == 'best_model':  # Skip to avoid duplication
                continue

            try:
                # Make predictions
                y_pred = model.predict(X_test)

                # Calculate metrics
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                # Calculate additional metrics
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

                results[model_name] = {
                    'r2': r2,
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'predictions': y_pred
                }

                print(f"✓ {model_name}: R² = {r2:.4f}, MAE = {mae:.4f}")

            except Exception as e:
                print(f"❌ Error evaluating {model_name}: {e}")
                continue

        self.results = results
        return results

    def create_performance_comparison_chart(self):
        """Create bar chart comparing model performance metrics"""
        if not self.results:
            print("No results available for performance comparison")
            return

        # Prepare data for plotting
        models = list(self.results.keys())
        r2_scores = [self.results[model]['r2'] for model in models]
        mae_scores = [self.results[model]['mae'] for model in models]
        rmse_scores = [self.results[model]['rmse'] for model in models]

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

        # R² Score comparison
        axes[0, 0].bar(models, r2_scores, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('R² Score Comparison', fontweight='bold')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)

        # Add value labels on bars
        for i, v in enumerate(r2_scores):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

        # MAE comparison
        axes[0, 1].bar(models, mae_scores, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Mean Absolute Error Comparison', fontweight='bold')
        axes[0, 1].set_ylabel('MAE (seconds)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)

        for i, v in enumerate(mae_scores):
            axes[0, 1].text(i, v + max(mae_scores) * 0.01, f'{v:.3f}', ha='center', va='bottom')

        # RMSE comparison
        axes[1, 0].bar(models, rmse_scores, color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('Root Mean Square Error Comparison', fontweight='bold')
        axes[1, 0].set_ylabel('RMSE (seconds)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)

        for i, v in enumerate(rmse_scores):
            axes[1, 0].text(i, v + max(rmse_scores) * 0.01, f'{v:.3f}', ha='center', va='bottom')

        # Combined metrics radar chart (normalized)
        angles = np.linspace(0, 2 * np.pi, 3, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        ax_radar = plt.subplot(2, 2, 4, projection='polar')

        # Normalize metrics for radar chart (higher is better)
        max_r2 = max(r2_scores)
        max_mae = max(mae_scores)
        max_rmse = max(rmse_scores)

        for i, model in enumerate(models):
            values = [
                r2_scores[i] / max_r2,  # R² (higher is better)
                1 - (mae_scores[i] / max_mae),  # MAE (lower is better, so invert)
                1 - (rmse_scores[i] / max_rmse)  # RMSE (lower is better, so invert)
            ]
            values += values[:1]  # Complete the circle

            ax_radar.plot(angles, values, 'o-', linewidth=2, label=model)
            ax_radar.fill(angles, values, alpha=0.25)

        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(['R²', 'MAE (inv)', 'RMSE (inv)'])
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('Normalized Performance Radar', fontweight='bold')
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        plt.tight_layout()
        plt.savefig(f"{self.output_path}model_performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Performance comparison chart saved to {self.output_path}model_performance_comparison.png")

    def create_prediction_accuracy_plots(self, X_test: np.ndarray, y_test: np.ndarray):
        """Create scatter plots showing prediction accuracy for each model"""
        if not self.results:
            print("No results available for prediction accuracy plots")
            return

        n_models = len(self.results)
        cols = 3
        rows = (n_models + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        fig.suptitle('Prediction Accuracy: Actual vs Predicted', fontsize=16, fontweight='bold')

        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)

        for idx, (model_name, result) in enumerate(self.results.items()):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]

            y_pred = result['predictions']

            # Scatter plot
            ax.scatter(y_test, y_pred, alpha=0.6, s=20)

            # Perfect prediction line
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

            # Add metrics to plot
            r2 = result['r2']
            mae = result['mae']
            ax.text(0.05, 0.95, f'R² = {r2:.3f}\nMAE = {mae:.3f}s',
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_xlabel('Actual Qualifying Time (seconds)')
            ax.set_ylabel('Predicted Qualifying Time (seconds)')
            ax.set_title(f'{model_name.replace("_", " ").title()}')
            ax.grid(True, alpha=0.3)
            ax.legend()

        # Hide empty subplots
        for idx in range(n_models, rows * cols):
            row = idx // cols
            col = idx % cols
            if rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)

        plt.tight_layout()
        plt.savefig(f"{self.output_path}prediction_accuracy_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Prediction accuracy plots saved to {self.output_path}prediction_accuracy_plots.png")

    def create_residual_analysis(self, y_test: np.ndarray):
        """Create residual plots for model analysis"""
        if not self.results:
            print("No results available for residual analysis")
            return

        n_models = len(self.results)
        fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 10))
        fig.suptitle('Residual Analysis', fontsize=16, fontweight='bold')

        if n_models == 1:
            axes = axes.reshape(-1, 1)

        for idx, (model_name, result) in enumerate(self.results.items()):
            y_pred = result['predictions']
            residuals = y_test - y_pred

            # Residuals vs Predicted
            axes[0, idx].scatter(y_pred, residuals, alpha=0.6, s=20)
            axes[0, idx].axhline(y=0, color='r', linestyle='--')
            axes[0, idx].set_xlabel('Predicted Values')
            axes[0, idx].set_ylabel('Residuals')
            axes[0, idx].set_title(f'{model_name.replace("_", " ").title()}\nResiduals vs Predicted')
            axes[0, idx].grid(True, alpha=0.3)

            # Residual distribution (histogram)
            axes[1, idx].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
            axes[1, idx].axvline(x=0, color='r', linestyle='--')
            axes[1, idx].set_xlabel('Residuals')
            axes[1, idx].set_ylabel('Frequency')
            axes[1, idx].set_title('Residual Distribution')
            axes[1, idx].grid(True, alpha=0.3)

            # Add statistics
            mean_residual = np.mean(residuals)
            std_residual = np.std(residuals)
            axes[1, idx].text(0.05, 0.95, f'Mean: {mean_residual:.3f}\nStd: {std_residual:.3f}',
                              transform=axes[1, idx].transAxes, verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(f"{self.output_path}residual_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Residual analysis saved to {self.output_path}residual_analysis.png")

    def create_feature_importance_plot(self):
        """Create feature importance plots for tree-based models"""
        if not self.models or not self.feature_columns:
            print("No models or feature columns available for feature importance")
            return

        # Find models with feature importance
        importance_models = {}
        for name, model in self.models.items():
            if name == 'best_model':
                continue
            if hasattr(model, 'feature_importances_'):
                importance_models[name] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance_models[name] = np.abs(model.coef_)

        if not importance_models:
            print("No models with feature importance available")
            return

        n_models = len(importance_models)
        fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 8))
        fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')

        if n_models == 1:
            axes = [axes]

        for idx, (model_name, importances) in enumerate(importance_models.items()):
            # Get top 15 features
            indices = np.argsort(importances)[::-1][:15]
            top_features = [self.feature_columns[i] for i in indices]
            top_importances = importances[indices]

            # Create horizontal bar plot
            y_pos = np.arange(len(top_features))
            axes[idx].barh(y_pos, top_importances, alpha=0.7)
            axes[idx].set_yticks(y_pos)
            axes[idx].set_yticklabels(top_features)
            axes[idx].invert_yaxis()
            axes[idx].set_xlabel('Importance')
            axes[idx].set_title(f'{model_name.replace("_", " ").title()}')
            axes[idx].grid(True, alpha=0.3)

            # Add value labels
            for i, v in enumerate(top_importances):
                axes[idx].text(v + max(top_importances) * 0.01, i, f'{v:.3f}',
                               va='center', ha='left')

        plt.tight_layout()
        plt.savefig(f"{self.output_path}feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Feature importance plot saved to {self.output_path}feature_importance.png")

    def create_error_distribution_analysis(self, y_test: np.ndarray):
        """Analyze error distributions across different ranges of target values"""
        if not self.results:
            print("No results available for error distribution analysis")
            return

        # Create bins for target values
        n_bins = 5
        bins = np.quantile(y_test, np.linspace(0, 1, n_bins + 1))
        bin_labels = [f'{bins[i]:.2f}-{bins[i + 1]:.2f}s' for i in range(n_bins)]

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Error Distribution Analysis', fontsize=16, fontweight='bold')

        # Prepare data for all models
        all_errors = {}
        all_abs_errors = {}

        for model_name, result in self.results.items():
            y_pred = result['predictions']
            errors = y_test - y_pred
            abs_errors = np.abs(errors)

            all_errors[model_name] = errors
            all_abs_errors[model_name] = abs_errors

        # 1. Error distribution by target value range
        bin_indices = np.digitize(y_test, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        model_names = list(all_abs_errors.keys())
        x = np.arange(n_bins)
        width = 0.8 / len(model_names)

        for i, model_name in enumerate(model_names):
            bin_errors = []
            for bin_idx in range(n_bins):
                mask = bin_indices == bin_idx
                if np.sum(mask) > 0:
                    bin_errors.append(np.mean(all_abs_errors[model_name][mask]))
                else:
                    bin_errors.append(0)

            axes[0, 0].bar(x + i * width, bin_errors, width, label=model_name, alpha=0.7)

        axes[0, 0].set_xlabel('Target Value Range')
        axes[0, 0].set_ylabel('Mean Absolute Error')
        axes[0, 0].set_title('MAE by Target Value Range')
        axes[0, 0].set_xticks(x + width * (len(model_names) - 1) / 2)
        axes[0, 0].set_xticklabels(bin_labels, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Box plot of errors for each model
        error_data = [all_errors[model] for model in model_names]
        axes[0, 1].boxplot(error_data, labels=[name.replace('_', '\n') for name in model_names])
        axes[0, 1].set_ylabel('Prediction Error (seconds)')
        axes[0, 1].set_title('Error Distribution by Model')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)

        # 3. Cumulative error distribution
        for model_name in model_names:
            sorted_abs_errors = np.sort(all_abs_errors[model_name])
            cumulative_pct = np.arange(1, len(sorted_abs_errors) + 1) / len(sorted_abs_errors) * 100
            axes[1, 0].plot(sorted_abs_errors, cumulative_pct, label=model_name, linewidth=2)

        axes[1, 0].set_xlabel('Absolute Error (seconds)')
        axes[1, 0].set_ylabel('Cumulative Percentage')
        axes[1, 0].set_title('Cumulative Error Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Error vs prediction confidence (using prediction variance as proxy)
        best_model_name = min(self.results.keys(), key=lambda x: self.results[x]['mae'])
        best_predictions = self.results[best_model_name]['predictions']
        best_errors = all_abs_errors[best_model_name]

        # Create prediction confidence bins
        pred_bins = np.quantile(best_predictions, np.linspace(0, 1, 6))
        pred_bin_indices = np.digitize(best_predictions, pred_bins) - 1
        pred_bin_indices = np.clip(pred_bin_indices, 0, 4)

        confidence_errors = []
        confidence_labels = []
        for i in range(5):
            mask = pred_bin_indices == i
            if np.sum(mask) > 0:
                confidence_errors.append(best_errors[mask])
                confidence_labels.append(f'{pred_bins[i]:.2f}-{pred_bins[i + 1]:.2f}s')

        axes[1, 1].boxplot(confidence_errors, labels=confidence_labels)
        axes[1, 1].set_xlabel('Predicted Value Range')
        axes[1, 1].set_ylabel('Absolute Error')
        axes[1, 1].set_title(f'Error vs Prediction Range\n(Best Model: {best_model_name})')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.output_path}error_distribution_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Error distribution analysis saved to {self.output_path}error_distribution_analysis.png")

    def create_summary_report(self):
        """Create a comprehensive summary report"""
        if not self.results:
            print("No results available for summary report")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Training Summary Report', fontsize=18, fontweight='bold')

        models = list(self.results.keys())

        # 1. Performance metrics table
        ax1.axis('tight')
        ax1.axis('off')

        table_data = []
        headers = ['Model', 'R²', 'MAE', 'RMSE', 'MAPE']

        for model in models:
            result = self.results[model]
            table_data.append([
                model.replace('_', ' ').title(),
                f"{result['r2']:.4f}",
                f"{result['mae']:.3f}s",
                f"{result['rmse']:.3f}s",
                f"{result['mape']:.2f}%"
            ])

        table = ax1.table(cellText=table_data, colLabels=headers,
                          cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # Color the best performing row
        best_model_idx = min(range(len(models)), key=lambda i: self.results[models[i]]['mae'])
        for j in range(len(headers)):
            table[(best_model_idx + 1, j)].set_facecolor('#90EE90')

        ax1.set_title('Performance Metrics Summary', fontweight='bold', pad=20)

        # 2. R² Score ranking
        r2_scores = [self.results[model]['r2'] for model in models]
        sorted_indices = np.argsort(r2_scores)[::-1]
        sorted_models = [models[i] for i in sorted_indices]
        sorted_r2 = [r2_scores[i] for i in sorted_indices]

        colors = plt.cm.RdYlGn([score for score in sorted_r2])
        bars = ax2.barh(range(len(sorted_models)), sorted_r2, color=colors)
        ax2.set_yticks(range(len(sorted_models)))
        ax2.set_yticklabels([model.replace('_', ' ').title() for model in sorted_models])
        ax2.set_xlabel('R² Score')
        ax2.set_title('Model Ranking by R² Score', fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, sorted_r2)):
            ax2.text(score + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{score:.3f}', va='center', ha='left')

        # 3. Error comparison
        mae_scores = [self.results[model]['mae'] for model in models]
        rmse_scores = [self.results[model]['rmse'] for model in models]

        x = np.arange(len(models))
        width = 0.35

        ax3.bar(x - width / 2, mae_scores, width, label='MAE', alpha=0.7)
        ax3.bar(x + width / 2, rmse_scores, width, label='RMSE', alpha=0.7)

        ax3.set_xlabel('Models')
        ax3.set_ylabel('Error (seconds)')
        ax3.set_title('Error Comparison (MAE vs RMSE)', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels([model.replace('_', '\n') for model in models])
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Key insights text
        ax4.axis('off')

        # Calculate insights
        best_model = min(models, key=lambda x: self.results[x]['mae'])
        best_r2 = self.results[best_model]['r2']
        best_mae = self.results[best_model]['mae']

        worst_model = max(models, key=lambda x: self.results[x]['mae'])
        worst_mae = self.results[worst_model]['mae']

        improvement = ((worst_mae - best_mae) / worst_mae) * 100

        insights_text = f"""
        KEY INSIGHTS:

        🏆 Best Performing Model: {best_model.replace('_', ' ').title()}
           • R² Score: {best_r2:.4f}
           • Mean Absolute Error: {best_mae:.3f} seconds

        📊 Performance Range:
           • Best MAE: {best_mae:.3f}s ({best_model.replace('_', ' ').title()})
           • Worst MAE: {worst_mae:.3f}s ({worst_model.replace('_', ' ').title()})
           • Improvement: {improvement:.1f}%

        🎯 Model Recommendations:
           • Use {best_model.replace('_', ' ').title()} for production
           • Consider ensemble methods if multiple models perform well
           • Monitor prediction confidence in different time ranges

        📈 Next Steps:
           • Collect more training data if R² < 0.8
           • Feature engineering for better performance
           • Hyperparameter optimization for top models
        """

        ax4.text(0.05, 0.95, insights_text, transform=ax4.transAxes,
                 fontsize=11, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.1))

        plt.tight_layout()
        plt.savefig(f"{self.output_path}summary_report.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Summary report saved to {self.output_path}summary_report.png")

    def run_complete_analysis(self, X_test: np.ndarray, y_test: np.ndarray):
        """Run complete analysis and generate all visualizations"""
        print("\n" + "=" * 60)
        print("STARTING COMPREHENSIVE MODEL ANALYSIS")
        print("=" * 60)

        # Evaluate models
        self.evaluate_models_on_data(X_test, y_test)

        if not self.results:
            print("❌ No results available. Cannot proceed with analysis.")
            return

        print(f"\n📊 Generating visualizations for {len(self.results)} models...")

        # Generate all visualizations
        self.create_performance_comparison_chart()
        self.create_prediction_accuracy_plots(X_test, y_test)
        self.create_residual_analysis(y_test)
        self.create_feature_importance_plot()
        self.create_error_distribution_analysis(y_test)
        self.create_summary_report()

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)
        print(f"📁 All charts saved to: {self.output_path}")
        print("\n📋 Generated files:")
        print("   • model_performance_comparison.png")
        print("   • prediction_accuracy_plots.png")
        print("   • residual_analysis.png")
        print("   • feature_importance.png")
        print("   • error_distribution_analysis.png")
        print("   • summary_report.png")

        # Print quick summary
        best_model = min(self.results.keys(), key=lambda x: self.results[x]['mae'])
        print(f"\n🏆 Best Model: {best_model.replace('_', ' ').title()}")
        print(f"   R² = {self.results[best_model]['r2']:.4f}")
        print(f"   MAE = {self.results[best_model]['mae']:.3f} seconds")


# Example usage function
def analyze_model_results(models_path="models/", output_path="analysis_charts/",
                          test_data_path=None, X_test=None, y_test=None):
    """
    Convenience function to run complete analysis

    Parameters:
    - models_path: Path to saved models directory
    - output_path: Path to save analysis charts
    - test_data_path: Path to test data CSV (optional)
    - X_test, y_test: Test data arrays (optional, alternative to test_data_path)
    """

    analyzer = ModelAnalysisVisualizer(models_path, output_path)

    # If test data is provided as file path
    if test_data_path and os.path.exists(test_data_path):
        print(f"Loading test data from {test_data_path}")
        test_df = pd.read_csv(test_data_path)
        # Assume last column is target, rest are features
        X_test = test_df.iloc[:, :-1].values
        y_test = test_df.iloc[:, -1].values

    # If test data arrays are provided
    elif X_test is not None and y_test is not None:
        print("Using provided test data arrays")
    else:
        print("❌ No test data provided. Please provide either test_data_path or X_test/y_test arrays")
        return

    # Run complete analysis
    analyzer.run_complete_analysis(X_test, y_test)

    return analyzer


if __name__ == "__main__":
    # Example usage - you'll need to provide your test data
    print("Model Analysis Visualizer")
    print("To use this script, call analyze_model_results() with your test data")
    print("\nExample:")
    print("analyzer = analyze_model_results(")
    print("    models_path='models/',")
    print("    output_path='analysis_charts/',")
    print("    X_test=your_X_test_array,")
    print("    y_test=your_y_test_array")
    print(")")