import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import requests
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("monitoring.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ModelMonitor:
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
        self.monitoring_dir = "monitoring"
        os.makedirs(self.monitoring_dir, exist_ok=True)

        # Track metrics over time
        self.metrics_history = []
        self.drift_history = []
        self.predictions_log = []

        # Reference data (baseline)
        self.reference_data = None

        logger.info("Model monitoring initialized")

    def load_reference_data(self, data_path="data/train_data.csv"):
        """Load reference data for comparison"""
        try:
            logger.info(f"Loading reference data from {data_path}")
            self.reference_data = pd.read_csv(data_path)

            # Calculate reference statistics
            self.ref_stats = {}
            for col in self.reference_data.columns:
                if col != "Potability":
                    self.ref_stats[col] = {
                        "mean": self.reference_data[col].mean(),
                        "std": self.reference_data[col].std(),
                        "min": self.reference_data[col].min(),
                        "max": self.reference_data[col].max(),
                        "q25": self.reference_data[col].quantile(0.25),
                        "q50": self.reference_data[col].quantile(0.50),
                        "q75": self.reference_data[col].quantile(0.75)
                    }

            logger.info(f"Reference data loaded: {self.reference_data.shape}")
            return True
        except Exception as e:
            logger.error(f"Failed to load reference data: {str(e)}")
            return False

    def get_model_metrics(self):
        """Get current model performance metrics"""
        try:
            response = requests.get(f"{self.api_url}/api/metrics")
            if response.status_code == 200:
                metrics = response.json()
                logger.info(f"Model metrics retrieved: {metrics}")

                # Add timestamp
                metrics["timestamp"] = datetime.now().isoformat()
                self.metrics_history.append(metrics)

                return metrics
            else:
                logger.error(
                    f"Failed to get metrics: HTTP {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error getting model metrics: {str(e)}")
            return None

    def get_recent_predictions(self):
        """Get recent model predictions"""
        try:
            response = requests.get(
                f"{self.api_url}/api/monitoring/recent_predictions")
            if response.status_code == 200:
                predictions = response.json().get("predictions", [])
                logger.info(f"Retrieved {len(predictions)} recent predictions")

                # Add to predictions log
                for pred in predictions:
                    if pred not in self.predictions_log:
                        self.predictions_log.append(pred)

                return predictions
            else:
                logger.error(
                    f"Failed to get predictions: HTTP {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error getting recent predictions: {str(e)}")
            return []

    def analyze_drift(self, prediction_data):
        """Analyze data drift by comparing recent predictions to reference data"""
        if self.reference_data is None or len(prediction_data) == 0:
            logger.warning(
                "Cannot analyze drift: missing reference data or predictions")
            return None

        # Convert prediction data to DataFrame
        pred_df = pd.DataFrame([p["input_data"] for p in prediction_data])

        if len(pred_df) < 10:
            logger.info("Not enough prediction data for drift analysis")
            return None

        # Calculate drift metrics
        drift_metrics = {}
        for col in pred_df.columns:
            if col in self.ref_stats:
                # Calculate statistics
                curr_mean = pred_df[col].mean()
                curr_std = pred_df[col].std()

                # Calculate drift
                mean_diff = abs(curr_mean - self.ref_stats[col]["mean"])
                mean_pct = mean_diff / self.ref_stats[col]["mean"] * 100

                # Normalize by standard deviation (Z-score of the difference)
                if self.ref_stats[col]["std"] > 0:
                    normalized_diff = mean_diff / self.ref_stats[col]["std"]
                else:
                    normalized_diff = 0

                drift_metrics[col] = {
                    "current_mean": curr_mean,
                    "reference_mean": self.ref_stats[col]["mean"],
                    "mean_difference": mean_diff,
                    "mean_difference_percent": mean_pct,
                    "normalized_difference": normalized_diff,
                    "drift_detected": normalized_diff > 1.0  # Threshold
                }

        # Overall drift score
        drift_scores = [m["normalized_difference"]
                        for m in drift_metrics.values()]
        overall_drift = sum(drift_scores) / \
            len(drift_scores) if drift_scores else 0

        result = {
            "timestamp": datetime.now().isoformat(),
            "sample_size": len(pred_df),
            "feature_drift": drift_metrics,
            "overall_drift_score": overall_drift,
            "drift_detected": overall_drift > 0.7  # Threshold for overall drift
        }

        logger.info(
            f"Drift analysis: score={overall_drift:.4f}, drift_detected={result['drift_detected']}")
        self.drift_history.append(result)

        return result

    def generate_monitoring_report(self):
        """Generate a comprehensive monitoring report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(self.monitoring_dir, f"report_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)

        # Recent predictions analysis
        recent_predictions = self.get_recent_predictions()

        # Model metrics over time
        metrics = self.get_model_metrics()

        # Data drift analysis
        drift_analysis = self.analyze_drift(recent_predictions)

        # Create report summary
        report = {
            "timestamp": timestamp,
            "metrics": metrics,
            "drift_analysis": drift_analysis,
            "predictions_count": len(recent_predictions)
        }

        # Save report as JSON
        with open(os.path.join(report_dir, "report.json"), "w") as f:
            json.dump(report, f, indent=4)

        # Generate visualizations if we have enough data
        if drift_analysis and metrics:
            self._generate_visualizations(report_dir, drift_analysis)

        logger.info(f"Monitoring report generated: {report_dir}")
        return report_dir

    def _generate_visualizations(self, report_dir, drift_analysis):
        """Generate monitoring visualizations"""
        # 1. Feature drift visualization
        if drift_analysis.get("feature_drift"):
            plt.figure(figsize=(12, 8))
            features = list(drift_analysis["feature_drift"].keys())
            drift_values = [drift_analysis["feature_drift"][f]
                            ["normalized_difference"] for f in features]

            # Sort by drift magnitude
            sorted_indices = np.argsort(drift_values)[::-1]
            features = [features[i] for i in sorted_indices]
            drift_values = [drift_values[i] for i in sorted_indices]

            # Plot
            plt.barh(features, drift_values, color=plt.cm.viridis(
                np.array(drift_values) / max(max(drift_values), 1)))
            plt.axvline(x=1.0, color='r', linestyle='--', alpha=0.7)
            plt.xlabel('Normalized Drift (Z-score)')
            plt.ylabel('Features')
            plt.title('Feature Drift Analysis')
            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, "feature_drift.png"))
            plt.close()

        # 2. Drift over time
        if len(self.drift_history) > 1:
            plt.figure(figsize=(10, 6))
            timestamps = [datetime.fromisoformat(
                d["timestamp"]) for d in self.drift_history]
            drift_scores = [d["overall_drift_score"]
                            for d in self.drift_history]

            plt.plot(timestamps, drift_scores, marker='o')
            plt.axhline(y=0.7, color='r', linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)
            plt.ylabel('Overall Drift Score')
            plt.title('Data Drift Over Time')
            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, "drift_over_time.png"))
            plt.close()

    def run_monitoring_cycle(self):
        """Run a full monitoring cycle"""
        logger.info("Starting monitoring cycle")

        # Check model health
        health_response = requests.get(f"{self.api_url}/api/health")
        if health_response.status_code != 200:
            logger.error(
                f"Model API health check failed: {health_response.status_code}")
            return False

        # Load reference data if not loaded
        if self.reference_data is None:
            self.load_reference_data()

        # Get model metrics
        self.get_model_metrics()

        # Get recent predictions
        predictions = self.get_recent_predictions()

        # Analyze drift
        self.analyze_drift(predictions)

        # Generate report
        report_dir = self.generate_monitoring_report()

        logger.info(f"Monitoring cycle completed: {report_dir}")
        return True


def run_continuous_monitoring(interval_minutes=60):
    """Run continuous monitoring at specified intervals"""
    monitor = ModelMonitor()
    monitor.load_reference_data()

    logger.info(
        f"Starting continuous monitoring every {interval_minutes} minutes")

    while True:
        try:
            monitor.run_monitoring_cycle()
        except Exception as e:
            logger.error(f"Error in monitoring cycle: {str(e)}")

        # Wait for next cycle
        logger.info(
            f"Waiting {interval_minutes} minutes until next monitoring cycle")
        time.sleep(interval_minutes * 60)


if __name__ == "__main__":
    # One-time monitoring run
    monitor = ModelMonitor()
    monitor.load_reference_data()
    monitor.run_monitoring_cycle()

    # Uncomment for continuous monitoring
    # run_continuous_monitoring(interval_minutes=60)
