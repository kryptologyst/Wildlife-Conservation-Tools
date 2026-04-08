"""
Evaluation Module for Wildlife Conservation Models

This module provides comprehensive evaluation metrics and analysis tools
for wildlife conservation threat prediction models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)


class ConservationEvaluator:
    """Comprehensive evaluator for wildlife conservation models."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.results = {}
        
    def evaluate_model(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        feature_importance: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single model comprehensively.
        
        Args:
            model_name: Name of the model
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            feature_names: Names of features (optional)
            feature_importance: Feature importance scores (optional)
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating {model_name}")
        
        # Basic classification metrics
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'specificity': self._calculate_specificity(y_true, y_pred),
            'balanced_accuracy': self._calculate_balanced_accuracy(y_true, y_pred)
        }
        
        # Probability-based metrics
        if y_proba is not None:
            metrics.update({
                'roc_auc': roc_auc_score(y_true, y_proba[:, 1]),
                'average_precision': average_precision_score(y_true, y_proba[:, 1]),
                'brier_score': self._calculate_brier_score(y_true, y_proba[:, 1])
            })
            
            # Conservation-specific metrics
            metrics.update(self._calculate_conservation_metrics(y_true, y_proba[:, 1]))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        metrics['true_negatives'] = cm[0, 0]
        metrics['false_positives'] = cm[0, 1]
        metrics['false_negatives'] = cm[1, 0]
        metrics['true_positives'] = cm[1, 1]
        
        # Store results
        self.results[model_name] = metrics
        
        return metrics
    
    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (true negative rate)."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    def _calculate_balanced_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate balanced accuracy."""
        recall = recall_score(y_true, y_pred, zero_division=0)
        specificity = self._calculate_specificity(y_true, y_pred)
        return (recall + specificity) / 2
    
    def _calculate_brier_score(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Calculate Brier score for probability calibration."""
        return np.mean((y_proba - y_true) ** 2)
    
    def _calculate_conservation_metrics(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        """Calculate conservation-specific evaluation metrics."""
        metrics = {}
        
        # High-confidence predictions (conservation alerts)
        high_conf_threshold = 0.8
        high_conf_mask = y_proba >= high_conf_threshold
        
        if np.any(high_conf_mask):
            high_conf_true = y_true[high_conf_mask]
            metrics['high_confidence_precision'] = np.mean(high_conf_true)
            metrics['high_confidence_recall'] = np.sum(high_conf_true) / np.sum(y_true)
        
        # Low-confidence predictions (safe zones)
        low_conf_threshold = 0.2
        low_conf_mask = y_proba <= low_conf_threshold
        
        if np.any(low_conf_mask):
            low_conf_true = y_true[low_conf_mask]
            metrics['low_confidence_precision'] = 1 - np.mean(low_conf_true)
            metrics['safe_zone_recall'] = np.sum(1 - low_conf_true) / np.sum(1 - y_true)
        
        # Conservation priority score
        metrics['conservation_priority_score'] = self._calculate_priority_score(y_true, y_proba)
        
        return metrics
    
    def _calculate_priority_score(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Calculate conservation priority score."""
        # Weight by actual threat level and prediction confidence
        weights = y_true * 2 + (1 - y_true) * 1  # Higher weight for actual threats
        priority_scores = y_proba * weights
        return np.mean(priority_scores)
    
    def create_leaderboard(self) -> pd.DataFrame:
        """Create a model leaderboard."""
        if not self.results:
            logger.warning("No results available for leaderboard")
            return pd.DataFrame()
        
        leaderboard_data = []
        for model_name, metrics in self.results.items():
            leaderboard_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'F1-Score': metrics['f1_score'],
                'ROC-AUC': metrics.get('roc_auc', np.nan),
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'Balanced Accuracy': metrics['balanced_accuracy'],
                'Conservation Priority Score': metrics.get('conservation_priority_score', np.nan)
            })
        
        leaderboard = pd.DataFrame(leaderboard_data)
        leaderboard = leaderboard.sort_values('ROC-AUC', ascending=False, na_last=True)
        
        return leaderboard
    
    def plot_confusion_matrices(self, save_path: Optional[Path] = None) -> None:
        """Plot confusion matrices for all models."""
        n_models = len(self.results)
        if n_models == 0:
            logger.warning("No results available for plotting")
            return
        
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, metrics) in enumerate(self.results.items()):
            cm = metrics['confusion_matrix']
            
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                ax=axes[i],
                xticklabels=['Low Risk', 'High Risk'],
                yticklabels=['Low Risk', 'High Risk']
            )
            axes[i].set_title(f'{model_name}\nConfusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrices saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curves(self, save_path: Optional[Path] = None) -> None:
        """Plot ROC curves for all models."""
        plt.figure(figsize=(8, 6))
        
        for model_name, metrics in self.results.items():
            if 'roc_auc' in metrics:
                # Note: This is a simplified version. In practice, you'd need the actual probabilities
                auc_score = metrics['roc_auc']
                plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
                plt.text(0.6, 0.1, f'{model_name}: AUC = {auc_score:.3f}')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Wildlife Conservation Models')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, model_name: str, feature_names: List[str], 
                              save_path: Optional[Path] = None) -> None:
        """Plot feature importance for a specific model."""
        if model_name not in self.results:
            logger.warning(f"No results available for {model_name}")
            return
        
        # This would need to be implemented based on the specific model
        # For now, we'll create a placeholder
        plt.figure(figsize=(10, 6))
        
        # Placeholder feature importance (in practice, get from model)
        importance_scores = np.random.random(len(feature_names))
        
        # Sort by importance
        sorted_indices = np.argsort(importance_scores)[::-1]
        sorted_features = [feature_names[i] for i in sorted_indices]
        sorted_scores = importance_scores[sorted_indices]
        
        plt.barh(range(len(sorted_features)), sorted_scores)
        plt.yticks(range(len(sorted_features)), sorted_features)
        plt.xlabel('Feature Importance')
        plt.title(f'Feature Importance - {model_name}')
        plt.gca().invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def generate_evaluation_report(self, output_dir: Path) -> None:
        """Generate comprehensive evaluation report."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create leaderboard
        leaderboard = self.create_leaderboard()
        leaderboard.to_csv(output_dir / 'model_leaderboard.csv', index=False)
        
        # Save detailed results
        import json
        with open(output_dir / 'detailed_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Generate plots
        self.plot_confusion_matrices(output_dir / 'confusion_matrices.png')
        self.plot_roc_curves(output_dir / 'roc_curves.png')
        
        logger.info(f"Evaluation report generated in {output_dir}")


def main():
    """Test the evaluation module."""
    from generate_data import WildlifeDataGenerator
    from feature_engineering import WildlifeFeatureEngineer
    from conservation_models import get_all_models
    
    # Generate and prepare data
    generator = WildlifeDataGenerator(seed=42)
    features_df, target_series = generator.generate_conservation_data(n_samples=500)
    spatial_df = generator.generate_spatial_data(n_samples=len(features_df))
    
    df = features_df.copy()
    df['threat_level'] = target_series
    df = pd.concat([df, spatial_df], axis=1)
    
    engineer = WildlifeFeatureEngineer()
    df_eng = engineer.engineer_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = engineer.prepare_modeling_data(df_eng)
    
    # Train and evaluate models
    evaluator = ConservationEvaluator()
    models = get_all_models()
    
    for name, model in models.items():
        print(f"\n🧪 Training and evaluating {name}...")
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        metrics = evaluator.evaluate_model(
            name, y_test, predictions, probabilities,
            feature_names=engineer.get_feature_importance_names()
        )
        
        print(f"✅ {name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
    
    # Create leaderboard
    leaderboard = evaluator.create_leaderboard()
    print("\n📊 Model Leaderboard:")
    print(leaderboard.to_string(index=False))


if __name__ == "__main__":
    main()
