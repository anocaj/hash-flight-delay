"""Model evaluation module for flight delay prediction.

This module provides evaluation functionality for the flight delay prediction project
"""

import os
import warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve, auc, confusion_matrix,
    classification_report, brier_score_loss, log_loss
)
from sklearn.calibration import calibration_curve
import lightgbm as lgb

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
    threshold: float = 0.5
) -> Dict[str, float]:
    """Evaluate a trained model with comprehensive metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        model_name: Name of the model for display
        threshold: Classification threshold
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Make predictions
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'predict') and hasattr(model, 'num_feature'):
        # LightGBM Booster
        y_pred_proba = model.predict(X_test)
    else:
        y_pred_proba = model.predict(X_test)
    
    y_pred_binary = (y_pred_proba >= threshold).astype(int)
    
    # Calculate comprehensive metrics
    metrics = {
        'auc_roc': roc_auc_score(y_test, y_pred_proba),
        'accuracy': accuracy_score(y_test, y_pred_binary),
        'precision': precision_score(y_test, y_pred_binary, zero_division=0),
        'recall': recall_score(y_test, y_pred_binary, zero_division=0),
        'f1_score': f1_score(y_test, y_pred_binary, zero_division=0),
        'brier_score': brier_score_loss(y_test, y_pred_proba),
        'log_loss': log_loss(y_test, y_pred_proba),
    }
    
    # Calculate Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    metrics['auc_pr'] = auc(recall, precision)
    
    # Calculate specificity (True Negative Rate)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary).ravel()
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Calculate balanced accuracy
    metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2
    
    # Calculate Matthews Correlation Coefficient
    mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    metrics['mcc'] = ((tp * tn) - (fp * fn)) / mcc_denom if mcc_denom != 0 else 0
    
    return metrics


def compare_models(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """Compare multiple models with comprehensive evaluation metrics.
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test target
        threshold: Classification threshold
        save_path: Optional path to save comparison table
        
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    print("Evaluating models...")
    print("=" * 60)
    
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        metrics = evaluate_model(model, X_test, y_test, model_name, threshold)
        metrics['model'] = model_name
        results.append(metrics)
        
        # Print key metrics
        print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"  AUC-PR:  {metrics['auc_pr']:.4f}")
        print(f"  F1:      {metrics['f1_score']:.4f}")
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.set_index('model')
    
    # Sort by AUC-ROC descending
    comparison_df = comparison_df.sort_values('auc_roc', ascending=False)
    
    # Display results
    print("\n" + "=" * 80)
    print("MODEL COMPARISON RESULTS")
    print("=" * 80)
    
    # Create a formatted display version
    display_df = comparison_df.round(4)
    print(display_df.to_string())
    
    # Print performance insights
    print_model_comparison_insights(comparison_df)
    
    # Save if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        comparison_df.to_csv(save_path)
        print(f"\nComparison results saved to: {save_path}")
    
    return comparison_df


def print_model_comparison_insights(comparison_df: pd.DataFrame) -> None:
    """Print insights from model comparison results.
    
    Args:
        comparison_df: DataFrame with model comparison results
    """
    best_model = comparison_df.index[0]
    best_auc = comparison_df.loc[best_model, 'auc_roc']
    
    print("\n" + "=" * 80)
    print("MODEL COMPARISON INSIGHTS")
    print("=" * 80)
    
    print(f"🏆 Best Performing Model: {best_model.replace('_', ' ').title()}")
    print(f"   AUC-ROC: {best_auc:.4f}")
    
    # Performance gaps
    print("\n📊 Performance Analysis:")
    for i, model_name in enumerate(comparison_df.index):
        if i == 0:
            continue  # Skip best model
        
        auc_gap = best_auc - comparison_df.loc[model_name, 'auc_roc']
        print(f"   {model_name.replace('_', ' ').title()}: {auc_gap:.4f} AUC points behind leader")
    
    # Metric-specific leaders
    print("\n🎯 Metric Leaders:")
    for metric in ['precision', 'recall', 'f1_score', 'balanced_accuracy']:
        if metric in comparison_df.columns:
            leader = comparison_df[metric].idxmax()
            value = comparison_df.loc[leader, metric]
            print(f"   {metric.replace('_', ' ').title()}: {leader.replace('_', ' ').title()} ({value:.4f})")
    
    # Model characteristics
    print("\n🔍 Model Characteristics:")
    for model_name in comparison_df.index:
        row = comparison_df.loc[model_name]
        if model_name == 'logistic_regression':
            print(f"   Logistic Regression: Linear, interpretable baseline")
        elif model_name == 'random_forest':
            print(f"   Random Forest: Ensemble method, handles feature interactions")
        elif model_name == 'lightgbm':
            print(f"   LightGBM: Gradient boosting, optimized for performance")
    
    print("\n💡 Recommendation:")
    if best_model == 'lightgbm':
        print("   LightGBM offers the best balance of predictive performance and efficiency")
        print("   for this flight delay prediction task. Its gradient boosting approach")
        print("   effectively captures complex patterns in weather, temporal, and operational data.")
    elif best_model == 'random_forest':
        print("   Random Forest provides excellent performance with good interpretability.")
        print("   The ensemble approach offers robust predictions across varying conditions.")
    else:
        print("   Logistic Regression provides the most interpretable model while")
        print("   maintaining competitive performance for this binary classification task.")


def create_model_comparison_visualization(
    comparison_df: pd.DataFrame,
    save_dir: Optional[str] = None
) -> Tuple[Figure, np.ndarray]:
    """Create comprehensive visualization of model comparison results.
    
    Args:
        comparison_df: DataFrame with model comparison results
        save_dir: Optional directory to save plots
        
    Returns:
        Tuple of (figure, axes)
    """
    # Select key metrics for visualization
    metrics_to_plot = ['auc_roc', 'auc_pr', 'f1_score', 'precision', 'recall', 'balanced_accuracy']
    available_metrics = [m for m in metrics_to_plot if m in comparison_df.columns]
    
    # Create subplots
    n_metrics = len(available_metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot each metric
    for i, metric in enumerate(available_metrics):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Create bar plot
        values = comparison_df[metric].values
        models = [name.replace('_', ' ').title() for name in comparison_df.index]
        
        bars = ax.bar(models, values, alpha=0.7)
        
        # Color the best performing model
        best_idx = comparison_df[metric].argmax()
        bars[best_idx].set_color('gold')
        bars[best_idx].set_alpha(1.0)
        
        # Formatting
        ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Hide empty subplots
    for i in range(n_metrics, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    # Save if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'model_comparison_metrics.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Model comparison visualization saved to: {save_path}")
    
    return fig, axes


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None
) -> str:
    """Generate detailed classification report.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        target_names: Optional names for classes
        
    Returns:
        Classification report as string
    """
    if target_names is None:
        target_names = ['No Delay', 'Delay']
    
    report = classification_report(
        y_true, y_pred,
        target_names=target_names,
        digits=4
    )
    
    return report


def plot_roc_pr_calibration(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str = "Model",
    save_dir: Optional[str] = None
) -> Tuple[Figure, np.ndarray]:
    """Plot ROC, Precision-Recall, and Calibration curves side-by-side.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        model_name: Name of the model
        save_dir: Optional directory to save plots
        
    Returns:
        Tuple of (figure, axes)
    """
    # Calculate curves
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
    brier = brier_score_loss(y_true, y_pred_proba)
    
    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # ROC Curve
    axes[0].plot(fpr, tpr, linewidth=2, label=f'AUC = {roc_auc:.3f}')
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title(f'ROC Curve - {model_name}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    axes[1].plot(recall, precision, linewidth=2, label=f'AP = {pr_auc:.3f}')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title(f'Precision-Recall Curve - {model_name}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Calibration Plot
    axes[2].plot(prob_pred, prob_true, marker='o', linewidth=2, markersize=8)
    axes[2].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[2].set_xlabel('Mean Predicted Probability')
    axes[2].set_ylabel('Fraction of Positives')
    axes[2].set_title(f'Calibration Plot - {model_name}\nBrier Score = {brier:.3f}')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{model_name.lower().replace(" ", "_")}_evaluation_curves.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Evaluation curves saved to: {save_path}")
    
    return fig, axes


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    normalize: bool = False,
    save_dir: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """Plot confusion matrix with annotations.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        model_name: Name of the model
        normalize: Whether to normalize the confusion matrix
        save_dir: Optional directory to save plot
        
    Returns:
        Tuple of (figure, axes)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = f'Normalized Confusion Matrix - {model_name}'
    else:
        fmt = 'd'
        title = f'Confusion Matrix - {model_name}'
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap='Blues',
        xticklabels=['No Delay', 'Delay'],
        yticklabels=['No Delay', 'Delay'],
        ax=ax
    )
    
    ax.set_title(title)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    
    # Save if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    return fig, ax


def analyze_threshold_impact(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    save_dir: Optional[str] = None
) -> Tuple[pd.DataFrame, Figure, Axes]:
    """Analyze and visualize how classification threshold affects model metrics.

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        thresholds: Optional array of thresholds to test (default 0.1–0.95)
        save_dir: Optional directory to save the output plot

    Returns:
        (results DataFrame, Figure, Axes)
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.05)

    results = []
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)

        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy
        })

    results_df = pd.DataFrame(results)

    # Single combined plot
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(results_df['threshold'], results_df['precision'], marker='o', label='Precision')
    ax.plot(results_df['threshold'], results_df['recall'], marker='s', label='Recall')
    ax.plot(results_df['threshold'], results_df['f1_score'], marker='^', label='F1-Score')
    ax.plot(results_df['threshold'], results_df['accuracy'], marker='D', label='Accuracy', linestyle='--', color='green')

    ax.set_title('Impact of Threshold on Model Performance', fontsize=14, fontweight='bold')
    ax.set_xlabel('Classification Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()

    # Save if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'threshold_analysis.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Threshold analysis saved to: {save_path}")

    return results_df, fig, ax



def get_feature_importance(
    model: Any,
    feature_names: List[str],
    importance_type: str = 'gain'
) -> pd.DataFrame:
    """Extract feature importance from trained model.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        importance_type: Type of importance ('gain', 'split', 'weight')
        
    Returns:
        DataFrame with feature importance
    """
    if hasattr(model, 'feature_importances_'):
        # Sklearn models
        importance = model.feature_importances_
    elif hasattr(model, 'feature_importance'):
        # LightGBM models
        importance = model.feature_importance(importance_type=importance_type)
    else:
        raise ValueError(f"Model type {type(model)} not supported for feature importance")
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False)
    importance_df['importance_normalized'] = importance_df['importance'] / importance_df['importance'].sum()
    
    return importance_df


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 20,
    importance_type: str = 'gain',
    save_dir: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """Plot feature importance from trained model.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        top_n: Number of top features to display
        importance_type: Type of importance ('gain', 'split', 'weight')
        save_dir: Optional directory to save plot
        
    Returns:
        Tuple of (figure, axes)
    """
    # Get feature importance
    importance_df = get_feature_importance(model, feature_names, importance_type)
    
    # Select top N features
    top_features = importance_df.head(top_n)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    
    bars = ax.barh(range(len(top_features)), top_features['importance'])
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel(f'Feature Importance ({importance_type})')
    ax.set_title(f'Top {top_n} Feature Importance')
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(top_features.iterrows()):
        ax.text(row['importance'], i, f' {row["importance"]:.0f}', 
                va='center', fontsize=9)
    
    # Invert y-axis to show most important at top
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Save if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'feature_importance_{importance_type}.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Feature importance plot saved to: {save_path}")
    
    return fig, ax


def comprehensive_model_evaluation(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: List[str],
    model_name: str = "Model",
    save_dir: Optional[str] = None,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """Perform comprehensive evaluation of a single model.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        feature_names: List of feature names
        model_name: Name of the model
        save_dir: Optional directory to save all outputs
        threshold: Classification threshold
        
    Returns:
        Dictionary containing all evaluation results
    """
    print(f"Performing comprehensive evaluation for {model_name}...")
    
    # Make predictions
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'predict') and hasattr(model, 'num_feature'):
        # LightGBM Booster
        y_pred_proba = model.predict(X_test)
    else:
        y_pred_proba = model.predict(X_test)
    
    y_pred_binary = (y_pred_proba >= threshold).astype(int)
    
    # Calculate all metrics
    metrics = evaluate_model(model, X_test, y_test, model_name, threshold)
    
    # Generate classification report
    class_report = generate_classification_report(y_test, y_pred_binary)
    
    # Create visualizations
    eval_fig, eval_axes = plot_roc_pr_calibration(y_test, y_pred_proba, model_name, save_dir)
    cm_fig, cm_ax = plot_confusion_matrix(y_test, y_pred_binary, model_name, save_dir=save_dir)
    
    # Feature importance (if supported)
    importance_df = None
    importance_fig = None
    try:
        importance_df = get_feature_importance(model, feature_names)
        importance_fig, importance_ax = plot_feature_importance(
            model, feature_names, save_dir=save_dir
        )
    except (ValueError, AttributeError):
        print(f"Feature importance not available for {model_name}")
    
    # Threshold analysis
    threshold_df, threshold_fig, threshold_axes = analyze_threshold_impact(
        y_test, y_pred_proba, save_dir=save_dir
    )
    
    # Package all results
    results = {
        'model_name': model_name,
        'metrics': metrics,
        'classification_report': class_report,
        'predictions': {
            'y_pred_proba': y_pred_proba,
            'y_pred_binary': y_pred_binary
        },
        'feature_importance': importance_df,
        'threshold_analysis': threshold_df,
        'figures': {
            'evaluation_curves': eval_fig,
            'confusion_matrix': cm_fig,
            'feature_importance': importance_fig,
            'threshold_analysis': threshold_fig
        }
    }
    
    # Print summary
    print(f"\n{model_name} Evaluation Summary:")
    print("-" * 50)
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"AUC-PR:  {metrics['auc_pr']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    
    return results