"""Model training module for flight delay prediction.

This module provides model training functionality for the flight delay prediction project,
containing the functions that are actually used in the analysis notebook.
"""

import os
import pickle
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression

# Import the better feature preparation function
from ..features.feature_engineering import prepare_features_for_modeling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import optuna
from optuna.samplers import TPESampler





def train_lightgbm_model(
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    params: Optional[Dict[str, Any]] = None,
    num_boost_round: int = 200,
    early_stopping_rounds: int = 20,
    verbose: bool = True
) -> lgb.LGBMClassifier:
    """Train a LightGBM model with specified parameters.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features (optional)
        y_val: Validation target (optional)
        params: Model parameters
        num_boost_round: Number of boosting rounds
        early_stopping_rounds: Early stopping rounds
        verbose: Whether to print training progress
        
    Returns:
        Trained LightGBM model
    """
    if params is None:
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_sets = [train_data]
    
    if X_val is not None and y_val is not None:
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        valid_sets.append(val_data)
    
    # Set up callbacks
    callbacks = []
    if early_stopping_rounds > 0:
        callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds))
    
    if verbose:
        print("Training LightGBM model...")
    
    # Train model
    model = lgb.train(
        params,
        train_data,
        valid_sets=valid_sets,
        num_boost_round=num_boost_round,
        callbacks=callbacks
    )
    
    return model


def train_baseline_models(
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    random_state: int = 42
) -> Dict[str, Any]:
    """Train baseline models for comparison.
    
    Args:
        X_train: Training features
        y_train: Training target
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary of trained baseline models
    """
    models = {}
    
    print("Training baseline models...")
    
    # Logistic Regression
    print("  - Training Logistic Regression...")
    lr_model = LogisticRegression(
        random_state=random_state,
        max_iter=1000,
        class_weight='balanced'
    )
    lr_model.fit(X_train, y_train)
    models['logistic_regression'] = lr_model
    
    # Random Forest
    print("  - Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    models['random_forest'] = rf_model
    
    return models


def hyperparameter_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
    timeout: Optional[int] = None,
    random_state: int = 42
) -> Dict[str, Any]:
    """Perform hyperparameter tuning using Optuna.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        n_trials: Number of optimization trials
        timeout: Timeout in seconds
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary with best parameters and study results
    """
    def objective(trial):
        # Suggest hyperparameters
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'verbose': -1
        }
        
        # Train model
        model = train_lightgbm_model(
            X_train, y_train, X_val, y_val,
            params=params,
            num_boost_round=200,
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Make predictions and return AUC
        y_pred = model.predict(X_val)
        auc = roc_auc_score(y_val, y_pred)
        
        return auc
    
    print(f"Starting hyperparameter tuning with {n_trials} trials...")
    
    # Create study
    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler
    )
    
    # Optimize
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    print(f"Best AUC: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")
    
    return {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'study': study
    }


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model"
) -> Dict[str, float]:
    """Evaluate a trained model on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        model_name: Name of the model for display
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Make predictions
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = model.predict(X_test)
    
    y_pred_binary = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    metrics = {
        'auc': roc_auc_score(y_test, y_pred_proba),
        'accuracy': accuracy_score(y_test, y_pred_binary),
        'precision': precision_score(y_test, y_pred_binary),
        'recall': recall_score(y_test, y_pred_binary),
        'f1_score': f1_score(y_test, y_pred_binary)
    }
    
    print(f"\n{model_name} Performance:")
    print("-" * 40)
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    return metrics


def compare_models(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    """Compare multiple models on test data.
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test target
        
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for model_name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, model_name)
        metrics['model'] = model_name
        results.append(metrics)
    
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.set_index('model')
    
    # Sort by AUC descending
    comparison_df = comparison_df.sort_values('auc', ascending=False)
    
    print("\nModel Comparison:")
    print("=" * 60)
    print(comparison_df.round(4))
    
    return comparison_df


def save_model(model: Any, filepath: Union[str, Path], model_type: str = "lightgbm") -> None:
    """Save a trained model to disk.
    
    Args:
        model: Trained model to save
        filepath: Path to save the model
        model_type: Type of model ('lightgbm', 'sklearn', or 'pickle')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if model_type == "lightgbm":
        model.save_model(str(filepath))
    elif model_type == "sklearn":
        joblib.dump(model, filepath)
    else:  # pickle
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
    
    print(f"Model saved to: {filepath}")


def load_model(filepath: Union[str, Path], model_type: str = "lightgbm") -> Any:
    """Load a trained model from disk.
    
    Args:
        filepath: Path to the saved model
        model_type: Type of model ('lightgbm', 'sklearn', or 'pickle')
        
    Returns:
        Loaded model
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    if model_type == "lightgbm":
        model = lgb.Booster(model_file=str(filepath))
    elif model_type == "sklearn":
        model = joblib.load(filepath)
    else:  # pickle
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
    
    print(f"Model loaded from: {filepath}")
    return model


def cross_validate_model(
    X: pd.DataFrame,
    y: pd.Series,
    params: Optional[Dict[str, Any]] = None,
    cv_folds: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """Perform cross-validation on the model.
    
    Args:
        X: Features
        y: Target
        params: Model parameters
        cv_folds: Number of CV folds
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary with CV results
    """
    if params is None:
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'verbose': -1
        }
    
    print(f"Performing {cv_folds}-fold cross-validation...")
    
    # Set up cross-validation
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    cv_scores = []
    fold_models = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"  Training fold {fold + 1}/{cv_folds}...")
        
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model
        model = train_lightgbm_model(
            X_train_fold, y_train_fold,
            X_val_fold, y_val_fold,
            params=params,
            verbose=False
        )
        
        # Evaluate
        y_pred = model.predict(X_val_fold)
        auc = roc_auc_score(y_val_fold, y_pred)
        cv_scores.append(auc)
        fold_models.append(model)
    
    cv_results = {
        'cv_scores': cv_scores,
        'mean_auc': np.mean(cv_scores),
        'std_auc': np.std(cv_scores),
        'fold_models': fold_models
    }
    
    print(f"Cross-validation results:")
    print(f"  Mean AUC: {cv_results['mean_auc']:.4f} (+/- {cv_results['std_auc']:.4f})")
    
    return cv_results


def train_final_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.2,
    use_hyperparameter_tuning: bool = False,
    tuning_trials: int = 50,
    random_state: int = 42
) -> Dict[str, Any]:
    """Train the final model with all components including baseline comparisons.
    
    Args:
        df: Complete dataset
        test_size: Proportion of data for testing
        val_size: Proportion of remaining data for validation
        use_hyperparameter_tuning: Whether to perform hyperparameter tuning
        tuning_trials: Number of trials for hyperparameter tuning
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary with all trained models and results
    """
    print("Starting comprehensive model training pipeline...")
    
    # Prepare features using the comprehensive feature engineering function
    X, feature_names = prepare_features_for_modeling(df)
    y = df['ARR_DEL15']  # Extract target separately
    print(f"Features prepared: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
    )
    
    print(f"Data split - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Train baseline models
    baseline_models = train_baseline_models(X_train, y_train, random_state)
    
    # Train LightGBM model
    best_params = None
    if use_hyperparameter_tuning:
        tuning_results = hyperparameter_tuning(
            X_train, y_train, X_val, y_val,
            n_trials=tuning_trials,
            random_state=random_state
        )
        best_params = tuning_results['best_params']
        best_params.update({
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'verbose': -1
        })
    
    lgb_model = train_lightgbm_model(
        X_train, y_train, X_val, y_val,
        params=best_params
    )
    
    # Combine all models
    all_models = baseline_models.copy()
    all_models['lightgbm'] = lgb_model
    
    # Compare models
    comparison_results = compare_models(all_models, X_test, y_test)

    # Package results
    results = {
        'models': all_models,
        'best_model': all_models[comparison_results.index[0]],  # Best performing model
        'best_model_name': comparison_results.index[0],
        'comparison_results': comparison_results,
        'feature_names': feature_names,
        'data_splits': {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
    }
    
    if use_hyperparameter_tuning:
        results['tuning_results'] = tuning_results
    
    print("\nModel training pipeline completed!")
    return results


