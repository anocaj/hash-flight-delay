"""Simple utility functions for the flight delay prediction project.

Essential utilities for a data science challenge:
- Directory management
- File saving/loading helpers
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, Union


def ensure_output_dirs(*paths: Union[str, Path]) -> None:
    """Ensure that output directories exist, creating them if necessary.
    
    Args:
        *paths: Variable number of directory paths to create
        
    Example:
        >>> ensure_output_dirs('outputs/figures', 'outputs/models')
    """
    for path in paths:
        if path:  # Skip None or empty paths
            Path(path).mkdir(parents=True, exist_ok=True)


def save_results(results: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """Save results dictionary to JSON file.
    
    Args:
        results: Dictionary containing results to save
        output_path: Path where to save the results
        
    Example:
        >>> results = {'model': 'LightGBM', 'auc_roc': 0.85, 'f1_score': 0.72}
        >>> save_results(results, 'outputs/reports/model_results.json')
    """
    output_path = Path(output_path)
    ensure_output_dirs(output_path.parent)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)


