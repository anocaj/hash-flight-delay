"""Visualization functions for flight delay analysis.

This module provides visualization functionality for the flight delay prediction project,
containing only the functions that are actually used in the analysis notebook.
"""

import os
import warnings
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def ensure_output_dir(save_path: str) -> None:
    """Ensure the output directory exists for saving plots.
    
    Args:
        save_path: Path where the plot will be saved
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)


def save_plot(fig: Figure, filename: str, output_dir: str = "outputs/figures") -> str:
    """Save a plot to the specified directory with consistent formatting.
    
    Args:
        fig: Matplotlib figure to save
        filename: Name of the file (with or without extension)
        output_dir: Directory to save the plot
        
    Returns:
        Full path where the plot was saved
    """
    # Ensure filename has .png extension
    if not filename.endswith('.png'):
        filename += '.png'
    
    # Create full path
    save_path = os.path.join(output_dir, filename)
    
    # Ensure directory exists
    ensure_output_dir(save_path)
    
    # Save with consistent settings
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    
    return save_path


def plot_delay_distribution(
    df: pd.DataFrame, 
    delay_columns: List[str] = ['DEP_DELAY'],
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Plot the distribution of flight delays for one or more columns side by side,
    and add a correlation scatter plot of DEP_DELAY vs ARR_DELAY.
    
    Args:
        df: DataFrame containing flight data
        delay_columns: List of column names containing delay values
        save_path: Optional path to save the plot
        
    Returns:
        fig, axes: Figure and list of Axes objects for the plots
    """
    # +1 column for correlation plot if both columns exist
    add_corr = {'DEP_DELAY', 'ARR_DELAY'}.issubset(df.columns)
    n_cols = len(delay_columns) + int(add_corr)

    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6), squeeze=False)
    axes = axes[0]  # flatten for 1 row

    # Plot histograms for each delay column
    for i, col in enumerate(delay_columns):
        sns.histplot(data=df, x=col, bins=50, ax=axes[i])
        axes[i].set_title(f'Distribution of {col.replace("_", " ").title()}')
        axes[i].set_xlabel('Delay (minutes)')
        axes[i].set_ylabel('Count')

    # Add correlation plot if both DEP_DELAY and ARR_DELAY exist
    if add_corr:
        sample_size = min(5000, len(df))
        sample_df = df.sample(sample_size, random_state=42)
        
        axes[-1].scatter(sample_df['DEP_DELAY'], sample_df['ARR_DELAY'], alpha=0.5, s=1)
        axes[-1].set_xlabel('Departure Delay (minutes)')
        axes[-1].set_ylabel('Arrival Delay (minutes)')
        axes[-1].set_title('Departure vs Arrival Delay Correlation')
        
        # Add correlation coefficient
        corr = df[['DEP_DELAY', 'ARR_DELAY']].corr().iloc[0, 1]
        axes[-1].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                     transform=axes[-1].transAxes, fontsize=12,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    
    if save_path:
        ensure_output_dir(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, axes


def plot_delays_by_carrier(
    df: pd.DataFrame, 
    delay_column: str = 'DEP_DELAY',
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot average delays by carrier with sample size annotations.
    
    Args:
        df: DataFrame containing flight data
        delay_column: Name of the column containing delay values
        save_path: Optional path to save the plot
        
    Returns:
        fig, ax: Figure and Axes objects for the plot
    """
    # Compute mean and sample size
    avg_delays = df.groupby('OP_CARRIER')[delay_column].agg(['mean', 'count']).reset_index()
    avg_delays = avg_delays.sort_values('mean', ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=avg_delays, x='OP_CARRIER', y='mean', ax=ax)

    ax.set_title(f'Average {delay_column.replace("_", " ").title()} by Carrier')
    ax.set_xlabel('Carrier')
    ax.set_ylabel('Average Delay (minutes)')

    # Add sample size annotations above bars
    for i, (mean, count) in enumerate(zip(avg_delays['mean'], avg_delays['count'])):
        ax.text(i, mean + 0.5, f'n={count:,}', ha='center', va='bottom', fontsize=9, color='black')

    if save_path:
        ensure_output_dir(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax


def plot_delays_by_hour(
    df: pd.DataFrame, 
    delay_column: str = 'DEP_DELAY',
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot average delays by hour of day.
    
    Args:
        df: DataFrame containing flight data
        delay_column: Name of the column containing delay values
        save_path: Optional path to save the plot
        
    Returns:
        fig, ax: Figure and Axes objects for the plot
    """
    hourly_delays = df.groupby('DEP_HOUR')[delay_column].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=hourly_delays, x='DEP_HOUR', y=delay_column, marker='o', ax=ax)
    ax.set_title(f'Average {delay_column.replace("_", " ").title()} by Hour')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average Delay (minutes)')
    ax.set_xticks(range(24))
    
    if save_path:
        ensure_output_dir(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax


def plot_correlation_matrix(
    df: pd.DataFrame, 
    columns: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot correlation matrix for selected features.
    
    Args:
        df: DataFrame containing flight data
        columns: List of columns to include in correlation matrix. If None, uses all numeric columns.
        save_path: Optional path to save the plot
        
    Returns:
        fig, ax: Figure and Axes objects for the plot
    """
    if columns is None:
        columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    corr_matrix = df[columns].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Feature Correlation Matrix')
    plt.tight_layout()
    
    if save_path:
        ensure_output_dir(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax


def plot_correlation_matrix_lg(df: pd.DataFrame, drop_features: List[str] = None, target: str = 'ARR_DEL15'):
    """
    Plot a correlation matrix for numeric features, optionally dropping specified features.
    Focus on correlations with the target variable.
    
    Args:
        df: DataFrame with features
        drop_features: List of feature names to exclude from the correlation matrix
        target: Target variable name
    
    Returns:
        fig, ax: matplotlib figure and axes
    """
    if drop_features is None:
        drop_features = []
    
    # Select numeric columns and drop specified features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features_to_use = [col for col in numeric_cols if col not in drop_features]
    
    # Calculate correlation matrix
    corr_matrix = df[features_to_use].corr()
    
    # Get correlations with target variable and sort by absolute value
    if target in corr_matrix.columns:
        target_corr = corr_matrix[target].abs().sort_values(ascending=False)
        # Reorder correlation matrix by target correlation strength
        ordered_features = target_corr.index.tolist()
        corr_matrix = corr_matrix.loc[ordered_features, ordered_features]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Create heatmap
    sns.heatmap(
        corr_matrix, 
        annot=False, 
        cmap='RdBu_r', 
        center=0, 
        square=True,
        fmt='.2f',
        cbar_kws={'shrink': 0.8},
        ax=ax
    )
    
    ax.set_title(f'Feature Correlation Matrix\n(Ordered by correlation strength with {target})', 
                fontsize=14, pad=20)
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    return fig, ax


def plot_weather_impact(
    df: pd.DataFrame, 
    weather_features: List[str], 
    delay_column: str = 'DEP_DELAY',
    ncols: int = 2,
    figsize_per_plot: Tuple[int, int] = (6, 4),
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, Union[plt.Axes, List[plt.Axes]]]:
    """
    Plot the relationship between weather features and delays in a grid layout.

    Args:
        df: DataFrame containing flight and weather data
        weather_features: List of weather-related column names
        delay_column: Name of the column containing delay values
        ncols: Number of columns in the grid
        figsize_per_plot: Size of each individual subplot (width, height)
        save_path: Optional path to save the plot
    """
    n_features = len(weather_features)
    nrows = (n_features + ncols - 1) // ncols  # round up
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per_plot[0]*ncols, figsize_per_plot[1]*nrows))
    
    # Flatten axes for easy iteration
    axes = axes.flatten() if n_features > 1 else [axes]

    for i, feature in enumerate(weather_features):
        sns.scatterplot(
            data=df.sample(min(1000, len(df))), 
            x=feature, 
            y=delay_column, 
            alpha=0.5,
            ax=axes[i]
        )
        axes[i].set_title(f'Impact of {feature.replace("_", " ").title()} on Delays')
        axes[i].set_xlabel(feature.replace('_', ' ').title())
        axes[i].set_ylabel('Delay (minutes)')

    # Hide any empty subplots if n_features < nrows*ncols
    for j in range(n_features, nrows*ncols):
        fig.delaxes(axes[j])

    plt.tight_layout()
    
    if save_path:
        ensure_output_dir(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, axes


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 20,
    save_dir: Optional[str] = None,
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot feature importance for a trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute or LightGBM model
        feature_names: List of feature names
        top_n: Number of top features to display
        save_dir: Directory to save the plot (deprecated, use save_path)
        save_path: Full path to save the plot
        
    Returns:
        fig, ax: Figure and Axes objects for the plot
    """
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        # Sklearn models
        importance = model.feature_importances_
    elif hasattr(model, 'feature_importance'):
        # LightGBM models
        importance = model.feature_importance(importance_type='gain')
    else:
        raise ValueError("Model does not have feature importance information")
    
    # Create DataFrame for easier handling
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(top_n)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    
    sns.barplot(
        data=importance_df,
        y='feature',
        x='importance',
        ax=ax,
        palette='viridis'
    )
    
    ax.set_title(f'Top {top_n} Feature Importance')
    ax.set_xlabel('Importance Score')
    ax.set_ylabel('Features')
    
    plt.tight_layout()
    
    # Handle saving (backward compatibility)
    if save_path:
        ensure_output_dir(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    elif save_dir:
        save_path = os.path.join(save_dir, 'feature_importance.png')
        ensure_output_dir(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax


def plot_missingness(df: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
    """Plot missingness patterns in the dataset.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        fig, ax: Figure and Axes objects for the plot
    """
    # Simple missingness visualization
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    if len(missing_data) == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No missing data found', ha='center', va='center', fontsize=14)
        ax.set_title('Data Completeness Analysis')
        return fig, ax
    
    fig, ax = plt.subplots(figsize=(10, 6))
    missing_data.plot(kind='bar', ax=ax)
    ax.set_title('Missing Data by Column')
    ax.set_xlabel('Columns')
    ax.set_ylabel('Number of Missing Values')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig, ax


def plot_top_airports_by_volume(df: pd.DataFrame, n: int = 20) -> Tuple[plt.Figure, plt.Axes]:
    """Plot top airports by flight volume.
    
    Args:
        df: DataFrame with flight data
        n: Number of top airports to show
        
    Returns:
        fig, ax: Figure and Axes objects for the plot
    """
    airport_counts = df['ORIGIN'].value_counts().head(n)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    airport_counts.plot(kind='bar', ax=ax)
    ax.set_title(f'Top {n} Airports by Flight Volume')
    ax.set_xlabel('Airport Code')
    ax.set_ylabel('Number of Flights')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig, ax


def plot_delay_rate_by_group(df: pd.DataFrame, group_col: str, top_n: int = 15) -> Tuple[plt.Figure, plt.Axes]:
    """Plot delay rates by a grouping column.
    
    Args:
        df: DataFrame with flight data
        group_col: Column to group by
        top_n: Number of top groups to show
        
    Returns:
        fig, ax: Figure and Axes objects for the plot
    """
    if 'ARR_DEL15' not in df.columns:
        # Create a simple delay indicator if not present
        delay_col = 'DEP_DELAY' if 'DEP_DELAY' in df.columns else 'ARR_DELAY'
        if delay_col in df.columns:
            df = df.copy()
            df['ARR_DEL15'] = (df[delay_col] >= 15).astype(int)
        else:
            raise ValueError("No delay information available")
    
    delay_rates = df.groupby(group_col)['ARR_DEL15'].agg(['count', 'mean']).reset_index()
    delay_rates = delay_rates[delay_rates['count'] >= 10]  # Filter for meaningful sample sizes
    delay_rates = delay_rates.sort_values('mean', ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(range(len(delay_rates)), delay_rates['mean'])
    ax.set_title(f'Delay Rate by {group_col} (Top {top_n})')
    ax.set_xlabel(group_col)
    ax.set_ylabel('Delay Rate')
    ax.set_xticks(range(len(delay_rates)))
    ax.set_xticklabels(delay_rates[group_col], rotation=45)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.1%}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig, ax


def plot_top_routes_delay(df: pd.DataFrame, n: int = 15, min_flights: int = 20) -> Tuple[plt.Figure, plt.Axes]:
    """Plot top routes by delay rate.
    
    Args:
        df: DataFrame with flight data
        n: Number of top routes to show
        min_flights: Minimum number of flights for a route to be included
        
    Returns:
        fig, ax: Figure and Axes objects for the plot
    """
    if 'ARR_DEL15' not in df.columns:
        # Create a simple delay indicator if not present
        delay_col = 'DEP_DELAY' if 'DEP_DELAY' in df.columns else 'ARR_DELAY'
        if delay_col in df.columns:
            df = df.copy()
            df['ARR_DEL15'] = (df[delay_col] >= 15).astype(int)
        else:
            raise ValueError("No delay information available")
    
    df['ROUTE'] = df['ORIGIN'] + '-' + df['DEST']
    route_stats = df.groupby('ROUTE')['ARR_DEL15'].agg(['count', 'mean']).reset_index()
    route_stats = route_stats[route_stats['count'] >= min_flights]
    route_stats = route_stats.sort_values('mean', ascending=False).head(n)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(range(len(route_stats)), route_stats['mean'])
    ax.set_title(f'Top {n} Routes by Delay Rate (min {min_flights} flights)')
    ax.set_xlabel('Route')
    ax.set_ylabel('Delay Rate')
    ax.set_xticks(range(len(route_stats)))
    ax.set_xticklabels(route_stats['ROUTE'], rotation=45)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.1%}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig, ax


def plot_inbound_delay_effect(df: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the effect of inbound delays on outbound delays.
    
    Args:
        df: DataFrame with flight data
        
    Returns:
        fig, ax: Figure and Axes objects for the plot
    """
    # This is a simplified version - would need more complex logic for real tail number tracking
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'DEP_DELAY' in df.columns and 'ARR_DELAY' in df.columns:
        # Simple correlation between arrival and departure delays
        delay_data = df[['DEP_DELAY', 'ARR_DELAY']].dropna()
        if len(delay_data) > 0:
            ax.scatter(delay_data['ARR_DELAY'], delay_data['DEP_DELAY'], alpha=0.1)
            ax.set_xlabel('Previous Arrival Delay (minutes)')
            ax.set_ylabel('Departure Delay (minutes)')
            ax.set_title('Relationship Between Arrival and Departure Delays')
        else:
            ax.text(0.5, 0.5, 'No delay data available', ha='center', va='center')
    else:
        ax.text(0.5, 0.5, 'Delay columns not found', ha='center', va='center')
    
    plt.tight_layout()
    return fig, ax


def plot_daily_delay_trend(df: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
    """Plot daily delay rate trends with special days marked.

    Args:
        df: DataFrame with flight data

    Returns:
        fig, ax: Figure and Axes objects for the plot
    """
    if 'FL_DATE' not in df.columns:
        raise ValueError("FL_DATE column not found")

    if 'ARR_DEL15' not in df.columns:
        # Create a simple delay indicator if not present
        delay_col = 'DEP_DELAY' if 'DEP_DELAY' in df.columns else 'ARR_DELAY'
        if delay_col in df.columns:
            df = df.copy()
            df['ARR_DEL15'] = (df[delay_col] >= 15).astype(int)
        else:
            raise ValueError("No delay information available")

    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
    daily_delays = df.groupby('FL_DATE')['ARR_DEL15'].mean().reset_index()

    # Define special days
    holiday_dates = {
        "New Year's Day": pd.Timestamp("2025-01-01"),
        "MLK Day": pd.Timestamp("2025-01-20"),
        "Super Bowl": pd.Timestamp("2025-02-09")  # Sunday before Feb 10
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(daily_delays['FL_DATE'], daily_delays['ARR_DEL15'], marker='o', label='Daily Delay Rate')

    # Add vertical lines for special days
    colors = ['red', 'orange', 'purple']
    for (holiday_name, holiday_date), color in zip(holiday_dates.items(), colors):
        if daily_delays['FL_DATE'].min() <= holiday_date <= daily_delays['FL_DATE'].max():
            ax.axvline(holiday_date, color=color, linestyle='--', alpha=0.7, linewidth=2, label=holiday_name)

    ax.set_title('Daily Delay Rate Trend with Special Events')
    ax.set_xlabel('Date')
    ax.set_ylabel('Delay Rate')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig, ax