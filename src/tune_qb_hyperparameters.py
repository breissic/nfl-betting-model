"""
Hyperparameter tuning for QB Elo system with adjustment method.
Tests different combinations of qb_weight, qb_k_factor, and qb_differential_threshold.
"""

import sqlite3
import pandas as pd
from qb_elo_model import QBEloModel
from elo_model import EloModel
import itertools
import sys
from io import StringIO

def run_backtest_with_params(db_path, seasons, qb_weight, qb_k_factor, qb_differential_threshold):
    """
    Run a backtest with specific QB Elo parameters using adjustment method.
    
    Parameters:
    -----------
    db_path : str
        Path to database
    seasons : list
        List of seasons to backtest
    qb_weight : float
        Multiplier for QB adjustment
    qb_k_factor : int
        QB rating change speed
    qb_differential_threshold : int
        Only apply QB Elo when abs(qb_diff) >= threshold
        
    Returns:
    --------
    dict
        Dictionary with accuracy and mae metrics
    """
    # Suppress debug output during backtest
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        # Initialize models with adjustment method
        qb_model = QBEloModel(
            qb_weight=qb_weight,
            qb_k_factor=qb_k_factor
        )
        elo_model = EloModel(
            qb_model=qb_model,
            qb_differential_threshold=qb_differential_threshold
        )

        # Run backtest
        results_df = elo_model.backtest(
            db_path=db_path,
            seasons=seasons
        )

        # Calculate metrics
        accuracy = float(results_df['predicted_correctly'].mean())
        mae = float((results_df['predicted_spread'] - results_df['actual_spread']).abs().mean())

        return {'accuracy': accuracy, 'mae': mae}

    finally:
        # Restore stdout
        sys.stdout = old_stdout

def tune_hyperparameters(db_path='data/nfl_betting.db', seasons=None):
    """
    Test multiple combinations of qb_weight, qb_k_factor, and qb_differential_threshold.
    Uses adjustment method: team_elo + (qb_elo - 1500) * qb_weight
    
    Parameters:
    -----------
    db_path : str
        Path to database
    seasons : list
        List of seasons to test (default: 2019-2024)
        
    Returns:
    --------
    pandas.DataFrame
        Results of all tested configurations
    """
    if seasons is None:
        seasons = [2019, 2020, 2021, 2022, 2023, 2024]

    print("=" * 70)
    print("QB ELO HYPERPARAMETER TUNING - ADJUSTMENT METHOD")
    print("=" * 70)
    print(f"Database: {db_path}")
    print(f"Seasons: {seasons}")
    print()
    print("Testing adjustment method: team_elo + (qb_elo - 1500) * qb_weight")
    print()
    print("Parameter ranges:")
    print("  qb_weight: Controls magnitude of QB impact")
    print("  qb_k_factor: Controls speed of QB rating changes")
    print("  qb_differential_threshold: Minimum QB gap to apply adjustment")
    print()

    # Parameter ranges to test based on previous analysis
    qb_weights = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    qb_k_factors = [15, 20, 25, 30]
    qb_differential_thresholds = [0, 25, 50, 75, 100]

    print(f"Testing {len(qb_weights)} qb_weight values: {qb_weights}")
    print(f"Testing {len(qb_k_factors)} qb_k_factor values: {qb_k_factors}")
    print(f"Testing {len(qb_differential_thresholds)} qb_differential_threshold values: {qb_differential_thresholds}")
    print(f"Total combinations: {len(qb_weights) * len(qb_k_factors) * len(qb_differential_thresholds)}")
    print()

    # Store results
    results_list = []

    # Test all combinations
    total = len(qb_weights) * len(qb_k_factors) * len(qb_differential_thresholds)
    count = 0

    for qb_weight, qb_k_factor, qb_threshold in itertools.product(qb_weights, qb_k_factors, qb_differential_thresholds):
        count += 1
        print(f"[{count}/{total}] qb_weight={qb_weight:.2f}, qb_k_factor={qb_k_factor}, threshold={qb_threshold}...", end=' ')

        try:
            results = run_backtest_with_params(db_path, seasons, qb_weight, qb_k_factor, qb_threshold)
            accuracy = results['accuracy']
            mae = results['mae']

            results_list.append({
                'qb_weight': qb_weight,
                'qb_k_factor': qb_k_factor,
                'qb_differential_threshold': qb_threshold,
                'accuracy': accuracy,
                'mae': mae
            })

            print(f"Acc: {accuracy:.3f}, MAE: {mae:.2f}")

        except Exception as e:
            print(f"ERROR: {str(e)}")
            continue

    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print()

    # Convert to dataframe
    results_df = pd.DataFrame(results_list)

    # Check if we have any results
    if len(results_df) == 0:
        print("ERROR: No successful backtest runs. Check for errors above.")
        return None

    # Sort by accuracy
    results_df = results_df.sort_values('accuracy', ascending=False)

    # Display top 15
    print("TOP 15 CONFIGURATIONS BY ACCURACY:")
    print()
    print(results_df.head(15).to_string(index=False))
    print()

    # Find best by accuracy
    best_acc = results_df.iloc[0]
    print("BEST CONFIGURATION (by accuracy):")
    print(f"  qb_weight: {best_acc['qb_weight']:.2f}")
    print(f"  qb_k_factor: {int(best_acc['qb_k_factor'])}")
    print(f"  qb_differential_threshold: {int(best_acc['qb_differential_threshold'])}")
    print(f"  Accuracy: {best_acc['accuracy']:.4f} ({best_acc['accuracy']*100:.2f}%)")
    print(f"  MAE: {best_acc['mae']:.2f}")
    print()

    # Find best by MAE
    best_mae = results_df.loc[results_df['mae'].idxmin()]
    print("BEST CONFIGURATION (by MAE):")
    print(f"  qb_weight: {best_mae['qb_weight']:.2f}")
    print(f"  qb_k_factor: {int(best_mae['qb_k_factor'])}") #type: ignore
    print(f"  qb_differential_threshold: {int(best_mae['qb_differential_threshold'])}") #type: ignore
    print(f"  Accuracy: {best_mae['accuracy']:.4f} ({best_mae['accuracy']*100:.2f}%)")
    print(f"  MAE: {best_mae['mae']:.2f}")
    print()

    # Save results
    output_file = 'data/qb_hyperparameter_tuning_adjustment.csv'
    results_df.to_csv(output_file, index=False)
    print(f"Full results saved to: {output_file}")
    print()

    # Analyze by threshold
    print("BEST ACCURACY BY QB DIFFERENTIAL THRESHOLD:")
    print()
    threshold_summary = results_df.groupby('qb_differential_threshold').agg({
        'accuracy': 'max',
        'mae': 'min'
    }).reset_index()
    threshold_summary.columns = ['threshold', 'best_accuracy', 'best_mae']
    print(threshold_summary.to_string(index=False))
    print()

    # Show top configuration for each threshold
    print("TOP CONFIGURATION FOR EACH THRESHOLD:")
    print()
    for threshold in qb_differential_thresholds:
        threshold_results = results_df[results_df['qb_differential_threshold'] == threshold]
        if len(threshold_results) > 0:
            best = threshold_results.nlargest(1, 'accuracy').iloc[0]
            print(f"Threshold {threshold:3d}: qb_weight={best['qb_weight']:.2f}, qb_k_factor={int(best['qb_k_factor']):2d} -> Acc={best['accuracy']:.4f} ({best['accuracy']*100:.2f}%)")
    print()

    # Analyze by qb_weight
    print("BEST ACCURACY BY QB WEIGHT:")
    print()
    weight_summary = results_df.groupby('qb_weight').agg({
        'accuracy': 'max',
        'mae': 'min'
    }).reset_index()
    weight_summary.columns = ['qb_weight', 'best_accuracy', 'best_mae']
    print(weight_summary.to_string(index=False))
    print()

    # Analyze by qb_k_factor
    print("BEST ACCURACY BY QB K-FACTOR:")
    print()
    k_summary = results_df.groupby('qb_k_factor').agg({
        'accuracy': 'max',
        'mae': 'min'
    }).reset_index()
    k_summary.columns = ['qb_k_factor', 'best_accuracy', 'best_mae']
    print(k_summary.to_string(index=False))
    print()

    print("=" * 70)
    print("TUNING COMPLETE")
    print("=" * 70)
    print()
    print("Key insights:")
    print("1. Compare best result to baseline (63.6%)")
    print("2. Check if specific threshold values work better")
    print("3. Look for patterns in qb_weight vs accuracy")
    print("4. Consider combining with rest/divisional features")

    return results_df

if __name__ == '__main__':
    tune_hyperparameters()