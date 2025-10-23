import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from elo_model import EloModel
from qb_elo_model import QBEloModel
from qb_data_loader import QBDataLoader
import pandas as pd
import sqlite3

def compare_with_vegas(results_df, db_path):
    """
    Compare model predictions with Vegas lines.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        Backtest results with predictions
    db_path : str
        Path to database with Vegas lines
        
    Returns:
    --------
    pandas.DataFrame
        Results with Vegas comparison data added
    """
    conn = sqlite3.connect(db_path)
    vegas_lines = pd.read_sql("""
        SELECT game_id, vegas_spread_close
        FROM games
    """, conn)
    conn.close()

    results = results_df.merge(vegas_lines, on='game_id', how='left')

    vegas_games = results[results['vegas_spread_close'].notna()].copy()

    if len(vegas_games) == 0:
        print("Warning: No Vegas lines found in database")
        return None

    vegas_games['vegas_spread_home'] = -vegas_games['vegas_spread_close']

    vegas_games['vegas_predicted_home_win'] = vegas_games['vegas_spread_home'] > 0
    vegas_games['vegas_correct'] = vegas_games['vegas_predicted_home_win'] == vegas_games['home_won']

    vegas_games['elo_predicted_home_win'] = vegas_games['predicted_spread'] > 0
    vegas_games['elo_correct'] = vegas_games['elo_predicted_home_win'] == vegas_games['home_won']

    vegas_games['elo_spread_error'] = abs(vegas_games['predicted_spread'] - vegas_games['actual_spread'])
    vegas_games['vegas_spread_error'] = abs(vegas_games['vegas_spread_home'] - vegas_games['actual_spread'])

    print("\n" + "="*60)
    print("COMPARISON WITH VEGAS")
    print("="*60)
    print(f"\nGames analyzed: {len(vegas_games)}")

    print(f"\n--- WIN/LOSS ACCURACY ---")
    elo_acc = vegas_games['elo_correct'].mean()
    vegas_acc = vegas_games['vegas_correct'].mean()
    print(f"Elo Model (with QB Elo): {elo_acc:.1%}")
    print(f"Vegas Lines:              {vegas_acc:.1%}")
    print(f"Difference:               {(elo_acc - vegas_acc)*100:+.1f} percentage points")

    print(f"\n--- SPREAD ACCURACY (MAE) ---")
    elo_mae = vegas_games['elo_spread_error'].mean()
    vegas_mae = vegas_games['vegas_spread_error'].mean()
    print(f"Elo Model (with QB Elo): {elo_mae:.2f} points")
    print(f"Vegas Lines:              {vegas_mae:.2f} points")
    print(f"Difference:               {(elo_mae - vegas_mae):+.2f} points")

    print(f"\n--- ACCURACY BY SEASON ---")
    season_comparison = vegas_games.groupby('season').agg({
        'elo_correct': 'mean',
        'vegas_correct': 'mean'
    }).round(3)
    season_comparison.columns = ['Elo Model', 'Vegas']
    print(season_comparison)

    return vegas_games


def analyze_qb_impact(results_df, qb_model):
    """
    Analyze how QB Elo affects predictions.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        Backtest results
    qb_model : QBEloModel
        Trained QB Elo model
    """
    print("\n" + "="*60)
    print("QB ELO IMPACT ANALYSIS")
    print("="*60)

    # Filter to games with QB data
    with_qbs = results_df[
        results_df['home_qb'].notna() & results_df['away_qb'].notna()
    ].copy()

    without_qbs = results_df[
        results_df['home_qb'].isna() | results_df['away_qb'].isna()
    ].copy()

    print(f"\nGames with both QBs: {len(with_qbs)}")
    print(f"  Model accuracy: {with_qbs['predicted_correctly'].mean():.1%}")

    print(f"\nGames missing QB data: {len(without_qbs)}")
    print(f"  Model accuracy: {without_qbs['predicted_correctly'].mean():.1%}")

    # Show top and bottom QBs by Elo
    if qb_model:
        qb_summary = qb_model.get_rating_summary()
        if len(qb_summary) > 0:
            print(f"\n--- TOP 10 QBs BY ELO ---")
            print(qb_summary.head(10).to_string(index=False))

            print(f"\n--- BOTTOM 10 QBs BY ELO ---")
            print(qb_summary.tail(10).to_string(index=False))

    # Analyze QB Elo gap vs win rate
    if 'home_qb_elo' in with_qbs.columns and 'away_qb_elo' in with_qbs.columns:
        with_qbs['qb_elo_diff'] = with_qbs['home_qb_elo'] - with_qbs['away_qb_elo']

        print(f"\n--- QB ELO DIFFERENTIAL IMPACT ---")
        for threshold in [50, 100, 150]:
            large_advantage = with_qbs[with_qbs['qb_elo_diff'] > threshold]
            if len(large_advantage) > 0:
                print(f"\nGames where home QB has >{threshold} Elo advantage: {len(large_advantage)}")
                print(f"  Home team won: {large_advantage['home_won'].mean():.1%}")
                print(f"  Model accuracy: {large_advantage['predicted_correctly'].mean():.1%}")


def test_qb_elo(load_qb_data=False, qb_weight=1.0, qb_k_factor=25, 
                qb_differential_threshold=50):
    """
    Test QB Elo with adjustment method and configurable parameters.
    
    Parameters:
    -----------
    load_qb_data : bool
        Whether to load QB data from nfl_data_py (default False)
    qb_weight : float
        Multiplier for QB adjustment (default 1.0)
    qb_k_factor : int
        QB rating change speed (default 25)
    qb_differential_threshold : int
        Only apply QB Elo when absolute difference >= threshold (default 50)
    """
    db_path = 'data/nfl_betting.db'
    seasons = [2019, 2020, 2021, 2022, 2023, 2024]

    print("="*60)
    print("QB ELO TEST - Adjustment Method with Threshold")
    print("="*60)
    print(f"Database: {db_path}")
    print(f"Seasons: {seasons}")
    print(f"QB Weight: {qb_weight}")
    print(f"QB K-Factor: {qb_k_factor}")
    print(f"QB Differential Threshold: {qb_differential_threshold}")
    print("QB ratings updated based on EPA, not wins/losses")
    print("Uses adjustment method: team_elo + (qb_elo - 1500) * qb_weight")
    print("="*60)

    # Step 1: Load QB data if requested
    if load_qb_data:
        print("\n### STEP 1: Loading QB Data ###")
        loader = QBDataLoader(db_path)
        loader.update_games_with_qb_data(seasons)
        loader.get_qb_summary(seasons)
    else:
        print("\n### STEP 1: Skipping QB data load (use load_qb_data=True to load) ###")
        loader = QBDataLoader(db_path)
        loader.get_qb_summary(seasons)

    # Step 2: Initialize models
    print("\n### STEP 2: Initializing Models ###")
    print(f"Creating QB Elo model with qb_weight={qb_weight}, qb_k_factor={qb_k_factor}")
    print(f"QB adjustment method: team_elo + (qb_elo - 1500) * {qb_weight}")
    print(f"QB differential threshold: {qb_differential_threshold}")

    qb_model = QBEloModel(
        starting_qb_elo=1500,
        qb_k_factor=qb_k_factor,
        qb_weight=qb_weight
    )

    elo_model = EloModel(
        k_factor=20,
        home_advantage=65,
        starting_elo=1500,
        rest_impact=0,  # Disable rest for clean QB comparison
        divisional_penalty=0,  # Disable divisional for clean QB comparison
        qb_model=qb_model,
        qb_differential_threshold=qb_differential_threshold
    )

    # Step 3: Run backtest
    print("\n### STEP 3: Running Backtest with QB Elo ###")
    results = elo_model.backtest(db_path, seasons)

    # Step 4: Evaluate results
    print("\n### STEP 4: Results ###")

    overall_accuracy = results['predicted_correctly'].mean()
    spread_mae = (results['predicted_spread'] - results['actual_spread']).abs().mean()

    print(f"\nOverall Performance:")
    print(f"  Win/Loss Accuracy: {overall_accuracy:.1%}")
    print(f"  Spread MAE: {spread_mae:.2f} points")

    vegas_results = compare_with_vegas(results, db_path)

    analyze_qb_impact(results, qb_model)

    # Save results
    output_file = 'data/backtest_qb_elo_adjustment.csv'
    results.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Save QB ratings
    qb_model.save_qb_ratings_to_db(db_path)

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Compare these results with your baseline (63.6%)")
    print("2. Try different qb_weight values (0.5, 1.0, 1.5, 2.0)")
    print("3. Try different thresholds (0, 25, 50, 75, 100)")
    print("4. Combine with rest/divisional features if beneficial")
    
    return results


if __name__ == "__main__":
    # Run test with default parameters
    # These defaults are based on analysis showing adjustment method with threshold works best
    test_qb_elo(
        load_qb_data=False,  # Set to True first time to load QB data
        qb_weight=1.0,
        qb_k_factor=25,
        qb_differential_threshold=50
    )
