from elo_model import EloModel
import pandas as pd
import sqlite3

def test_backtest_with_vegas():
    model = EloModel(k_factor=20, home_advantage=65, starting_elo=1500)
    
    # Run backtest
    print("Running backtest...")
    results = model.backtest('data/nfl_betting.db', seasons=[2019, 2020, 2021, 2022, 2023, 2024])
    
    # Load Vegas lines from database
    conn = sqlite3.connect('data/nfl_betting.db')
    vegas_lines = pd.read_sql("""
        SELECT game_id, vegas_spread_close, vegas_total
        FROM games
    """, conn)
    conn.close()
    
    # Merge with backtest results
    results = results.merge(vegas_lines, on='game_id', how='left')
    
    # Filter to games with Vegas data
    vegas_games = results[results['vegas_spread_close'].notna()].copy()
    
    print(f"\n{'='*60}")
    print("VEGAS COMPARISON")
    print(f"{'='*60}")
    print(f"\nGames with Vegas lines: {len(vegas_games)}")
    
    # FIX: Convert Vegas spread to same convention as Elo
    # Vegas: negative = home favored
    # Elo: positive = home favored
    # So flip Vegas sign
    vegas_games['vegas_spread_home_perspective'] = -vegas_games['vegas_spread_close']
    
    # Calculate predictions
    vegas_games['vegas_predicted_home_win'] = vegas_games['vegas_spread_home_perspective'] > 0
    vegas_games['vegas_correct'] = vegas_games['vegas_predicted_home_win'] == vegas_games['home_won']
    
    vegas_games['elo_predicted_home_win'] = vegas_games['predicted_spread'] > 0
    vegas_games['elo_correct'] = vegas_games['elo_predicted_home_win'] == vegas_games['home_won']
    
    print(f"\n--- WIN/LOSS ACCURACY ---")
    print(f"My Elo Model:    {vegas_games['elo_correct'].mean():.1%}")
    print(f"Vegas Lines:     {vegas_games['vegas_correct'].mean():.1%}")
    
    # Spread comparison
    vegas_games['elo_spread_error'] = abs(vegas_games['predicted_spread'] - vegas_games['actual_spread'])
    vegas_games['vegas_spread_error'] = abs(vegas_games['vegas_spread_home_perspective'] - vegas_games['actual_spread'])
    
    print(f"\n--- SPREAD ACCURACY (MAE) ---")
    print(f"My Elo Model:    {vegas_games['elo_spread_error'].mean():.2f} points")
    print(f"Vegas Lines:     {vegas_games['vegas_spread_error'].mean():.2f} points")
    
    # Agreement analysis
    elo_pick = vegas_games['predicted_spread'] > 0
    vegas_pick = vegas_games['vegas_spread_home_perspective'] > 0
    agreement = elo_pick == vegas_pick
    
    print(f"\n--- AGREEMENT ---")
    print(f"Games where Elo & Vegas agree on winner: {agreement.mean():.1%}")
    print(f"When we agree, my accuracy: {vegas_games[agreement]['elo_correct'].mean():.1%}")
    print(f"When we disagree, my accuracy: {vegas_games[~agreement]['elo_correct'].mean():.1%}")
    
    # Find biggest edges (where spreads differ most)
    vegas_games['spread_difference'] = abs(vegas_games['predicted_spread'] - vegas_games['vegas_spread_home_perspective'])
    big_disagreements = vegas_games.nlargest(10, 'spread_difference')
    
    print(f"\n--- MY 10 BIGGEST DISAGREEMENTS WITH VEGAS ---")
    print(big_disagreements[['date', 'home_team', 'away_team', 
                              'predicted_spread', 'vegas_spread_home_perspective', 
                              'actual_spread', 'elo_correct']].to_string(index=False))
    
    # Accuracy by season
    print(f"\n--- ACCURACY BY SEASON ---")
    season_comparison = vegas_games.groupby('season').agg({
        'elo_correct': 'mean',
        'vegas_correct': 'mean'
    }).round(3)
    season_comparison.columns = ['My Elo', 'Vegas']
    print(season_comparison)
    
    # Save extended results
    results.to_csv('data/backtest_with_vegas.csv', index=False)
    print(f"\n✓ Results saved to data/backtest_with_vegas.csv")

if __name__ == "__main__":
    test_backtest_with_vegas()