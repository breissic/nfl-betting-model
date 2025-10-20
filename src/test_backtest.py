from elo_model import EloModel

def test_backtest():
    model = EloModel(k_factor=20, home_advantage=65, starting_elo=1500)
    
    # Run backtest
    results = model.backtest('data/nfl_betting.db', seasons=[2019, 2020, 2021, 2022, 2023, 2024])
    
    # Quick stats
    print(f"\nTotal games: {len(results)}")
    print(f"Prediction accuracy: {results['predicted_correctly'].mean():.1%}")
    print(f"\nSample predictions:")
    print(results[['date', 'home_team', 'away_team', 'predicted_home_win_prob', 
                   'predicted_spread', 'actual_spread', 'predicted_correctly']].head(10))
    print("\nPredictions from Week 10, 2024 (after ratings diverged):")
    print(results[results['season'] == 2024][['date', 'home_team', 'away_team', 
                                                'predicted_home_win_prob', 
                                                'predicted_spread', 'actual_spread']].tail(10))
    
    # Mean Absolute Error on spreads
    results['spread_error'] = abs(results['predicted_spread'] - results['actual_spread'])
    print(f"\nMean Absolute Error (spread): {results['spread_error'].mean():.2f} points")

    # Accuracy by season
    print("\nAccuracy by season:")
    print(results.groupby('season')['predicted_correctly'].mean())

    # Check calibration
    print("\nPredictions where model was >70% confident:")
    confident = results[results['predicted_home_win_prob'] > 0.7]
    print(f"  Games: {len(confident)}")
    print(f"  Actual accuracy: {confident['predicted_correctly'].mean():.1%}")
        
    # Save to CSV for analysis
    results.to_csv('data/backtest_results.csv', index=False)
    print("\nResults saved to data/backtest_results.csv")

if __name__ == "__main__":
    test_backtest()