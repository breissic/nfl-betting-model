from elo_model import EloModel

def test_elo():
    """Test the Elo model on historical data"""
    
    # Initialize model
    model = EloModel(k_factor=20, home_advantage=65, starting_elo=1500)
    
    # Run simulation
    print("Running Elo simulation...")
    model.run_simulation('data/nfl_betting.db', seasons=[2019, 2020, 2021, 2022, 2023, 2024])
    
    # Save results
    model.save_ratings_to_db('data/nfl_betting.db')
    
    # Print final ratings (sorted by rating)
    print("\nTop 10 teams by final Elo rating:")
    sorted_teams = sorted(model.ratings.items(), key=lambda x: x[1], reverse=True)
    for i, (team, rating) in enumerate(sorted_teams[:10], 1):
        print(f"{i}. {team}: {rating:.1f}")
    
    print(f"\nBottom 10 teams:")
    for i, (team, rating) in enumerate(sorted_teams[-10:], 1):
        print(f"{i}. {team}: {rating:.1f}")

if __name__ == "__main__":
    test_elo()