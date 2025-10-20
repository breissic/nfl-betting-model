"""
Quick test to verify data loader works
"""

from data_loader import NFLDataLoader

def test_basic_load():
    """Test loading a small amount of data"""
    loader = NFLDataLoader('data/test.db')
    
    try:
        # Load teams
        teams = loader.load_teams()
        print(f"Loaded {len(teams)} teams")
        print("Sample teams:", teams.head())
        
        # Load just 2024 season
        games = loader.load_games([2024])
        print(f"\nLoaded {len(games)} games")
        print("Sample games:", games.head())
        
    finally:
        loader.close()

if __name__ == "__main__":
    test_basic_load()