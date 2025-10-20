import pandas as pd
import sqlite3
from datetime import datetime

TEAM_NAME_MAPPING = {
    'Arizona Cardinals': 'ARI',
    'Atlanta Falcons': 'ATL',
    'Baltimore Ravens': 'BAL',
    'Buffalo Bills': 'BUF',
    'Carolina Panthers': 'CAR',
    'Chicago Bears': 'CHI',
    'Cincinnati Bengals': 'CIN',
    'Cleveland Browns': 'CLE',
    'Dallas Cowboys': 'DAL',
    'Denver Broncos': 'DEN',
    'Detroit Lions': 'DET',
    'Green Bay Packers': 'GB',
    'Houston Texans': 'HOU',
    'Indianapolis Colts': 'IND',
    'Jacksonville Jaguars': 'JAX',
    'Kansas City Chiefs': 'KC',
    'Las Vegas Raiders': 'LV',
    'Los Angeles Chargers': 'LAC',
    'Los Angeles Rams': 'LA',  # They use LA in nfl_data_py
    'Miami Dolphins': 'MIA',
    'Minnesota Vikings': 'MIN',
    'New England Patriots': 'NE',
    'New Orleans Saints': 'NO',
    'New York Giants': 'NYG',
    'New York Jets': 'NYJ',
    'Oakland Raiders': 'OAK',  # Pre-2020
    'Philadelphia Eagles': 'PHI',
    'Pittsburgh Steelers': 'PIT',
    'San Diego Chargers': 'SD',  # Pre-2017
    'San Francisco 49ers': 'SF',
    'Seattle Seahawks': 'SEA',
    'St. Louis Rams': 'STL',  # Pre-2016
    'Tampa Bay Buccaneers': 'TB',
    'Tennessee Titans': 'TEN',
    'Washington Commanders': 'WAS',
    'Washington Football Team': 'WAS',
    'Washington Redskins': 'WAS',
}

def import_vegas_lines(csv_path='data/raw/nfl_betting_kaggle.csv', db_path='data/nfl_betting.db'):
    """Import Vegas lines from Kaggle CSV into database"""
    
    print("Loading Vegas lines data...")
    df = pd.read_csv(csv_path)
    
    # Filter to 2019+ only
    df = df[df['schedule_season'] >= 2019].copy()
    
    # Map team names to abbreviations
    df['home_team_abbr'] = df['team_home'].map(TEAM_NAME_MAPPING)
    df['away_team_abbr'] = df['team_away'].map(TEAM_NAME_MAPPING)
    
    # Check for unmapped teams
    unmapped_home = df[df['home_team_abbr'].isna()]['team_home'].unique()
    unmapped_away = df[df['away_team_abbr'].isna()]['team_away'].unique()
    if len(unmapped_home) > 0 or len(unmapped_away) > 0:
        print(f"WARNING: Unmapped teams found!")
        print(f"  Home: {unmapped_home}")
        print(f"  Away: {unmapped_away}")
        return
    
    # Convert date format: "9/5/2019" -> "2019-09-05"
    df['date'] = pd.to_datetime(df['schedule_date']).dt.strftime('%Y-%m-%d')
    
    # Convert spread to home team perspective
    # If home team is favorite (team_favorite_id == home), spread is negative
    # If away team is favorite, spread is positive (from home's perspective)
    def convert_spread_to_home_perspective(row) -> float | None:
        if pd.isna(row['spread_favorite']):
            return None
        
        if row['team_favorite_id'] == row['home_team_abbr']:
            # Home team is favorite, spread is already negative
            return float(row['spread_favorite'])
        elif row['team_favorite_id'] == row['away_team_abbr']:
            # Away team is favorite, flip the sign
            return float(-row['spread_favorite'])
        else:
            # Pick'em or unclear
            return None
    
    df['home_spread'] = df.apply(convert_spread_to_home_perspective, axis=1) #type: ignore
    
    # Parse over/under (it's stored as object, convert to float)
    df['total'] = pd.to_numeric(df['over_under_line'], errors='coerce')
    
    print(f"Loaded {len(df)} games from 2019+")
    print("\nSample with converted spreads:")
    print(df[['date', 'home_team_abbr', 'away_team_abbr', 'team_favorite_id', 
              'spread_favorite', 'home_spread', 'total']].head(10))
    
    # Now update database
    conn = sqlite3.connect(db_path)
    
    matched = 0
    unmatched = 0
    
    for _, row in df.iterrows():
        # Find matching game in database
        result = conn.execute("""
            SELECT game_id FROM games
            WHERE date = ? 
            AND home_team_id = ? 
            AND away_team_id = ?
        """, (row['date'], row['home_team_abbr'], row['away_team_abbr'])).fetchone()
        
        if result:
            game_id = result[0]
            # Update Vegas lines
            conn.execute("""
                UPDATE games 
                SET vegas_spread_close = ?, vegas_total = ?
                WHERE game_id = ?
            """, (row['home_spread'], row['total'], game_id))
            matched += 1
        else:
            unmatched += 1
    
    conn.commit()
    conn.close()
    
    print(f"\n✓ Matched {matched} games")
    print(f"✗ Unmatched {unmatched} games")
    print(f"\nVegas lines imported successfully!")

if __name__ == "__main__":
    import_vegas_lines()