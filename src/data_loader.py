import nfl_data_py as nfl
import pandas as pd
import sqlite3
from datetime import datetime
from pathlib import Path

class NFLDataLoader:
    # Load NFL data into database

    def __init__(self, db_path='data/nfl_betting.db'):
        self.db_path = db_path

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(db_path)
        print(f"DEBUG: Database connected: {db_path}")

    def initialize_schema(self):
        # Create database tables with proper schema
        print("Initializing database schema...")
        
        # Drop existing tables if they exist
        self.conn.execute("DROP TABLE IF EXISTS predictions")
        self.conn.execute("DROP TABLE IF EXISTS elo_ratings")
        self.conn.execute("DROP TABLE IF EXISTS games")
        self.conn.execute("DROP TABLE IF EXISTS teams")
        
        # Create tables with proper constraints
        self.conn.execute("""
            CREATE TABLE teams(
                team_id TEXT PRIMARY KEY,
                team_name TEXT NOT NULL,
                conference TEXT,
                division TEXT
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE games(
                game_id INTEGER PRIMARY KEY AUTOINCREMENT,
                season INTEGER NOT NULL,
                week INTEGER NOT NULL,
                date TEXT NOT NULL,
                home_team_id TEXT NOT NULL,
                away_team_id TEXT NOT NULL,
                home_score INTEGER,
                away_score INTEGER,
                vegas_spread_open REAL,
                vegas_spread_close REAL,
                vegas_total REAL,
                FOREIGN KEY(home_team_id) REFERENCES teams(team_id),
                FOREIGN KEY(away_team_id) REFERENCES teams(team_id)
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE elo_ratings(
                team_id TEXT NOT NULL,
                date TEXT NOT NULL,
                elo_rating REAL NOT NULL,
                season INTEGER NOT NULL,
                week INTEGER NOT NULL,
                PRIMARY KEY(team_id, season, week),
                FOREIGN KEY(team_id) REFERENCES teams(team_id)
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE predictions(
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id INTEGER NOT NULL,
                model_version TEXT NOT NULL,
                predicted_spread REAL,
                predicted_total REAL,
                confidence REAL,
                timestamp INTEGER NOT NULL,
                FOREIGN KEY(game_id) REFERENCES games(game_id),
                UNIQUE(game_id, model_version)
            )
        """)
        
        # Create indexes
        self.conn.execute("CREATE INDEX idx_games_date ON games(date)")
        self.conn.execute("CREATE INDEX idx_games_season_week ON games(season, week)")
        self.conn.execute("CREATE INDEX idx_games_teams ON games(home_team_id, away_team_id)")
        self.conn.execute("CREATE INDEX idx_elo_team_date ON elo_ratings(team_id, date)")
        
        self.conn.commit()
        print("✓ Schema initialized")

    def load_teams(self):
        # Load NFL teams into teams table

        print("DEBUG: Attempting to load team data")

        try:
            teams_raw = nfl.import_team_desc()

            teams_df = pd.DataFrame({
                'team_id': teams_raw['team_abbr'],
                'team_name': teams_raw['team_name'],
                'conference': teams_raw['team_conf'],
                'division': teams_raw['team_division']
            })

            teams_df = teams_df.drop_duplicates(subset=['team_id'])
            teams_df.to_sql('teams', self.conn, if_exists='append', index=False)

            print(f'DEBUG: Loaded {len(teams_df)} teams succesfully')
            return teams_df
        except Exception as e:
            print(f"DEBUG: Error loading teams: {e}")
            raise

    def load_games(self, seasons):
        # Load game schedules and results into games table

        print(f"DEBUG: Loading games for seasons: {seasons}")

        try:
            schedules_raw = nfl.import_schedules(seasons)

            # game_type: REG = reg season, the others are playoff
            schedules_raw = schedules_raw[schedules_raw['game_type'].isin(['REG', 'WC', 'DIV', 'CON', 'SB'])]

            games_df = pd.DataFrame({
                'season': schedules_raw['season'],
                'week': schedules_raw['week'],
                'date': schedules_raw['gameday'],  # YYYY-MM-DD
                'home_team_id': schedules_raw['home_team'],
                'away_team_id': schedules_raw['away_team'],
                'home_score': schedules_raw['home_score'],
                'away_score': schedules_raw['away_score'],
                # TODO: Add Vegas lines
                'vegas_spread_open': None,
                'vegas_spread_close': None,
                'vegas_total': None
            })

            games_df = games_df.sort_values(['date', 'season', 'week']).reset_index(drop=True)

            games_df.to_sql('games', self.conn, if_exists='append', index=False)

            completed_games = games_df['home_score'].notna().sum()
            upcoming_games = games_df['home_score'].isna().sum()

            print(f"DEBUG: Loaded {len(games_df)} games\n\t-{completed_games} completed games\n\t-{upcoming_games} upcoming games")
            return games_df
        
        except Exception as e:
            print(f"Error loading games: {e}")
            raise

    def load_vegas_lines(self):
        """
        TODO: Use Odds API or find a dataset for vegas lines
        """
        pass

    def get_data_summary(self):
        # Spit out a summary of the loaded data

        print("\n" + "="*50)
        print("DATABASE SUMMARY")
        print("="*50)
        
        # Teams
        teams_count = pd.read_sql("SELECT COUNT(*) as count FROM teams", self.conn).iloc[0]['count']
        print(f"\nTeams: {teams_count}")
        
        # Games by season
        games_by_season = pd.read_sql("""
            SELECT season, COUNT(*) as games
            FROM games
            GROUP BY season
            ORDER BY season
        """, self.conn)
        print(f"\nGames by season:")
        print(games_by_season.to_string(index=False))
        
        # Completed vs upcoming
        summary = pd.read_sql("""
            SELECT 
                COUNT(*) as total_games,
                SUM(CASE WHEN home_score IS NOT NULL THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN home_score IS NULL THEN 1 ELSE 0 END) as upcoming
            FROM games
        """, self.conn)
        print(f"\nGame status:")
        print(summary.to_string(index=False))
        
        # Date range
        date_range = pd.read_sql("""
            SELECT MIN(date) as earliest, MAX(date) as latest
            FROM games
        """, self.conn)
        print(f"\nDate range: {date_range.iloc[0]['earliest']} to {date_range.iloc[0]['latest']}")
        
        print("="*50 + "\n")

    def validate_data(self):
        # Data quality checks


        print("\nRunning data quality checks...")
        
        issues = []
        
        # Check 1: All teams in games exist in teams table
        orphaned_teams = pd.read_sql("""
            SELECT DISTINCT team_id
            FROM (
                SELECT home_team_id as team_id FROM games
                UNION
                SELECT away_team_id as team_id FROM games
            )
            WHERE team_id NOT IN (SELECT team_id FROM teams)
        """, self.conn)
        
        if len(orphaned_teams) > 0:
            issues.append(f"Found {len(orphaned_teams)} teams in games not in teams table: {orphaned_teams['team_id'].tolist()}")
        
        # Check 2: No negative scores
        negative_scores = pd.read_sql("""
            SELECT COUNT(*) as count
            FROM games
            WHERE home_score < 0 OR away_score < 0
        """, self.conn).iloc[0]['count']
        
        if negative_scores > 0:
            issues.append(f"Found {negative_scores} games with negative scores")
        
        # Check 3: No games where team plays itself
        self_games = pd.read_sql("""
            SELECT COUNT(*) as count
            FROM games
            WHERE home_team_id = away_team_id
        """, self.conn).iloc[0]['count']
        
        if self_games > 0:
            issues.append(f"Found {self_games} games where team plays itself")
        
        # Check 4: Reasonable score ranges (0-100)
        unreasonable_scores = pd.read_sql("""
            SELECT COUNT(*) as count
            FROM games
            WHERE home_score > 100 OR away_score > 100
        """, self.conn).iloc[0]['count']
        
        if unreasonable_scores > 0:
            issues.append(f"Found {unreasonable_scores} games with scores > 100")
        
        # Report results
        if issues:
            print("Data quality issues found:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print("All data quality checks passed!")
            return True
    
    def close(self):
        """Close database connection"""
        self.conn.close()
        print("Database connection closed")

def main():
    # load data

    # Initialize loader
    loader = NFLDataLoader('data/nfl_betting.db')
    
    try:
        loader.initialize_schema()

        # Load teams
        loader.load_teams()
        
        # Load games for 2019-2024 seasons
        seasons = [2019, 2020, 2021, 2022, 2023, 2024]
        loader.load_games(seasons)
        
        # Validate data quality
        loader.validate_data()
        
        # Print summary
        loader.get_data_summary()
        
        print("\nData loading complete!")
        
    except Exception as e:
        print(f"\nData loading failed: {e}")
        raise
    
    finally:
        loader.close()


if __name__ == "__main__":
    main()