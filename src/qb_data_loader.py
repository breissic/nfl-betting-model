import nfl_data_py as nfl
import pandas as pd
import sqlite3
from collections import Counter

class QBDataLoader:
    # load QB data for each game from nfl_data_py

    def __init__(self, db_path):
        self.db_path = db_path

    def add_qb_columns_to_games(self):
        # add home_qb and away_qb columns to games table

        conn = sqlite3.connect(self.db_path)

        cursor = conn.execute("PRAGMA table_info(games)")
        existing_columns = [row[1] for row in cursor.fetchall()]

        columns_to_add = [
            ('home_qb', 'TEXT'),
            ('away_qb', 'TEXT'),
            ('home_qb_epa', 'REAL'),
            ('away_qb_epa', 'REAL'),
            ('home_qb_pass_attempts', 'INTEGER'),
            ('away_qb_pass_attempts', 'INTEGER')
        ]

        for col_name, col_type in columns_to_add:
            if col_name not in existing_columns:
                conn.execute(f"ALTER TABLE games ADD COLUMN {col_name} {col_type}")
                print(f"Added {col_name} column")

        conn.commit()
        conn.close()

    def load_qb_data_for_seasons(self, seasons):
        # fetch play-by-play data and extract starting QBs and performance stats
        # this uses nfl_data_py to get detailed play data

        print(f"Loading QB data for seasons: {seasons}")
        print("This may take a few minutes...")

        try:
            # import play-by-play data for all seasons
            pbp_data = nfl.import_pbp_data(seasons, downcast=True)
            print(f"Loaded {len(pbp_data)} plays")

            # filter to plays with passer names (passing plays)
            passing_plays = pbp_data[pbp_data['passer_player_name'].notna()].copy()

            # check which EPA column exists
            if 'qb_epa' in passing_plays.columns:
                epa_col = 'qb_epa'
            elif 'epa' in passing_plays.columns:
                epa_col = 'epa'
            else:
                print("Warning: No EPA column found in play-by-play data")
                epa_col = None

            # group by season, week, team, QB to get performance stats
            agg_dict = {'pass_attempt': 'sum'}
            if epa_col:
                agg_dict[epa_col] = 'sum'

            qb_stats = passing_plays.groupby(['season', 'week', 'posteam', 'passer_player_name']).agg(agg_dict).reset_index()

            # rename columns
            col_rename = {
                'passer_player_name': 'qb_name',
                'posteam': 'team',
                'pass_attempt': 'pass_attempts'
            }
            if epa_col:
                col_rename[epa_col] = 'total_epa'

            qb_stats = qb_stats.rename(columns=col_rename)

            # if no EPA column, set to 0
            if 'total_epa' not in qb_stats.columns:
                qb_stats['total_epa'] = 0

            # for each game, select the QB with most pass attempts as the starter
            qb_by_game = qb_stats.loc[qb_stats.groupby(['season', 'week', 'team'])['pass_attempts'].idxmax()]

            print(f"Identified QBs for {len(qb_by_game)} team-games with performance stats")

            return qb_by_game

        except Exception as e:
            print(f"Error loading QB data: {e}")
            print("You may need to install nfl_data_py: pip install nfl_data_py")
            return None

    def update_games_with_qb_data(self, seasons):
        # load QB data and update games table

        # add columns if needed
        self.add_qb_columns_to_games()

        # load QB data
        qb_data = self.load_qb_data_for_seasons(seasons)

        if qb_data is None:
            print("Could not load QB data")
            return

        # load games from database
        conn = sqlite3.connect(self.db_path)
        games_df = pd.read_sql("""
            SELECT game_id, season, week, home_team_id, away_team_id
            FROM games
            WHERE season IN ({})
        """.format(','.join(map(str, seasons))), conn)

        print(f"Updating {len(games_df)} games with QB data...")

        updates_made = 0
        for idx, game in games_df.iterrows():
            # find home QB - match by season, week, and team
            home_qb_row = qb_data[
                (qb_data['season'] == game['season']) &
                (qb_data['week'] == game['week']) &
                (qb_data['team'] == game['home_team_id'])
            ]

            if len(home_qb_row) > 0:
                home_qb = home_qb_row['qb_name'].iloc[0]
                home_qb_epa = home_qb_row['total_epa'].iloc[0]
                home_qb_attempts = home_qb_row['pass_attempts'].iloc[0]
            else:
                home_qb = None
                home_qb_epa = None
                home_qb_attempts = None

            # find away QB - match by season, week, and team
            away_qb_row = qb_data[
                (qb_data['season'] == game['season']) &
                (qb_data['week'] == game['week']) &
                (qb_data['team'] == game['away_team_id'])
            ]

            if len(away_qb_row) > 0:
                away_qb = away_qb_row['qb_name'].iloc[0]
                away_qb_epa = away_qb_row['total_epa'].iloc[0]
                away_qb_attempts = away_qb_row['pass_attempts'].iloc[0]
            else:
                away_qb = None
                away_qb_epa = None
                away_qb_attempts = None

            # update database - convert to native Python types to avoid binary storage
            if home_qb or away_qb:
                # convert numpy/pandas types to native Python float/int
                home_epa_val = float(home_qb_epa) if home_qb_epa is not None and not pd.isna(home_qb_epa) else None
                away_epa_val = float(away_qb_epa) if away_qb_epa is not None and not pd.isna(away_qb_epa) else None
                home_att_val = int(home_qb_attempts) if home_qb_attempts is not None and not pd.isna(home_qb_attempts) else None
                away_att_val = int(away_qb_attempts) if away_qb_attempts is not None and not pd.isna(away_qb_attempts) else None

                conn.execute("""
                    UPDATE games
                    SET home_qb = ?, away_qb = ?,
                        home_qb_epa = ?, away_qb_epa = ?,
                        home_qb_pass_attempts = ?, away_qb_pass_attempts = ?
                    WHERE game_id = ?
                """, (home_qb, away_qb, home_epa_val, away_epa_val,
                      home_att_val, away_att_val, game['game_id']))
                updates_made += 1

            if (idx + 1) % 100 == 0: #type: ignore
                print(f"  Updated {idx + 1}/{len(games_df)} games...") #type: ignore

        conn.commit()
        conn.close()

        print(f"Updated {updates_made} games with QB data")

    def get_qb_summary(self, seasons):
        # print summary of QB data

        conn = sqlite3.connect(self.db_path)

        summary = pd.read_sql("""
            SELECT
                COUNT(*) as total_games,
                COUNT(home_qb) as games_with_home_qb,
                COUNT(away_qb) as games_with_away_qb,
                COUNT(CASE WHEN home_qb IS NOT NULL AND away_qb IS NOT NULL THEN 1 END) as games_with_both_qbs
            FROM games
            WHERE season IN ({})
            AND home_score IS NOT NULL
        """.format(','.join(map(str, seasons))), conn)

        print("\n" + "="*60)
        print("QB DATA SUMMARY")
        print("="*60)
        print(f"Total games: {summary['total_games'].iloc[0]}")
        print(f"Games with home QB: {summary['games_with_home_qb'].iloc[0]}")
        print(f"Games with away QB: {summary['games_with_away_qb'].iloc[0]}")
        print(f"Games with both QBs: {summary['games_with_both_qbs'].iloc[0]}")
        print("="*60)

        # show top QBs by games played
        top_qbs = pd.read_sql("""
            SELECT qb_name, COUNT(*) as games
            FROM (
                SELECT home_qb as qb_name FROM games WHERE home_qb IS NOT NULL
                UNION ALL
                SELECT away_qb as qb_name FROM games WHERE away_qb IS NOT NULL
            )
            GROUP BY qb_name
            ORDER BY games DESC
            LIMIT 15
        """, conn)

        print("\nTop 15 QBs by games:")
        print(top_qbs.to_string(index=False))

        conn.close()


def main():
    # load QB data into database

    db_path = 'data/nfl_betting.db'
    seasons = [2019, 2020, 2021, 2022, 2023, 2024]

    print("QB Data Loader")
    print("="*60)
    print(f"Database: {db_path}")
    print(f"Seasons: {seasons}")
    print("="*60)

    loader = QBDataLoader(db_path)
    loader.update_games_with_qb_data(seasons)
    loader.get_qb_summary(seasons)

    print("\nQB data loading complete!")


if __name__ == "__main__":
    main()
