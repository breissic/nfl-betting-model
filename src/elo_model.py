import pandas as pd
import numpy as np
import sqlite3 
from datetime import datetime

class EloModel:
    # NFL Team Elo rating system

    def __init__(self, k_factor=20, home_advantage=65, starting_elo=1500):
        # initialize elo model with hyperparameters

        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.starting_elo = starting_elo

        self.ratings = {}
        self.rating_history = []

    def initialize_teams(self, team_ids, season, start_date):
        for team in team_ids:
            self.ratings[team] = self.starting_elo

            self.rating_history.append({
                'team_id': team,
                'elo_rating': self.starting_elo,
                'season': season,
                'week': 0,
                'date': start_date
            })

    def calculate_expected_score(self, rating_diff):
        """
        1 / (1 + 10^(-rating_diff/400))

        returns a float between 0 and 1 representing win probability
        """
        return 1 / (1 + 10**(-rating_diff/400)) # TODO make this 400 into a calculated number based on ~25-30 per point spread
    
    def get_rating_diff(self, home_team, away_team):
        """
        rating difference with home team advantage factored in

        positive rating difference facors home team
        """
        return (self.ratings[home_team] + self.home_advantage) - self.ratings[away_team]
    
    def update_ratings_after_game(self, home_team, away_team, home_score, away_score, date, season, week):
        """
        updates elo for both teams after a game
        
        new rating = old_rating + k(actual-expected)
        """
        diff = self.get_rating_diff(home_team, away_team)
        xScore = self.calculate_expected_score(diff)

        if home_score > away_score:
            actual = 1
        elif home_score < away_score:
            actual = 0
        else:
            actual = 0.5

        change = self.k_factor * (actual - xScore)
        old_rating_home = self.ratings[home_team]
        new_rating_home = old_rating_home + change
        old_rating_away = self.ratings[away_team]
        new_rating_away = old_rating_away - change

        self.rating_history.append({
            'team_id': home_team,
            'elo_rating': new_rating_home,
            'date': date,
            'season': season,
            'week': week
        })

        self.rating_history.append({
            'team_id': away_team,
            'elo_rating': new_rating_away,
            'date': date,
            'season': season,
            'week': week
        })

        self.ratings[home_team] = new_rating_home
        self.ratings[away_team] = new_rating_away

    def run_simulation(self, db_path, seasons):
        """
        run elo sim 
        """
        # open database connection
        conn = sqlite3.connect(db_path)

        # initialize teams (team_ids from teams table) which tracks their current and historical elos
        teams_df = pd.read_sql('SELECT team_id FROM teams', conn)
        team_ids = teams_df['team_id'].tolist()

        first_game = pd.read_sql("""
                                 SELECT MIN(date) as start_date FROM games WHERE season IN ({})
                                 """.format(','.join(map(str, seasons))), conn)
        start_date = first_game.iloc[0]['start_date']

        self.initialize_teams(team_ids, seasons[0], start_date)

        # start running through the games table based on the provided seasons in ascending order, 
        # calling update ratings after game for each game and teams involved
        games_df = pd.read_sql("""
            SELECT * FROM games
            WHERE season IN ({}) 
            AND home_score IS NOT NULL  
            ORDER BY date, game_id
        """.format(','.join(map(str, seasons))), conn)

        debug_count = 0
        for idx, game in games_df.iterrows():
            self.update_ratings_after_game(game['home_team_id'], game['away_team_id'],
                                           game['home_score'], game['away_score'],
                                           game['date'], game['season'], game['week'])
            debug_count += 1
            if debug_count % 100 == 0:
                print(f"Processed {debug_count}/{len(games_df)} games...")
            
        # boom we have elo ratings for each nfl team
        conn.close()

    def save_ratings_to_db(self, db_path):
        """
        save calculated elo ratings to the elo_ratings table in db
        """
        conn = sqlite3.connect(db_path)

        ratings_df = pd.DataFrame(self.rating_history)

        conn.execute('DELETE FROM elo_ratings')

        ratings_df.to_sql('elo_ratings', conn, if_exists='append', index=False)

        conn.close()

        print(f"DEBUG: Saved {len(ratings_df)} rating records")