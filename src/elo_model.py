import pandas as pd
import numpy as np
import sqlite3 
from datetime import datetime

class EloModel:
    """
    NFL Team Elo rating system with contextual features and optional QB Elo.
    Supports adjustment method for QB ratings where QB impact is added to team rating.
    """

    def __init__(self, k_factor=20, home_advantage=65, starting_elo=1500,
                 rest_impact=1.0, divisional_penalty=0, qb_model=None,
                 qb_differential_threshold=50):
        """
        Initialize elo model with hyperparameters.
        
        Parameters:
        -----------
        k_factor : int
            How quickly team ratings change (default 20)
        home_advantage : int
            Home field advantage in Elo points (default 65)
        starting_elo : int
            Initial Elo for all teams (default 1500)
        rest_impact : float
            Elo adjustment per day of rest advantage (default 1.0)
        divisional_penalty : int
            Reduction in effective elo difference for divisional games (default 0)
        qb_model : QBEloModel or None
            Optional QBEloModel instance for QB-adjusted predictions
        qb_differential_threshold : int
            Only apply QB Elo when abs(qb_diff) >= this value (default 50)
            0 = always apply, higher values = only for big QB gaps
        """

        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.starting_elo = starting_elo
        self.rest_impact = rest_impact
        self.divisional_penalty = divisional_penalty
        self.qb_model = qb_model
        self.qb_differential_threshold = qb_differential_threshold

        self.ratings = {}
        self.rating_history = []

    def initialize_teams(self, team_ids, season, start_date):
        """
        Initialize team ratings to starting Elo.
        
        Parameters:
        -----------
        team_ids : list
            List of team IDs
        season : int
            Starting season
        start_date : str
            Date of first game
        """
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
        Calculate win probability from Elo rating difference.
        
        Formula: 1 / (1 + 10^(-rating_diff/400))
        
        Parameters:
        -----------
        rating_diff : float
            Elo rating difference (home - away)
            
        Returns:
        --------
        float
            Win probability between 0 and 1
        """
        return 1 / (1 + 10**(-rating_diff/400))
    
    def get_rating_diff(self, home_team, away_team, rest_advantage=0, is_divisional=False,
                        home_qb=None, away_qb=None, season=None, week=None):
        """
        Calculate rating difference with home advantage and contextual adjustments.
        Uses QB adjustment method when QB model is available.

        Parameters:
        -----------
        home_team : str
            Home team ID
        away_team : str
            Away team ID
        rest_advantage : int
            Days of rest advantage for home team (default 0)
        is_divisional : bool
            Whether this is a divisional game (default False)
        home_qb : str or None
            Home team QB name
        away_qb : str or None
            Away team QB name
        season : int or None
            Season year (for QB initialization)
        week : int or None
            Week number (for QB initialization)

        Returns:
        --------
        float
            Rating difference with positive value favoring home team
        """
        # Get base team ratings
        home_team_elo = self.ratings[home_team]
        away_team_elo = self.ratings[away_team]

        # Apply QB adjustment if QB model is available and differential threshold is met
        home_adjusted = home_team_elo
        away_adjusted = away_team_elo

        if self.qb_model is not None and home_qb is not None and away_qb is not None:
            # Get QB adjustments (pass season/week for smart backup initialization)
            home_qb_adjustment = self.qb_model.get_qb_adjustment(home_qb, season, week)
            away_qb_adjustment = self.qb_model.get_qb_adjustment(away_qb, season, week)

            # Check if QB differential meets threshold
            qb_differential = abs(home_qb_adjustment - away_qb_adjustment)

            if qb_differential >= self.qb_differential_threshold:
                # Apply QB adjustments using adjustment method
                home_adjusted = home_team_elo + home_qb_adjustment
                away_adjusted = away_team_elo + away_qb_adjustment

        # Base rating difference with home advantage
        base_diff = (home_adjusted + self.home_advantage) - away_adjusted

        # Adjust for rest days
        rest_adjustment = rest_advantage * self.rest_impact

        # Adjust for divisional games - reduce the gap between teams
        if is_divisional:
            if base_diff > 0:
                base_diff = max(0, base_diff - self.divisional_penalty)
            else:
                base_diff = min(0, base_diff + self.divisional_penalty)

        return base_diff + rest_adjustment
    
    def update_ratings_after_game(self, home_team, away_team, home_score, away_score, date, season, week,
                                   home_qb=None, away_qb=None):
        """
        Update Elo ratings for both teams after a game.
        QB ratings are NOT updated here - they are updated separately in backtest()
        based on EPA performance, not game outcome.
        
        Parameters:
        -----------
        home_team : str
            Home team ID
        away_team : str
            Away team ID
        home_score : int
            Home team score
        away_score : int
            Away team score
        date : str
            Game date
        season : int
            Season year
        week : int
            Week number
        home_qb : str or None
            Home team QB name
        away_qb : str or None
            Away team QB name
        """
        home_elo_before = self.ratings[home_team]
        away_elo_before = self.ratings[away_team]

        diff = self.get_rating_diff(home_team, away_team, home_qb=home_qb, away_qb=away_qb,
                                    season=season, week=week)
        expected = self.calculate_expected_score(diff)

        if home_score > away_score:
            actual = 1
            winner_elo = home_elo_before
            loser_elo = away_elo_before
        elif home_score < away_score:
            actual = 0
            winner_elo = away_elo_before
            loser_elo = home_elo_before
        else:
            actual = 0.5
            winner_elo = home_elo_before
            loser_elo = away_elo_before

        score_diff = abs(home_score - away_score)
        elo_diff = winner_elo - loser_elo
        mov_multiplier = np.log(score_diff + 1) * (2.2 / ((elo_diff * 0.001) + 2.2))
        effective_k = self.k_factor * mov_multiplier

        change = effective_k * (actual - expected)
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
        Run Elo simulation on historical games.
        
        Parameters:
        -----------
        db_path : str
            Path to database
        seasons : list
            List of seasons to simulate
        """
        # Open database connection
        conn = sqlite3.connect(db_path)

        # Initialize teams
        teams_df = pd.read_sql('SELECT team_id FROM teams', conn)
        team_ids = teams_df['team_id'].tolist()

        first_game = pd.read_sql("""
                                 SELECT MIN(date) as start_date FROM games WHERE season IN ({})
                                 """.format(','.join(map(str, seasons))), conn)
        start_date = first_game.iloc[0]['start_date']

        self.initialize_teams(team_ids, seasons[0], start_date)

        # Process games in chronological order
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
            
        conn.close()

    def save_ratings_to_db(self, db_path):
        """
        Save calculated Elo ratings to database.
        
        Parameters:
        -----------
        db_path : str
            Path to database
        """
        conn = sqlite3.connect(db_path)

        ratings_df = pd.DataFrame(self.rating_history)

        conn.execute('DELETE FROM elo_ratings')

        ratings_df.to_sql('elo_ratings', conn, if_exists='append', index=False)

        conn.close()

        print(f"Saved {len(ratings_df)} rating records")

    def predict_game(self, home_team, away_team, rest_advantage=0, is_divisional=False,
                     home_qb=None, away_qb=None, season=None, week=None):
        """
        Predict game outcome using current Elo ratings.

        Parameters:
        -----------
        home_team : str
            Home team ID
        away_team : str
            Away team ID
        rest_advantage : int
            Days of rest advantage for home team (default 0)
        is_divisional : bool
            Whether this is a divisional game (default False)
        home_qb : str or None
            Home team QB name
        away_qb : str or None
            Away team QB name
        season : int or None
            Season year (for QB initialization)
        week : int or None
            Week number (for QB initialization)

        Returns:
        --------
        tuple
            (win_probability, predicted_spread)
        """

        diff = self.get_rating_diff(home_team, away_team, rest_advantage, is_divisional,
                                    home_qb, away_qb, season, week)

        xWinProb = self.calculate_expected_score(diff)

        predicted_spread = diff / 25  # 25 ELO = 1 point in the spread

        return (xWinProb, predicted_spread)

    def backtest(self, db_path, seasons):
        """
        Run backtest to evaluate predictions.
        
        Parameters:
        -----------
        db_path : str
            Path to database
        seasons : list
            List of seasons to backtest
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with predicted vs actual outcomes
        """
        # Open database connection
        conn = sqlite3.connect(db_path)

        # Initialize teams
        teams_df = pd.read_sql('SELECT team_id FROM teams', conn)
        team_ids = teams_df['team_id'].tolist()

        first_game = pd.read_sql("""
                                 SELECT MIN(date) as start_date FROM games WHERE season IN ({})
                                 """.format(','.join(map(str, seasons))), conn)
        start_date = first_game.iloc[0]['start_date']

        self.initialize_teams(team_ids, seasons[0], start_date)

        games_df = pd.read_sql("""
            SELECT * FROM games
            WHERE season IN ({}) 
            AND home_score IS NOT NULL  
            ORDER BY date, game_id
        """.format(','.join(map(str, seasons))), conn)

        predictions = []
        current_season = None

        for i, (_, game) in enumerate(games_df.iterrows()):
            if current_season is not None and game['season'] != current_season:
                self.apply_season_reversion()
                if self.qb_model is not None:
                    self.qb_model.apply_season_reversion()

            current_season = game['season']

            # Get contextual features
            rest_advantage = game.get('rest_advantage', 0) or 0
            is_divisional = bool(game.get('is_divisional', 0))
            home_qb = game.get('home_qb')
            away_qb = game.get('away_qb')
            season = game['season']
            week = game['week']

            win_prob, pred_spread = self.predict_game(
                game['home_team_id'],
                game['away_team_id'],
                rest_advantage=rest_advantage,
                is_divisional=is_divisional,
                home_qb=home_qb,
                away_qb=away_qb,
                season=season,
                week=week
            )

            pred_dict = {
                'game_id': game['game_id'],
                'date': game['date'],
                'season': game['season'],
                'week': game['week'],
                'home_team': game['home_team_id'],
                'away_team': game['away_team_id'],
                'home_elo_before': self.ratings[game['home_team_id']],
                'away_elo_before': self.ratings[game['away_team_id']],
                'rest_advantage': rest_advantage,
                'is_divisional': is_divisional,
                'predicted_home_win_prob': win_prob,
                'predicted_spread': pred_spread,
                'actual_home_score': game['home_score'],
                'actual_away_score': game['away_score'],
                'actual_spread': game['home_score'] - game['away_score'],
                'home_won': game['home_score'] > game['away_score'],
                'predicted_correctly': (win_prob > 0.5 and game['home_score'] > game['away_score']) or
                                    (win_prob < 0.5 and game['home_score'] < game['away_score'])
            }

            # Add QB info if available
            if self.qb_model is not None:
                pred_dict['home_qb'] = home_qb
                pred_dict['away_qb'] = away_qb
                if home_qb:
                    pred_dict['home_qb_elo'] = self.qb_model.get_qb_rating(home_qb, season, week)
                if away_qb:
                    pred_dict['away_qb_elo'] = self.qb_model.get_qb_rating(away_qb, season, week)

            predictions.append(pred_dict)

            # Update team ratings
            self.update_ratings_after_game(game['home_team_id'], game['away_team_id'],
                                           game['home_score'], game['away_score'],
                                           game['date'], game['season'], game['week'],
                                           home_qb, away_qb)

            # Update QB ratings based on EPA performance (if QB model is enabled)
            if self.qb_model is not None:
                home_qb_epa = game.get('home_qb_epa')
                away_qb_epa = game.get('away_qb_epa')
                home_qb_attempts = game.get('home_qb_pass_attempts')
                away_qb_attempts = game.get('away_qb_pass_attempts')

                self.qb_model.update_qb_rating_from_epa(
                    home_qb, home_qb_epa, home_qb_attempts,
                    game['date'], game['season'], game['week']
                )
                self.qb_model.update_qb_rating_from_epa(
                    away_qb, away_qb_epa, away_qb_attempts,
                    game['date'], game['season'], game['week']
                )
            
        conn.close()
        return pd.DataFrame(predictions)
    
    def apply_season_reversion(self, reversion_factor=1.0):
        """
        Apply season-end regression toward mean.
        
        Parameters:
        -----------
        reversion_factor : float
            How much to preserve (1.0 = no reversion, 0.0 = full reversion to mean)
        """
        mean_rating = 1500

        for team in self.ratings:
            deviation = self.ratings[team] - mean_rating
            self.ratings[team] = mean_rating + (deviation * reversion_factor)

        print(f"Season reversion applied with factor {reversion_factor}")