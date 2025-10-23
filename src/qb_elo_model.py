import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime

class QBEloModel:
    """
    Track individual QB Elo ratings alongside team ratings using adjustment method.
    Uses performance-based updates (EPA) instead of win/loss.
    
    Key difference from composite approach:
    - Adjustment: team_elo + (qb_elo - 1500) * qb_weight
    - Composite: team_weight * team_elo + qb_weight * qb_elo
    
    Adjustment preserves team signal while adding QB impact.
    """

    def __init__(self, starting_qb_elo=1500, qb_k_factor=25, qb_weight=1.0,
                 backup_qb_elo=1400):
        """
        Initialize QB Elo model with adjustment method.

        Parameters:
        -----------
        starting_qb_elo : float
            Initial Elo rating for starter QBs (default 1500)
        qb_k_factor : int
            How quickly QB ratings change (default 25, up from 12 for faster convergence)
        qb_weight : float
            Multiplier for QB adjustment (default 1.0)
            Higher values = more QB impact
            qb_weight=1.0 means 100 QB Elo difference = 100 Elo adjustment = 4 points spread
        backup_qb_elo : float
            Initial Elo rating for backup QBs (default 1400)
            Used when QB first appears mid-season (after week 4)
        """

        self.starting_qb_elo = starting_qb_elo
        self.backup_qb_elo = backup_qb_elo
        self.qb_k_factor = qb_k_factor
        self.qb_weight = qb_weight

        self.qb_ratings = {}
        self.qb_rating_history = []
        self.qb_first_appearance = {}  # Track when QBs first appear

    def get_qb_rating(self, qb_name, season=None, week=None):
        """
        Get current rating for a QB, initialize if new.

        Parameters:
        -----------
        qb_name : str
            Name of the quarterback
        season : int, optional
            Current season (for initialization logic)
        week : int, optional
            Current week (for initialization logic)

        Returns:
        --------
        float
            Current Elo rating for the QB
        """
        if qb_name not in self.qb_ratings:
            # Initialize new QB
            self._initialize_qb(qb_name, season, week)

        return self.qb_ratings[qb_name]

    def _initialize_qb(self, qb_name, season, week):
        """
        Initialize a new QB with appropriate starting Elo.

        Logic:
        - If QB first appears in weeks 1-4: Likely a starter → starting_qb_elo (1500)
        - If QB first appears after week 4: Likely a backup → backup_qb_elo (1400)
        - This prevents backups from being overrated when they first appear

        Parameters:
        -----------
        qb_name : str
            Name of the quarterback
        season : int or None
            Current season
        week : int or None
            Current week
        """
        # Record first appearance
        if season is not None and week is not None:
            self.qb_first_appearance[qb_name] = {'season': season, 'week': week}

        # Determine starting Elo based on when QB first appears
        if week is not None and week > 4:
            # Mid-season appearance = likely backup
            initial_elo = self.backup_qb_elo
        else:
            # Early season or unknown = assume starter
            initial_elo = self.starting_qb_elo

        self.qb_ratings[qb_name] = initial_elo

    def get_qb_adjustment(self, qb_name, season=None, week=None):
        """
        Calculate QB Elo adjustment to add to team rating.
        Uses adjustment method: (qb_elo - 1500) * qb_weight

        Parameters:
        -----------
        qb_name : str
            Name of the quarterback
        season : int, optional
            Current season (for initialization if QB is new)
        week : int, optional
            Current week (for initialization if QB is new)

        Returns:
        --------
        float
            Elo adjustment to add to team rating
            Positive if QB above average, negative if below
        """
        if qb_name is None or qb_name == '' or pd.isna(qb_name):
            # No QB data, return 0 adjustment
            return 0.0

        qb_elo = self.get_qb_rating(qb_name, season, week)
        adjustment = (qb_elo - self.starting_qb_elo) * self.qb_weight

        return adjustment

    def calculate_expected_epa_per_attempt(self, qb_elo):
        """
        Calculate expected EPA per pass attempt based on QB Elo rating.
        
        Formula: (qb_elo - 1500) / 1333
        Maps: 1700 Elo -> +0.15 EPA/att, 1500 Elo -> 0.0 EPA/att, 1300 Elo -> -0.15 EPA/att
        
        Parameters:
        -----------
        qb_elo : float
            Current QB Elo rating
            
        Returns:
        --------
        float
            Expected EPA per pass attempt
        """
        return (qb_elo - self.starting_qb_elo) / 1333.0

    def update_qb_rating_from_epa(self, qb_name, actual_epa, pass_attempts, date, season, week):
        """
        Update QB Elo rating based on EPA performance (not wins/losses).

        Parameters:
        -----------
        qb_name : str
            Name of the quarterback
        actual_epa : float
            Total EPA generated by QB in the game
        pass_attempts : int
            Number of pass attempts
        date : str
            Game date
        season : int
            NFL season year
        week : int
            Week number
        """
        if qb_name is None or qb_name == '' or pd.isna(qb_name):
            return

        # Convert to numeric types
        try:
            pass_attempts = float(pass_attempts) if pass_attempts is not None else None
            actual_epa = float(actual_epa) if actual_epa is not None else None
        except (ValueError, TypeError):
            # Silently skip if conversion fails
            return

        if pass_attempts is None or pass_attempts == 0 or pd.isna(pass_attempts):
            return

        if actual_epa is None or pd.isna(actual_epa):
            return

        current_rating = self.get_qb_rating(qb_name, season, week)

        # Calculate expected EPA based on current rating
        expected_epa_per_att = self.calculate_expected_epa_per_attempt(current_rating)
        expected_total_epa = expected_epa_per_att * pass_attempts

        # Calculate actual EPA per attempt
        actual_epa_per_att = actual_epa / pass_attempts

        # Performance differential (normalized by attempts)
        epa_diff = actual_epa - expected_total_epa

        # Scale K-factor by pass attempts (more attempts = more reliable signal)
        # Cap at 50 attempts to avoid huge swings
        attempt_factor = min(pass_attempts / 35.0, 1.5)
        effective_k = self.qb_k_factor * attempt_factor

        # Calculate rating change
        # Scale epa_diff to be similar magnitude to win/loss (roughly 0 to 1 range)
        # Typical game: -10 to +10 total EPA, so divide by 15 to get roughly -0.67 to +0.67
        normalized_performance = np.tanh(epa_diff / 15.0)
        change = effective_k * normalized_performance

        # Update rating
        new_rating = current_rating + change
        self.qb_ratings[qb_name] = new_rating

        # Record history
        self.qb_rating_history.append({
            'qb_name': qb_name,
            'qb_elo': new_rating,
            'date': date,
            'season': season,
            'week': week,
            'change': change,
            'actual_epa': actual_epa,
            'expected_epa': expected_total_epa,
            'pass_attempts': pass_attempts
        })

    def apply_season_reversion(self, reversion_factor=0.5):
        """
        Apply regression to mean for QBs at season end.
        QBs regress more than teams due to higher variance.
        
        Parameters:
        -----------
        reversion_factor : float
            How much to regress toward mean (0.0 = full regression, 1.0 = no regression)
            Default 0.5 = regress 50% of deviation
        """
        mean_rating = self.starting_qb_elo

        for qb in self.qb_ratings:
            deviation = self.qb_ratings[qb] - mean_rating
            self.qb_ratings[qb] = mean_rating + (deviation * reversion_factor)

        print(f"QB season reversion applied with factor {reversion_factor}")

    def get_rating_summary(self):
        """
        Get summary of all QB ratings sorted by rating.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with qb_name and qb_elo columns, sorted by rating descending
        """
        if not self.qb_ratings:
            return pd.DataFrame()

        summary = pd.DataFrame([
            {'qb_name': qb, 'qb_elo': elo}
            for qb, elo in self.qb_ratings.items()
        ]).sort_values('qb_elo', ascending=False)

        return summary

    def save_qb_ratings_to_db(self, db_path):
        """
        Save QB rating history to database.
        
        Parameters:
        -----------
        db_path : str
            Path to SQLite database
        """
        if not self.qb_rating_history:
            print("No QB rating history to save")
            return

        conn = sqlite3.connect(db_path)

        # Drop and recreate qb_elo_ratings table to ensure schema is correct
        conn.execute('DROP TABLE IF EXISTS qb_elo_ratings')
        conn.execute("""
            CREATE TABLE qb_elo_ratings (
                qb_name TEXT NOT NULL,
                qb_elo REAL NOT NULL,
                date TEXT NOT NULL,
                season INTEGER NOT NULL,
                week INTEGER NOT NULL,
                change REAL,
                actual_epa REAL,
                expected_epa REAL,
                pass_attempts INTEGER,
                PRIMARY KEY(qb_name, season, week)
            )
        """)

        # Save history
        ratings_df = pd.DataFrame(self.qb_rating_history)
        ratings_df.to_sql('qb_elo_ratings', conn, if_exists='append', index=False)

        conn.close()
        print(f"Saved {len(ratings_df)} QB rating records")