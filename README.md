# NFL Betting Model - Elo-Based Prediction System

A machine learning project that predicts NFL game outcomes using an Elo rating system with margin-of-victory adjustments. Built from scratch as a learning project to understand sports betting models and eventually compete with Vegas lines.

## Project Goals

- Build a predictive model for NFL games using historical data
- Compare model performance against Vegas betting lines
- Learn statistical modeling, backtesting, and sports analytics
- Foundation for future ML improvements (XGBoost, ensemble models)

## Current Performance

**Model Accuracy (2019-2024):**
- **Win/Loss Predictions:** 63.7% (vs Vegas 66.6%)
- **Spread Error (MAE):** 10.44 points (vs Vegas 9.81 points)
- **Agreement with Vegas:** 81.6% of games

**By Season:**
```
Season | My Model | Vegas
2019   | 59.6%    | 64.8%
2020   | 65.7%    | 67.7%
2021   | 60.3%    | 63.0%
2022   | 64.6%    | 67.9%
2023   | 64.1%    | 65.2%
2024   | 67.5%    | 70.8%
```

## Architecture

### Database Schema
```sql
teams          - 36 NFL teams with divisions/conferences
games          - 1,675 games (2019-2024) with scores and Vegas lines
elo_ratings    - Historical Elo ratings for all teams
predictions    - Model predictions (future use)
```

### Elo Model Features

**Core Algorithm:**
- Standard Elo with 400-point scaling
- K-factor: 20 (how quickly ratings change)
- Home field advantage: 65 Elo points (~2.6 point spread)

**Advanced Features:**
- **Margin of Victory (MOV) multiplier:** Blowouts matter more than close games
  - Formula: `ln(score_diff + 1) * (2.2 / ((winner_elo - loser_elo) * 0.001 + 2.2))`
  - Autocorrelation adjustment: Expected blowouts → smaller rating change
- **Season reversion:** Optional (currently disabled for best performance)

## Getting Started

### Prerequisites
```bash
Python 3.11+
pip install -r requirements.txt
```

### Setup

1. **Clone and install:**
```bash
git clone <your-repo>
cd nfl-betting-model
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Load data:**
```bash
# Populate database with NFL games
python src/data_loader.py

# Import Vegas lines (requires Kaggle dataset)
# Download from: https://www.kaggle.com/datasets/tobycrabtree/nfl-scores-and-betting-data
# Place in data/ folder as spreadspoke_scores.csv
python src/import_vegas_lines.py
```

3. **Run backtest:**
```bash
python src/test_backtest.py
```

## Project Structure
```
nfl-betting-model/
├── data/
│   ├── nfl_betting.db          # SQLite database
│   └── raw/
|       └── spreadspoke_scores.csv  # Vegas lines (not included)
├── src/
│   ├── data_loader.py          # Load NFL game data
│   ├── elo_model.py            # Core Elo rating system
│   ├── import_vegas_lines.py   # Import Vegas betting lines
│   └── test_backtest.py        # Backtesting & evaluation
├── notebooks/                   # Analysis notebooks (future)
├── requirements.txt
├── .gitignore
└── README.md
```

## Usage

### Basic Prediction
```python
from elo_model import EloModel

model = EloModel(k_factor=20, home_advantage=65, starting_elo=1500)
model.run_simulation('data/nfl_betting.db', seasons=[2019, 2020, 2021, 2022, 2023, 2024])

# Predict a game
win_prob, predicted_spread = model.predict_game('KC', 'BUF')
print(f"Chiefs win probability: {win_prob:.1%}")
print(f"Predicted spread: {predicted_spread:.1f}")
```

### Backtesting
```python
# Run full backtest with Vegas comparison
model = EloModel()
results = model.backtest('data/nfl_betting.db', seasons=[2019, 2020, 2021, 2022, 2023, 2024])

# Results saved to data/backtest_with_vegas.csv
```

## Key Findings

1. **Simple Elo is surprisingly effective** - Within 3% of Vegas on win/loss predictions
2. **MOV adjustment matters** - Improved accuracy by 2.4% over basic Elo
3. **Season reversion hurts** - NFL teams are more persistent year-to-year than expected
4. **Model improves over time** - 59.6% (2019) → 67.5% (2024) as ratings stabilize
5. **Agreement with Vegas is high** - 81.6% pick the same winner

## What I Learned

- Elo rating systems and probabilistic prediction
- Backtesting methodology and avoiding lookahead bias
- Database design for time-series sports data
- Sports betting market efficiency (Vegas is really good!)
- Python data science stack (pandas, numpy, sqlite3)

## Future Improvements

**Phase 1: Feature Engineering**
- [ ] Player injuries and roster data
- [ ] Weather conditions for outdoor games
- [ ] Rest days and travel distance
- [ ] Divisional/rivalry game indicators
- [ ] Quarterback-specific adjustments

**Phase 2: Advanced Models**
- [ ] XGBoost ensemble using Elo + features
- [ ] Neural network for non-linear patterns
- [ ] Separate models for spreads vs totals
- [ ] Meta-model combining multiple approaches

**Phase 3: Deployment**
- [ ] Weekly predictions for upcoming games
- [ ] Web interface for friends to compete
- [ ] Real-time line tracking and alerts
- [ ] Portfolio/bankroll simulation

## Data Sources

- **Game Data:** [nfl_data_py](https://pypi.org/project/nfl-data-py/)
- **Vegas Lines:** [Kaggle - NFL Scores and Betting Data](https://www.kaggle.com/datasets/tobycrabtree/nfl-scores-and-betting-data)

## Acknowledgments

Built as a learning project combining interests in:
- Sports analytics
- Machine learning
- Statistical modeling
- Data engineering

Inspired by FiveThirtyEight's NFL Elo ratings and the challenge of beating the betting market.

## License

Personal project for educational purposes.

---
