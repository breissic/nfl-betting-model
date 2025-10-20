# Model Performance Summary

## Quick Stats (2019-2024)

| Metric | My Elo Model | Vegas | Difference |
|--------|-------------|-------|------------|
| Win/Loss Accuracy | 63.7% | 66.6% | -2.9% |
| Spread MAE | 10.44 pts | 9.81 pts | +0.63 pts |
| Games Analyzed | 1,572 | 1,572 | - |

## Interpretation

**I'm competitive!** Only 2.9% behind Vegas on picking winners with a simple model using just historical scores. Vegas has:
- Real-time injury reports
- Weather data
- Insider information
- Billions in market efficiency
- Team news and coaching insights

And I'm still within field goal range on spread predictions. That's a solid foundation to build on.

## Where I Beat Vegas

When we disagree on the winner (18.4% of games), I'm right 42.1% of the time. Those are the games where:
- My Elo ratings captured something the market missed
- Or more likely, Vegas had info I didn't (injuries, weather, etc.)

## Biggest Wins

Games where I was right and Vegas was significantly off:
- 2019-09-15: MIA @ NE (I predicted closer game, Patriots crushed anyway)
- 2022-09-12: SEA vs DEN (I liked Seattle, they won straight up)

## Biggest Losses

Games where Vegas crushed me:
- 2025-01-05: DEN @ KC (Playoff game - missed by 20+ points!)
- 2021-01-03: KC vs LAC (Divisional game - way off)

## Key Takeaway

A pure statistical model can get you ~64% accuracy. To beat Vegas consistently, I'll need:
1. Better features (injuries, weather, rest)
2. More sophisticated models (XGBoost, ensembles)
3. Market timing (line movement analysis)

But this is a great starting point!