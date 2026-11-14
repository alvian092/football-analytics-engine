# Football Analytics Engine

A data-first pre-match football analytics project. The first milestone is a transparent, testable probability engine—not a tipster and not a profit guarantee.

## Initial scope

- Historical home/away attack and defence strengths with recency weighting and shrinkage.\n- Automatic pre-match lambda estimates using only matches dated before the fixture cutoff.
- Poisson scoreline matrix with optional Dixon–Coles low-score correction.
- Derived 1X2, BTTS, and goal-total probabilities.
- Fair odds, expected value, and Kelly calculations for simple binary outcomes.
- Automated unit tests and a Streamlit starter dashboard.

## Project status

**Milestone 1 — historical lambda estimator (initial).** The engine now includes a transparent, recency-weighted home/away strength estimator with shrinkage toward competition averages, strict pre-match date filtering, and minimum-history safeguards. It does not fetch live data, price Asian quarter-lines, or claim validated betting performance.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pytest
streamlit run app/dashboard.py
```

## Model use and validation

1. Treat estimated lambdas as model outputs, not facts. Record cutoff date, competition, sample sizes, and estimator settings.
2. Never train or evaluate using information that was unavailable at the prediction timestamp.
3. Keep chronological train/test splits; do not randomly shuffle time-series match data.
4. Evaluate probability quality with Brier score, log loss, and calibration—not accuracy alone.
5. Track bookmaker, market, odds timestamp, and closing price for every recorded forecast.
6. Use **NO BET** when data quality is weak, the model is uncalibrated, or estimated value is not positive after uncertainty and costs.

All outputs are estimates. This software does not guarantee profit and should initially be used in research/paper-trading mode.
