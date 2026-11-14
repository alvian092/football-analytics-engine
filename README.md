# Football Analytics Engine

A data-first pre-match football analytics project. The first milestone is a transparent, testable probability engine—not a tipster and not a profit guarantee.

## Initial scope

- Historical home/away attack and defence strengths with recency weighting and shrinkage.
- Automatic pre-match lambda estimates using only matches dated before the fixture cutoff.
- Poisson scoreline matrix with optional Dixon–Coles low-score correction.
- One-call pre-match prediction pipeline joining historical lambda estimation and score probabilities.
- Chronological walk-forward backtesting with strict same-day/future-data exclusion and comparison against a smoothed historical 1X2 baseline.
- Derived 1X2, BTTS, and goal-total probabilities.
- Fair odds, expected value, and Kelly calculations for simple binary outcomes.
- Automated unit tests and a Streamlit starter dashboard.

## Project status

**Milestone 2 — initial walk-forward backtest.** The engine can now evaluate chronological 1X2 forecasts, skip fixtures without enough prior history, and compare Brier Score / Log Loss against a smoothed historical-frequency baseline. This is an evaluation framework, not evidence of predictive edge or profitability.

**Milestone 1 — historical lambda estimator (initial).** The engine now includes a transparent, recency-weighted home/away strength estimator with shrinkage toward competition averages, strict pre-match date filtering, and minimum-history safeguards. It does not fetch live data, price Asian quarter-lines, or claim validated betting performance.

## Run a walk-forward backtest

Prepare a CSV using the columns in `data/raw/matches_template.csv` (the xG columns are optional), then run:

```bash
python run_backtest.py --input data/raw/matches.csv --competition "Premier League"
```

The command writes `data/processed/backtest/metrics.json` and `predictions.csv`. Omit `--competition` to evaluate each competition separately within its own history. The default minimum history is 30 earlier league matches and 3 relevant home/away matches per team; early fixtures are skipped rather than guessed. Use a sufficiently large, correctly licensed historical dataset before interpreting the metrics. Synthetic test fixtures only verify software behavior.

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


## Import manually downloaded Football-Data.co.uk CSVs

Football-Data.co.uk provides season-by-season CSV files for private personal
betting analysis and academic research. Its published fair-use policy permits
manual downloads or at most light automated downloads (no more than twice per
week); this project therefore **does not scrape or download files**. Download
the CSVs manually from https://football-data.co.uk/downloadm.php and keep the
original files in a folder, for example:

```text
data/raw/football-data/
  1011/
    E0.csv
    E1.csv
  1112/
    E0.csv
    E1.csv
```

The importer also accepts filenames such as `E0_1011.csv`. It converts
Football-Data columns (`Date`, `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`)
to the engine schema, drops unplayed fixtures with both scores blank, retains
the original statistics/odds columns, and records the source file and source.
Files must contain their season code in the filename or parent folder.

```bash
python import_football_data.py --input-dir data/raw/football-data --output data/raw/matches.csv
python run_backtest.py --input data/raw/matches.csv
```

League code mapping currently covers the main European divisions listed in
Football-Data's archive. Other division codes are retained as
`Football-Data <code>` and should be reviewed before analysis. Historical
coverage and odds/statistics columns vary by competition and season. Do not
treat missing odds or statistics as zero. Keep original CSVs as source records
and check the provider's current terms before each data collection.

## Google Colab workflow (Android-friendly)

Open the private-data loader notebook in Google Colab:

[Open Football Analytics Engine data loader in Colab](https://colab.research.google.com/github/alvian092/football-analytics-engine/blob/main/notebooks/Football_Analytics_Engine_Private_Data_Loader.ipynb)

The notebook mounts the user's Google Drive, downloads one season archive per run from the official Football-Data.co.uk archive, stores raw files in the user's private Drive, normalizes all collected seasons, and runs the walk-forward backtest. It enforces a maximum of two automated archive downloads in any rolling seven-day period. Start with season code `1011`; subsequent codes are `1112`, `1213`, ..., `2627`.

Do not publish or upload the provider's raw CSV/ZIP files to this public repository. Keep the Drive folder private and use the data only in accordance with the provider's current personal-use/fair-use terms. The notebook does not promise predictive edge or profitability.
