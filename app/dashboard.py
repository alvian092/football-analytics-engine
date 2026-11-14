"""Starter Streamlit dashboard for the pre-match probability engine."""
import streamlit as st

from engine.poisson import match_probabilities, score_matrix, total_goals_probabilities
from engine.value import expected_value, fair_odds

st.set_page_config(page_title="Football Analytics Engine", page_icon="⚽", layout="wide")
st.title("⚽ Football Analytics Engine")
st.caption("Pre-match research dashboard · estimates, not guarantees")

with st.sidebar:
    st.header("Match model inputs")
    home = st.text_input("Home team", "Home")
    away = st.text_input("Away team", "Away")
    lambda_home = st.number_input("Home expected goals (λ)", min_value=0.0, value=1.50, step=0.05)
    lambda_away = st.number_input("Away expected goals (λ)", min_value=0.0, value=1.10, step=0.05)
    rho = st.number_input("Dixon–Coles rho", min_value=-0.20, max_value=0.20, value=0.0, step=0.01)
    max_goals = st.slider("Maximum goals per team", 6, 15, 10)
    total_line = st.selectbox("Goal total line", [1.5, 2.5, 3.5, 4.5], index=1)
    odds = st.number_input("Decimal odds to evaluate (optional)", min_value=1.01, value=2.00, step=0.01)
    market_probability = st.number_input("Your estimated selection probability", min_value=0.01, max_value=0.99, value=0.55, step=0.01)

matrix = score_matrix(lambda_home, lambda_away, max_goals=max_goals, rho=rho)
probs = match_probabilities(matrix)
totals = total_goals_probabilities(matrix, total_line)
left, middle, right = st.columns(3)
left.metric(f"{home} win", f"{probs['home_win']:.1%}")
middle.metric("Draw", f"{probs['draw']:.1%}")
right.metric(f"{away} win", f"{probs['away_win']:.1%}")

st.subheader("Other market probabilities")
c1, c2, c3 = st.columns(3)
c1.metric("BTTS — Yes", f"{probs['btts_yes']:.1%}")
c2.metric(f"Over {total_line}", f"{totals['over']:.1%}")
c3.metric(f"Under {total_line}", f"{totals['under']:.1%}")

st.subheader("Price check")
ev = expected_value(market_probability, odds)
st.write(f"Fair odds: **{fair_odds(market_probability):.3f}**")
st.write(f"Expected value: **{ev:+.2%}** per unit stake")
st.warning("The probability entered in the price check is your own estimate. Do not treat the sample default as a model recommendation.")

st.subheader("Most likely scorelines")
scores = []
for h in range(matrix.shape[0]):
    for a in range(matrix.shape[1]):
        scores.append((h, a, float(matrix[h, a])))
for h, a, p in sorted(scores, key=lambda item: item[2], reverse=True)[:8]:
    st.write(f"{home} {h}–{a} {away}: **{p:.2%}**")

st.info("This starter version accepts lambdas manually. Historical data ingestion, lambda estimation, Asian handicap settlement, and walk-forward validation are planned next.")
