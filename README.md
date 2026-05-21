# World Cup 2026 Predictor

A data-driven project to predict the 2026 FIFA World Cup results and top scorer probabilities.

The project uses free data sources, team-strength modeling, Monte Carlo tournament simulation, and later player-level scoring models once final squads are available.

## Goals

- Predict group-stage outcomes.
- Estimate qualification probabilities for each round.
- Simulate the full World Cup bracket.
- Estimate winner probabilities.
- Estimate Golden Boot and top scorer probabilities.

## Planned methodology

1. Collect free international football data.
2. Build a baseline Elo-based match model.
3. Simulate the tournament using Monte Carlo methods.
4. Backtest the model on previous tournaments.
5. Add squad-level features once final squads are released.
6. Add player-level top scorer predictions.

## Project structure

```text
data/
    raw/          Original downloaded data
    interim/      Intermediate cleaned files
    processed/    Final modeling datasets
    external/     Manually added external reference files

src/
    data/         Data collection and cleaning
    models/       Match and player models
    simulation/   Tournament simulation
    evaluation/   Backtesting and metrics
    visualization/ Plotting utilities

notebooks/        Exploratory notebooks
outputs/          Predictions, figures, and tables
app/              Optional Streamlit dashboard
tests/            Unit tests

## Prediction stages

The project is designed around three prediction stages.

### Stage 1: Before or during the group stage

Predict:

- group-stage qualification probabilities
- World Cup winner probabilities
- Golden Boot probabilities

Run:

```bash
python scripts/run_stage.py --stage 1