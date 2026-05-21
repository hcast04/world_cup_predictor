# World Cup 2026 Predictor

A data-driven project to predict the 2026 FIFA World Cup group stage, World Cup winner probabilities, and Golden Boot probabilities.

The project is designed as a **stage-based forecasting engine**. Before the tournament, it simulates the group stage and an approximate knockout path. During the tournament, actual results and manually entered knockout fixtures can be added so predictions can be updated round by round.

## Goals

- Predict group-stage results and qualification probabilities.
- Estimate World Cup winner probabilities.
- Estimate Golden Boot and top scorer probabilities.
- Support manual updates once real match results and knockout fixtures are known.
- Build a reproducible data pipeline using free data sources where possible.
- Backtest baseline models on historical international matches.
- Train reusable models from processed historical match features.

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
```

For a faster test:

```bash
python scripts/run_stage.py --stage 1 --n-simulations 1000
```

You can also run Stage 1 through the full pipeline:

```bash
python scripts/run_pipeline.py --stage 1 --n-simulations 1000
```

### Stage 2: First knockout stage

After the group stage is complete, manually fill:

```text
data/raw/manual_knockout_fixtures.csv
```

with the Round of 32 fixtures.

Then run:

```bash
python scripts/run_stage.py --stage 2
```

or:

```bash
python scripts/run_pipeline.py --stage 2
```

### Stage 3: Later knockout stages

As later fixtures become known, update:

```text
data/raw/manual_knockout_fixtures.csv
```

with round-of-16, quarter-final, semi-final, third-place, and final fixtures.

Then run:

```bash
python scripts/run_stage.py --stage 3
```

or:

```bash
python scripts/run_pipeline.py --stage 3
```

## Full pipeline

The full pipeline regenerates derived data, validates inputs, and runs the selected prediction stage.

Run full Stage 1 pipeline:

```bash
python scripts/run_pipeline.py --stage 1
```

Run faster Stage 1 test:

```bash
python scripts/run_pipeline.py --stage 1 --n-simulations 1000
```

Run Stage 1 with a specific model:

```bash
python scripts/run_pipeline.py --stage 1 --model-type elo_poisson
python scripts/run_pipeline.py --stage 1 --model-type strength_baseline
```

Skip data regeneration and only rerun predictions:

```bash
python scripts/run_pipeline.py --stage 1 --skip-build
```

## Match models

The simulator currently supports multiple match-model variants.

### `elo_poisson`

Uses model-ready Elo ratings to convert team-strength differences into expected goals.

```bash
python scripts/run_pipeline.py --stage 1 --model-type elo_poisson
```

### `strength_baseline`

Uses historical attack and defence strengths estimated from international results.

```bash
python scripts/run_pipeline.py --stage 1 --model-type strength_baseline
```

### Recent-form model

The project also includes a reusable recent-form logistic-regression model trained on historical match features. It is currently used for backtesting and model development, and can later be wired into the simulator as another model option.

## Entering known match results

During the tournament, actual group-stage results can be entered in:

```text
data/raw/manual_match_results.csv
```

Example:

```csv
match_id,stage,group,date,team_a,team_b,goals_a,goals_b,status
1,group,A,2026-06-11,Mexico,South Africa,2,0,played
```

When Stage 1 is rerun, known group-stage results are used directly and only unplayed matches are simulated.

```bash
python scripts/run_stage.py --stage 1
```

## Entering knockout fixtures

Once knockout fixtures are known, enter them in:

```text
data/raw/manual_knockout_fixtures.csv
```

Example:

```csv
match_id,stage,date,team_a,team_b,venue,city,country,status
R32_01,round_of_32,2026-06-28,Spain,Canada,TBD,TBD,TBD,scheduled
R32_02,round_of_32,2026-06-28,France,South Africa,TBD,TBD,TBD,scheduled
```

Then run:

```bash
python scripts/run_stage.py --stage 2
```

For later knockout rounds, use stages such as:

```text
round_of_16
quarter_final
semi_final
third_place
final
```

and run:

```bash
python scripts/run_stage.py --stage 3
```

## Current workflow

```text
Before tournament:
simulate group stage, winner, and Golden Boot

During group stage:
use played results plus simulate remaining group matches

After group stage:
manually enter Round of 32 fixtures and predict first knockout round

During knockouts:
manually update fixtures and rerun Stage 3 predictions
```

## Data acquisition

The project supports free historical international football data from the `martj42/international_results` repository.

Download historical match results, goal scorers, and shootouts:

```bash
python -m src.data.collect_historical_results
```

Validate downloaded historical data:

```bash
python scripts/validate_historical_data.py
```

Build model-ready historical matches:

```bash
python -m src.data.build_historical_matches
```

Summarize historical matches:

```bash
python scripts/summarize_historical_data.py
```

The historical data is used for future model training, calibration, recent-form features, and backtesting.

## Recent-form features

Historical results can be transformed into model-ready recent-form features:

```bash
python -m src.data.build_match_features
```

Summarize the resulting feature table:

```bash
python scripts/summarize_match_features.py
```

Output:

```text
data/processed/match_features.csv
```

The feature table includes recent-form variables such as:

```text
team_a_goals_for_last_5
team_a_goals_against_last_5
team_a_points_last_10
team_b_goals_for_last_10
team_b_goal_diff_last_10
points_diff_last_10
goal_diff_diff_last_10
win_rate_diff_last_10
```

## Backtesting

The project includes a first historical baseline model trained on international matches since 2010.

Build historical matches first:

```bash
python -m src.data.build_historical_matches
```

Run the strength baseline backtest:

```bash
python scripts/backtest_strength_baseline.py
```

Outputs:

```text
outputs/tables/strength_baseline_backtest_predictions.csv
outputs/tables/strength_baseline_backtest_metrics.csv
outputs/tables/team_goal_strengths.csv
```

Current strength baseline:

- fits simple team attack and defence indices from historical goals
- trains on matches before 2022
- tests on matches from 2022 onward
- evaluates with Brier score, log loss, accuracy, and Poisson score log loss

## Recent-form model backtest

Run a recent-form logistic-regression backtest:

```bash
python scripts/backtest_recent_form_model.py
```

Compare it against naive baselines:

```bash
python scripts/compare_backtest_baselines.py
```

Outputs:

```text
outputs/tables/recent_form_model_backtest_predictions.csv
outputs/tables/recent_form_model_backtest_metrics.csv
outputs/tables/backtest_model_comparison.csv
```

The recent-form model currently uses features such as recent points, recent goal difference, recent goals for/against, win-rate differences, neutral venue status, and tournament type.

## Training the recent-form model

The project includes a reusable multinomial logistic-regression model trained on recent-form features from historical international matches.

Build features:

```bash
python -m src.data.build_match_features
```

Train the reusable model artifact:

```bash
python scripts/train_recent_form_model.py
```

Outputs:

```text
models/recent_form_model.joblib
models/recent_form_model_features.json
outputs/tables/recent_form_model_training_metrics.csv
```

The trained artifact is not required to be committed because it can be regenerated from the raw and processed data.

## Comparing model variants

Stage 1 can currently be run with two match models:

```bash
python scripts/run_pipeline.py --stage 1 --model-type elo_poisson
python scripts/run_pipeline.py --stage 1 --model-type strength_baseline
```

Compare winner probabilities between both models:

```bash
python scripts/compare_stage1_models.py
python scripts/make_model_comparison_report.py
```

Outputs:

```text
outputs/tables/stage1_model_comparison_winner_probs.csv
outputs/tables/stage1_model_comparison_report.md
```

## Planned methodology

1. Collect free international football data.
2. Build a baseline Elo-based match model.
3. Simulate group-stage outcomes using Monte Carlo simulation.
4. Estimate winner probabilities using an approximate knockout simulation.
5. Add player-level Golden Boot prediction.
6. Build recent-form features from historical international results.
7. Backtest baseline and recent-form models.
8. Train reusable recent-form models.
9. Replace placeholder ratings and player inputs with real data.
10. Add squad-level and player-level features once final squads are released.

## Project structure

```text
data/
    raw/           Original downloaded or manually entered data
    interim/       Intermediate cleaned files
    processed/     Final modeling datasets
    external/      External reference files

src/
    data/          Data collection, loading, and feature building
    models/        Match, player, and recent-form models
    simulation/    Group, knockout, and tournament simulation
    evaluation/    Backtesting and metrics
    visualization/ Reports and plots
    utils/         Shared helper functions

scripts/           Command-line scripts
notebooks/         Exploratory notebooks
outputs/           Predictions, figures, and tables
models/            Regenerable trained model artifacts
app/               Optional Streamlit dashboard
tests/             Unit tests
```

## Main input files

```text
data/raw/groups_2026.csv
data/raw/teams_2026.csv
data/raw/elo_ratings_seed.csv
data/processed/elo_ratings_model.csv
data/raw/players_2026_seed.csv
data/raw/manual_match_results.csv
data/raw/manual_knockout_fixtures.csv
data/raw/historical_results.csv
data/raw/historical_goalscorers.csv
data/raw/historical_shootouts.csv
```

## Main processed files

```text
data/processed/historical_matches.csv
data/processed/match_features.csv
data/processed/team_goal_strengths_model.csv
data/processed/elo_ratings_model.csv
```

## Main output files

```text
outputs/predictions/group_stage_qualification_probabilities.csv
outputs/predictions/stage1_winner_predictions.csv
outputs/predictions/stage1_golden_boot_predictions.csv
outputs/predictions/stage1_summary.md
outputs/predictions/stage2_knockout_predictions.csv
outputs/predictions/stage3_knockout_predictions.csv
outputs/tables/recent_form_model_backtest_metrics.csv
outputs/tables/backtest_model_comparison.csv
outputs/figures/stage1_winner_probabilities.png
outputs/figures/stage1_golden_boot_probabilities.png
outputs/figures/group_stage_qualification_probabilities.png
```

## Setup

Create and activate a virtual environment.

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the local package in editable mode:

```bash
pip install -e .
```

## Useful commands

Generate teams from groups:

```bash
python -m src.data.generate_teams_from_groups
```

Generate group fixtures from groups:

```bash
python -m src.data.generate_group_fixtures
```

Create model-ready Elo ratings:

```bash
python -m src.data.fill_missing_elo
```

Build historical matches:

```bash
python -m src.data.build_historical_matches
```

Build recent-form match features:

```bash
python -m src.data.build_match_features
```

Build team goal strengths:

```bash
python -m src.data.build_team_strengths
```

Train recent-form model:

```bash
python scripts/train_recent_form_model.py
```

Validate data:

```bash
python scripts/validate_raw_data.py
```

Audit rating sources:

```bash
python scripts/audit_elo_sources.py
```

Run Stage 1:

```bash
python scripts/run_stage.py --stage 1
```

Run Stage 2:

```bash
python scripts/run_stage.py --stage 2
```

Run Stage 3:

```bash
python scripts/run_stage.py --stage 3
```

Run full Stage 1 pipeline:

```bash
python scripts/run_pipeline.py --stage 1 --n-simulations 1000
```

## Important data caveat

At the current development stage, many team ratings and player inputs may be temporary placeholders. The pipeline is structurally complete, but predictions should only be interpreted once real ratings, squad information, and player data have been added.

Historical data is useful for backtesting, calibration, and recent-form features, but final 2026 predictions should combine:

```text
current team ratings
recent international form
current squads
player-level scoring data
tournament simulation
manual updates during the tournament
```
