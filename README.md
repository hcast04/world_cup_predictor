# World Cup 2026 Predictor

A data-driven project to predict the 2026 FIFA World Cup group stage, World Cup winner probabilities, and Golden Boot probabilities.

The project is designed as a stage-based forecasting engine. Before the tournament, it simulates the group stage and an approximate knockout path. During the tournament, actual match results and knockout fixtures can be entered manually so predictions can be updated round by round.

## Goals

- Predict group-stage outcomes and qualification probabilities.
- Estimate World Cup winner probabilities.
- Estimate Golden Boot and top scorer probabilities.
- Support manual updates once real match results and knockout fixtures are known.
- Use free and reproducible data sources where possible.
- Build a project that can later be improved with real squads, player data, and backtesting.

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

You can also run the full pipeline, including data generation and validation:

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

Or run the pipeline:

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

Or run the pipeline:

```bash
python scripts/run_pipeline.py --stage 3
```

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

Then run:

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

## Planned methodology

1. Collect free international football data.
2. Build a baseline Elo-based match model.
3. Simulate group-stage outcomes using Monte Carlo simulation.
4. Estimate winner probabilities using an approximate knockout simulation.
5. Add player-level Golden Boot prediction.
6. Replace placeholder ratings and player inputs with real data.
7. Backtest the model on previous international tournaments.
8. Add squad-level and player-level features once final squads are released.

## Project structure

```text
data/
    raw/            Original downloaded or manually entered data
    interim/        Intermediate cleaned files
    processed/      Final modeling datasets
    external/       External reference files

src/
    data/           Data collection and loading
    models/         Match and player models
    simulation/     Group, knockout, and tournament simulation
    evaluation/     Backtesting and metrics
    visualization/  Reports and plotting utilities
    utils/          Shared helper functions

scripts/            Command-line scripts
notebooks/          Exploratory notebooks
outputs/            Predictions, figures, and tables
app/                Optional Streamlit dashboard
tests/              Unit tests
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

## Main output files

```text
outputs/predictions/group_stage_qualification_probabilities.csv
outputs/predictions/stage1_winner_predictions.csv
outputs/predictions/stage1_golden_boot_predictions.csv
outputs/predictions/stage1_summary.md
outputs/predictions/stage2_knockout_predictions.csv
outputs/predictions/stage2_knockout_summary.md
outputs/predictions/stage3_knockout_predictions.csv
outputs/predictions/stage3_knockout_summary.md
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

## Data acquisition

The project currently supports free historical international football data from the `martj42/international_results` repository.

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

The historical data is used for future model training, calibration, and backtesting. Current Stage 1 predictions still use the baseline Elo-Poisson model until training code is added.

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

Validate data:

```bash
python scripts/validate_raw_data.py
```

Audit rating sources:

```bash
python scripts/audit_elo_sources.py
```

Run full Stage 1 pipeline:

```bash
python scripts/run_pipeline.py --stage 1
```

Run faster Stage 1 test:

```bash
python scripts/run_pipeline.py --stage 1 --n-simulations 1000
```

Run Stage 2 pipeline:

```bash
python scripts/run_pipeline.py --stage 2
```

Run Stage 3 pipeline:

```bash
python scripts/run_pipeline.py --stage 3
```

Skip data regeneration and only rerun predictions:

```bash
python scripts/run_pipeline.py --stage 1 --skip-build
```

Create Stage 1 plots from existing predictions:

```bash
python scripts/make_stage1_plots.py
```

## Modeling caveats

At the current development stage, many team ratings and player inputs may be temporary placeholders. The pipeline is structurally complete, but predictions should only be interpreted once real ratings, squad information, and player data have been added.

The current Stage 1 winner prediction uses an approximate generic knockout bracket. Once real knockout fixtures are known, manually enter them in `data/raw/manual_knockout_fixtures.csv` and use Stage 2 or Stage 3.

## Roadmap

Short-term next steps:

1. Replace temporary Elo defaults with real team ratings.
2. Add FIFA ranking points as an alternative team-strength input.
3. Use historical results for model calibration and backtesting.
4. Add real player inputs for Golden Boot candidates.
5. Improve the winner model with a more accurate bracket structure.
6. Add a Streamlit dashboard for interactive exploration.
