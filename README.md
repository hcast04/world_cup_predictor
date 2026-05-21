# World Cup 2026 Predictor

A data-driven project to predict the 2026 FIFA World Cup results, World Cup winner probabilities, and Golden Boot probabilities.

The project is designed as a stage-based forecasting engine. Before the tournament, it simulates the group stage and an approximate knockout path. During the tournament, actual results and knockout fixtures can be entered manually so predictions can be updated round by round.

## Goals

- Predict group-stage results and qualification probabilities.
- Estimate World Cup winner probabilities.
- Estimate Golden Boot and top scorer probabilities.
- Support manual updates once real match results and knockout fixtures are known.
- Use free and reproducible data sources where possible.

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