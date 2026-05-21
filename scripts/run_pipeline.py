import argparse
import subprocess
import sys


def run_command(command: list[str], description: str) -> None:
    """
    Run a command and stop the pipeline if it fails.
    """
    print(f"\n{description}")
    print("-" * len(description))
    print(" ".join(command))

    result = subprocess.run(command, check=False)

    if result.returncode != 0:
        raise RuntimeError(
            f"Pipeline failed during: {description}\n"
            f"Command: {' '.join(command)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the World Cup prediction pipeline.")
    parser.add_argument(
        "--stage",
        type=int,
        required=True,
        choices=[1, 2, 3],
        help="Prediction stage to run: 1, 2, or 3.",
    )
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=10_000,
        help="Number of Monte Carlo simulations for Stage 1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=21,
        help="Random seed.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip derived data generation and validation.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="elo_poisson",
        choices=["elo_poisson", "strength_baseline"],
        help="Match model to use.",
    )

    args = parser.parse_args()

    python = sys.executable

    if not args.skip_build:
        run_command(
            [python, "-m", "src.data.generate_teams_from_groups"],
            "Generating teams from groups",
        )

        run_command(
            [python, "-m", "src.data.generate_group_fixtures"],
            "Generating group fixtures",
        )

        run_command(
            [python, "-m", "src.data.fill_missing_elo"],
            "Creating model-ready Elo ratings",
        )
        run_command(
            [python, "-m", "src.data.build_historical_matches"],
            "Building model-ready historical matches",
        )
        run_command(
            [python, "-m", "src.data.build_match_features"],
            "Building recent-form match features",
        )
        run_command(
            [python, "scripts/train_recent_form_model.py"],
            "Training recent-form model",
        )
        run_command(
            [python, "-m", "src.data.build_team_strengths"],
            "Building team goal strengths",
        )
        run_command(
            [python, "scripts/validate_raw_data.py"],
            "Validating raw/model data",
        )

        run_command(
            [python, "scripts/audit_elo_sources.py"],
            "Auditing Elo sources",
        )

    stage_command = [
        python,
        "scripts/run_stage.py",
        "--stage",
        str(args.stage),
        "--seed",
        str(args.seed),
        "--model-type",
        args.model_type,
    ]

    if args.stage == 1:
        stage_command.extend(["--n-simulations", str(args.n_simulations)])

    run_command(
        stage_command,
        f"Running Stage {args.stage} predictions",
    )

    if args.stage == 1:
        run_command(
            [python, "scripts/make_stage1_plots.py"],
            "Creating Stage 1 plots",
        )

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()