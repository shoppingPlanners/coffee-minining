#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NB_DIR = os.path.join(ROOT, "notebooks")
NOTEBOOKS = [
    "01_data_understanding.ipynb",
    "02_data_preprocessing.ipynb",
    "03_exploratory_analysis.ipynb",
    "04_feature_engineering.ipynb",
    "05_model_building.ipynb",
    "09_final_pipeline.ipynb",
    "06_model_evaluation.ipynb",
    "07_model_explainability.ipynb",
    "08_recommendation_engine.ipynb",
]

def run_notebooks(nbs):
    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--inplace",
        "--execute",
    ] + nbs
    print("Executing:", " ".join(nbs))
    subprocess.check_call(cmd, cwd=NB_DIR)


def main():
    parser = argparse.ArgumentParser(description="Run project pipeline notebooks")
    parser.add_argument("--run-all", action="store_true", help="Run all notebooks in order")
    parser.add_argument("--only", nargs="*", help="Run only these notebooks (filenames)")
    args = parser.parse_args()

    if args.run_all:
        run_notebooks(NOTEBOOKS)
        return

    if args.only:
        unknown = [nb for nb in args.only if not os.path.exists(os.path.join(NB_DIR, nb))]
        if unknown:
            print("Notebook(s) not found:", ", ".join(unknown))
            sys.exit(1)
        run_notebooks(args.only)
        return

    parser.print_help()


if __name__ == "__main__":
    main()


