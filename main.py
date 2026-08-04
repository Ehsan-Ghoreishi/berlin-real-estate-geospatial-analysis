"""
main.py
=======
Runs the full pipeline end to end:

    extract -> transform -> load -> model -> policy_analysis -> visualize

Usage:
    python3 main.py
"""

import time
from src import extract, transform, load, model, policy_analysis, visualize


def run_pipeline():
    t0 = time.time()

    print("\n=== [1/6] EXTRACT ===")
    extract.run()

    print("\n=== [2/6] TRANSFORM ===")
    transform.run()

    print("\n=== [3/6] LOAD ===")
    load.run()

    print("\n=== [4/6] MODEL ===")
    model.run()

    print("\n=== [5/6] POLICY ANALYSIS ===")
    policy_analysis.run()

    print("\n=== [6/6] VISUALIZE ===")
    visualize.run()

    print(f"\nPipeline complete in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    run_pipeline()
