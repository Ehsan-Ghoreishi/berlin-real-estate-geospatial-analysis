"""
main.py
=======
Runs the full ETL + ML + visualization pipeline end to end:

    extract -> transform -> load -> model -> visualize

Usage:
    python3 main.py
"""

import time
from src import extract, transform, load, model, visualize


def run_pipeline():
    t0 = time.time()

    print("\n=== [1/5] EXTRACT ===")
    extract.run()

    print("\n=== [2/5] TRANSFORM ===")
    transform.run()

    print("\n=== [3/5] LOAD ===")
    load.run()

    print("\n=== [4/5] MODEL ===")
    model.run()

    print("\n=== [5/5] VISUALIZE ===")
    visualize.run()

    print(f"\nPipeline complete in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    run_pipeline()
