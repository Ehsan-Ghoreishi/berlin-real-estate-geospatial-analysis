"""
main.py
=======
Runs the full pipeline end to end:

    extract -> transform -> load -> model -> policy_analysis -> visualize

Usage:
    python3 main.py
"""

import logging
import time

from src import extract, load, model, policy_analysis, transform, visualize

logger = logging.getLogger(__name__)


def run_pipeline():
    t0 = time.time()

    logger.info("=== [1/6] EXTRACT ===")
    extract.run()

    logger.info("=== [2/6] TRANSFORM ===")
    transform.run()

    logger.info("=== [3/6] LOAD ===")
    load.run()

    logger.info("=== [4/6] MODEL ===")
    model.run()

    logger.info("=== [5/6] POLICY ANALYSIS ===")
    policy_analysis.run()

    logger.info("=== [6/6] VISUALIZE ===")
    visualize.run()

    logger.info(f"Pipeline complete in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)-22s %(message)s",
        datefmt="%H:%M:%S",
    )
    run_pipeline()
