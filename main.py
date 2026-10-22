"""
main.py
=======
Runs the full pipeline end to end:

    extract -> transform -> load -> model -> policy_analysis
             -> yield_analysis -> visualize

Usage:
    python3 main.py                          # run every stage
    python3 main.py --stages model visualize # run a subset, in pipeline order
    python3 main.py --list                   # show available stages
"""

import argparse
import logging
import time

from src import (
    extract,
    load,
    model,
    policy_analysis,
    transform,
    visualize,
    yield_analysis,
)

logger = logging.getLogger(__name__)

STAGES = {
    "extract": extract.run,
    "transform": transform.run,
    "load": load.run,
    "model": model.run,
    "policy_analysis": policy_analysis.run,
    "yield_analysis": yield_analysis.run,
    "visualize": visualize.run,
}


def run_pipeline(stages=None):
    selected = [s for s in STAGES if stages is None or s in stages]
    t0 = time.time()

    for i, name in enumerate(selected, start=1):
        logger.info(f"=== [{i}/{len(selected)}] {name.upper().replace('_', ' ')} ===")
        stage_t0 = time.time()
        STAGES[name]()
        logger.info(f"{name} finished in {time.time() - stage_t0:.1f}s")

    logger.info(f"Pipeline complete in {time.time() - t0:.1f}s")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Berlin real estate ETL + ML pipeline")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=list(STAGES),
        metavar="STAGE",
        help="subset of stages to run (always executed in pipeline order)",
    )
    parser.add_argument("--list", action="store_true", help="list stages and exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)-22s %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.list:
        print("\n".join(STAGES))
    else:
        run_pipeline(args.stages)
