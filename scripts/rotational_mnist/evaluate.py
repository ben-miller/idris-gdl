"""Entry point for evaluating rotational MNIST models."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.config import configure_logging
from lib.evaluation import main

if __name__ == "__main__":
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Evaluate rotational MNIST models"
    )
    parser.add_argument(
        "models",
        nargs="*",
        default=["baseline", "augmented", "e2_simple"],
        help="Models to evaluate: baseline, augmented, e2_simple (default: all three)",
    )
    args = parser.parse_args()

    main(models=args.models)
