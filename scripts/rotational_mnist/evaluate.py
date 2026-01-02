"""Entry point for evaluating rotational MNIST models."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.config import configure_logging
from lib.evaluation import main

if __name__ == "__main__":
    configure_logging()
    main()
