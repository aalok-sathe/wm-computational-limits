#!/usr/bin/env python3
"""
run as: `python -m workingmem [-h]`
"""

import logging
import os

from workingmem import entrypoint

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger("workingmem")
LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logger.setLevel(LOGLEVEL)


if __name__ == "__main__":
    entrypoint()
