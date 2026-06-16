#!/usr/bin/env python3
"""
run as: `python -m workingmem [-h]`
"""

import logging
import os

from workingmem.cli import entrypoint

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
)
_logger = logging.getLogger("workingmem")
_LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
_logger.setLevel(_LOGLEVEL)


if __name__ == "__main__":
    entrypoint()
