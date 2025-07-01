import typing
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger("workingmem")
logger.setLevel(logging.INFO)


def print_gpu_mem(obj: typing.Any = None):
    """
    Print the GPU memory usage.
    """
    import torch

    if torch.cuda.is_available():
        logger.info(
            f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB, "
            f"reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB"
        )
        if obj is not None:
            logger.info(
                f"GPU memory allocated for {obj.__class__.__name__}: "
                f"{torch.cuda.memory_allocated(obj) / 1024**3:.2f} GB"
            )
    else:
        logger.info("No GPU available; no memory report.")
