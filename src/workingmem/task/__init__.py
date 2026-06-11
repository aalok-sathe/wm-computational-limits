"""
include: ./README.md
"""

from workingmem.task.SIR.SIR import SIRDataset, SIRConfig
from workingmem.task.interface import _T_dataset_or_collection_of_datasets

__all__ = [
    "SIRDataset",
    "SIRConfig",
    "_T_dataset_or_collection_of_datasets",
]
