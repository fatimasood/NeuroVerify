"""
Trustify: Fuzzy Logic-Based Hybrid Framework for Detecting Fake News
Package initialization
"""

__version__ = "1.0.0"
__author__ = "Fatima Masood(Reproduce paper by Bharati et al., 2026)"
__email__ = "thejuniordeve@gmail.com"

from src.models.trustify import TrustifyModel
from src.preprocessing.preprocess import PreprocessPipeline
from src.training.trainer import Trainer
from src.evaluation.evaluate import Evaluator

__all__ = [
    'TrustifyModel',
    'PreprocessPipeline',
    'Trainer',
    'Evaluator',
]