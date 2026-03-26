"""
Predição de Sobreviventes do Titanic

Projeto de Machine Learning para prever a sobrevivência dos passageiros do Titanic.
"""

__version__ = "2.0.0"
__author__ = "Ron Bragaglia"
__email__ = "ronbragaglia@gmail.com"

from .data_loader import DataLoader
from .preprocessor import DataPreprocessor
from .model import TitanicModel
from .evaluator import ModelEvaluator
from .predictor import TitanicPredictor

__all__ = [
    "DataLoader",
    "DataPreprocessor",
    "TitanicModel",
    "ModelEvaluator",
    "TitanicPredictor",
]
