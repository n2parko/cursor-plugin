"""Citation-prediction autoresearch loop.

Predicts a paper's future citation count (a proxy for impact) from
publication-time signals, and iteratively improves the prediction strategy in a
karpathy/autoresearch-style loop.
"""
from .paper import Paper
from .predictor import Strategy, Prediction, predict_paper
from .llm import LLMClient

__all__ = ["Paper", "Strategy", "Prediction", "predict_paper", "LLMClient"]
__version__ = "0.1.0"
