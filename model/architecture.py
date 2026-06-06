"""Re-exports FraudMLP from client.model.

The model is defined in client/model.py so the FL client can import it
without depending on the model/ package. This module re-exports it so
evaluation and serving code can use a clean model.architecture import path.
"""
from client.model import FraudMLP, INPUT_DIM

__all__ = ["FraudMLP", "INPUT_DIM"]