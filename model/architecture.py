"""⚠️ DEPRECATED: Re-export module superseded by direct imports

DO NOT USE. This file is a legacy re-export layer.

Instead of:
    from model.architecture import FraudMLP, INPUT_DIM

Use:
    from src.model.fraud_mlp import FraudMLP, INPUT_DIM

This file will be removed in a future version.
Can be safely deleted once all imports are migrated.
"""
from src.model.fraud_mlp import FraudMLP, INPUT_DIM

__all__ = ["FraudMLP", "INPUT_DIM"]