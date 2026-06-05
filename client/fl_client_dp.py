# client/fl_client_dp.py  (extends FraudClient)
import os
from opacus import PrivacyEngine
from client.fl_client import FraudClient

class FraudClientDP(FraudClient):
    def __init__(self, model, train_loader, val_loader,
                 max_grad_norm=1.0, noise_mult=0.8, **kwargs):
        super().__init__(model, train_loader, val_loader, **kwargs)
        if os.environ.get("USE_DP", "false").lower() == "true":
            engine = PrivacyEngine()
            self.model, self.optimizer, self.train_loader = engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_loader,
                noise_multiplier=noise_mult,
                max_grad_norm=max_grad_norm,
            )
            self._dp_engine = engine
            print(f"[DP] Privacy engine active: noise={noise_mult}, clip={max_grad_norm}")
        else:
            self._dp_engine = None
            print("[DP] Differential privacy disabled")

    def get_epsilon(self, delta=1e-5):
        if self._dp_engine:
            return self._dp_engine.get_epsilon(delta)
        return float("inf")
