# algorithms/max_snr.py
from __future__ import annotations
import numpy as np

class MaxSNRPolicy:
    """학습 불필요한 규칙 정책: UE별 SNR 최대 SBS 선택"""
    def train(self, *args, **kwargs): return self
    def act(self, obs_us: np.ndarray) -> np.ndarray:
        return 1 + np.argmax(obs_us, axis=1)
    def save(self, path: str): pass
    def load(self, path: str): return self
