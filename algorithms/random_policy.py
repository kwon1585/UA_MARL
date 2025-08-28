# algorithms/random_policy.py
from __future__ import annotations
import numpy as np

class RandomPolicy:
    """랜덤 정책: 탐험 증가를 위한 정책"""
    
    def __init__(self, num_sbs: int):
        self.num_sbs = num_sbs
        self.name = "Random"
        self.is_trained = True
    
    def act(self, obs: np.ndarray) -> np.ndarray:
        """
        랜덤하게 SBS 선택
        
        Args:
            obs: (U, S) SNR dB - 각 UE별 각 SBS의 SNR
        
        Returns:
            actions: (U,) int - 각 UE의 SBS 선택 (0=미서빙, 1~S=SBS)
        """
        U = obs.shape[0]
        return np.random.randint(0, self.num_sbs + 1, size=U)
    
    def train(self):
        """정책 학습 (필요 없음)"""
        return self
