# algorithms/epsilon_greedy_policy.py
from __future__ import annotations
import numpy as np

class EpsilonGreedyPolicy:
    """ε-greedy 정책: MaxSNR 기반이지만 일정 확률로 랜덤 선택"""
    
    def __init__(self, num_sbs: int, epsilon: float = 0.3):
        self.num_sbs = num_sbs
        self.epsilon = epsilon
        self.name = "EpsilonGreedy"
        self.is_trained = True
    
    def act(self, obs: np.ndarray) -> np.ndarray:
        """
        ε-greedy 방식으로 SBS 선택
        
        Args:
            obs: (U, S) SNR dB - 각 UE별 각 SBS의 SNR
        
        Returns:
            actions: (U,) int - 각 UE의 SBS 선택 (0=미서빙, 1~S=SBS)
        """
        U = obs.shape[0]
        actions = np.zeros(U, dtype=np.int64)
        
        for ue_id in range(U):
            if np.random.random() < self.epsilon:
                # ε 확률로 랜덤 선택
                actions[ue_id] = np.random.randint(0, self.num_sbs + 1)
            else:
                # (1-ε) 확률로 MaxSNR 선택
                actions[ue_id] = 1 + np.argmax(obs[ue_id])
        
        return actions
    
    def train(self):
        """정책 학습 (필요 없음)"""
        return self
