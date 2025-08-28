# algorithms/load_balancing_policy.py
from __future__ import annotations
import numpy as np
from typing import Tuple

class LoadBalancingPolicy:
    """로드 밸런싱 정책: 빔 제한을 고려한 균등 분배"""
    
    def __init__(self, num_sbs: int, beam_limits: Tuple[int, ...]):
        self.num_sbs = num_sbs
        self.beam_limits = beam_limits
        self.name = "LoadBalancing"
        self.is_trained = True
    
    def act(self, obs: np.ndarray) -> np.ndarray:
        """
        로드 밸런싱 방식으로 SBS 선택
        
        Args:
            obs: (U, S) SNR dB - 각 UE별 각 SBS의 SNR
        
        Returns:
            actions: (U,) int - 각 UE의 SBS 선택 (0=미서빙, 1~S=SBS)
        """
        U = obs.shape[0]
        actions = np.zeros(U, dtype=np.int64)
        
        # 각 SBS의 현재 할당된 UE 수
        sbs_usage = np.zeros(self.num_sbs)
        
        # SNR 순으로 UE 정렬 (높은 순)
        ue_snr_pairs = [(i, np.max(obs[i])) for i in range(U)]
        ue_snr_pairs.sort(key=lambda x: x[1], reverse=True)
        
        for ue_id, _ in ue_snr_pairs:
            # 가장 적게 사용된 SBS 찾기
            best_sbs = np.argmin(sbs_usage)
            
            if sbs_usage[best_sbs] < self.beam_limits[best_sbs]:
                # 해당 SBS에 할당 가능
                actions[ue_id] = best_sbs + 1
                sbs_usage[best_sbs] += 1
            else:
                # 모든 SBS가 가득 찬 경우
                actions[ue_id] = 0
        
        return actions
    
    def train(self):
        """정책 학습 (필요 없음)"""
        return self
