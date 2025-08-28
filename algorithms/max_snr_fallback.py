# algorithms/max_snr_fallback.py
from __future__ import annotations
import numpy as np
from typing import Optional

class MaxSNRWithFallbackPolicy:
    """
    MaxSNR with Fallback: 최고 SNR SBS가 거부되면 대안 SBS 선택
    빔 제한, 간섭 등을 고려한 더 현실적인 정책
    """
    
    def __init__(self, beam_limits=None):
        self.beam_limits = beam_limits or [2, 3, 3]  # 기본값
        self.name = "MaxSNR-WithFallback"
        self.is_trained = True
    
    def act(self, obs_us: np.ndarray, admitted: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fallback이 있는 MaxSNR 정책
        
        Args:
            obs_us: (U, S) SNR dB - 각 UE별 각 SBS의 SNR
            admitted: (U, S) bool - 각 UE별 각 SBS의 서빙 허용 여부 (선택사항)
        
        Returns:
            actions: (U,) int - 각 UE의 SBS 선택 (0=미서빙, 1~S=SBS)
        """
        U, S = obs_us.shape
        actions = np.zeros(U, dtype=np.int64)
        
        for ue_id in range(U):
            # 1차 선택: 최고 SNR SBS
            best_sbs = 1 + np.argmax(obs_us[ue_id])
            actions[ue_id] = best_sbs
        
        # Fallback 적용 (admitted 정보가 있는 경우)
        if admitted is not None:
            actions = self._apply_fallback(obs_us, actions, admitted)
        
        return actions
    
    def act_with_fallback(self, obs_us: np.ndarray) -> np.ndarray:
        """
        개선된 데이터 수집기를 위한 통합 인터페이스
        Beam limit을 고려한 실제 Fallback 로직 구현
        """
        U, S = obs_us.shape
        actions = np.zeros(U, dtype=np.int64)
        
        # 1차: Max-SNR 선택
        for ue_id in range(U):
            best_sbs = 1 + np.argmax(obs_us[ue_id])
            actions[ue_id] = best_sbs
        
        # 2차: Beam limit 제약 확인 및 Fallback
        actions = self._apply_beam_limit_fallback(obs_us, actions)
        
        return actions
    
    def _apply_fallback(self, obs_us: np.ndarray, actions: np.ndarray, admitted: np.ndarray) -> np.ndarray:
        """
        거부된 UE들에 대해 대안 SBS 선택
        
        Args:
            obs_us: (U, S) SNR dB
            actions: (U,) 현재 선택된 SBS
            admitted: (U, S) 서빙 허용 여부
        
        Returns:
            updated_actions: (U,) Fallback 적용된 SBS 선택
        """
        U, S = obs_us.shape
        updated_actions = actions.copy()
        
        for ue_id in range(U):
            # 현재 선택된 SBS에서 서빙받지 못한 경우
            current_sbs = actions[ue_id] - 1  # 0-based index
            if current_sbs >= 0 and admitted[ue_id, current_sbs] == 0:
                # SNR 순으로 정렬 (높은 순)
                sorted_sbs = np.argsort(-obs_us[ue_id])
                
                # 2번째, 3번째로 좋은 SBS 순차적으로 시도
                fallback_found = False
                for sbs_idx in sorted_sbs[1:]:  # 1번째(최고)는 이미 시도했음
                    if admitted[ue_id, sbs_idx] == 1:
                        # 해당 SBS에서 서빙 가능
                        updated_actions[ue_id] = sbs_idx + 1
                        fallback_found = True
                        break
                
                if not fallback_found:
                    # 모든 SBS에서 서빙 불가능
                    updated_actions[ue_id] = 0
        
        return updated_actions
    
    def _apply_beam_limit_fallback(self, obs_us: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        Beam limit 제약을 고려한 Fallback 구현
        """
        U, S = obs_us.shape
        updated_actions = actions.copy()
        
        # 각 SBS별로 beam limit 확인
        for sbs_id in range(S):
            # 해당 SBS를 선택한 UE들
            ue_selected = np.where(updated_actions == sbs_id + 1)[0]
            
            if len(ue_selected) > self.beam_limits[sbs_id]:
                # Beam limit 초과: SNR이 낮은 UE들을 다른 SBS로 이동
                ue_snr_pairs = [(ue_id, obs_us[ue_id, sbs_id]) for ue_id in ue_selected]
                ue_snr_pairs.sort(key=lambda x: x[1])  # SNR 낮은 순으로 정렬
                
                # Beam limit을 초과하는 UE들
                excess_ues = ue_snr_pairs[self.beam_limits[sbs_id]:]
                
                for ue_id, _ in excess_ues:
                    # 대안 SBS 찾기 (SNR 순으로)
                    alternative_sbs = self._find_alternative_sbs(ue_id, obs_us, updated_actions)
                    if alternative_sbs > 0:
                        updated_actions[ue_id] = alternative_sbs
                    else:
                        updated_actions[ue_id] = 0  # 서빙 불가능
        
        return updated_actions
    
    def _find_alternative_sbs(self, ue_id: int, obs_us: np.ndarray, current_actions: np.ndarray) -> int:
        """
        UE에 대한 대안 SBS 찾기
        """
        U, S = obs_us.shape
        
        # SNR 순으로 SBS 정렬 (높은 순)
        sbs_snr_pairs = [(sbs_id, obs_us[ue_id, sbs_id]) for sbs_id in range(S)]
        sbs_snr_pairs.sort(key=lambda x: x[1], reverse=True)
        
        for sbs_id, _ in sbs_snr_pairs:
            if sbs_id + 1 == current_actions[ue_id]:
                continue  # 현재 선택된 SBS는 건너뛰기
            
            # 해당 SBS의 현재 부하 확인
            current_load = np.sum(current_actions == sbs_id + 1)
            
            if current_load < self.beam_limits[sbs_id]:
                return sbs_id + 1  # 사용 가능한 SBS
        
        return 0  # 사용 가능한 SBS 없음
    
    def train(self):
        """정책 학습 (필요 없음)"""
        return self
