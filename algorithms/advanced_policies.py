# algorithms/advanced_policies.py
from __future__ import annotations
import numpy as np
from typing import Optional, List, Tuple
import random

class CooperativePolicy:
    """
    UE 간 협력을 고려한 정책
    간섭을 최소화하고 전체 시스템 성능을 최적화
    """
    
    def __init__(self, beam_limits: Optional[List[int]] = None):
        self.name = "Cooperative"
        self.is_trained = True
        self.beam_limits = beam_limits or [2, 3, 3]
    
    def act_with_fallback(self, obs_us: np.ndarray) -> np.ndarray:
        """
        협력적 UE 선택 정책
        """
        U, S = obs_us.shape
        actions = np.zeros(U, dtype=np.int64)
        
        # 1차: 간섭을 고려한 SBS 선택
        for ue_id in range(U):
            # 간섭을 최소화하는 SBS 선택
            best_sbs = self._select_cooperative_sbs(ue_id, obs_us, actions)
            actions[ue_id] = best_sbs
        
        # 2차: Beam limit 제약 확인 및 Fallback
        actions = self._apply_beam_limit_fallback(obs_us, actions)
        
        return actions
    
    def _select_cooperative_sbs(self, ue_id: int, obs_us: np.ndarray, current_actions: np.ndarray) -> int:
        """
        간섭을 최소화하는 SBS 선택
        """
        U, S = obs_us.shape
        
        # 각 SBS의 간섭 점수 계산
        interference_scores = np.zeros(S)
        
        for sbs_id in range(S):
            # 해당 SBS의 현재 부하
            current_load = np.sum(current_actions == sbs_id + 1)
            
            # SNR 기반 점수
            snr_score = obs_us[ue_id, sbs_id]
            
            # 부하 기반 점수 (부하가 적을수록 높은 점수)
            load_score = 1.0 / (1.0 + current_load)
            
            # 간섭 점수 (SNR + 부하 고려)
            interference_scores[sbs_id] = snr_score * load_score
        
        # 최고 점수 SBS 선택
        best_sbs = 1 + np.argmax(interference_scores)
        return best_sbs
    
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
                # Beam limit 초과: 협력 점수가 낮은 UE들을 다른 SBS로 이동
                ue_coop_pairs = []
                for ue_id in ue_selected:
                    # 협력 점수 계산
                    snr_score = obs_us[ue_id, sbs_id]
                    load_score = 1.0 / (1.0 + len(ue_selected))
                    coop_score = snr_score * load_score
                    ue_coop_pairs.append((ue_id, coop_score))
                
                ue_coop_pairs.sort(key=lambda x: x[1])  # 협력 점수 낮은 순으로 정렬
                
                # Beam limit을 초과하는 UE들
                excess_ues = ue_coop_pairs[self.beam_limits[sbs_id]:]
                
                for ue_id, _ in excess_ues:
                    # 대안 SBS 찾기
                    alternative_sbs = self._find_alternative_sbs(ue_id, obs_us, updated_actions)
                    if alternative_sbs > 0:
                        updated_actions[ue_id] = alternative_sbs
                    else:
                        updated_actions[ue_id] = 0
        
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


class LoadBalancingWithFallback:
    """
    부하 분산 + Fallback 정책
    SBS 간 부하를 균등하게 분산하고 Fallback 지원
    """
    
    def __init__(self, beam_limits: Optional[List[int]] = None):
        self.name = "LoadBalancing-WithFallback"
        self.is_trained = True
        self.beam_limits = beam_limits or [2, 3, 3]
    
    def act_with_fallback(self, obs_us: np.ndarray) -> np.ndarray:
        """
        부하 분산 기반 SBS 선택
        """
        U, S = obs_us.shape
        actions = np.zeros(U, dtype=np.int64)
        
        # 1차: 부하 분산 기반 선택
        for ue_id in range(U):
            # 부하가 가장 적은 SBS 선택 (SNR도 고려)
            best_sbs = self._select_load_balanced_sbs(ue_id, obs_us, actions)
            actions[ue_id] = best_sbs
        
        # 2차: Beam limit 제약 확인 및 Fallback
        actions = self._apply_beam_limit_fallback(obs_us, actions)
        
        return actions
    
    def _select_load_balanced_sbs(self, ue_id: int, obs_us: np.ndarray, current_actions: np.ndarray) -> int:
        """
        부하 분산을 고려한 SBS 선택
        """
        U, S = obs_us.shape
        
        # 각 SBS의 부하 점수 계산
        load_scores = np.zeros(S)
        
        for sbs_id in range(S):
            # 해당 SBS의 현재 부하
            current_load = np.sum(current_actions == sbs_id + 1)
            
            # SNR 기반 점수
            snr_score = obs_us[ue_id, sbs_id]
            
            # 부하 기반 점수 (부하가 적을수록 높은 점수)
            load_score = 1.0 / (1.0 + current_load)
            
            # 최종 점수 (SNR + 부하 고려)
            load_scores[sbs_id] = snr_score * load_score
        
        # 최고 점수 SBS 선택
        best_sbs = 1 + np.argmax(load_scores)
        return best_sbs
    
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
                # Beam limit 초과: 부하 분산 점수가 낮은 UE들을 다른 SBS로 이동
                ue_load_pairs = []
                for ue_id in ue_selected:
                    # 부하 분산 점수 계산
                    snr_score = obs_us[ue_id, sbs_id]
                    load_score = 1.0 / (1.0 + len(ue_selected))
                    load_balance_score = snr_score * load_score
                    ue_load_pairs.append((ue_id, load_balance_score))
                
                ue_load_pairs.sort(key=lambda x: x[1])  # 부하 분산 점수 낮은 순으로 정렬
                
                # Beam limit을 초과하는 UE들
                excess_ues = ue_load_pairs[self.beam_limits[sbs_id]:]
                
                for ue_id, _ in excess_ues:
                    # 대안 SBS 찾기
                    alternative_sbs = self._find_alternative_sbs(ue_id, obs_us, updated_actions)
                    if alternative_sbs > 0:
                        updated_actions[ue_id] = alternative_sbs
                    else:
                        updated_actions[ue_id] = 0
        
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


class AdaptivePolicy:
    """
    환경 변화에 적응하는 정책
    SNR 변화와 SBS 부하에 따라 동적으로 정책 조정
    """
    
    def __init__(self, beam_limits: Optional[List[int]] = None):
        self.name = "Adaptive"
        self.is_trained = True
        self.beam_limits = beam_limits or [2, 3, 3]
        self.policy_history = []
    
    def act_with_fallback(self, obs_us: np.ndarray) -> np.ndarray:
        """
        적응적 정책 선택
        """
        U, S = obs_us.shape
        actions = np.zeros(U, dtype=np.int64)
        
        # 환경 상태 분석
        env_state = self._analyze_environment(obs_us)
        
        # 환경 상태에 따른 정책 선택
        if env_state == "high_interference":
            # 간섭이 높은 경우: 협력 정책
            actions = self._cooperative_selection(obs_us, actions)
        elif env_state == "high_load":
            # 부하가 높은 경우: 부하 분산 정책
            actions = self._load_balancing_selection(obs_us, actions)
        else:
            # 일반적인 경우: Max-SNR + Fallback
            actions = self._max_snr_selection(obs_us, actions)
        
        # Beam limit 제약 확인 및 Fallback
        actions = self._apply_beam_limit_fallback(obs_us, actions)
        
        # 정책 히스토리 업데이트
        self.policy_history.append(env_state)
        
        return actions
    
    def _analyze_environment(self, obs_us: np.ndarray) -> str:
        """
        환경 상태 분석
        """
        U, S = obs_us.shape
        
        # SNR 분산 계산
        snr_variance = np.var(obs_us)
        
        # 평균 SNR 계산
        mean_snr = np.mean(obs_us)
        
        # 간섭 지수 계산
        interference_index = snr_variance / (mean_snr + 1e-6)
        
        if interference_index > 0.5:
            return "high_interference"
        elif mean_snr < 80:  # 낮은 SNR
            return "high_load"
        else:
            return "normal"
    
    def _cooperative_selection(self, obs_us: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        협력적 선택
        """
        U, S = obs_us.shape
        
        for ue_id in range(U):
            # 간섭을 최소화하는 SBS 선택
            interference_scores = np.zeros(S)
            
            for sbs_id in range(S):
                snr_score = obs_us[ue_id, sbs_id]
                current_load = np.sum(actions == sbs_id + 1)
                load_score = 1.0 / (1.0 + current_load)
                interference_scores[sbs_id] = snr_score * load_score
            
            best_sbs = 1 + np.argmax(interference_scores)
            actions[ue_id] = best_sbs
        
        return actions
    
    def _load_balancing_selection(self, obs_us: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        부하 분산 선택
        """
        U, S = obs_us.shape
        
        for ue_id in range(U):
            # 부하가 가장 적은 SBS 선택
            load_scores = np.zeros(S)
            
            for sbs_id in range(S):
                snr_score = obs_us[ue_id, sbs_id]
                current_load = np.sum(actions == sbs_id + 1)
                load_score = 1.0 / (1.0 + current_load)
                load_scores[sbs_id] = snr_score * load_score
            
            best_sbs = 1 + np.argmax(load_scores)
            actions[ue_id] = best_sbs
        
        return actions
    
    def _max_snr_selection(self, obs_us: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        Max-SNR 선택
        """
        U, S = obs_us.shape
        
        for ue_id in range(U):
            best_sbs = 1 + np.argmax(obs_us[ue_id])
            actions[ue_id] = best_sbs
        
        return actions
    
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
                # Beam limit 초과: 적응적 점수가 낮은 UE들을 다른 SBS로 이동
                ue_adaptive_pairs = []
                for ue_id in ue_selected:
                    # 적응적 점수 계산
                    snr_score = obs_us[ue_id, sbs_id]
                    load_score = 1.0 / (1.0 + len(ue_selected))
                    adaptive_score = snr_score * load_score
                    ue_adaptive_pairs.append((ue_id, adaptive_score))
                
                ue_adaptive_pairs.sort(key=lambda x: x[1])  # 적응적 점수 낮은 순으로 정렬
                
                # Beam limit을 초과하는 UE들
                excess_ues = ue_adaptive_pairs[self.beam_limits[sbs_id]:]
                
                for ue_id, _ in excess_ues:
                    # 대안 SBS 찾기
                    alternative_sbs = self._find_alternative_sbs(ue_id, obs_us, updated_actions)
                    if alternative_sbs > 0:
                        updated_actions[ue_id] = alternative_sbs
                    else:
                        updated_actions[ue_id] = 0
        
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
