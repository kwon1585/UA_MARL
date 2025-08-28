# ua_env.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from algorithms.max_snr import MaxSNRPolicy

@dataclass
class UAConfig:
    num_ue: int = 20
    num_sbs: int = 3
    beam_limits: Tuple[int, ...] = (2, 3, 3)
    area_size: Tuple[float, float] = (100.0, 100.0)
    carrier_freq_ghz: float = 28.0
    bandwidth_mhz: float = 100.0
    noise_figure_db: float = 7.0
    tx_power_dbm: float = 30.0
    pathloss_exp: float = 2.0
    shadowing_std_db: float = 6.0
    fading_scale_db: float = 3.0
    snr_clip: Optional[Tuple[float, float]] = (-30.0, 120.0)  # 클리핑 범위 확장
    episode_length: int = 100
    seed: Optional[int] = None

class UAEnv:
    """
    Obs:  (U,S) SNR[dB]
    Act:  (U,)  in {0..S} (0=미서빙/MBS, 1..S=SBS)
    Reward: sum-rate (Shannon); info['per_ue_rate'] 제공
    """
    def __init__(self, cfg: UAConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.t = 0
        self._init_layout()

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self._init_layout()
        return self._draw_snr_db()

    def step(self, actions: np.ndarray, use_mixed_reward: bool = False, lambda_weight: float = 0.5):
        assert actions.shape == (self.cfg.num_ue,), "actions must be (U,)"
        snr_db = self._draw_snr_db()
        admitted, chosen = self._apply_beam_limits(actions, snr_db)
        sinr_lin, per_ue_rate = self._compute_rates(snr_db, admitted)
        
        if use_mixed_reward:
            # Mixed reward 계산: 각 UE별로 개별 리워드 반환
            mixed_rewards = self._compute_mixed_rewards(actions, snr_db, admitted, per_ue_rate, lambda_weight)
            reward = mixed_rewards
        else:
            # 기존 방식: 전체 sum_rate 반환
            reward = float(per_ue_rate.sum())
        
        self.t += 1
        done = self.t >= self.cfg.episode_length
        next_obs = self._draw_snr_db()
        info = {
            "per_ue_rate": per_ue_rate.astype(np.float32),
            "per_ue_sinr_db": (10.0*np.log10(np.clip(sinr_lin, 1e-12, None))).astype(np.float32),
            "admitted": admitted,
            "chosen": chosen,
            "sum_rate": float(per_ue_rate.sum()),
        }
        return next_obs, reward, done, info

    def flatten_obs(self, obs_snr_db: np.ndarray) -> np.ndarray:
        return obs_snr_db.astype(np.float32).reshape(-1)

    def visualize_layout(self, actions: Optional[np.ndarray] = None, admitted: Optional[np.ndarray] = None, 
                        save_path: Optional[str] = None, show_plot: bool = True):
        """
        UE와 SBS의 위치, 연결 상태, 커버리지를 시각화
        
        Args:
            actions: (U,) 각 UE의 SBS 선택
            admitted: (U, S) 각 UE별 각 SBS의 서빙 허용 여부
            save_path: 저장할 파일 경로
            show_plot: 플롯을 화면에 표시할지 여부
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        W, H = self.cfg.area_size
        
        # SBS 위치 및 커버리지 표시
        colors = ['red', 'blue', 'green']
        for i, (pos, color) in enumerate(zip(self.sbs_pos, colors)):
            # SBS 위치 (삼각형)
            ax.scatter(pos[0], pos[1], c=color, marker='^', s=200, 
                      edgecolors='black', linewidth=2, label=f'SBS{i+1}')
            
            # 커버리지 원 (SNR > 0dB인 영역)
            # 간단한 커버리지 반경 계산 (경로손실 기반)
            coverage_radius = 30.0  # 대략적인 커버리지 반경
            circle = patches.Circle(pos, coverage_radius, fill=False, 
                                  linestyle='--', color=color, alpha=0.7)
            ax.add_patch(circle)
        
        # UE 위치 및 연결 상태 표시
        if actions is not None and admitted is not None:
            # 연결된 UE들
            for ue_id in range(self.cfg.num_ue):
                ue_pos = self.ue_pos[ue_id]
                
                # 어떤 SBS에 연결되었는지 확인
                connected_sbs = None
                if actions[ue_id] > 0:  # SBS 선택됨
                    sbs_idx = actions[ue_id] - 1
                    if admitted[ue_id, sbs_idx] == 1:  # 서빙 허용됨
                        connected_sbs = sbs_idx
                
                if connected_sbs is not None:
                    # 연결된 UE (녹색 원)
                    ax.scatter(ue_pos[0], ue_pos[1], c='green', s=100, 
                              alpha=0.8, label='Connected UE' if ue_id == 0 else "")
                    
                    # 연결선 그리기
                    sbs_pos = self.sbs_pos[connected_sbs]
                    ax.plot([ue_pos[0], sbs_pos[0]], [ue_pos[1], sbs_pos[1]], 
                           'gray', linestyle='--', alpha=0.6)
                    
                    # UE 라벨
                    ax.annotate(f'UE{ue_id}', (ue_pos[0], ue_pos[1]), 
                               xytext=(5, 5), textcoords='offset points', 
                               fontsize=8, alpha=0.8)
                else:
                    # 연결되지 않은 UE (빨간색 원)
                    ax.scatter(ue_pos[0], ue_pos[1], c='red', s=100, 
                              alpha=0.6, label='Unconnected UE' if ue_id == 0 else "")
                    
                    # UE 라벨
                    ax.annotate(f'UE{ue_id}', (ue_pos[0], ue_pos[1]), 
                               xytext=(5, 5), textcoords='offset points', 
                               fontsize=8, alpha=0.6)
        else:
            # 단순히 UE 위치만 표시
            for ue_id, ue_pos in enumerate(self.ue_pos):
                ax.scatter(ue_pos[0], ue_pos[1], c='gray', s=80, alpha=0.7)
                ax.annotate(f'UE{ue_id}', (ue_pos[0], ue_pos[1]), 
                           xytext=(3, 3), textcoords='offset points', 
                           fontsize=7, alpha=0.7)
        
        # 축 설정
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(f'UE and SBS Layout (Episode {self.t})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 저장
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Layout visualization saved to: {save_path}")
        
        # 표시
        if show_plot:
            plt.show()
        else:
            plt.close()

    # ---------- internals ----------
    def _init_layout(self):
        U, S = self.cfg.num_ue, self.cfg.num_sbs
        W, H = self.cfg.area_size
        
        # SBS 위치를 더 다양하게 배치 (에피소드마다 약간씩 다르게)
        if hasattr(self, 'base_seed'):
            # 기존 시드가 있으면 약간씩 변화
            sbs_rng = np.random.default_rng(self.base_seed + self.t // 10)
        else:
            # 첫 번째 에피소드
            self.base_seed = self.cfg.seed if self.cfg.seed is not None else 42
            sbs_rng = np.random.default_rng(self.base_seed)
        
        # SBS 위치에 약간의 랜덤성 추가
        gx = np.linspace(W*0.2, W*0.8, S)
        gy = np.full(S, H*0.4)
        
        # SBS 위치에 ±10% 랜덤 변화
        gx += sbs_rng.uniform(-W*0.05, W*0.05, S)
        gy += sbs_rng.uniform(-H*0.05, H*0.05, S)
        
        self.sbs_pos = np.stack([gx, gy], axis=1)
        
        # UE 위치를 더 다양하게 배치 (클러스터링 효과 추가)
        self.ue_pos = np.zeros((U, 2))
        
        # 일부 UE는 SBS 주변에 클러스터링
        cluster_ues = U // 2
        for i in range(cluster_ues):
            # 랜덤하게 SBS 선택
            sbs_idx = sbs_rng.integers(0, S)
            # SBS 주변에 클러스터링 (가우시안 분포)
            cluster_std = min(W, H) * 0.1  # 영역의 10%
            self.ue_pos[i] = sbs_rng.normal(self.sbs_pos[sbs_idx], cluster_std)
        
        # 나머지 UE는 전체 영역에 균등 분포
        for i in range(cluster_ues, U):
            self.ue_pos[i] = self.rng.uniform([W*0.1, H*0.1], [W*0.9, H*0.9])
        
        # 위치를 영역 내로 클리핑
        self.ue_pos[:, 0] = np.clip(self.ue_pos[:, 0], W*0.05, W*0.95)
        self.ue_pos[:, 1] = np.clip(self.ue_pos[:, 1], H*0.05, H*0.95)
        
        # 거리 및 경로손실 계산
        self.dist = np.linalg.norm(self.ue_pos[:,None,:]-self.sbs_pos[None,:,:], axis=2) + 1e-6
        
        # 1m 기준 자유공간 손실 계산 (32.4 + 20*log10(f_GHz))
        # f_GHz = 28.0 GHz
        f_ghz = self.cfg.carrier_freq_ghz
        pl_1m = 32.4 + 20.0 * np.log10(f_ghz)
        
        # 경로손실 = PL(1m) + 10*n*log10(d)
        self.pathloss_db = pl_1m + 10.0 * self.cfg.pathloss_exp * np.log10(self.dist)
        
        # 섀도잉 (UE별로 독립적)
        self.shadow_db = (self.rng.normal(0.0, self.cfg.shadowing_std_db, size=(U,self.cfg.num_sbs))
                          if self.cfg.shadowing_std_db>0 else np.zeros((U,self.cfg.num_sbs)))
        
        self.noise_dbm = self._thermal_noise_dbm(self.cfg.bandwidth_mhz) + self.cfg.noise_figure_db

    def _thermal_noise_dbm(self, bandwidth_mhz: float) -> float:
        kt_db = -174.0; bw_hz = bandwidth_mhz*1e6
        return kt_db + 10.0*np.log10(bw_hz)

    def _draw_snr_db(self) -> np.ndarray:
        U, S = self.cfg.num_ue, self.cfg.num_sbs
        fading_db = (self.rng.normal(0.0, self.cfg.fading_scale_db, size=(U,S))
                     if self.cfg.fading_scale_db>0 else np.zeros((U,S)))
        
        # SNR 계산 과정 디버깅
        rx_dbm = self.cfg.tx_power_dbm - (self.pathloss_db + self.shadow_db) + fading_db
        snr_db = rx_dbm - self.noise_dbm
        
        # 디버깅 정보 출력 (첫 번째 호출에서만)
        if not hasattr(self, '_debug_printed'):
            print(f"[SNR Debug] tx_power_dbm: {self.cfg.tx_power_dbm}")
            print(f"[SNR Debug] noise_dbm: {self.noise_dbm}")
            print(f"[SNR Debug] PL(1m) at {self.cfg.carrier_freq_ghz}GHz: {32.4 + 20.0*np.log10(self.cfg.carrier_freq_ghz):.2f} dB")
            print(f"[SNR Debug] pathloss_db range: {self.pathloss_db.min():.2f} to {self.pathloss_db.max():.2f}")
            print(f"[SNR Debug] shadow_db range: {self.shadow_db.min():.2f} to {self.shadow_db.max():.2f}")
            print(f"[SNR Debug] fading_db range: {fading_db.min():.2f} to {fading_db.max():.2f}")
            print(f"[SNR Debug] rx_dbm range: {rx_dbm.min():.2f} to {rx_dbm.max():.2f}")
            print(f"[SNR Debug] raw SNR range: {snr_db.min():.2f} to {snr_db.max():.2f}")
            self._debug_printed = True
        
        if self.cfg.snr_clip is not None:
            lo, hi = self.cfg.snr_clip
            snr_db = np.clip(snr_db, lo, hi)
        
        return snr_db

    def _apply_beam_limits(self, actions: np.ndarray, snr_db: np.ndarray):
        U, S = self.cfg.num_ue, self.cfg.num_sbs
        chosen = actions.astype(int).copy()
        admitted = np.zeros((U,S), dtype=np.int32)
        for j in range(1, S+1):
            users = np.where(chosen==j)[0]
            if users.size==0: continue
            K = int(self.cfg.beam_limits[j-1]) if j-1 < len(self.cfg.beam_limits) else users.size
            order = users[np.argsort(-snr_db[users, j-1])]
            keep = order[:K]; admitted[keep, j-1]=1
            drop = order[K:]; 
            if drop.size>0: chosen[drop]=0
        return admitted, chosen

    def _compute_mixed_rewards(self, actions: np.ndarray, snr_db: np.ndarray, admitted: np.ndarray, per_ue_rate: np.ndarray, lambda_weight: float):
        """
        개선된 Mixed Reward 계산: 로그 스케일링 + 정규화 + 비선형 결합
        r̃_i = sqrt(personal_reward² + joint_reward²)
        """
        U = self.cfg.num_ue
        mixed_rewards = np.zeros(U, dtype=np.float32)
        
        # Max-SNR 정책 준비
        max_snr_policy = MaxSNRPolicy()
        
        # 현재 전체 시스템 성능 (G_t)
        current_sum_rate = float(per_ue_rate.sum())
        
        for i in range(U):
            # UE i의 원래 리워드 (r_i)
            original_reward = float(per_ue_rate[i])
            
            # UE i가 서빙 안함일 때 Max-SNR으로 최적화
            actions_modified = actions.copy()
            actions_modified[i] = 0  # UE i 연결 해제
            
            # Max-SNR 정책으로 최적 액션 재계산
            optimal_actions = max_snr_policy.act(snr_db)
            actions_modified[i] = optimal_actions[i]  # Max-SNR 기반 최적 선택
            
            # 수정된 행동으로 성능 재계산
            admitted_modified, _ = self._apply_beam_limits(actions_modified, snr_db)
            _, per_ue_rate_modified = self._compute_rates(snr_db, admitted_modified)
            modified_sum_rate = float(per_ue_rate_modified.sum())
            
            # 차이 리워드: 현재 vs Max-SNR 기반 최적
            difference_reward = current_sum_rate - modified_sum_rate
            
            # 개선된 리워드 계산: 로그 스케일링 + 정규화 + 비선형 결합
            
            # 1. 개인 리워드: 로그 스케일링으로 극단값 방지
            if original_reward > 0:
                personal_reward = np.log10(np.maximum(original_reward, 1e6))  # 6~9 범위
            else:
                personal_reward = 0.0  # 0일 때는 0
            
            # 2. 조인트 리워드: 차이를 -1~1 범위로 정규화 (tanh 사용)
            # 차이가 너무 클 수 있으므로 스케일링 계수 조정
            scale_factor = max(1e8, abs(difference_reward) * 0.1)  # 적응적 스케일링
            joint_reward = np.tanh(difference_reward / scale_factor)  # -1~1 범위
            
            # 3. 비선형 결합: 유클리드 거리 기반
            if personal_reward == 0 and joint_reward == 0:
                mixed_rewards[i] = 0.0
            else:
                mixed_rewards[i] = np.sign(personal_reward + joint_reward) * np.sqrt(personal_reward**2 + joint_reward**2)
        
        return mixed_rewards

    def _compute_rates(self, snr_db: np.ndarray, admitted: np.ndarray):
        noise_mw = 10**(self.noise_dbm/10.0)
        snr_lin = 10**(snr_db/10.0); rx = snr_lin*noise_mw
        active = (admitted.sum(axis=0)>0).astype(rx.dtype)
        signal = (rx*admitted).sum(axis=1)
        interf = (rx*active[None,:]).sum(axis=1) - signal
        sinr = np.where(signal>0.0, signal/(interf+noise_mw), 0.0)
        B = self.cfg.bandwidth_mhz*1e6
        per_ue_rate = B*np.log2(1.0+sinr)
        per_ue_rate[admitted.sum(axis=1)==0]=0.0
        return sinr, per_ue_rate
