# make_dataset_improved.py
from __future__ import annotations
import os, numpy as np
from typing import Dict, List, Tuple
from ua_marl import build_common_parser, build_env, dataset_path_for_agent
from algorithms.max_snr_fallback import MaxSNRWithFallbackPolicy
from algorithms.random_policy import RandomPolicy
from algorithms.epsilon_greedy_policy import EpsilonGreedyPolicy
from algorithms.load_balancing_policy import LoadBalancingPolicy
import random

class ImprovedDataCollector:
    """개선된 데이터 수집기: 데이터 품질과 일관성 향상"""
    
    def __init__(self, env, num_episodes: int, episode_length: int):
        self.env = env
        self.num_episodes = num_episodes
        self.episode_length = episode_length
        self.policies = self._create_diverse_policies()
        
    def _create_diverse_policies(self) -> List:
        """다양한 정책을 생성하여 탐험 증가"""
        policies = []
        
        # 1. MaxSNR 정책
        policies.append(MaxSNRWithFallbackPolicy())
        
        # 2. 랜덤 정책
        policies.append(RandomPolicy(self.env.cfg.num_sbs))
        
        # 3. ε-greedy 정책 (MaxSNR 기반)
        policies.append(EpsilonGreedyPolicy(self.env.cfg.num_sbs, epsilon=0.3))
        
        # 4. Load balancing 정책
        policies.append(LoadBalancingPolicy(self.env.cfg.num_sbs, self.env.cfg.beam_limits))
        
        return policies
    
    def collect_rollouts(self) -> Tuple[Dict, Tuple]:
        """개선된 데이터 수집: 정책 다양화 및 데이터 일관성 보장"""
        
        # UE별 개별 데이터
        ue_data = {i: {
            'obs': [], 'act': [], 'reward': [], 'next_obs': [], 
            'episode_id': [], 'step_id': [], 'policy_id': []
        } for i in range(self.env.cfg.num_ue)}
        
        # 전체 환경 데이터 (UE 간 상호작용 정보 포함)
        env_data = {
            'obs': [], 'act': [], 'reward': [], 'next_obs': [], 
            'done': [], 'per_ue_rate': [], 'episode_id': [], 'step_id': [],
            'ue_positions': [], 'sbs_positions': [], 'admitted_matrix': []
        }
        
        episode_count = 0
        
        for episode in range(self.num_episodes):
            # 에피소드마다 다른 정책 사용 (탐험 증가)
            policy = random.choice(self.policies)
            policy_id = self.policies.index(policy)
            
            obs = self.env.reset(seed=episode + 42)  # 재현 가능한 시드
            
            for step in range(self.episode_length):
                # 정책에 따른 액션 선택
                if hasattr(policy, 'act_with_fallback'):
                    actions = policy.act_with_fallback(obs)
                else:
                    actions = policy.act(obs)
                
                # 환경 스텝 실행 (Mixed Reward 사용)
                next_obs, reward, done, info = self.env.step(actions, use_mixed_reward=True, lambda_weight=0.5)
                
                # 데이터 저장 (UE별) - Mixed Reward 처리
                for ue_id in range(self.env.cfg.num_ue):
                    ue_data[ue_id]['obs'].append(obs[ue_id].copy())
                    ue_data[ue_id]['act'].append(actions[ue_id])
                    # Mixed Reward: reward가 배열이면 해당 UE의 보상만 추출
                    if isinstance(reward, np.ndarray) and reward.ndim > 0:
                        ue_reward = reward[ue_id]
                    else:
                        ue_reward = info['per_ue_rate'][ue_id]
                    ue_data[ue_id]['reward'].append(ue_reward)
                    ue_data[ue_id]['next_obs'].append(next_obs[ue_id].copy())
                    ue_data[ue_id]['episode_id'].append(episode)
                    ue_data[ue_id]['step_id'].append(step)
                    ue_data[ue_id]['policy_id'].append(policy_id)
                
                # 환경 데이터 저장 (UE 간 상호작용 정보 포함) - Mixed Reward 처리
                env_data['obs'].append(self.env.flatten_obs(obs))
                env_data['act'].append(actions.astype(np.int64))
                # Mixed Reward: reward가 배열이면 전체 시스템 성능으로 저장
                if isinstance(reward, np.ndarray) and reward.ndim > 0:
                    system_reward = float(reward.sum())  # 전체 Mixed Reward 합
                else:
                    system_reward = float(reward)
                env_data['reward'].append(np.array([system_reward], np.float32))
                env_data['next_obs'].append(self.env.flatten_obs(next_obs))
                env_data['done'].append(np.array([done], bool))
                env_data['per_ue_rate'].append(info['per_ue_rate'].astype(np.float32))
                env_data['episode_id'].append(episode)
                env_data['step_id'].append(step)
                env_data['ue_positions'].append(self.env.ue_pos.copy())
                env_data['sbs_positions'].append(self.env.sbs_pos.copy())
                env_data['admitted_matrix'].append(info['admitted'].copy())
                
                obs = next_obs
                if done:
                    episode_count += 1
                    break
        
        # numpy 배열로 변환
        for ue_id in ue_data:
            for key in ['obs', 'act', 'reward', 'next_obs', 'episode_id', 'step_id', 'policy_id']:
                ue_data[ue_id][key] = np.asarray(ue_data[ue_id][key])
        
        # 환경 데이터도 numpy 배열로 변환
        for key in env_data:
            if key in ['ue_positions', 'sbs_positions', 'admitted_matrix']:
                env_data[key] = np.asarray(env_data[key])
            else:
                env_data[key] = np.asarray(env_data[key])
        
        return ue_data, env_data

def save_improved_datasets(args, ue_data: Dict, env_data: Dict):
    """개선된 데이터셋 저장: UE별 + 통합 환경 데이터"""
    
    # 1. UE별 개별 데이터 저장 (기존 방식 유지)
    for ue_id in range(args.num_ue):
        data_i = {
            "obs": ue_data[ue_id]['obs'],
            "act": ue_data[ue_id]['act'],
            "reward": ue_data[ue_id]['reward'],
            "next_obs": ue_data[ue_id]['next_obs'],
            "terminal": np.zeros(len(ue_data[ue_id]['obs']), bool),  # 모든 스텝 False
            "episode_id": ue_data[ue_id]['episode_id'],
            "step_id": ue_data[ue_id]['step_id'],
            "policy_id": ue_data[ue_id]['policy_id'],
            "meta_U": np.array([args.num_ue], np.int32),
            "meta_S": np.array([args.num_sbs], np.int32),
            "meta_agent_id": np.array([ue_id], np.int32),
        }
        
        out_path = dataset_path_for_agent(args, ue_id)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(out_path, **data_i)
        print(f"[data] saved agent {ue_id} -> {out_path} (shape: {ue_data[ue_id]['obs'].shape})")
    
    # 2. 통합 환경 데이터 저장 (UE 간 상호작용 정보 포함)
    env_data_path = os.path.join(args.dataset_root, "ua_mix_environment.npz")
    np.savez_compressed(env_data_path, **env_data)
    print(f"[data] saved environment data -> {env_data_path}")
    
    # 3. 데이터 품질 리포트 생성
    generate_data_quality_report(args, ue_data, env_data)

def generate_data_quality_report(args, ue_data: Dict, env_data: Dict):
    """데이터 품질 리포트 생성"""
    
    report_path = os.path.join(args.dataset_root, "data_quality_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("=== UA Mixed Reward Dataset Quality Report ===\n\n")
        
        # 기본 정보
        f.write(f"Dataset Configuration:\n")
        f.write(f"  - Number of UEs: {args.num_ue}\n")
        f.write(f"  - Number of SBSs: {args.num_sbs}\n")
        f.write(f"  - Episodes: {args.num_episodes}\n")
        f.write(f"  - Episode length: {args.episode_length}\n")
        f.write(f"  - Total samples: {len(env_data['obs'])}\n\n")
        
        # 정책 분포
        policy_counts = {}
        for ue_id in ue_data:
            for policy_id in ue_data[ue_id]['policy_id']:
                policy_counts[policy_id] = policy_counts.get(policy_id, 0) + 1
        
        f.write("Policy Distribution:\n")
        # 정책 이름 리스트 (정책 순서와 일치해야 함)
        policy_names = [
            "MaxSNR-WithFallback",
            "Random",
            "EpsilonGreedy",
            "LoadBalancing"
        ]
        for policy_id, count in sorted(policy_counts.items()):
            f.write(f"  - {policy_names[policy_id]}: {count} samples\n")
        f.write("\n")
        
        # 데이터 다양성 분석
        f.write("Data Diversity Analysis:\n")
        
        # SNR 범위
        all_snr = np.concatenate([ue_data[ue_id]['obs'] for ue_id in ue_data])
        f.write(f"  - SNR range: {all_snr.min():.2f} to {all_snr.max():.2f} dB\n")
        f.write(f"  - SNR std: {all_snr.std():.2f} dB\n")
        
        # 액션 분포
        all_actions = np.concatenate([ue_data[ue_id]['act'] for ue_id in ue_data])
        unique_actions, action_counts = np.unique(all_actions, return_counts=True)
        f.write(f"  - Action distribution: {dict(zip(unique_actions, action_counts))}\n")
        
        # 보상 분포 (Mixed Reward)
        all_rewards = np.concatenate([ue_data[ue_id]['reward'] for ue_id in ue_data])
        f.write(f"  - Mixed Reward range: {all_rewards.min():.2e} to {all_rewards.max():.2e}\n")
        f.write(f"  - Mixed Reward std: {all_rewards.std():.2e}\n")
        f.write(f"  - Mixed Reward mean: {all_rewards.mean():.2e}\n")
        
        # 시스템 성능 (전체 Mixed Reward 합)
        system_rewards = env_data['reward'].flatten()
        f.write(f"  - System Mixed Reward range: {system_rewards.min():.2e} to {system_rewards.max():.2e}\n")
        f.write(f"  - System Mixed Reward mean: {system_rewards.mean():.2e}\n")
        
        # 에피소드별 데이터 분포
        episode_counts = {}
        for ue_id in ue_data:
            for episode_id in ue_data[ue_id]['episode_id']:
                episode_counts[episode_id] = episode_counts.get(episode_id, 0) + 1
        
        f.write(f"  - Episodes covered: {len(episode_counts)}\n")
        f.write(f"  - Samples per episode: {np.mean(list(episode_counts.values())):.1f}\n")
    
    print(f"[data] quality report saved to: {report_path}")

def main():
    p = build_common_parser("UA Improved Dataset Maker")
    args = p.parse_args()

    # 디렉토리 생성
    os.makedirs(args.dataset_root, exist_ok=True)
    
    env = build_env(args)
    
    print(f"[data] UA Mixed Reward 데이터셋 생성 시작 (Policy Mix)")
    print(f"[data] - 에피소드: {args.num_episodes}, 스텝: {args.episode_length}")
    print(f"[data] - 정책 다양화: MaxSNR, Random, EpsilonGreedy, LoadBalancing")
    print(f"[data] - Mixed Reward: λ=0.5, 스케일 매칭 및 단위 축소 적용")
    print(f"[data] - 데이터 품질 향상 및 UE 간 상호작용 정보 보존")

    # 개선된 데이터 수집기 생성
    collector = ImprovedDataCollector(env, args.num_episodes, args.episode_length)
    
    # 데이터 수집
    ue_data, env_data = collector.collect_rollouts()
    
    print(f"[data] 데이터 수집 완료: {len(env_data['obs'])}개 샘플")
    print(f"[data] UE별 데이터 분할 및 저장 중...")
    
    # 개선된 데이터셋 저장
    save_improved_datasets(args, ue_data, env_data)
    
    print("[data] UA Mixed Reward 데이터셋 생성 완료!")

if __name__ == "__main__":
    main()
