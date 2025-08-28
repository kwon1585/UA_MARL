# make_dataset_maxsnrfb.py
from __future__ import annotations
import numpy as np
import os
import pickle
from typing import List, Dict, Tuple
from ua_env import UAEnv, UAConfig
from algorithms.max_snr_fallback import MaxSNRWithFallbackPolicy
import argparse



class DatasetCollector:
    """MaxSNRWithFallback 정책만 사용하는 데이터셋 수집기"""
    
    def __init__(self, env: UAEnv):
        self.env = env
        self.policies = self._create_diverse_policies()
        self.policy_names = [p.name for p in self.policies]
    
    def _create_diverse_policies(self) -> List:
        """MaxSNRWithFallback 정책만 사용하여 일관된 데이터 생성"""
        policies = []
        
        # MaxSNR with Fallback 정책만 사용 (가중치 높임)
        policies.append(MaxSNRWithFallbackPolicy(self.env.cfg.beam_limits))
        policies.append(MaxSNRWithFallbackPolicy(self.env.cfg.beam_limits))  # 중복 추가로 가중치 증가
        policies.append(MaxSNRWithFallbackPolicy(self.env.cfg.beam_limits))  # 더 추가
        
        return policies
    
    def collect_dataset(self, num_episodes: int) -> List[Dict]:
        """데이터셋 수집"""
        dataset = []
        
        for ep in range(num_episodes):
            print(f"에피소드 {ep+1}/{num_episodes} 수집 중...")
            
            # 환경 초기화
            obs = self.env.reset(seed=self.env.cfg.seed + ep)
            
            for step in range(self.env.cfg.episode_length):
                # 정책별로 액션 생성
                for policy_idx, policy in enumerate(self.policies):
                    # 정책별 액션 생성
                    actions = policy.act_with_fallback(obs)
                    
                    # 환경에서 한 스텝 진행 (MaxSNR_fallback 사용)
                    next_obs, rewards, done, info = self.env.step(actions, use_mixed_reward=True, lambda_weight=0.5)
                    
                    # 데이터 저장
                    episode_data = {
                        'episode': ep,
                        'step': step,
                        'policy': policy.name,
                        'policy_idx': policy_idx,
                        'obs': obs.copy(),
                        'actions': actions.copy(),
                        'rewards': rewards if isinstance(rewards, np.ndarray) else np.array(rewards),
                        'next_obs': next_obs.copy(),
                        'done': done,
                        'info': info
                    }
                    dataset.append(episode_data)
                    
                    # 다음 관찰로 업데이트
                    obs = next_obs
                    
                    if done:
                        break
        
        return dataset
    
    def save_dataset(self, dataset: List[Dict], dataset_root: str):
        """데이터셋 저장"""
        os.makedirs(dataset_root, exist_ok=True)
        
        # 전체 데이터셋 저장
        with open(os.path.join(dataset_root, 'dataset.pkl'), 'wb') as f:
            pickle.dump(dataset, f)
        
        # UE별 데이터셋 분리
        U = self.env.cfg.num_ue
        for ue_id in range(U):
            # FedCombo가 기대하는 형식으로 데이터 구성
            obs = np.array([data['obs'][ue_id] for data in dataset])
            act = np.array([data['actions'][ue_id] for data in dataset])
            # Mixed Reward 처리: rewards가 배열이면 해당 UE의 보상만 추출
            reward = np.array([float(data['rewards'][ue_id]) if isinstance(data['rewards'], np.ndarray) and data['rewards'].ndim > 0 else float(data['rewards']) for data in dataset])
            next_obs = np.array([data['next_obs'][ue_id] for data in dataset])
            terminals = np.zeros(len(dataset), dtype=bool)
            timeouts = np.zeros(len(dataset), dtype=bool)
            
            # FedCombo가 기대하는 형식: ua_agent_{i}.npz
            ue_file = os.path.join(dataset_root, f'ua_agent_{ue_id}.npz')
            np.savez_compressed(ue_file, 
                               obs=obs,
                               act=act,
                               reward=reward,
                               next_obs=next_obs,
                               terminals=terminals,
                               timeouts=timeouts)
        
        # 데이터 품질 리포트 생성
        self._generate_quality_report(dataset, dataset_root)
        
        print(f"데이터셋 저장 완료: {dataset_root}")
        print(f"총 {len(dataset)}개 샘플, {U}개 UE")
    
    def _generate_quality_report(self, dataset: List[Dict], dataset_root: str):
        """데이터 품질 리포트 생성"""
        report = []
        report.append("=== UA MaxSNR_fallback 데이터셋 품질 리포트 ===")
        report.append(f"총 샘플 수: {len(dataset)}")
        report.append(f"에피소드 수: {len(set(d['episode'] for d in dataset))}")
        report.append(f"스텝 수: {len(set(d['step'] for d in dataset))}")
        report.append(f"정책 수: {len(self.policies)}")
        report.append("")
        
        # 정책별 사용 통계
        policy_stats = {}
        for data in dataset:
            policy = data['policy']
            if policy not in policy_stats:
                policy_stats[policy] = 0
            policy_stats[policy] += 1
        
        report.append("정책별 사용 통계:")
        for policy, count in policy_stats.items():
            percentage = (count / len(dataset)) * 100
            report.append(f"  {policy}: {count} ({percentage:.1f}%)")
        
        report.append("")
        
        # 보상 통계
        all_rewards = [d['rewards'] for d in dataset]
        # 보상이 스칼라인지 배열인지 확인하여 처리
        flat_rewards = []
        for rewards in all_rewards:
            if np.isscalar(rewards):
                flat_rewards.append(float(rewards))
            elif rewards.ndim == 0:
                flat_rewards.append(float(rewards.item()))
            else:
                flat_rewards.extend([float(r) for r in rewards])
        
        if flat_rewards:
            report.append("보상 통계:")
            report.append(f"  평균: {np.mean(flat_rewards):.2f}")
            report.append(f"  표준편차: {np.std(flat_rewards):.2f}")
            report.append(f"  최소: {np.min(flat_rewards):.2f}")
            report.append(f"  최대: {np.max(flat_rewards):.2f}")
        else:
            report.append("보상 통계: 데이터 없음")
        
        # 리포트 저장
        report_path = os.path.join(dataset_root, 'data_quality_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))

def main():
    parser = argparse.ArgumentParser(description="MaxSNRWithFallback 기반 데이터셋 수집")
    parser.add_argument("--dataset_root", type=str, required=True, help="데이터셋 저장 경로")
    parser.add_argument("--num_episodes", type=int, default=10, help="수집할 에피소드 수")
    parser.add_argument("--num_ue", type=int, default=20, help="UE 수")
    parser.add_argument("--num_sbs", type=int, default=3, help="SBS 수")
    parser.add_argument("--episode_length", type=int, default=100, help="에피소드 길이")
    parser.add_argument("--seed", type=int, default=1, help="랜덤 시드")
    
    args = parser.parse_args()
    
    # 환경 설정
    config = UAConfig(
        num_ue=args.num_ue,
        num_sbs=args.num_sbs,
        episode_length=args.episode_length,
        seed=args.seed
    )
    
    # 환경 생성
    env = UAEnv(config)
    
    # 데이터 수집기 생성
    collector = DatasetCollector(env)
    
    # 데이터셋 수집
   
    dataset = collector.collect_dataset(args.num_episodes)
    
    # 데이터셋 저장
    collector.save_dataset(dataset, args.dataset_root)
    
    print("UA MaxSNR_fallback dataset is done!")

if __name__ == "__main__":
    main()