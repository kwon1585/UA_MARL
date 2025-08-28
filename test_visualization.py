# test_visualization.py
"""
UA 환경의 문제 상황들과 개선된 상황을 비교 시각화
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ua_env import UAEnv, UAConfig
from algorithms.max_snr_fallback import MaxSNRWithFallbackPolicy
from algorithms.random_policy import RandomPolicy
from algorithms.epsilon_greedy_policy import EpsilonGreedyPolicy
from algorithms.load_balancing_policy import LoadBalancingPolicy
import os

def create_problem_scenarios():
    """문제 상황들을 생성하는 함수들"""
    
    def create_original_env():
        """기존 환경 (문제 상황)"""
        cfg = UAConfig(
            num_ue=20,
            num_sbs=3,
            beam_limits=(2, 3, 3),
            area_size=(100.0, 100.0),
            snr_clip=(-30.0, 50.0),  # 문제: 너무 좁은 SNR 범위
            episode_length=100,
            seed=42
        )
        return UAEnv(cfg)
    
    def create_improved_env():
        """개선된 환경"""
        cfg = UAConfig(
            num_ue=20,
            num_sbs=3,
            beam_limits=(2, 3, 3),
            area_size=(100.0, 100.0),
            snr_clip=(-30.0, 120.0),  # 개선: 넓은 SNR 범위
            episode_length=100,
            seed=42
        )
        return UAEnv(cfg)
    
    return create_original_env(), create_improved_env()

def visualize_snr_clipping_problem():
    """SNR 클리핑 문제 시각화"""
    print("=== SNR 클리핑 문제 시각화 ===")
    
    # 두 환경 생성
    original_env = create_problem_scenarios()[0]
    improved_env = create_problem_scenarios()[1]
    
    # 동일한 시드로 초기화
    original_obs = original_env.reset(seed=42)
    improved_obs = improved_env.reset(seed=42)
    
    # SNR 분포 비교
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 기존 환경 SNR 분포
    axes[0].hist(original_obs.flatten(), bins=30, alpha=0.7, color='red', edgecolor='black')
    axes[0].axvline(x=50, color='red', linestyle='--', linewidth=2, label='Clipping at 50dB')
    axes[0].set_xlabel('SNR (dB)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('기존 환경: SNR 클리핑 문제\n(50dB에서 잘림)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 개선된 환경 SNR 분포
    axes[1].hist(improved_obs.flatten(), bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[1].axvline(x=120, color='green', linestyle='--', linewidth=2, label='Clipping at 120dB')
    axes[1].set_xlabel('SNR (dB)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('개선된 환경: 넓은 SNR 범위\n(120dB까지 허용)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 통계 정보 출력
    print(f"기존 환경 SNR 범위: {original_obs.min():.2f} ~ {original_obs.max():.2f} dB")
    print(f"개선된 환경 SNR 범위: {improved_obs.min():.2f} ~ {improved_obs.max():.2f} dB")
    print(f"기존 환경 SNR 표준편차: {original_obs.std():.2f} dB")
    print(f"개선된 환경 SNR 표준편차: {improved_obs.std():.2f} dB")

def visualize_data_consistency_problem():
    """데이터 일관성 문제 시각화"""
    print("\n=== 데이터 일관성 문제 시각화 ===")
    
    # 기존 환경에서 여러 에피소드 실행
    original_env = create_problem_scenarios()[0]
    policy = MaxSNRWithFallbackPolicy()
    
    # 3개 에피소드 실행
    episode_data = []
    for episode in range(3):
        obs = original_env.reset(seed=episode + 100)  # 다른 시드
        episode_rewards = []
        
        for step in range(20):  # 20스텝만 실행
            actions = policy.act(obs)
            obs, reward, done, info = original_env.step(actions)
            episode_rewards.append(reward)
            if done:
                break
        
        episode_data.append(episode_rewards)
    
    # 보상 패턴 비교
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 에피소드별 보상 패턴
    for i, rewards in enumerate(episode_data):
        axes[0].plot(rewards, label=f'Episode {i+1}', alpha=0.8, linewidth=2)
    
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Sum Rate (bits/s)')
    axes[0].set_title('기존 환경: 에피소드별 보상 패턴\n(시드마다 다른 결과)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 보상 분포 비교
    all_rewards = np.concatenate(episode_data)
    axes[1].hist(all_rewards, bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[1].set_xlabel('Sum Rate (bits/s)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('기존 환경: 보상 분포\n(불안정한 패턴)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"에피소드별 보상 표준편차: {[np.std(rewards) for rewards in episode_data]}")
    print(f"전체 보상 표준편차: {np.std(all_rewards):.2e}")

def visualize_ue_interaction_loss():
    """UE 간 상호작용 정보 손실 문제 시각화"""
    print("\n=== UE 간 상호작용 정보 손실 문제 시각화 ===")
    
    # 기존 환경과 개선된 환경 비교
    original_env = create_problem_scenarios()[0]
    improved_env = create_problem_scenarios()[1]
    
    # 동일한 시드로 초기화
    original_obs = original_env.reset(seed=42)
    improved_obs = improved_env.reset(seed=42)
    
    # 정책 실행
    policy = MaxSNRWithFallbackPolicy()
    original_actions = policy.act(original_obs)
    improved_actions = policy.act(improved_obs)
    
    # 환경 스텝 실행
    original_next_obs, original_reward, original_done, original_info = original_env.step(original_actions)
    improved_next_obs, improved_reward, improved_done, improved_info = improved_env.step(improved_actions)
    
    # UE 위치와 연결 상태 시각화
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # 기존 환경 (문제 상황)
    plot_ue_sbs_layout(axes[0], original_env, original_actions, original_info, 
                       "기존 환경: UE별 분리된 데이터\n(상호작용 정보 손실)")
    
    # 개선된 환경
    plot_ue_sbs_layout(axes[1], improved_env, improved_actions, improved_info, 
                       "개선된 환경: 통합 데이터\n(상호작용 정보 보존)")
    
    plt.tight_layout()
    plt.show()
    
    # 데이터 구조 비교
    print("=== 데이터 구조 비교 ===")
    print("기존 환경:")
    print(f"  - UE별 개별 데이터만 존재")
    print(f"  - UE 간 상호작용 정보 없음")
    print(f"  - admitted_matrix 정보 손실")
    
    print("\n개선된 환경:")
    print(f"  - UE별 개별 데이터 + 통합 환경 데이터")
    print(f"  - UE 간 상호작용 정보 보존")
    print(f"  - admitted_matrix: {improved_info['admitted'].shape}")
    print(f"  - UE 위치: {improved_env.ue_pos.shape}")
    print(f"  - SBS 위치: {improved_env.sbs_pos.shape}")

def plot_ue_sbs_layout(ax, env, actions, info, title):
    """UE와 SBS 레이아웃 플롯"""
    W, H = env.cfg.area_size
    
    # SBS 위치 및 커버리지
    colors = ['red', 'blue', 'green']
    for i, (pos, color) in enumerate(zip(env.sbs_pos, colors)):
        # SBS 위치 (삼각형)
        ax.scatter(pos[0], pos[1], c=color, marker='^', s=200, 
                  edgecolors='black', linewidth=2, label=f'SBS{i+1}')
        
        # 커버리지 원
        coverage_radius = 30.0
        circle = patches.Circle(pos, coverage_radius, fill=False, 
                              linestyle='--', color=color, alpha=0.7)
        ax.add_patch(circle)
    
    # UE 위치 및 연결 상태
    admitted = info['admitted']
    for ue_id in range(env.cfg.num_ue):
        ue_pos = env.ue_pos[ue_id]
        action = actions[ue_id]
        
        if action > 0:  # SBS 선택됨
            sbs_idx = action - 1
            if admitted[ue_id, sbs_idx] == 1:  # 서빙 허용됨
                ax.scatter(ue_pos[0], ue_pos[1], c='green', s=100, alpha=0.8)
                # 연결선 그리기
                sbs_pos = env.sbs_pos[sbs_idx]
                ax.plot([ue_pos[0], sbs_pos[0]], [ue_pos[1], sbs_pos[1]], 
                       'gray', linestyle='--', alpha=0.6)
            else:
                ax.scatter(ue_pos[0], ue_pos[1], c='orange', s=100, alpha=0.6)
        else:
            ax.scatter(ue_pos[0], ue_pos[1], c='red', s=100, alpha=0.6)
        
        # UE 라벨
        ax.annotate(f'UE{ue_id}', (ue_pos[0], ue_pos[1]), 
                   xytext=(3, 3), textcoords='offset points', fontsize=7, alpha=0.8)
    
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

def visualize_policy_diversity_problem():
    """정책 다양성 부족 문제 시각화"""
    print("\n=== 정책 다양성 부족 문제 시각화 ===")
    
    # 기존 환경에서 단일 정책 사용
    env = create_problem_scenarios()[0]
    policy = MaxSNRWithFallbackPolicy()
    
    # 100번 실행하여 액션 분포 확인
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    for _ in range(100):
        obs = env.reset(seed=np.random.randint(1000))
        actions = policy.act(obs)
        for action in actions:
            action_counts[action] += 1
    
    # 액션 분포 시각화
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 단일 정책 액션 분포
    action_labels = ['No Service', 'SBS1', 'SBS2', 'SBS3']
    axes[0].bar(action_counts.keys(), action_counts.values(), alpha=0.7, color='red', edgecolor='black')
    axes[0].set_xlabel('Action')
    axes[0].set_ylabel('Count')
    axes[0].set_title('기존 환경: 단일 정책 액션 분포\n(MaxSNR 정책만 사용)')
    axes[0].set_xticks(list(action_counts.keys()))
    axes[0].set_xticklabels(action_labels)
    axes[0].grid(True, alpha=0.3)
    
    # 다양한 정책 시뮬레이션 (개선된 상황)
    diverse_action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    # 여러 정책을 실제로 사용
    max_snr_policy = MaxSNRWithFallbackPolicy()
    random_policy = RandomPolicy(env.cfg.num_sbs)
    epsilon_policy = EpsilonGreedyPolicy(env.cfg.num_sbs, epsilon=0.3)
    load_balancing_policy = LoadBalancingPolicy(env.cfg.num_sbs, env.cfg.beam_limits)
    
    policies = [max_snr_policy, random_policy, epsilon_policy, load_balancing_policy]
    
    for _ in range(100):
        obs = env.reset(seed=np.random.randint(1000))
        
        # 랜덤하게 정책 선택
        selected_policy = np.random.choice(policies)
        actions = selected_policy.act(obs)
        
        for action in actions:
            diverse_action_counts[action] += 1
    
    # 다양한 정책 액션 분포
    axes[1].bar(diverse_action_counts.keys(), diverse_action_counts.values(), alpha=0.7, color='green', edgecolor='black')
    axes[1].set_xlabel('Action')
    axes[1].set_ylabel('Count')
    axes[1].set_title('개선된 환경: 다양한 정책 액션 분포\n(4가지 정책 사용)')
    axes[1].set_xticks(list(diverse_action_counts.keys()))
    axes[1].set_xticklabels(action_labels)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=== 정책 다양성 비교 ===")
    print("기존 환경 (단일 정책):")
    for action, count in action_counts.items():
        print(f"  {action_labels[action]}: {count}")
    
    print("\n개선된 환경 (다양한 정책):")
    for action, count in diverse_action_counts.items():
        print(f"  {action_labels[action]}: {count}")

def main():
    """메인 함수"""
    print("UA 환경 문제 상황 시각화 시작")
    print("=" * 50)
    
    # 디렉토리 생성 (이미지 저장하지 않음)
    
    # 1. SNR 클리핑 문제
    visualize_snr_clipping_problem()
    
    # 2. 데이터 일관성 문제
    visualize_data_consistency_problem()
    
    # 3. UE 간 상호작용 정보 손실 문제
    visualize_ue_interaction_loss()
    
    # 4. 정책 다양성 부족 문제
    visualize_policy_diversity_problem()
    
    print("\n" + "=" * 50)
    print("모든 문제 상황 시각화 완료!")

if __name__ == "__main__":
    main()
