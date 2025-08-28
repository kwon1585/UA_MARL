#!/usr/bin/env python3
"""
FedCombo with Federated Learning
- 병렬로 UE들의 World Model을 FedAvg
- Q-네트워크는 개별적으로 학습
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple, Optional
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# 기존 FedCombo에서 사용하는 모델들 import
from .fed_combo import WorldModel, QNetwork

class FedComboFederatedPolicy:
    """
    FedCombo with Federated Learning
    - World Model: 병렬 FedAvg로 연합학습
    - Q-Network: 개별 UE별로 독립 학습
    """
    
    def __init__(self, model_dir: Optional[str] = None, num_ue: int = 20):
        self.model_dir = model_dir
        self.num_ue = num_ue
        self.is_trained = False
        
        # UE별 모델들
        self.ue_models: Dict[int, Dict] = {}
        
        # 하이퍼파라미터
        self.hidden_dim = 128
        self.lr_wm = 1e-3
        self.lr_q = 1e-3
        self.batch_size = 64
        self.epochs_wm = 100
        self.epochs_q = 300
        
        # 연합학습 파라미터
        self.fed_rounds = 100  # 연합학습 라운드 수
        self.local_epochs = 5  # 각 UE당 로컬 학습 에포크 수
        self.fed_avg_freq = 1  # 몇 번의 로컬 학습 후 FedAvg할지
        
        
        # 정규화 파라미터 (SNR만, 보상은 원본 그대로 사용)
        self.snr_mean = 0.0
        self.snr_std = 1.0
        
        # 글로벌 World Model (FedAvg 결과)
        self.global_world_model = None
        
    def _compute_normalization_stats(self, dataset_files: List[str]) -> None:
        """데이터셋 전체에서 SNR 정규화 통계 계산"""
        print("[FedCombo-Federated] SNR 정규화 통계 계산 중...")
        
        all_snr = []
        
        for file_path in dataset_files:
            if os.path.exists(file_path):
                data = np.load(file_path)
                all_snr.extend(data['obs'].flatten())
        
        if all_snr:
            self.snr_mean = float(np.mean(all_snr))
            self.snr_std = float(np.std(all_snr))
            if self.snr_std < 1e-8:
                self.snr_std = 1.0
                print(f"[FedCombo-Federated] SNR 표준편차가 너무 작음, 1.0으로 설정")
            elif self.snr_std < 0.1:
                # SNR 범위가 너무 좁으면 표준편차를 인위적으로 확대
                self.snr_std = max(0.1, abs(self.snr_mean) * 0.1)
                print(f"[FedCombo-Federated] SNR 범위가 좁음, 표준편차를 {self.snr_std:.4f}로 조정")
        
        print(f"[FedCombo-Federated] SNR: mean={self.snr_mean:.4f}, std={self.snr_std:.4f}")
        print(f"[FedCombo-Federated] 보상 정규화 제거 - 원본 보상 그대로 사용")
    
    def _normalize_snr(self, snr_db: np.ndarray) -> np.ndarray:
        """SNR Z-score 정규화"""
        return (snr_db - self.snr_mean) / self.snr_std
    
    def _normalize_reward(self, reward: np.ndarray) -> np.ndarray:
        """보상 정규화 제거 - 원본 보상 그대로 사용"""
        return reward.astype(np.float32)
    
    def _load_ue_dataset(self, dataset_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """단일 UE의 데이터셋 로드 및 정규화"""
        if not os.path.exists(dataset_file):
            return np.array([]), np.array([]), np.array([]), np.ndarray([])
            
        data = np.load(dataset_file)
        obs = data['obs']  # (T, S) - SNR dB
        acts = data['act']  # (T,)
        rews = data['reward']  # (T,) - 전송률 bps
        next_obs = data['next_obs']  # (T, S) - SNR dB
        
        # SNR과 보상 모두 정규화 적용
        obs_norm = self._normalize_snr(obs)
        next_obs_norm = self._normalize_snr(next_obs)
        rews_norm = self._normalize_reward(rews)
        
        return obs_norm, acts, rews_norm, next_obs_norm
    
    def _create_ue_rollout_buffer(self, obs: np.ndarray, acts: np.ndarray, 
                                 rews: np.ndarray, num_sbs: int) -> List[Dict]:
        """단일 UE의 롤아웃 버퍼 생성"""
        buffer = []
        
        for i in range(len(obs)):
            # 현재 상태
            current_obs = obs[i]  # (S,) - 정규화된 SNR
            action = acts[i]  # scalar
            reward = rews[i]  # scalar - 정규화된 보상
            
            # 액션을 one-hot으로 변환
            action_onehot = np.zeros(num_sbs + 1)  # 0=미서빙, 1~S=SBS
            action_onehot[action] = 1
            
            # 버퍼에 추가
            buffer.append({
                'obs': current_obs.astype(np.float32),
                'action': action_onehot.astype(np.float32),
                'reward': np.array([reward], dtype=np.float32)
            })
        
        return buffer
    
    def _train_ue_world_model_local(self, ue_id: int, buffer: List[Dict], device: str, 
                                   global_model: Optional[WorldModel] = None) -> Tuple[WorldModel, float]:
        """단일 UE의 World Model 로컬 학습"""
        if not buffer:
            return None, float('inf')
            
        # 데이터 준비
        obs_list = [b['obs'] for b in buffer]
        actions_list = [b['action'] for b in buffer]
        rewards_list = [b['reward'] for b in buffer]
        
        obs = torch.tensor(np.array(obs_list), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(actions_list), dtype=torch.float32).to(device)
        rewards = torch.tensor(np.array(rewards_list), dtype=torch.float32).to(device)
        
        # 데이터로더
        dataset = TensorDataset(obs, actions, rewards)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 모델 초기화 (글로벌 모델이 있으면 복사)
        obs_dim = obs.shape[1]
        action_dim = actions.shape[1]
        
        if global_model is not None:
            world_model = copy.deepcopy(global_model)
        else:
            world_model = WorldModel(obs_dim, action_dim, self.hidden_dim).to(device)
        
        optimizer_wm = optim.Adam(world_model.parameters(), lr=self.lr_wm)
        
        # 로컬 학습 (Early Stopping 적용)
        world_model.train()
        best_loss = float('inf')
        patience_counter = 0
        patience = 10  # 연속으로 10번 loss가 안 낮아지면 중단
        
        for epoch in range(self.local_epochs):
            total_loss = 0.0
            for batch_obs, batch_acts, batch_rewards in dataloader:
                optimizer_wm.zero_grad()
                
                # 예측 (reward만)
                pred_rewards = world_model(batch_obs, batch_acts)
                
                # 손실 계산
                reward_loss = nn.MSELoss()(pred_rewards, batch_rewards)
                
                # 역전파
                reward_loss.backward()
                optimizer_wm.step()
                
                total_loss += reward_loss.item()
            
            avg_loss = total_loss / len(dataloader)
            
            # Early Stopping 체크
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"UE {ue_id} World Model Epoch {epoch+1}/{self.epochs_wm}, Loss: {avg_loss:.6f}, Patience: {patience_counter}/{patience}")
            
            # Early stopping 조건 확인
            if patience_counter >= patience:
                print(f"UE {ue_id} World Model Early Stopping at epoch {epoch+1} (Loss: {avg_loss:.6f})")
                break
        
        return world_model, best_loss
    
    def _federated_average_world_models(self, ue_world_models: Dict[int, WorldModel], 
                                       ue_buffer_sizes: Dict[int, int]) -> WorldModel:
        """UE들의 World Model을 FedAvg로 통합"""
        print("[FedCombo-Federated] World Model FedAvg 시작...")
        
        # 첫 번째 모델을 기준으로 글로벌 모델 생성
        first_model = next(iter(ue_world_models.values()))
        global_model = copy.deepcopy(first_model)
        
        # 전체 데이터 크기 계산
        total_samples = sum(ue_buffer_sizes.values())
        
        # FedAvg 수행
        with torch.no_grad():
            for param_name, global_param in global_model.named_parameters():
                # 가중 평균 계산
                weighted_sum = torch.zeros_like(global_param)
                
                for ue_id, ue_model in ue_world_models.items():
                    ue_param = ue_model.state_dict()[param_name]
                    weight = ue_buffer_sizes[ue_id] / total_samples
                    weighted_sum += weight * ue_param
                
                global_param.copy_(weighted_sum)
        
        print(f"[FedCombo-Federated] World Model FedAvg 완료 (총 {len(ue_world_models)}개 UE)")
        return global_model
    
    def _train_ue_q_network(self, ue_id: int, buffer: List[Dict], num_sbs: int, device: str):
        """특정 UE의 Q-네트워크 학습 (개별 학습)"""
        if not buffer:
            return None, None
            
        # 데이터 준비
        obs_list = [b['obs'] for b in buffer]
        actions_list = [b['action'] for b in buffer]
        rewards_list = [b['reward'] for b in buffer]
        
        obs = torch.tensor(np.array(obs_list), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(actions_list), dtype=torch.float32).to(device)
        rewards = torch.tensor(np.array(rewards_list), dtype=torch.float32).to(device)
        
        # 데이터로더
        dataset = TensorDataset(obs, actions, rewards)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 모델 초기화
        obs_dim = obs.shape[1]
        num_actions = num_sbs + 1
        
        q_network = QNetwork(obs_dim, num_actions, self.hidden_dim).to(device)
        optimizer_q = optim.Adam(q_network.parameters(), lr=self.lr_q)
        
        # 학습
        q_network.train()
        best_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(self.epochs_q):
            total_loss = 0.0
            for batch_obs, batch_acts, batch_rewards in dataloader:
                optimizer_q.zero_grad()
                
                # Q-값 예측
                q_values = q_network(batch_obs)
                
                # 선택된 액션의 Q-값 추출
                selected_actions = torch.argmax(batch_acts, dim=1)
                q_selected = q_values.gather(1, selected_actions.unsqueeze(1))
                
                # 손실 계산 (MSE)
                loss = nn.MSELoss()(q_selected, batch_rewards)
                
                # 역전파
                loss.backward()
                optimizer_q.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            
            # Early Stopping 체크
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"UE {ue_id} Q-Network Epoch {epoch+1}/{self.epochs_q}, Loss: {avg_loss:.6f}, Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"UE {ue_id} Q-Network Early Stopping at epoch {epoch+1} (Loss: {avg_loss:.6f})")
                break
        
        return q_network, optimizer_q
    
    def train(self, dataset_files: List[str], num_sbs: int, device: str = 'cpu'):
        """연합학습으로 FedCombo 모델 학습"""
        print(f"[FedCombo-Federated] 연합학습 시작 (UE: {self.num_ue}, SBS: {num_sbs})")
        
        # 1. 정규화 통계 계산
        self._compute_normalization_stats(dataset_files)
        
        # 2. UE별 데이터 로드 및 버퍼 생성
        print("[FedCombo-Federated] UE별 데이터 로드 중...")
        ue_buffers = {}
        ue_buffer_sizes = {}
        
        for ue_id in range(self.num_ue):
            dataset_file = dataset_files[ue_id]
            obs, acts, rews, _ = self._load_ue_dataset(dataset_file)
            
            if len(obs) > 0:
                buffer = self._create_ue_rollout_buffer(obs, acts, rews, num_sbs)
                ue_buffers[ue_id] = buffer
                ue_buffer_sizes[ue_id] = len(buffer)
                print(f"  UE {ue_id}: {len(buffer)}개 샘플")
            else:
                print(f"  UE {ue_id}: 데이터 없음")
        
        # 3. 연합학습 (World Model)
        print(f"\n[FedCombo-Federated] World Model 연합학습 시작 (라운드: {self.fed_rounds})")
        
        # 초기 글로벌 모델 생성
        first_ue_id = next(iter(ue_buffers.keys()))
        obs_dim = len(ue_buffers[first_ue_id][0]['obs'])
        action_dim = num_sbs + 1
        self.global_world_model = WorldModel(obs_dim, action_dim, self.hidden_dim).to(device)
        
        # 연합학습 (손실이 안 줄어들 때까지 반복)
        fed_round = 0
        best_global_loss = float('inf')
        patience_counter = 0
        patience = 5  # 연속으로 3번 손실이 안 줄어들면 중단
        
        while fed_round < self.fed_rounds:
            fed_round += 1
            print(f"\n--- 연합학습 라운드 {fed_round} ---")
            
            # 병렬로 UE별 로컬 학습
            ue_world_models = {}
            ue_losses = {}
            
            with ThreadPoolExecutor(max_workers=min(8, len(ue_buffers))) as executor:
                # 로컬 학습 작업 제출
                future_to_ue = {}
                for ue_id, buffer in ue_buffers.items():
                    future = executor.submit(
                        self._train_ue_world_model_local, 
                        ue_id, buffer, device, self.global_world_model
                    )
                    future_to_ue[future] = ue_id
                
                # 결과 수집
                for future in as_completed(future_to_ue):
                    ue_id = future_to_ue[future]
                    try:
                        world_model, loss = future.result()
                        if world_model is not None:
                            ue_world_models[ue_id] = world_model
                            ue_losses[ue_id] = loss
                            print(f"  UE {ue_id}: 로컬 학습 완료 (Loss: {loss:.6f})")
                    except Exception as e:
                        print(f"  UE {ue_id}: 로컬 학습 실패 - {e}")
            
            # FedAvg로 글로벌 모델 업데이트
            if ue_world_models:
                self.global_world_model = self._federated_average_world_models(
                    ue_world_models, ue_buffer_sizes
                )
                
                # 평균 손실 계산
                avg_loss = np.mean(list(ue_losses.values()))
                print(f"  평균 로컬 손실: {avg_loss:.6f}")
                
                # Early Stopping 체크 (글로벌 손실 기준)
                if avg_loss < best_global_loss:
                    best_global_loss = avg_loss
                    patience_counter = 0
                    print(f"  [개선] 새로운 최고 손실: {best_global_loss:.6f}")
                else:
                    patience_counter += 1
                    print(f"  [유지] 손실 개선 없음 ({patience_counter}/{patience})")
                
                # Early stopping 조건 확인
                if patience_counter >= patience:
                    print(f"  [조기 종료] 연속 {patience}번 손실 개선 없음")
                    break
        
        # 4. Q-네트워크 개별 학습
        print(f"\n[FedCombo-Federated] Q-네트워크 개별 학습 시작...")
        
        for ue_id, buffer in ue_buffers.items():
            print(f"  UE {ue_id} Q-네트워크 학습 중...")
            q_network, optimizer_q = self._train_ue_q_network(ue_id, buffer, num_sbs, device)
            
            if q_network is not None:
                self.ue_models[ue_id] = {
                    'world_model': copy.deepcopy(self.global_world_model),
                    'q_network': q_network,
                    'is_trained': True
                }
                print(f"    완료")
            else:
                print(f"    실패")
        
        # 5. 학습 완료
        self.is_trained = len(self.ue_models) > 0
        print(f"\n[FedCombo-Federated] 연합학습 완료!")
        print(f"  총 연합학습 라운드: {fed_round}")
        print(f"  최종 평균 손실: {best_global_loss:.6f}")
        print(f"  학습된 UE 수: {len(self.ue_models)}")
        print(f"  글로벌 World Model: {'성공' if self.global_world_model else '실패'}")
    
    def _plan_with_world_model(self, ue_id: int, obs: np.ndarray, world_model: WorldModel, 
                              num_sbs: int, planning_horizon: int = 3) -> int:
        """World Model을 사용한 Planning으로 최적 액션 선택"""
        device = next(world_model.parameters()).device
        
        # 현재 상태를 정규화
        obs_norm = self._normalize_snr(obs.reshape(1, -1))
        obs_tensor = torch.tensor(obs_norm, dtype=torch.float32).to(device)
        
        best_action = 0
        best_total_reward = float('-inf')
        
        # 가능한 모든 액션에 대해 planning
        for action in range(num_sbs + 1):  # 0=미서빙, 1~S=SBS
            total_reward = 0.0
            current_obs = obs_tensor.clone()
            
            # Planning horizon만큼 미래 예측
            for step in range(planning_horizon):
                # 액션을 one-hot으로 변환
                action_onehot = torch.zeros(1, num_sbs + 1, device=device)
                action_onehot[0, action] = 1.0
                
                # World Model로 reward 예측
                with torch.no_grad():
                    predicted_reward = world_model(current_obs, action_onehot)
                    total_reward += predicted_reward.item()
                
                # 간단한 상태 전이 (실제로는 더 복잡할 수 있음)
                # 여기서는 현재 obs를 그대로 유지 (간단한 가정)
            
            # 최고 총 보상을 가진 액션 선택
            if total_reward > best_total_reward:
                best_total_reward = total_reward
                best_action = action
        
        return best_action
    
    def act(self, obs_us: np.ndarray) -> np.ndarray:
        """액션 선택 (연합학습된 World Model + 개별 Q-네트워크 사용)"""
        if not self.is_trained:
            # 학습되지 않은 경우 Max-SNR fallback
            actions = np.array([1 + np.argmax(obs_us[i]) for i in range(self.num_ue)])
            return actions
        
        actions = np.zeros(self.num_ue, dtype=int)
        
        for ue_id in range(self.num_ue):
            if ue_id in self.ue_models and self.ue_models[ue_id].get('is_trained', False):
                try:
                    # World Model로 planning
                    world_model = self.ue_models[ue_id]['world_model']
                    planned_action = self._plan_with_world_model(ue_id, obs_us[ue_id], world_model, obs_us.shape[1])
                    
                    # Q-네트워크로 액션 검증/개선
                    q_network = self.ue_models[ue_id]['q_network']
                    obs_norm = self._normalize_snr(obs_us[ue_id].reshape(1, -1))
                    obs_tensor = torch.tensor(obs_norm, dtype=torch.float32)
                    
                    with torch.no_grad():
                        q_values = q_network(obs_tensor)
                        q_action = torch.argmax(q_values).item()
                    
                    # World Model의 예측을 그대로 사용 (액션 0 포함)
                    actions[ue_id] = planned_action
                        
                except Exception as e:
                    # World Model 예측 실패 시 Max-SNR fallback
                    maxsnr_action = 1 + np.argmax(obs_us[ue_id])
                    actions[ue_id] = maxsnr_action
                    if ue_id < 3:  # 디버깅 로그
                        print(f"UE {ue_id}: World Model 예측 실패, Max-SNR fallback 사용: {maxsnr_action} (에러: {e})")
            else:
                # 학습되지 않은 UE는 Max-SNR fallback
                actions[ue_id] = 1 + np.argmax(obs_us[ue_id])
        
        return actions
    
    def save(self, path: str):
        """UE별 모델 및 글로벌 World Model 저장"""
        if not self.is_trained:
            return
            
        save_dict = {
            'num_ue': self.num_ue,
            'ue_models': {},
            'global_world_model': self.global_world_model.state_dict() if self.global_world_model else None,
            'normalization': {
                'snr_mean': self.snr_mean,
                'snr_std': self.snr_std
            },
            'fed_params': {
                'fed_rounds': self.fed_rounds,
                'local_epochs': self.local_epochs,
                'fed_avg_freq': self.fed_avg_freq
            }
        }
        
        for ue_id, ue_model in self.ue_models.items():
            if ue_model.get('is_trained', False):
                save_dict['ue_models'][str(ue_id)] = {
                    'q_network_state_dict': ue_model['q_network'].state_dict(),
                    'is_trained': True,
                    'hyperparams': {
                        'hidden_dim': self.hidden_dim,
                        'lr_wm': self.lr_wm,
                        'lr_q': self.lr_q,
                        'batch_size': self.batch_size,
                        'epochs_wm': self.epochs_wm,
                        'epochs_q': self.epochs_q
                    }
                }
        
        torch.save(save_dict, os.path.join(path, 'fedcombo_federated_models.pth'))
        print(f"[FedCombo-Federated] 모델 저장 완료: {path}")
    
    def load(self, path: str):
        """UE별 모델 및 글로벌 World Model 로드"""
        model_path = os.path.join(path, 'fedcombo_federated_models.pth')
        if not os.path.exists(model_path):
            print(f"[FedCombo-Federated] 모델 파일 없음: {model_path}")
            return self
            
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 정규화 파라미터 복원
        if 'normalization' in checkpoint:
            norm = checkpoint['normalization']
            self.snr_mean = norm.get('snr_mean', self.snr_mean)
            self.snr_std = norm.get('snr_std', self.snr_std)
            print(f"[FedCombo-Federated] 정규화 설정 복원: SNR(mean={self.snr_mean:.4f}, std={self.snr_std:.4f})")
        
        # 연합학습 파라미터 복원
        if 'fed_params' in checkpoint:
            fed_params = checkpoint['fed_params']
            self.fed_rounds = fed_params.get('fed_rounds', self.fed_rounds)
            self.local_epochs = fed_params.get('local_epochs', self.local_epochs)
            self.fed_avg_freq = fed_params.get('fed_avg_freq', self.fed_avg_freq)
            print(f"[FedCombo-Federated] 연합학습 파라미터 복원: rounds={self.fed_rounds}, local_epochs={self.local_epochs}")
        
        # 글로벌 World Model 복원
        if 'global_world_model' in checkpoint and checkpoint['global_world_model'] is not None:
            obs_dim = 3  # 기본값 (SBS 수)
            action_dim = 4  # 기본값 (SBS 수 + 1)
            self.global_world_model = WorldModel(obs_dim, action_dim, self.hidden_dim)
            self.global_world_model.load_state_dict(checkpoint['global_world_model'])
            print(f"[FedCombo-Federated] 글로벌 World Model 복원 완료")
        
        # UE별 Q-네트워크 복원
        for ue_id_str, ue_checkpoint in checkpoint['ue_models'].items():
            ue_id = int(ue_id_str)
            
            if ue_checkpoint.get('is_trained', False):
                # 하이퍼파라미터 복원
                hp = ue_checkpoint.get('hyperparams', {})
                self.hidden_dim = hp.get('hidden_dim', self.hidden_dim)
                self.lr_wm = hp.get('lr_wm', self.lr_wm)
                self.lr_q = hp.get('lr_q', self.lr_q)
                self.batch_size = hp.get('batch_size', self.batch_size)
                self.epochs_wm = hp.get('epochs_wm', self.epochs_wm)
                self.epochs_q = hp.get('epochs_q', self.epochs_q)
                
                # Q-네트워크 복원
                obs_dim = 3  # 기본값 (SBS 수)
                num_actions = 4  # 기본값 (SBS 수 + 1)
                
                q_network = QNetwork(obs_dim, num_actions, self.hidden_dim)
                q_network.load_state_dict(ue_checkpoint['q_network_state_dict'])
                
                self.ue_models[ue_id] = {
                    'world_model': copy.deepcopy(self.global_world_model),
                    'q_network': q_network,
                    'is_trained': True
                }
        
        self.is_trained = any(model.get('is_trained', False) for model in self.ue_models.values())
        print(f"[FedCombo-Federated] 모델 로드 완료: {path}, {len(self.ue_models)}개 UE 모델")
        return self
