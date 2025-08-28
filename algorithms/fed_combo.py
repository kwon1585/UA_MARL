# algorithms/fed_combo.py
from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Optional, Dict, Tuple

class WorldModel(nn.Module):
    """월드모델: (obs, action) -> reward 예측 (Planning용)"""
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # 인코더: obs + action -> hidden
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 예측기: hidden -> reward만
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # reward만
        )
        
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # obs: (batch, obs_dim), action: (batch, action_dim)
        x = torch.cat([obs, action], dim=1)
        hidden = self.encoder(x)
        reward = self.predictor(hidden)
        
        return reward

class QNetwork(nn.Module):
    """Q-네트워크: obs -> Q-values for each action"""
    def __init__(self, obs_dim: int, num_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.q_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.q_net(obs)

class FedComboPolicy:
    """
    FedCombo: 각 UE별로 독립적인 월드모델 + 오프라인 롤아웃 + 정책 학습
    """
    def __init__(self, model_dir: Optional[str] = None, num_ue: int = 20):
        self.model_dir = model_dir
        self.num_ue = num_ue
        self.is_trained = False
        
        # UE별 모델들 (각 UE마다 독립적인 모델)
        self.ue_models: Dict[int, Dict] = {}
        
        # 하이퍼파라미터
        self.hidden_dim = 128
        self.lr_wm = 1e-3
        self.lr_q = 1e-3
        self.batch_size = 64
        self.epochs_wm = 100
        self.epochs_q = 300
        
        # 정규화 파라미터 (SNR만, 보상은 원본 그대로 사용)
        self.snr_mean = 0.0
        self.snr_std = 1.0
        
    def _compute_normalization_stats(self, dataset_files: List[str]) -> None:
        """데이터셋 전체에서 정규화 통계 계산"""
        print("[FedCombo] 정규화 통계 계산 중...")
        
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
                print(f"[FedCombo] SNR 표준편차가 너무 작음, 1.0으로 설정")
            elif self.snr_std < 0.1:
                # SNR 범위가 너무 좁으면 표준편차를 인위적으로 확대
                self.snr_std = max(0.1, abs(self.snr_mean) * 0.1)
                print(f"[FedCombo] SNR 범위가 좁음, 표준편차를 {self.snr_std:.4f}로 조정")
        
        print(f"[FedCombo] SNR: mean={self.snr_mean:.4f}, std={self.snr_std:.4f}")
        print(f"[FedCombo] 보상 정규화 제거 - 원본 보상 그대로 사용")
    
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
        """단일 UE의 롤아웃 버퍼 생성 (next_obs 제거)"""
        buffer = []
        
        for i in range(len(obs)):
            # 현재 상태
            current_obs = obs[i]  # (S,) - 정규화된 SNR
            action = acts[i]  # scalar
            reward = rews[i]  # scalar - 정규화된 보상
            
            # 액션을 one-hot으로 변환
            action_onehot = np.zeros(num_sbs + 1)  # 0=미서빙, 1~S=SBS
            action_onehot[action] = 1
            
            # 버퍼에 추가 (next_obs 제거, 보상은 정규화된 값)
            buffer.append({
                'obs': current_obs.astype(np.float32),
                'action': action_onehot.astype(np.float32),
                'reward': np.array([reward], dtype=np.float32)
            })
        
        return buffer
    
    def _train_ue_world_model(self, ue_id: int, buffer: List[Dict], device: str):
        """특정 UE의 월드모델 학습"""
        if not buffer:
            return None, None
            
        # 데이터 준비 (next_obs 제거)
        obs_list = [b['obs'] for b in buffer]
        actions_list = [b['action'] for b in buffer]
        rewards_list = [b['reward'] for b in buffer]
        
        obs = torch.tensor(np.array(obs_list), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(actions_list), dtype=torch.float32).to(device)
        rewards = torch.tensor(np.array(rewards_list), dtype=torch.float32).to(device)
        
        # 데이터로더 (next_obs 제거)
        dataset = TensorDataset(obs, actions, rewards)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 모델 초기화
        obs_dim = obs.shape[1]
        action_dim = actions.shape[1]
        world_model = WorldModel(obs_dim, action_dim, self.hidden_dim).to(device)
        optimizer_wm = optim.Adam(world_model.parameters(), lr=self.lr_wm)
        
        # 학습 (Early Stopping 적용)
        world_model.train()
        best_loss = float('inf')
        patience_counter = 0
        patience = 10  # 연속으로 10번 loss가 안 낮아지면 중단
        
        for epoch in range(self.epochs_wm):
            total_loss = 0.0
            for batch_obs, batch_acts, batch_rewards in dataloader:
                optimizer_wm.zero_grad()
                
                # 예측 (reward만)
                pred_rewards = world_model(batch_obs, batch_acts)
                
                # 손실 계산 (reward만)
                reward_loss = nn.MSELoss()(pred_rewards, batch_rewards)
                total_loss_wm = reward_loss
                
                # 역전파
                total_loss_wm.backward()
                optimizer_wm.step()
                
                total_loss += total_loss_wm.item()
            
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
        
        return world_model, optimizer_wm
    
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
    
    def _train_ue_q_network(self, ue_id: int, buffer: List[Dict], num_sbs: int, device: str):
        """특정 UE의 Q-네트워크 학습"""
        if not buffer:
            return None, None
            
        # 데이터 준비
        obs_list = [b['obs'] for b in buffer]
        actions_list = [b['action'] for b in buffer]
        rewards_list = [b['reward'] for b in buffer]
        
        obs = torch.tensor(np.array(obs_list), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(actions_list), dtype=torch.float32).to(device)
        rewards = torch.tensor(np.array(rewards_list), dtype=torch.float32).to(device)
        
        # 액션을 인덱스로 변환
        action_indices = torch.argmax(actions, dim=1)
        
        # 데이터로더
        dataset = TensorDataset(obs, action_indices, rewards)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 모델 초기화
        obs_dim = obs.shape[1]
        num_actions = num_sbs + 1  # 0=미서빙, 1~S=SBS
        q_network = QNetwork(obs_dim, num_actions, self.hidden_dim).to(device)
        optimizer_q = optim.Adam(q_network.parameters(), lr=self.lr_q)
        
        # 학습 (Early Stopping 적용)
        q_network.train()
        best_loss = float('inf')
        patience_counter = 0
        patience = 10  # 연속으로 3번 loss가 안 낮아지면 중단
        
        for epoch in range(self.epochs_q):
            total_loss = 0.0
            for batch_obs, batch_acts, batch_rewards in dataloader:
                optimizer_q.zero_grad()
                
                # Q-값 예측
                q_values = q_network(batch_obs)
                
                # 선택된 액션의 Q-값
                q_selected = q_values.gather(1, batch_acts.unsqueeze(1))
                
                # 손실 계산 (MSE with rewards as targets)
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
            
            if (epoch + 1) % 20 == 0:
                print(f"UE {ue_id} Q-Network Epoch {epoch+1}/{self.epochs_q}, Loss: {avg_loss:.6f}, Patience: {patience_counter}/{patience}")
            
            # Early stopping 조건 확인
            if patience_counter >= patience:
                print(f"UE {ue_id} Q-Network Early Stopping at epoch {epoch+1} (Loss: {avg_loss:.6f})")
                break
        
        return q_network, optimizer_q
    
    def train(self,
              dataset_files: List[str],
              num_ue: int,
              num_sbs: int,
              seed: int = 1,
              device: str = "cpu",
              **kwargs):
        """각 UE별로 독립적인 FedCombo 학습"""
        print(f"[FedCombo] UE별 독립 학습 시작: {len(dataset_files)}개 데이터셋, {num_ue}UE, {num_sbs}SBS")
        
        # 시드 설정
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # 1. 전체 데이터셋에서 정규화 통계 계산
        self._compute_normalization_stats(dataset_files)
        
        # 각 UE별로 독립적으로 학습
        for ue_id in range(num_ue):
            dataset_file = dataset_files[ue_id] if ue_id < len(dataset_files) else None
            
            if dataset_file and os.path.exists(dataset_file):
                print(f"[FedCombo] UE {ue_id} 학습 시작...")
                
                # 2. UE별 데이터 로드 및 정규화
                obs, acts, rews, next_obs = self._load_ue_dataset(dataset_file)
                if len(obs) == 0:
                    print(f"[FedCombo] UE {ue_id}: 데이터 없음, 건너뜀")
                    continue
                
                print(f"[FedCombo] UE {ue_id} 데이터 로드 완료: {len(obs)}개 샘플")
                print(f"[FedCombo] UE {ue_id} 정규화된 SNR 범위: [{obs.min():.4f}, {obs.max():.4f}]")
                print(f"[FedCombo] UE {ue_id} 정규화된 보상 범위: [{rews.min():.4f}, {rews.max():.4f}]")
                
                # 3. UE별 롤아웃 버퍼 생성 (next_obs 제거)
                rollout_buffer = self._create_ue_rollout_buffer(
                    obs, acts, rews, num_sbs
                )
                print(f"[FedCombo] UE {ue_id} 롤아웃 버퍼 생성 완료: {len(rollout_buffer)}개 샘플")
                
                # 4. UE별 월드모델 학습
                print(f"[FedCombo] UE {ue_id} 월드모델 학습 시작...")
                world_model, optimizer_wm = self._train_ue_world_model(ue_id, rollout_buffer, device)
                
                # 5. UE별 Q-네트워크 학습
                print(f"[FedCombo] UE {ue_id} Q-네트워크 학습 시작...")
                q_network, optimizer_q = self._train_ue_q_network(ue_id, rollout_buffer, num_sbs, device)
                
                # 6. UE별 모델 저장
                if world_model is not None and q_network is not None:
                    self.ue_models[ue_id] = {
                        'world_model': world_model,
                        'q_network': q_network,
                        'optimizer_wm': optimizer_wm,
                        'optimizer_q': optimizer_q,
                        'rollout_buffer': rollout_buffer,
                        'is_trained': True
                    }
                    print(f"[FedCombo] UE {ue_id} 학습 완료!")
                else:
                    print(f"[FedCombo] UE {ue_id} 학습 실패")
            else:
                print(f"[FedCombo] UE {ue_id}: 데이터셋 파일 없음, 건너뜀")
        
        # 7. 전체 모델 저장
        if self.model_dir:
            os.makedirs(self.model_dir, exist_ok=True)
            self.save(self.model_dir)
        
        self.is_trained = any(model.get('is_trained', False) for model in self.ue_models.values())
        print(f"[FedCombo] 전체 학습 완료! {sum(1 for m in self.ue_models.values() if m.get('is_trained', False))}/{num_ue} UE 학습됨")
        return self

    def act(self, obs_us: np.ndarray) -> np.ndarray:
        """각 UE별로 독립적인 정책으로 액션 선택"""
        if not self.is_trained:
            # 안전 fallback: Max-SNR (원본 SNR 사용)
            return 1 + np.argmax(obs_us, axis=1)
        
        U = obs_us.shape[0]  # UE 수
        actions = np.zeros(U, dtype=np.int64)
        
        # 각 UE별로 독립적으로 액션 선택
        for ue_id in range(U):
            if ue_id in self.ue_models and self.ue_models[ue_id].get('is_trained', False):
                try:
                    # World Model을 사용한 Planning으로 액션 선택
                    world_model = self.ue_models[ue_id]['world_model']
                    ue_obs = obs_us[ue_id]  # (S,)
                    
                    # Planning으로 최적 액션 선택
                    planned_action = self._plan_with_world_model(
                        ue_id, ue_obs, world_model, obs_us.shape[1], planning_horizon=3
                    )
                    
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
        """UE별 모델 저장"""
        if not self.is_trained:
            return
            
        save_dict = {
            'num_ue': self.num_ue,
            'ue_models': {},
            'normalization': {
                'snr_mean': self.snr_mean,
                'snr_std': self.snr_std
            }
        }
        
        for ue_id, ue_model in self.ue_models.items():
            if ue_model.get('is_trained', False):
                save_dict['ue_models'][str(ue_id)] = {
                    'world_model_state_dict': ue_model['world_model'].state_dict(),
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
        
        torch.save(save_dict, os.path.join(path, 'fedcombo_ue_models.pth'))
        print(f"[FedCombo] UE별 모델 저장 완료: {path}")

    def load(self, path: str):
        """UE별 모델 로드"""
        model_path = os.path.join(path, 'fedcombo_ue_models.pth')
        if not os.path.exists(model_path):
            print(f"[FedCombo] UE별 모델 파일 없음: {model_path}")
            return self
            
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 정규화 파라미터 복원 (SNR만)
        if 'normalization' in checkpoint:
            norm = checkpoint['normalization']
            self.snr_mean = norm.get('snr_mean', self.snr_mean)
            self.snr_std = norm.get('snr_std', self.snr_std)
            print(f"[FedCombo] 정규화 설정 복원: SNR(mean={self.snr_mean:.4f}, std={self.snr_std:.4f})")
        
        # UE별 모델 복원
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
                
                # 모델 복원
                obs_dim = 3  # 기본값 (SBS 수)
                action_dim = 4  # 기본값 (SBS 수 + 1)
                num_actions = 4  # 기본값 (SBS 수 + 1)
                
                world_model = WorldModel(obs_dim, action_dim, self.hidden_dim)
                world_model.load_state_dict(ue_checkpoint['world_model_state_dict'])
                
                q_network = QNetwork(obs_dim, num_actions, self.hidden_dim)
                q_network.load_state_dict(ue_checkpoint['q_network_state_dict'])
                
                self.ue_models[ue_id] = {
                    'world_model': world_model,
                    'q_network': q_network,
                    'is_trained': True
                }
        
        self.is_trained = any(model.get('is_trained', False) for model in self.ue_models.values())
        print(f"[FedCombo] UE별 모델 로드 완료: {path}, {len(self.ue_models)}개 UE 모델")
        return self
