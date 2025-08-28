# UA (User Association) - Federated Learning for Wireless Networks

## 🌐 Environment
- **Network Topology**: 20 UEs, 3 SBSs, 100m x 100m area
- **Frequency**: 28 GHz (mmWave)
- **Channel Model**: Path loss + Shadowing + Fading
- **Path Loss**: 61.34 dB at 1m, 28 GHz
- **Shadowing**: -16.27 to 12.71 dB (σ = 8 dB)
- **Fading**: -7.40 to 7.64 dB (Rayleigh)
- **SNR Range**: 8.80 ~ 59.62 dB
- **Episode Length**: 100 steps
- **Evaluation**: 20 episodes

## 🧠 Algorithms

### FedCombo (Standard)
- **Architecture**: World Model + Q-Network (Independent per UE)
- **Training**: Sequential UE training
- **World Model**: 100 epochs, patience 10, hidden_dim=64
- **Q-Network**: 300 epochs, patience 10, hidden_dim=64
- **Normalization**: Z-score (SNR), Raw rewards
- **Learning Rate**: World Model (1e-3), Q-Network (1e-3)
- **Batch Size**: 32

### FedCombo-Federated
- **Architecture**: Federated World Model + Individual Q-Networks
- **Training**: FedAvg for World Models, Parallel execution
- **Federated Rounds**: 100 max, Early stopping (patience 5)
- **Local Epochs**: 5 per round
- **Q-Network**: Individual training per UE
- **Parallel Workers**: ThreadPoolExecutor (max 8)
- **FedAvg Frequency**: Every round

## 🎯 Key Features

### Action Space & Policy
- **Action Space**: 0 (no service) + SBS selection (1-3)
- **Policy Type**: Stochastic with World Model planning
- **Planning Horizon**: 3 steps ahead
- **Fallback**: Max-SNR when World Model fails

### Hybrid Reward Function
- **Personal Reward**: `log10(max(original_reward, 1e6))` (0 if original=0)
- **Joint Reward**: `tanh(difference_reward / scale_factor)`
  - `scale_factor = max(1e8, |difference_reward| * 0.1)`
- **Combination**: `sign(personal + joint) * sqrt(personal² + joint²)`
- **Difference Reward**: `G_t - G_t^(-i)` (system performance without UE i)
- **G(-i) Calculation**: Max-SNR re-optimization when UE i removed

### Early Stopping Mechanism
- **Local Training**: Patience 10 (World Model), 10 (Q-Network)
- **Federated Training**: Patience 5 (global loss improvement)
- **Stopping Criterion**: Consecutive epochs without loss improvement

### Data Collection & Preprocessing
- **Policy**: MaxSNR with Fallback
- **Dataset**: 1660 samples per UE
- **Normalization**: Z-score for SNR, Raw for rewards
- **Buffer**: Rollout buffer with (obs, action, reward) tuples

## 📊 Output Files
- `step_by_step_comparison.csv`: Per-step performance comparison
- `episode_summary.csv`: Episode-wise returns and statistics
- `detailed_actions_and_connections.csv`: Action details and connectivity
- `experiment_summary.json`: Overall statistics and confidence intervals
- `figs/`: Performance plots (step overlay, cumulative advantage, episode returns)

## 🚀 Usage
```bash
# Standard FedCombo
python ua_marl.py --fc_train --fc_type standard --save_plots

# Federated FedCombo
python ua_marl.py --fc_train --fc_type federated --save_plots

# Evaluation only (no training)
python ua_marl.py --fc_type [standard|federated] --save_plots
```

## 🔧 Dependencies
- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computations
- **Matplotlib**: Plotting and visualization
- **Pandas**: Data manipulation
- **Conda environment**: `d4rl` (includes PyTorch, CUDA support)

## 📈 Performance Metrics
- **Sum Rate**: Total system throughput
- **Admit Rate**: Percentage of connected UEs
- **Fairness**: Jain's fairness index
- **Return**: Cumulative reward per episode
- **Confidence Intervals**: 95% CI for statistical significance
