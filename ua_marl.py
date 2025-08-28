# ua_marl.py
# UA Head-to-Head Compare with FedCombo (Standard/Federated)
# 
# 사용법:
#   # 기존 FedCombo 사용
#   python ua_marl.py --fc_train --fc_type standard
#   
#   # 연합학습 FedCombo 사용  
#   python ua_marl.py --fc_train --fc_type federated
#   
#   # 저장된 모델 사용
#   python ua_marl.py --fc_model_dir models/fedcombo --fc_type federated
#
from __future__ import annotations
import argparse, os, json, csv, math
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from ua_env import UAEnv, UAConfig
from algorithms.max_snr import MaxSNRPolicy
from algorithms.fed_combo import FedComboPolicy
from algorithms.fed_combo_federated import FedComboFederatedPolicy

# ---------- CLI(기존 유지; 옵션 추가 안 함) ----------
def build_common_parser(desc: str = "UA Head-to-Head Compare"):
    p = argparse.ArgumentParser(description=desc)
    # paths
    p.add_argument("--dataset_root", type=str, default="datasets/ua_maxsnrfb")
    p.add_argument("--logdir", type=str, default="results")
    # env (UE == Agent)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--num_ue", type=int, default=20)
    p.add_argument("--num_sbs", type=int, default=3)
    p.add_argument("--beam_limits", type=int, nargs="+", default=[2,3,3])
    p.add_argument("--area_size", type=float, nargs=2, default=[100.0, 100.0])
    p.add_argument("--carrier_ghz", type=float, default=28.0)
    p.add_argument("--bandwidth_mhz", type=float, default=100.0)
    p.add_argument("--noise_figure_db", type=float, default=7.0)
    p.add_argument("--tx_power_dbm", type=float, default=30.0)
    p.add_argument("--pathloss_exp", type=float, default=2.0)
    p.add_argument("--shadowing_std_db", type=float, default=6.0)
    p.add_argument("--fading_scale_db", type=float, default=3.0)
    p.add_argument("--snr_clip", type=float, nargs=2, default=[-30.0, 70.0])
    # dataset
    p.add_argument("--num_episodes", type=int, default=5)
    # eval/report
    p.add_argument("--eval_episodes", type=int, default=20)
    p.add_argument("--episode_length", type=int, default=100)
    p.add_argument("--save_plots", action="store_true")
    # fed-combo
    p.add_argument("--fc_model_dir", type=str, default=None)
    p.add_argument("--fc_train", action="store_true", help="비교 전 FedCombo 학습 실행")
    p.add_argument("--fc_type", type=str, default="standard", choices=["standard", "federated"], 
                   help="FedCombo 타입: standard(기존) 또는 federated(연합학습)")
    return p

# ---------- 유틸 ----------
def ensure_dirs(*paths: str):
    for p in paths: os.makedirs(p, exist_ok=True)

def get_dataset_name(dataset_root: str) -> str:
    """데이터셋 루트 경로에서 데이터셋 이름 추출"""
    # datasets/ua_mixed_reward -> ua_mixed_reward
    return os.path.basename(dataset_root)

def outdirs(args) -> Tuple[str, str]:
    dataset_name = get_dataset_name(args.dataset_root)
    
    # FedCombo 타입에 따라 폴더 분리
    if args.fc_type == "federated":
        base = os.path.join(args.logdir, f"{dataset_name}_federated")
    else:
        base = os.path.join(args.logdir, dataset_name)
    
    fig = os.path.join(base, "figs")
    ensure_dirs(base, fig)
    return base, fig

def dataset_path_for_agent(args, i: int) -> str:
    return os.path.join(args.dataset_root, f"ua_agent_{i}.npz")

def build_env(args) -> UAEnv:
    ec = UAConfig(
        num_ue=args.num_ue,
        num_sbs=args.num_sbs,
        beam_limits=tuple(args.beam_limits),
        area_size=tuple(args.area_size),
        carrier_freq_ghz=args.carrier_ghz,
        bandwidth_mhz=args.bandwidth_mhz,
        noise_figure_db=args.noise_figure_db,
        tx_power_dbm=args.tx_power_dbm,
        pathloss_exp=args.pathloss_exp,
        shadowing_std_db=args.shadowing_std_db,
        fading_scale_db=args.fading_scale_db,
        snr_clip=tuple(args.snr_clip),
        episode_length=args.episode_length,
        seed=args.seed,
    )
    return UAEnv(ec)

def jains_index(x: np.ndarray) -> float:
    x = x[np.asarray(x) > 0.0]
    if x.size == 0: return 0.0
    s1 = float(np.sum(x)); s2 = float(np.sum(x*x)); n = float(x.size)
    return (s1*s1) / (n * s2 + 1e-12)

def _write_csv(path: str, rows: List[Dict]):
    if not rows:
        with open(path, "w") as f: f.write("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

def _ci95(vals: List[float]) -> Dict[str, float]:
    x = np.asarray(vals, dtype=np.float64)
    n = int(x.size)
    if n == 0: return {"low": 0.0, "high": 0.0}
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1)) if n > 1 else 0.0
    half = 1.96 * s / max(1.0, np.sqrt(n))
    return {"low": m - half, "high": m + half}

# ---------- 핵심: 동일 스텝 SNR로 동시 비교 ----------
def eval_on_snr(env: UAEnv, snr_db: np.ndarray, actions: np.ndarray) -> Dict[str, float]:
    # 내부 함수로 보상 구성 (env.step을 쓰지 않고 동일 SNR에서 두 정책 비교)
    admitted, _ = env._apply_beam_limits(actions, snr_db)
    sinr_lin, per_ue_rate = env._compute_rates(snr_db, admitted)
    sum_rate = float(per_ue_rate.sum())
    admit_rate = float((admitted.sum(axis=1) > 0).mean())
    
    # Fairness 측정: 연결 성공한 UE들만 대상으로
    # admitted.sum(axis=1) > 0인 UE들만 선택 (최소 하나의 SBS에 연결된 UE)
    connected_ue_mask = admitted.sum(axis=1) > 0
    if connected_ue_mask.any():
        # 연결 성공한 UE들의 rate만 사용
        connected_rates = per_ue_rate[connected_ue_mask]
        fairness = float(jains_index(connected_rates))
    else:
        # 연결된 UE가 없으면 fairness = 0
        fairness = 0.0
    
    return {"sum_rate": sum_rate, "admit_rate": admit_rate, "fairness_jain": fairness}

def head_to_head(env: UAEnv, pol_maxsnr, pol_fed, episodes: int, save_visualizations: bool = True) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """스텝별로 같은 SNR을 공유해 두 정책을 동시에 평가. env.reset으로 에피소드 경계만 설정."""
    U = env.cfg.num_ue
    steps_rows: List[Dict] = []
    ep_rows: List[Dict] = []
    visualization_data: List[Dict] = []
    
    for ep in range(episodes):
        env.reset(seed=env.cfg.seed + ep)  # UE/BS 고정(에피소드마다 재샘플), 페이딩은 스텝마다 새로
        ret_max, ret_fed = 0.0, 0.0
        
        for t in range(env.cfg.episode_length):
            # 같은 스텝 SNR을 두 정책이 공유
            snr_db = env._draw_snr_db()
            obs = snr_db  # (U,S)
            a_max = pol_maxsnr.act(obs)
            
            a_fed = pol_fed.act(obs)

            m_max = eval_on_snr(env, snr_db, a_max)
            m_fed = eval_on_snr(env, snr_db, a_fed)
            ret_max += m_max["sum_rate"]; ret_fed += m_fed["sum_rate"]


            
            steps_rows.append({
                "episode": ep, "step": t,
                "maxsnr_sum_rate": m_max["sum_rate"],
                "fed_sum_rate": m_fed["sum_rate"],
                "maxsnr_admit_rate": m_max["admit_rate"],
                "fed_admit_rate": m_fed["admit_rate"],
                "maxsnr_fairness": m_max["fairness_jain"],
                "fed_fairness": m_fed["fairness_jain"],
                "winner": "FedCombo" if m_fed["sum_rate"] > m_max["sum_rate"] else ("Max-SNR" if m_max["sum_rate"] > m_fed["sum_rate"] else "Tie"),
                "diff_sum_rate": m_fed["sum_rate"] - m_max["sum_rate"],
            })
            
            # 시각화 데이터 저장 (--save_plots 옵션 시에만)
            if save_visualizations:
                # Max-SNR 정책의 연결 결과 계산
                admitted_max, _ = env._apply_beam_limits(a_max, obs)
                maxsnr_connected = (admitted_max.sum(axis=1) > 0).tolist()
                maxsnr_sum_rate = float(env._compute_rates(obs, admitted_max)[1].sum())
                
                # FedCombo 정책의 연결 결과 계산
                admitted_fed, _ = env._apply_beam_limits(a_fed, obs)
                fedcombo_connected = (admitted_fed.sum(axis=1) > 0).tolist()
                fedcombo_sum_rate = float(env._compute_rates(obs, admitted_fed)[1].sum())
                
                visualization_data.append({
                    "episode": ep,
                    "step": t,
                    "maxsnr_actions": a_max.tolist(),
                    "fedcombo_actions": a_fed.tolist(),
                    "maxsnr_connected": maxsnr_connected,  # 실제 연결 상태
                    "fedcombo_connected": fedcombo_connected,  # 실제 연결 상태
                    "maxsnr_sum_rate": maxsnr_sum_rate,  # 실제 sum rate
                    "fedcombo_sum_rate": fedcombo_sum_rate,  # 실제 sum rate
                    "ue_positions": env.ue_pos.tolist(),
                    "sbs_positions": env.sbs_pos.tolist()
                })

        ep_rows.append({
            "episode": ep,
            "return_maxsnr": ret_max,
            "return_fed": ret_fed,
            "return_diff": ret_fed - ret_max
        })
    
    return ep_rows, steps_rows, visualization_data

# ---------- 플롯(고정 출력: 추가 옵션 없이 생성) ----------
def plot_ue_sbs_associations(env, obs, actions, episode, step, policy_name, out_png: str):
    """UE와 SBS의 연결 관계를 시각화"""
    plt.figure(figsize=(10, 8))
    
    # SBS 위치 (삼각형으로 표시)
    sbs_pos = env.sbs_pos
    for i, pos in enumerate(sbs_pos):
        plt.plot(pos[0], pos[1], '^', markersize=15, color='white', 
                markeredgecolor='black', markeredgewidth=2, label=f'SBS{i+1}' if i == 0 else "")
    
    # UE 위치 (원으로 표시)
    ue_pos = env.ue_pos
    for i, pos in enumerate(ue_pos):
        plt.plot(pos[0], pos[1], 'o', markersize=8, color='green', label=f'UE{i}' if i == 0 else "")
    
    # 연결 관계 표시 (액션에 따라)
    for ue_id, action in enumerate(actions):
        if action > 0:  # 액션이 0이 아닌 경우 (연결된 경우)
            sbs_id = action - 1  # 액션은 1부터 시작하므로 0부터로 변환
            ue_pos_ue = ue_pos[ue_id]
            sbs_pos_sbs = sbs_pos[sbs_id]
            
            # 연결선 그리기
            plt.plot([ue_pos_ue[0], sbs_pos_sbs[0]], 
                    [ue_pos_ue[1], sbs_pos_sbs[1]], 
                    '--', color='gray', alpha=0.7, linewidth=1)
    
    # SBS 커버리지 영역 표시 (대략적인 원)
    for i, pos in enumerate(sbs_pos):
        # 간단한 커버리지 원 (반지름 30)
        circle = plt.Circle(pos, 30, fill=False, linestyle='--', color='blue', alpha=0.3)
        plt.gca().add_patch(circle)
    
    plt.xlim(0, env.cfg.area_size[0])
    plt.ylim(0, env.cfg.area_size[1])
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'{policy_name} - Episode {episode}, Step {step}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()

def plot_step_overlay(steps_rows: List[Dict], out_png: str, ep: int = 0):
    xs = [r["step"] for r in steps_rows if r["episode"] == ep]
    ys_m = [r["maxsnr_sum_rate"] for r in steps_rows if r["episode"] == ep]
    ys_f = [r["fed_sum_rate"] for r in steps_rows if r["episode"] == ep]
    plt.figure(figsize=(7,4))
    plt.plot(xs, ys_m, label="Max-SNR", linewidth=1.5)
    plt.plot(xs, ys_f, label="FedCombo", linewidth=1.5)
    plt.xlabel("Step"); plt.ylabel("Sum-Rate (bps)"); plt.title(f"Episode {ep}: Step-wise Sum-Rate")
    plt.legend(); plt.tight_layout(); plt.savefig(out_png); plt.close()

def plot_cumdiff(steps_rows: List[Dict], out_png: str, ep: int = 0):
    ys = [r["diff_sum_rate"] for r in steps_rows if r["episode"] == ep]
    xs = list(range(len(ys)))
    cum = np.cumsum(ys)
    plt.figure(figsize=(7,4))
    plt.plot(xs, cum, linewidth=2)
    plt.xlabel("Step"); plt.ylabel("Cumulative (Fed - Max) Sum-Rate")
    plt.title(f"Episode {ep}: Cumulative Advantage (FedCombo vs Max-SNR)")
    plt.tight_layout(); plt.savefig(out_png); plt.close()

def plot_episode_bar(ep_rows: List[Dict], out_png: str):
    means = [np.mean([r["return_maxsnr"] for r in ep_rows]),
             np.mean([r["return_fed"] for r in ep_rows])]
    stds  = [np.std([r["return_maxsnr"] for r in ep_rows], ddof=1) if len(ep_rows)>1 else 0.0,
             np.std([r["return_fed"] for r in ep_rows], ddof=1) if len(ep_rows)>1 else 0.0]
    plt.figure(figsize=(6,4))
    x = np.arange(2)
    plt.bar(x, means, yerr=stds, capsize=4, tick_label=["Max-SNR","FedCombo"])
    plt.ylabel("Episodic Return (sum of sum-rates)")
    plt.title("Mean ± SD over Episodes")
    plt.tight_layout(); plt.savefig(out_png); plt.close()

# ---------- 메인 ----------
def main():
    p = build_common_parser()
    args = p.parse_args()

    base_dir, fig_dir = outdirs(args)

    # 데이터 파일 존재 체크(UE별) — 학습형 알고리즘 대비
    ds_files = []
    for i in range(args.num_ue):
        pth = dataset_path_for_agent(args, i)
        if os.path.exists(pth): ds_files.append(pth)

    # 알고리즘 준비
    pol_max = MaxSNRPolicy().train()
    
    # FedCombo 정책 선택 (기존 또는 연합학습)
    if args.fc_type == "federated":
        print(f"[info] FedCombo-Federated 정책 사용")
        pol_fed = FedComboFederatedPolicy(model_dir=args.fc_model_dir, num_ue=args.num_ue)
    else:
        print(f"[info] FedCombo-Standard 정책 사용")
        pol_fed = FedComboPolicy(model_dir=args.fc_model_dir)
    
    if args.fc_train:
        if args.fc_type == "federated":
            pol_fed.train(dataset_files=ds_files, num_sbs=args.num_sbs, device=args.device)
        else:
            pol_fed.train(dataset_files=ds_files, num_ue=args.num_ue, num_sbs=args.num_sbs, seed=args.seed, device=args.device)

    # 동일 환경에서 헤드-투-헤드
    env = build_env(args)
    ep_rows, steps_rows, visualization_data = head_to_head(env, pol_max, pol_fed, args.eval_episodes, 
                                                          save_visualizations=args.save_plots)

    # 저장: CSV 파일들을 더 직관적인 이름으로 저장
    _write_csv(os.path.join(base_dir, "step_by_step_comparison.csv"), steps_rows)
    _write_csv(os.path.join(base_dir, "episode_summary.csv"), ep_rows)
    
    # 시각화 데이터 저장 (--save_plots 옵션 시에만)
    if args.save_plots and visualization_data:
        _write_csv(os.path.join(base_dir, "detailed_actions_and_connections.csv"), visualization_data)
        print(f"[info] 상세 액션 및 연결 데이터 저장 완료: {len(visualization_data)}개 스텝")

    ret_max = [r["return_maxsnr"] for r in ep_rows]
    ret_fed = [r["return_fed"] for r in ep_rows]
    summ = {
        "episodes": args.eval_episodes,
        "maxsnr": {"mean": float(np.mean(ret_max)),
                   "std": float(np.std(ret_max, ddof=1)) if len(ret_max)>1 else 0.0,
                   "ci95": _ci95(ret_max)},
        "fedcombo": {"mean": float(np.mean(ret_fed)),
                     "std": float(np.std(ret_fed, ddof=1)) if len(ret_fed)>1 else 0.0,
                     "ci95": _ci95(ret_fed)},
        "avg_return_diff": float(np.mean(np.asarray(ret_fed) - np.asarray(ret_max))),
    }
    with open(os.path.join(base_dir, "experiment_summary.json"), "w") as f:
        json.dump({"args": vars(args), "summary": summ}, f, indent=2)

    # 그림(고정 출력: 추가 옵션 없이 생성)
    plot_step_overlay(steps_rows, os.path.join(fig_dir, "step_overlay_ep0.png"), ep=0)
    plot_cumdiff(steps_rows, os.path.join(fig_dir, "cumulative_advantage_ep0.png"), ep=0)
    plot_episode_bar(ep_rows, os.path.join(fig_dir, "episode_returns_bar.png"))

    print(f"[done] head-to-head results → {base_dir}")
    if args.save_plots and visualization_data:
        print(f"[info] 상세 데이터 및 시각화 → {fig_dir} (총 {len(visualization_data)}개 스텝 데이터)")

if __name__ == "__main__":
    main()
