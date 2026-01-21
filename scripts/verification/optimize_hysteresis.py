import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from dataclasses import dataclass

# =========================================================
#  Standalone Definitions (To avoid Isaac Lab dependencies)
# =========================================================

@dataclass
class PamHysteresisModelCfg:
    """PAMのヒステリシスモデル設定"""
    hysteresis_width: float = 0.1
    curve_shape_param: float = 2.0

class PamHysteresisModel(nn.Module):
    """
    Additive Hysteresis Model (Play Operator)
    圧力に対する「摩擦（一定の圧力幅）」を再現するモデル
    (porcaro_rl/actions/pam_dynamics.py からロジックを抜粋)
    """
    def __init__(self, cfg: PamHysteresisModelCfg, device: str = 'cpu'):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.last_output = None

    def reset_idx(self, env_ids: torch.Tensor):
        if self.last_output is not None:
            self.last_output[env_ids] = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Play Operator Logic:
        y(t) = min( x(t) + r, max( x(t) - r, y(t-1) ) )
        """
        # 初回初期化: 入力値に合わせて初期化
        if self.last_output is None or self.last_output.shape != x.shape:
            self.last_output = x.clone()

        # ヒステリシス幅 (全幅) から 半幅 (r) を計算
        r = self.cfg.hysteresis_width / 2.0
        
        # Play Operator 計算
        lower_bound = x - r
        upper_bound = x + r
        
        # 前回の値を上下限でクリップする
        output = torch.max(lower_bound, torch.min(upper_bound, self.last_output))
        
        # 状態更新
        self.last_output = output.clone()
        
        return output

# =========================================================
#  Main Optimization Script
# =========================================================

# --- 設定 ---
DATA_DIR = "./external_data/jetson_project"
TARGET_FREQ = "0.5Hz" # 準静的パラメータを決めるため、最も低速なデータを使用

def load_data():
    # ファイル検索
    pattern = os.path.join(DATA_DIR, f"data_exp2_hysteresis_{TARGET_FREQ}_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        # パスが見つからない場合のフォールバック（カレントディレクトリ確認）
        print(f"[WARN] No files found in {pattern}. Checking current dir...")
        files = sorted(glob.glob(f"data_exp2_hysteresis_{TARGET_FREQ}_*.csv"))
        if not files:
            raise FileNotFoundError(f"No files found for {TARGET_FREQ}")
    
    target_file = files[-1]
    print(f"[INFO] Loading: {target_file}")
    df = pd.read_csv(target_file)
    
    # 安定した後半データを使用 (最初の20%をカット)
    start_idx = int(len(df) * 0.2)
    df = df.iloc[start_idx:].reset_index(drop=True)
    
    # 入力: 差圧 (P_DF - P_F)
    # 出力: 角度 (Angle)
    u_data = (df['meas_pres_DF'] - df['meas_pres_F']).values
    y_data = df['meas_ang_wrist'].values
    
    return u_data, y_data

def run_model(width, u_tensor, device='cpu'):
    """指定されたwidthでモデルを実行"""
    cfg = PamHysteresisModelCfg(hysteresis_width=float(width))
    model = PamHysteresisModel(cfg, device=device)
    
    # 時系列シミュレーション
    # (Play Operatorは履歴依存なのでシーケンシャルに計算)
    
    # バッチサイズ1として処理するため次元追加せずにループで回す（可読性重視）
    # 初期化: 最初の入力値を初期状態とする
    model.last_output = torch.tensor([u_tensor[0]], dtype=torch.float32, device=device)
    
    h_vals = []
    # 高速化のため torch.no_grad は外側で適用済み前提
    for t in range(len(u_tensor)):
        x_t = u_tensor[t:t+1] # shape [1]
        h_t = model(x_t)
        h_vals.append(h_t.item())
        
    return np.array(h_vals)

def objective_function(params, u_data, y_data):
    """最小化する目的関数 (MSE)"""
    width = params[0]
    scale = params[1] # 角度への変換ゲイン
    bias = params[2]  # オフセット
    
    if width < 0: return 1e9 # 制約
    
    u_tensor = torch.tensor(u_data, dtype=torch.float32)
    h_eff = run_model(width, u_tensor)
    
    # 線形回帰: y_pred = scale * h_eff + bias
    y_pred = scale * h_eff + bias
    
    mse = np.mean((y_data - y_pred)**2)
    return mse

def main():
    try:
        u_data, y_data = load_data()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    print(f"[INFO] Optimizing hysteresis parameters for {len(u_data)} steps...")
    
    # 初期値推定 [width, scale, bias]
    # width: 0.1 (MPa), scale: 100 (deg/MPa), bias: -30 (deg)
    initial_guess = [0.1, 100.0, -30.0]
    
    # 最適化実行 (Nelder-Mead法)
    result = minimize(
        objective_function, 
        initial_guess, 
        args=(u_data, y_data),
        method='Nelder-Mead',
        tol=1e-4
    )
    
    best_width = result.x[0]
    best_scale = result.x[1]
    best_bias = result.x[2]
    
    print("-" * 40)
    print(f"Optimization Result:")
    print(f"  Best Hysteresis Width : {best_width:.4f} MPa")
    print(f"  Scale Factor          : {best_scale:.2f}")
    print(f"  Bias                  : {best_bias:.2f}")
    print(f"  Final MSE             : {result.fun:.4f}")
    print("-" * 40)
    
    # --- 結果のプロット ---
    u_tensor = torch.tensor(u_data, dtype=torch.float32)
    h_eff = run_model(best_width, u_tensor)
    y_pred = best_scale * h_eff + best_bias
    
    plt.figure(figsize=(12, 6))
    
    # 左: 時系列比較
    plt.subplot(1, 2, 1)
    plt.title("Time Series Comparison")
    plt.plot(y_data, label="Real Angle", color='black', alpha=0.6)
    plt.plot(y_pred, label=f"Sim Angle (Width={best_width:.3f})", color='red', linestyle='--')
    plt.ylabel("Angle [deg]")
    plt.xlabel("Step")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 右: ヒステリシスループ比較
    plt.subplot(1, 2, 2)
    plt.title(f"Hysteresis Loop Fitting ({TARGET_FREQ})")
    plt.plot(u_data, y_data, label="Real Data", color='blue', alpha=0.3)
    plt.plot(u_data, y_pred, label="Sim Model", color='orange', alpha=0.8)
    plt.xlabel("Diff Pressure (P_DF - P_F) [MPa]")
    plt.ylabel("Angle [deg]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存
    if not os.path.exists("analysis_results"):
        os.makedirs("analysis_results")
    out_path = "analysis_results/hysteresis_optimization.png"
    plt.savefig(out_path, dpi=300)
    print(f"[INFO] Saved plot to {out_path}")
    print(f"\n[ACTION] Update 'actuator_cfg.py' -> PamHysteresisModelCfg -> hysteresis_width = {best_width:.4f}")

if __name__ == "__main__":
    main()