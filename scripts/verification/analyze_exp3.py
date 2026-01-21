import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 設定 ---
DATA_DIR = "./external_data/jetson_project"
OUTPUT_DIR = "./analysis_results"

# 筋肉ごとの設定 (ファイル名パターン, 監視対象の角度カラム, 動きの方向)
TARGETS = {
    "PAMDF": {
        "pattern": "data_exp3_slack_pamdf_*.csv",
        "angle_col": "meas_ang_wrist",
        "pressure_col": "meas_pres_DF",
        "direction": "change", # 変化すればOK（初期位置からの移動）
        "threshold_deg": 1.0   # 動き出しとみなす角度変化量 [deg]
    },
    "PAMF": {
        "pattern": "data_exp3_slack_pamf_*.csv",
        "angle_col": "meas_ang_wrist",
        "pressure_col": "meas_pres_F",
        "direction": "change",
        "threshold_deg": 1.0
    },
    "PAMG": {
        "pattern": "data_exp3_slack_pamg_*.csv",
        "angle_col": "meas_ang_hand", # グリップ角度
        "pressure_col": "meas_pres_G",
        "direction": "change",
        "threshold_deg": 2.0   # グリップはノイズが大きい可能性があるので少し大きめに
    }
}

def analyze_slack():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    results = {}

    for label, config in TARGETS.items():
        # ファイル検索
        search_path = os.path.join(DATA_DIR, config["pattern"])
        files = sorted(glob.glob(search_path))
        
        if not files:
            print(f"[WARN] No file found for {label}")
            continue
        
        target_file = files[-1]
        print(f"[INFO] Analyzing {label}: {os.path.basename(target_file)}")
        
        df = pd.read_csv(target_file)
        
        # データ抽出
        time = df['timestamp'].values
        time = time - time[0] # 0秒開始に補正
        pressure = df[config['pressure_col']].values
        angle = df[config['angle_col']].values
        
        # --- 動き出し検出ロジック ---
        # 最初の0.5秒の平均角度を「初期位置」とする
        init_idx_end = int(len(df) * 0.05) # 最初の5%
        if init_idx_end == 0: init_idx_end = 5
        
        initial_angle = np.mean(angle[:init_idx_end])
        
        # 初期位置からの偏差が閾値を超えた最初のインデックスを探す
        diff = np.abs(angle - initial_angle)
        move_idx = np.argmax(diff > config['threshold_deg'])
        
        # まだ動いていない（閾値を超えていない）場合は検出失敗
        if move_idx == 0 and diff[0] <= config['threshold_deg']:
            print(f"  -> No significant movement detected for {label}.")
            engagement_pressure = None
            engagement_angle = None
            engagement_time = None
        else:
            engagement_pressure = pressure[move_idx]
            engagement_angle = angle[move_idx]
            engagement_time = time[move_idx]
            
            print(f"  -> Movement detected at T={engagement_time:.2f}s")
            print(f"     Pressure: {engagement_pressure:.4f} MPa")
            print(f"     Angle   : {engagement_angle:.2f} deg")
            
            results[label] = {
                "pressure": engagement_pressure,
                "angle": engagement_angle,
                "initial_angle": initial_angle
            }

        # --- プロット作成 ---
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # 圧力 (左軸)
        color = 'tab:blue'
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Pressure [MPa]', color=color)
        ax1.plot(time, pressure, color=color, label='Pressure')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, linestyle=':', alpha=0.6)
        
        # 動き出しポイントの表示
        if engagement_time is not None:
            ax1.axvline(engagement_time, color='red', linestyle='--', alpha=0.8, label='Engagement Point')
            ax1.scatter([engagement_time], [engagement_pressure], color='red', zorder=5)
            ax1.text(engagement_time, engagement_pressure + 0.02, 
                     f"  {engagement_pressure:.3f} MPa", color='red', fontweight='bold')

        # 角度 (右軸)
        ax2 = ax1.twinx() 
        color = 'tab:orange'
        ax2.set_ylabel('Angle [deg]', color=color) 
        ax2.plot(time, angle, color=color, label='Angle', linewidth=2)
        ax2.tick_params(axis='y', labelcolor=color)
        
        # 初期角度ライン
        ax2.axhline(initial_angle, color='gray', linestyle='-', alpha=0.5, linewidth=1)
        ax2.axhline(initial_angle + config['threshold_deg'], color='gray', linestyle=':', alpha=0.5)
        ax2.axhline(initial_angle - config['threshold_deg'], color='gray', linestyle=':', alpha=0.5)

        plt.title(f"Exp3 Slack Analysis: {label}")
        fig.tight_layout()
        
        save_path = os.path.join(OUTPUT_DIR, f"Exp3_Slack_{label}.png")
        plt.savefig(save_path)
        print(f"  -> Plot saved to {save_path}")
        plt.close()

    return results

if __name__ == "__main__":
    results = analyze_slack()
    
    print("\n" + "="*40)
    print("SUMMARY: Engagement Pressures (Slack Thresholds)")
    print("="*40)
    for k, v in results.items():
        print(f"{k:5s} : {v['pressure']:.4f} MPa (moved from {v['initial_angle']:.1f} deg)")
    print("="*40)