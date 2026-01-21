import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- 設定 ---
# データが格納されているディレクトリ (Knowledgeの構造に合わせる)
DATA_DIR = "./external_data/jetson_project"
OUTPUT_DIR = "./analysis_results"

# ファイル名のパターンとラベルのマッピング
FILE_PATTERNS = {
    "0.5Hz": "data_exp2_hysteresis_0.5Hz_*.csv",
    "1.0Hz": "data_exp2_hysteresis_1.0Hz_*.csv",
    "2.0Hz": "data_exp2_hysteresis_2.0Hz_*.csv",
    "3.0Hz": "data_exp2_hysteresis_3.0Hz_*.csv",
}

def analyze_hysteresis():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"[INFO] Created output directory: {OUTPUT_DIR}")

    plt.figure(figsize=(10, 8))
    
    # プロット用の色設定 (周波数が高いほど濃い色/熱い色にするなど)
    colors = {"0.5Hz": "blue", "1.0Hz": "green", "2.0Hz": "orange", "3.0Hz": "red"}
    
    found_any = False

    for freq_label, pattern in FILE_PATTERNS.items():
        # ファイル検索
        search_path = os.path.join(DATA_DIR, pattern)
        files = glob.glob(search_path)
        
        if not files:
            print(f"[WARN] No file found for {freq_label} (pattern: {pattern})")
            continue
        
        # 複数見つかった場合は最新のものを使用（タイムスタンプがファイル名末尾にある前提）
        target_file = sorted(files)[-1]
        print(f"[INFO] Processing {freq_label}: {os.path.basename(target_file)}")
        found_any = True

        try:
            # CSV読み込み
            df = pd.read_csv(target_file)
            
            # --- データ前処理 ---
            # 必要なカラムの確認
            req_cols = ['meas_pres_DF', 'meas_pres_F', 'meas_ang_wrist']
            if not all(col in df.columns for col in req_cols):
                print(f"[ERROR] Missing columns in {target_file}. Skipping.")
                continue

            # 最初の数秒（過渡期）をカットする場合 (例: 最初の2秒=200step)
            # 安定したループを見るため、データ後半を使うのが一般的
            start_idx = int(len(df) * 0.2) 
            df_stable = df.iloc[start_idx:]

            # --- 軸データの計算 ---
            # 横軸: 差圧 (P_DF - P_F)
            # DF(伸筋)が正、F(屈筋)が負の方向と定義した場合の「正味の圧力入力」
            diff_pressure = df_stable['meas_pres_DF'] - df_stable['meas_pres_F']
            
            # 縦軸: 手首角度 (meas_ang_wrist) [deg]
            wrist_angle = df_stable['meas_ang_wrist']

            # --- プロット ---
            plt.plot(diff_pressure, wrist_angle, 
                     label=f"{freq_label}", 
                     color=colors.get(freq_label, 'black'),
                     alpha=0.7, linewidth=1.5)

        except Exception as e:
            print(f"[ERROR] Failed to process {target_file}: {e}")

    if not found_any:
        print("[ERROR] No data files found. Check DATA_DIR.")
        return

    # --- グラフ装飾 ---
    plt.title("Hysteresis Loop: Differential Pressure vs Wrist Angle", fontsize=16)
    plt.xlabel("Differential Pressure (P_DF - P_F) [MPa]", fontsize=14)
    plt.ylabel("Wrist Angle [deg]", fontsize=14)
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=12)
    
    # 保存
    save_path = os.path.join(OUTPUT_DIR, "Exp2_Hysteresis_Comparison.png")
    plt.savefig(save_path, dpi=300)
    print(f"\n[SUCCESS] Plot saved to: {save_path}")
    # plt.show() # ローカル実行時用

if __name__ == "__main__":
    analyze_hysteresis()