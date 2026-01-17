# scripts/verification/quick_check_hysteresis.py
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# 1. パスの自動解決
# このスクリプトのあるディレクトリ (.../scripts/verification/)
script_dir = os.path.dirname(os.path.abspath(__file__))
# プロジェクトルート (.../porcaro_rl/) まで遡る
project_root = os.path.dirname(os.path.dirname(script_dir))

# データの場所を探す（プロジェクト内にある場合と、外にある場合の両方を考慮）
possible_paths = [
    # パターンA: porcaro_rl/external_data/jetson_project/...
    os.path.join(project_root, "external_data", "jetson_project", "01-11-hysteresis.csv"),
    # パターンB: porcaro_rl/../../external_data/... (sibling directory)
    os.path.join(project_root, "..", "external_data", "jetson_project", "01-11-hysteresis.csv"),
]

csv_path = None
for p in possible_paths:
    if os.path.exists(p):
        csv_path = p
        break

if csv_path is None:
    print("[Error] CSV file not found.")
    print("Searched locations:")
    for p in possible_paths:
        print(f" - {p}")
    sys.exit(1)

print(f"Loading data from: {csv_path}")

# 2. データ読み込み
df = pd.read_csv(csv_path)

# 3. 差圧 vs 角度 のプロット
# 平行なループ（太さが一定）なら「摩擦モデル（Play Operator）」が正解です
# 扇型（圧力が高いほど太い）なら「比率モデル」が正解です
df['p_diff'] = df['meas_pres_DF'] - df['meas_pres_F']

plt.figure(figsize=(8,6))
plt.scatter(df['p_diff'], df['meas_ang_wrist'], alpha=0.1, s=1, c='black', label='Real Data')
plt.title("Hysteresis Loop Check")
plt.xlabel("Differential Pressure (P_DF - P_F) [MPa]")
plt.ylabel("Wrist Angle [deg]")
plt.legend()
plt.grid(True)
plt.show()