import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 設定
# ==========================================
# 解析対象のファイル（前回アップロードされたファイルを指定）
FILE_PATH = "external_data/jetson_project/data_exp1_step_pamf_1768881636.csv"
OUTPUT_DIR = "analysis_results_42steps"

def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        # テスト用にダミーなどを使わず終了
        return None
    df = pd.read_csv(filepath)
    # 相対時間計算
    if 'timestamp' in df.columns:
        df['time'] = df['timestamp'] - df['timestamp'].iloc[0]
    return df

def identify_step_parameters(segment, p_start_cmd, p_end_cmd, t_cmd_change):
    """
    1つのステップ区間データから時定数とむだ時間を同定する
    """
    # ターゲット圧力（指令値）
    target_val = p_end_cmd
    start_val = segment['meas_pres_F'].iloc[0] # 計測値の初期値
    
    # 変化の方向と大きさ
    delta_p = target_val - start_val
    is_inflation = delta_p > 0
    
    # ノイズ対策: 変化幅があまりに小さい場合は解析不可とする (0.05MPa未満など)
    if abs(delta_p) < 0.03:
        return None

    # 1. むだ時間 (L) の特定
    # 基準: 変化幅の 10% を超えた時点を「反応開始」とみなす
    threshold_ratio = 0.10
    threshold_val = start_val + delta_p * threshold_ratio
    
    if is_inflation:
        reaction_points = segment[segment['meas_pres_F'] >= threshold_val]
    else:
        reaction_points = segment[segment['meas_pres_F'] <= threshold_val]
        
    if reaction_points.empty:
        return None # 反応しきれなかった場合
    
    t_reaction = reaction_points['time'].iloc[0]
    dead_time = t_reaction - t_cmd_change
    
    # むだ時間が負になる（ノイズでたまたま超えていた）場合の補正
    if dead_time < 0: dead_time = 0.0

    # 2. 時定数 (tau) の特定
    # 基準: 一次遅れ系において、反応開始から (Target - Start) の 63.2% に達するまでの時間
    # 実際には Startは「反応開始時の圧力」としたほうが正確だが、ここでは全体の変化幅で見る
    target_63_ratio = 0.632
    target_63_val = start_val + delta_p * target_63_ratio
    
    if is_inflation:
        tau_points = segment[segment['meas_pres_F'] >= target_63_val]
    else:
        tau_points = segment[segment['meas_pres_F'] <= target_63_val]
        
    if tau_points.empty:
        # 63%に達しなかった場合（3秒で収束しなかった場合など）
        tau = np.nan
    else:
        t_63 = tau_points['time'].iloc[0]
        # 時定数は「反応開始」からの経過時間
        tau = t_63 - t_reaction

    return {
        "direction": "Inflation" if is_inflation else "Deflation",
        "p_start_cmd": p_start_cmd,
        "p_end_cmd": p_end_cmd,
        "delta_p": delta_p,
        "dead_time": dead_time,
        "time_constant": tau,
        "t_cmd_change": t_cmd_change
    }

def main():
    df = load_data(FILE_PATH)
    if df is None: return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 指令値の変化点（ステップ開始点）を検出
    # 指令値の差分をとり、非ゼロの場所を探す
    df['cmd_diff'] = df['cmd_F'].diff().fillna(0)
    change_indices = df.index[df['cmd_diff'].abs() > 0.01].tolist()
    
    results = []
    
    print(f"Detected {len(change_indices)} step transitions.")
    
    for i, idx in enumerate(change_indices):
        # 区間の定義: 今回の変化点から、次の変化点（またはデータ末尾）まで
        next_idx = change_indices[i+1] if i+1 < len(change_indices) else len(df)-1
        
        # 解析には「変化直前」の情報が必要
        if idx == 0: continue # 先頭はいきなり始まるのでスキップ（または初期値0と仮定）
        
        # ステップ情報
        p_start = df['cmd_F'].iloc[idx-1]
        p_end = df['cmd_F'].iloc[idx]
        t_change = df['time'].iloc[idx]
        
        # データ切り出し（少しマージンを持って次の変化の直前まで）
        segment = df.iloc[idx : next_idx].copy()
        
        # 解析実行
        res = identify_step_parameters(segment, p_start, p_end, t_change)
        if res:
            results.append(res)

    # 結果をDataFrame化
    res_df = pd.DataFrame(results)
    
    # CSV保存
    save_csv_path = os.path.join(OUTPUT_DIR, "step_response_parameters.csv")
    res_df.to_csv(save_csv_path, index=False)
    print(f"\nAnalysis complete. Results saved to: {save_csv_path}")
    print(res_df[['direction', 'p_start_cmd', 'p_end_cmd', 'dead_time', 'time_constant']].head(10))

    # ==========================================
    # 可視化: パラメータの傾向分析
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # グループ分け
    inf_df = res_df[res_df['direction'] == 'Inflation']
    def_df = res_df[res_df['direction'] == 'Deflation']
    
    # 1. 時定数 vs 到達圧力 (Target Pressure)
    ax = axes[0, 0]
    ax.scatter(inf_df['p_end_cmd'], inf_df['time_constant'], c='r', label='Inflation', marker='^')
    ax.scatter(def_df['p_end_cmd'], def_df['time_constant'], c='b', label='Deflation', marker='v')
    ax.set_title('Time Constant vs Target Pressure')
    ax.set_xlabel('Target Pressure [MPa]')
    ax.set_ylabel('Time Constant [s]')
    ax.legend()
    ax.grid(True)

    # 2. むだ時間 vs 開始圧力 (Start Pressure) -> 排気遅れは開始圧に依存しやすい
    ax = axes[0, 1]
    ax.scatter(inf_df['p_start_cmd'], inf_df['dead_time'], c='r', label='Inflation', marker='^')
    ax.scatter(def_df['p_start_cmd'], def_df['dead_time'], c='b', label='Deflation', marker='v')
    ax.set_title('Dead Time vs Start Pressure')
    ax.set_xlabel('Start Pressure [MPa]')
    ax.set_ylabel('Dead Time [s]')
    ax.legend()
    ax.grid(True)
    
    # 3. 時定数 vs 圧力変化幅 (Delta P)
    ax = axes[1, 0]
    ax.scatter(inf_df['delta_p'].abs(), inf_df['time_constant'], c='r', label='Inflation', marker='^')
    ax.scatter(def_df['delta_p'].abs(), def_df['time_constant'], c='b', label='Deflation', marker='v')
    ax.set_title('Time Constant vs Step Size')
    ax.set_xlabel('Step Size (|Delta P|) [MPa]')
    ax.set_ylabel('Time Constant [s]')
    ax.legend()
    ax.grid(True)
    
    # 4. 全体の応答波形（確認用・最初の5個）
    ax = axes[1, 1]
    plot_df = df[df['time'] < 15.0] # 最初の15秒
    ax.plot(plot_df['time'], plot_df['cmd_F'], 'k--', label='CMD')
    ax.plot(plot_df['time'], plot_df['meas_pres_F'], 'g', label='MEAS')
    ax.set_title('First 15s Response (Check)')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Pressure [MPa]')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "step_parameter_analysis.png"))
    print(f"Graph saved to: {os.path.join(OUTPUT_DIR, 'step_parameter_analysis.png')}")

if __name__ == "__main__":
    main()