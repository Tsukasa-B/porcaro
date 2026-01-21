import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 設定
# ==========================================
FILE_PATH = "external_data/jetson_project/data_exp1_step_full_perm_1768886252.csv"
OUTPUT_DIR = "analysis_full_perm"
LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

def load_and_preprocess(filepath):
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        return None
    df = pd.read_csv(filepath)
    if 'timestamp' in df.columns:
        df['time'] = df['timestamp'] - df['timestamp'].iloc[0]
    return df

def identify_step_params(segment, t_change, p_start, p_end):
    """ 1つのステップ区間のパラメータを計算 """
    # データ抽出 (変化点以降)
    data = segment[segment['time'] >= t_change].copy()
    if data.empty: return None

    # 目標値と開始値
    meas_start = data['meas_pres_F'].iloc[0]
    delta_p = p_end - meas_start
    
    # ノイズ対策: 変化幅が小さすぎる場合はスキップ
    if abs(delta_p) < 0.02:
        return None

    is_inflation = delta_p > 0
    
    # --- 1. むだ時間 (L) ---
    # 変化幅の 10% を超えた時点を「反応開始」とする
    threshold = meas_start + delta_p * 0.10
    
    if is_inflation:
        react_points = data[data['meas_pres_F'] >= threshold]
    else:
        react_points = data[data['meas_pres_F'] <= threshold]
        
    if react_points.empty:
        return None
        
    t_reaction = react_points['time'].iloc[0]
    L = t_reaction - t_change
    L = max(0.0, L)

    # --- 2. 時定数 (tau) ---
    # 反応開始から、(Target - Start) の 63.2% に達するまでの時間
    target_63 = meas_start + delta_p * 0.632
    
    if is_inflation:
        tau_points = data[data['meas_pres_F'] >= target_63]
    else:
        tau_points = data[data['meas_pres_F'] <= target_63]
        
    if tau_points.empty:
        tau = np.nan
    else:
        t_63 = tau_points['time'].iloc[0]
        tau = t_63 - t_reaction
        tau = max(0.01, tau)

    return L, tau

def plot_heatmap(ax, matrix, title, cmap, vmin, vmax):
    """ Matplotlibだけでヒートマップを描画する関数 """
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
    
    # 軸の設定
    ax.set_xticks(np.arange(len(LEVELS)))
    ax.set_yticks(np.arange(len(LEVELS)))
    ax.set_xticklabels(LEVELS)
    ax.set_yticklabels(LEVELS)
    ax.set_xlabel("Target Pressure [MPa]")
    ax.set_ylabel("Start Pressure [MPa]")
    ax.set_title(title)
    
    # グリッドライン
    ax.set_xticks(np.arange(len(LEVELS)+1)-0.5, minor=True)
    ax.set_yticks(np.arange(len(LEVELS)+1)-0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 数値をマス目に表示
    for i in range(len(LEVELS)):
        for j in range(len(LEVELS)):
            val = matrix.iloc[i, j]
            if not np.isnan(val):
                text_color = "white" if (val - vmin) / (vmax - vmin) > 0.5 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=text_color, fontsize=9)
    
    # カラーバー
    plt.colorbar(im, ax=ax)

def main():
    df = load_and_preprocess(FILE_PATH)
    if df is None: return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- トランジション検出 ---
    df['cmd_diff'] = df['cmd_F'].diff().fillna(0)
    change_indices = df.index[df['cmd_diff'].abs() > 0.02].tolist()
    
    results = []
    
    print(f"Total detected command changes: {len(change_indices)}")
    
    for idx in change_indices:
        t_change = df['time'].iloc[idx]
        p_prev = df['cmd_F'].iloc[idx-1]
        p_curr = df['cmd_F'].iloc[idx]
        
        # サイクル内時刻 (2.5s ~ 4.5s の間の変化のみを「計測ステップ」とみなす)
        cycle_time = t_change % 6.0
        
        if 2.0 <= cycle_time <= 4.5:
            end_search_idx = min(idx + 300, len(df))
            segment = df.iloc[idx : end_search_idx]
            
            res = identify_step_params(segment, t_change, p_prev, p_curr)
            
            if res:
                L, tau = res
                results.append({
                    'p_start': round(p_prev, 1),
                    'p_end': round(p_curr, 1),
                    'L': L,
                    'tau': tau
                })

    results_df = pd.DataFrame(results)
    print(f"Successfully analyzed {len(results_df)} transitions.")
    results_df.to_csv(os.path.join(OUTPUT_DIR, "identified_params.csv"), index=False)

    # --- マトリクス生成 ---
    tau_matrix = pd.DataFrame(index=LEVELS, columns=LEVELS, dtype=float)
    L_matrix = pd.DataFrame(index=LEVELS, columns=LEVELS, dtype=float)
    
    for _, row in results_df.iterrows():
        s = row['p_start']
        e = row['p_end']
        if s in LEVELS and e in LEVELS:
            tau_matrix.at[s, e] = row['tau']
            L_matrix.at[s, e] = row['L']

    # 欠損値補完（可視化用）
    tau_matrix_vis = tau_matrix.copy()
    L_matrix_vis = L_matrix.copy()
    tau_matrix_vis.fillna(0.0, inplace=True)
    L_matrix_vis.fillna(0.0, inplace=True)

    # --- ヒートマップ描画 (Matplotlib版) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    plot_heatmap(axes[0], tau_matrix_vis, "Time Constant (tau) [s]", "YlOrRd", 0.0, 0.3)
    plot_heatmap(axes[1], L_matrix_vis, "Dead Time (L) [s]", "YlGnBu", 0.0, 0.15)
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "dynamics_heatmap_mpl.png")
    plt.savefig(plot_path)
    print(f"Heatmap saved to: {plot_path}")

    # --- コード生成 (PyTorch Tensor形式) ---
    tau_matrix.fillna(0.04, inplace=True) # コード用補完
    L_matrix.fillna(0.00, inplace=True)
    
    print("\n" + "="*30)
    print(" [Copy & Paste Code]")
    print("="*30)
    
    def print_tensor(df, name):
        print(f"{name} = torch.tensor([")
        for idx, row in df.iterrows():
            vals = ", ".join([f"{x:.4f}" for x in row.values])
            print(f"    [{vals}], # Start={idx}")
        print("], dtype=torch.float32, device=self.device)")

    print_tensor(tau_matrix, "TAU_TABLE")
    print("")
    print_tensor(L_matrix, "DEADTIME_TABLE")
    print("="*30)

if __name__ == "__main__":
    main()