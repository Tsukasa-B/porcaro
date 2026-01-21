import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# 保存先ディレクトリの作成
OUTPUT_DIR = "analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 物理パラメータテーブル (Smoothing適用済み)
# self.device を削除しました
TAU_TABLE = torch.tensor([
    [0.0010, 0.0200, 0.0404, 0.0404, 0.0808, 0.0909, 0.1313], # Start=0.0
    [0.0908, 0.0010, 0.0404, 0.0405, 0.0505, 0.0908, 0.0808], # Start=0.1
    [0.0404, 0.1313, 0.0010, 0.0913, 0.0404, 0.0811, 0.0808], # Start=0.2
    [0.0404, 0.0808, 0.0910, 0.0010, 0.0809, 0.0403, 0.0808], # Start=0.3
    [0.0909, 0.0909, 0.0404, 0.1717, 0.0010, 0.0914, 0.0808], # Start=0.4
    [0.0404, 0.0909, 0.0406, 0.0404, 0.0404, 0.0010, 0.0404], # Start=0.5
    [0.0404, 0.0909, 0.0505, 0.0404, 0.0505, 0.0200, 0.0010], # Start=0.6
], dtype=torch.float32)

DEADTIME_TABLE = torch.tensor([
    [0.0000, 0.0809, 0.0908, 0.0404, 0.0404, 0.0807, 0.0404], # Start=0.0
    [0.0000, 0.0000, 0.0505, 0.0505, 0.0808, 0.0404, 0.0908], # Start=0.1
    [0.0809, 0.0808, 0.0000, 0.0000, 0.0911, 0.0505, 0.0505], # Start=0.2
    [0.0404, 0.0910, 0.1313, 0.0000, 0.0000, 0.0807, 0.0404], # Start=0.3
    [0.0404, 0.0808, 0.0808, 0.0000, 0.0000, 0.0405, 0.0909], # Start=0.4
    [0.0808, 0.0808, 0.0913, 0.0808, 0.0909, 0.0000, 0.0810], # Start=0.5
    [0.0809, 0.0808, 0.0813, 0.0908, 0.0807, 0.1313, 0.0000], # Start=0.6
], dtype=torch.float32)

LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

# 2. ヒートマップ作成関数 (論文用スタイル)
def plot_heatmap(matrix, title, cbar_label, filename, vmin=None, vmax=None):
    # フォント設定
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
    
    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = plt.cm.YlOrRd
    
    # テンソルをnumpyに変換してプロット
    im = ax.imshow(matrix.numpy(), cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
    
    # 軸設定
    ax.set_xticks(np.arange(len(LEVELS)))
    ax.set_yticks(np.arange(len(LEVELS)))
    ax.set_xticklabels(LEVELS)
    ax.set_yticklabels(LEVELS)
    ax.set_xlabel("Target Pressure [MPa]", fontweight='bold')
    ax.set_ylabel("Start Pressure [MPa]", fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=15)
    
    # グリッド
    ax.set_xticks(np.arange(len(LEVELS)+1)-0.5, minor=True)
    ax.set_yticks(np.arange(len(LEVELS)+1)-0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 数値書き込み
    for i in range(len(LEVELS)):
        for j in range(len(LEVELS)):
            val = matrix[i, j].item()
            text_color = "white" if (val - (vmin or 0))/( (vmax or 1) - (vmin or 0)) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=9)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

if __name__ == "__main__":
    plot_heatmap(TAU_TABLE, "Time Constant ($\\tau$)", "Time Constant [s]", "tau_heatmap.png", vmin=0.03, vmax=0.15)
    plot_heatmap(DEADTIME_TABLE, "Dead Time ($L$)", "Dead Time [s]", "deadtime_heatmap.png", vmin=0.0, vmax=0.10)