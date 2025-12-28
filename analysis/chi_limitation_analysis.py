"""
視覺化展示 TRG 中的兩種限制
"""
import numpy as np
import matplotlib.pyplot as plt
from trg_final_project import _ising_local_tensor, _trg_step

Tc = 2.0 / np.log(1 + np.sqrt(2))
beta = 1.0 / Tc
tensor = _ising_local_tensor(beta, 1.0, 0.0)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 歸一化
g0 = float(np.einsum('abab->', tensor, optimize=True))
tensor = tensor / g0

step_labels = ['Step 1', 'Step 2', 'Step 3', 'Step 4']

for idx, step in enumerate([1, 2, 3, 4]):
    ax = axes[idx // 2, idx % 2]
    
    coarse, chi_eff, s_spectrum = _trg_step(tensor, chi=64)
    s_norm = s_spectrum / s_spectrum[0]
    
    # 繪製奇異值頻譜
    indices = np.arange(len(s_norm)) + 1
    ax.semilogy(indices, s_norm, 'o-', markersize=4, linewidth=1.5, label='Singular Values')
    
    # 標記截斷線
    chi_lines = [4, 8, 16, 32]
    colors = ['red', 'orange', 'green', 'blue']
    for chi, color in zip(chi_lines, colors):
        if chi <= len(s_norm):
            ax.axvline(chi, color=color, linestyle='--', alpha=0.6, 
                      linewidth=2, label=f'χ={chi}')
    
    # 標記「幾何限制」
    geom_limit = len(s_spectrum)
    if geom_limit < 64:
        ax.axvline(geom_limit, color='black', linestyle=':', 
                  linewidth=3, label=f'幾何限制 ({geom_limit})')
    
    # 標記「數值門檻」
    ax.axhline(1e-6, color='gray', linestyle='-.', alpha=0.5, 
              label='數值門檻 (10⁻⁶)')
    
    ax.set_xlabel('Singular Value Index', fontsize=11)
    ax.set_ylabel('Normalized Singular Value', fontsize=11)
    ax.set_title(f'{step_labels[idx]}: {len(s_spectrum)} 個奇異值 (幾何限制)', 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_ylim([1e-20, 2])
    
    # 更新張量
    g_n = float(np.einsum('abab->', coarse, optimize=True))
    tensor = coarse / g_n

plt.suptitle('TRG 的兩種限制：幾何限制 vs 數值截斷', 
            fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('chi_limitation_visualization.pdf', dpi=300, bbox_inches='tight')
print('圖表已儲存至 chi_limitation_visualization.pdf')
print()
print('解讀：')
print('  - 黑色虛線：幾何限制（SVD 最多產生這麼多奇異值）')
print('  - 灰色線：數值門檻 10⁻⁶（低於此值視為數值誤差）')
print('  - 彩色虛線：不同 chi 的截斷位置')
print()
print('觀察：')
print('  Step 1: 只有 4 個奇異值 → chi>4 被幾何限制')
print('  Step 2: 雖有 16 個，但 S[4+] < 10⁻¹⁸ → chi=4 已足夠')
print('  Step 3: S[4+] 仍是機器精度 → chi>4 改善微小')
print('  Step 4: 終於有顯著的中等奇異值 → chi=8 開始有用')
