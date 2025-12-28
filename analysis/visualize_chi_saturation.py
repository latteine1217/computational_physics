"""
視覺化 χ 飽和現象的關鍵圖表
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from trg_final_project import _ising_local_tensor, _trg_step, TRGFlow

# 設定
Tc = 2.0 / np.log(1 + np.sqrt(2))
beta = 1.0 / Tc
tensor = _ising_local_tensor(beta, 1.0, 0.0)

print("=" * 70)
print("生成 χ 飽和現象視覺化圖表")
print("=" * 70)
print()

# 創建圖表
fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

# ==================== 圖 1: 奇異值頻譜（4 步驟）====================
print("生成圖 1: 奇異值頻譜...")

g0 = float(np.einsum('abab->', tensor, optimize=True))
t = tensor / g0

for step_idx, step in enumerate([1, 2, 3, 4]):
    ax = fig.add_subplot(gs[step_idx // 2, step_idx % 2])

    coarse, chi_eff, s_spectrum = _trg_step(t, chi=64)
    s_norm = s_spectrum / s_spectrum[0]

    # 繪製奇異值
    indices = np.arange(len(s_norm)) + 1
    ax.semilogy(indices, s_norm, 'o-', markersize=4, linewidth=1.5,
                color='navy', label='Singular Values')

    # 標記不同 χ 的截斷線
    chi_markers = [4, 8, 16, 32]
    colors = ['red', 'orange', 'green', 'blue']
    for chi, color in zip(chi_markers, colors):
        if chi <= len(s_norm):
            ax.axvline(chi, color=color, linestyle='--', alpha=0.6,
                      linewidth=2, label=f'χ={chi}')

    # 標記數值門檻
    ax.axhline(1e-6, color='gray', linestyle='-.', alpha=0.5,
              linewidth=2, label='Threshold (10⁻⁶)')
    ax.axhline(1e-12, color='lightgray', linestyle=':', alpha=0.5,
              linewidth=2, label='Noise (10⁻¹²)')

    ax.set_xlabel('Singular Value Index', fontsize=11)
    ax.set_ylabel('Normalized S / S[0]', fontsize=11)
    ax.set_title(f'Step {step}: {len(s_norm)} singular values',
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_ylim([1e-20, 2])

    # 更新張量
    g_n = float(np.einsum('abab->', coarse, optimize=True))
    t = coarse / g_n

print("  完成")

# ==================== 圖 2: χ vs 誤差 ====================
print("生成圖 2: χ vs 誤差曲線...")

ax2 = fig.add_subplot(gs[:, 2])

chi_values = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64]
iterations = 15
errors = []

print(f"  測試 χ = {chi_values}")

for chi in chi_values:
    flow = TRGFlow(tensor.copy(), beta, chi)
    error_history = flow.run(iterations, verbose=False)
    final_error = error_history[-1]
    errors.append(final_error)
    print(f"    χ={chi:2d}: {final_error:.4f}%")

# 主曲線
ax2.plot(chi_values, errors, 'o-', linewidth=2.5, markersize=8,
         color='darkred', label='TRG Error')

# 標記關鍵點
critical_points = {
    4: ('χ=4\n幾何限制解除', 'bottom'),
    8: ('χ=8\n最優平衡點', 'top'),
    16: ('χ=16\n邊際收益遞減', 'bottom'),
    32: ('χ=32\n飽和區', 'top')
}

for chi, (text, va) in critical_points.items():
    if chi in chi_values:
        idx = chi_values.index(chi)
        offset = 0.15 if va == 'top' else -0.15
        ax2.annotate(text, xy=(chi, errors[idx]),
                    xytext=(chi, errors[idx] * (1 + offset)),
                    ha='center', va=va, fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.5',
                             facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->',
                                   connectionstyle='arc3,rad=0'))

# 飽和線
saturation_level = errors[-1]
ax2.axhline(saturation_level, color='gray', linestyle='--',
           alpha=0.6, linewidth=2,
           label=f'Saturation (~{saturation_level:.2f}%)')

ax2.set_xlabel('Bond Dimension χ', fontsize=12, fontweight='bold')
ax2.set_ylabel('Relative Error (%)', fontsize=12, fontweight='bold')
ax2.set_title('Error Saturation vs χ (T=Tc, 15 iterations)',
             fontsize=13, fontweight='bold')
ax2.set_xscale('log', base=2)
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3, which='both')
ax2.legend(fontsize=10)

# 設定刻度
ax2.set_xticks(chi_values)
ax2.set_xticklabels([str(x) for x in chi_values])

print("  完成")

# 添加總標題
fig.suptitle('TRG χ Saturation Phenomenon: Geometric & Numerical Limitations',
            fontsize=15, fontweight='bold', y=0.98)

# 儲存
plt.savefig('chi_saturation_comprehensive.pdf', dpi=300, bbox_inches='tight')
print()
print("=" * 70)
print("圖表已儲存: chi_saturation_comprehensive.pdf")
print("=" * 70)
print()

# 輸出數值總結
print("數值總結：")
print("-" * 70)
print(f"{'χ':<8} {'誤差 (%)':<12} {'相對改善 (%)':<15} {'效益/成本':<12}")
print("-" * 70)

for i, (chi, err) in enumerate(zip(chi_values, errors)):
    if i == 0:
        improvement = "--"
        efficiency = "--"
    else:
        imp = (errors[i-1] - err) / errors[i-1] * 100
        # 成本正比於 χ³
        cost_ratio = (chi / chi_values[i-1]) ** 3
        eff = imp / cost_ratio if cost_ratio > 0 else 0
        improvement = f"{imp:>6.2f}"
        efficiency = f"{eff:>6.2f}"

    print(f"{chi:<8} {err:<12.4f} {improvement:<15} {efficiency:<12}")

print("-" * 70)
print()
print("結論：")
print("  1. χ=4→8:  改善顯著，效益高")
print("  2. χ=8→16: 改善中等，成本可接受")
print("  3. χ>16:   邊際收益遞減，不推薦")
print("  4. χ>32:   完全飽和，浪費計算資源")
print("=" * 70)
