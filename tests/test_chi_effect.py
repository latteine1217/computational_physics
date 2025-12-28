"""
驗證 χ 效果的實驗：需要足夠的迭代次數
"""
import numpy as np
from trg_final_project import _ising_local_tensor, TRGFlow
import matplotlib.pyplot as plt

Tc = 2.0 / np.log(1 + np.sqrt(2))
beta = 1.0 / Tc
tensor = _ising_local_tensor(beta, 1.0, 0.0)

print('=== 驗證：χ 的效果需要足夠迭代 ===\n')

# 測試不同迭代次數
iteration_counts = [5, 10, 15, 20]
chi_values = [4, 8, 16, 32]

plt.figure(figsize=(10, 6))

for chi in chi_values:
    errors_at_different_iterations = []
    
    for max_iter in iteration_counts:
        flow = TRGFlow(tensor.copy(), beta, chi)
        errors = flow.run(max_iter, verbose=False)
        errors_at_different_iterations.append(errors[-1])
    
    print(f'χ={chi:2d}: {[f"{e:.4f}%" for e in errors_at_different_iterations]}')
    plt.semilogy(iteration_counts, errors_at_different_iterations, 'o-', 
                 label=f'χ={chi}', markersize=8, linewidth=2)

plt.xlabel('Iteration Count', fontsize=12)
plt.ylabel('Relative Error (%)', fontsize=12)
plt.title('χ Effect vs Iteration Count (T=Tc)', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('chi_effect_vs_iterations.pdf', dpi=300, bbox_inches='tight')
print(f'\n圖表已儲存至 chi_effect_vs_iterations.pdf')
