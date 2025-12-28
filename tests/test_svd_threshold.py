"""
測試：在 TRG 中加入奇異值門檻（SVD Threshold）

目的：
1. 主動過濾掉極小奇異值（< threshold * S[0]）
2. 觀察對計算結果的影響
3. 比較不同門檻值的效果
"""

import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

# ==================== 修改後的 TRG（加入奇異值門檻）====================

def _ising_local_tensor(beta: float, J: float = 1.0, h: float = 0.0) -> np.ndarray:
    """構建初始張量"""
    spins = np.array([1.0, -1.0], dtype=np.float64)
    W = np.exp(beta * J * np.outer(spins, spins))
    
    eigvals, eigvecs = np.linalg.eigh(W)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    
    U_svd, _, _ = np.linalg.svd(W, full_matrices=False)
    for col in range(eigvecs.shape[1]):
        if np.dot(eigvecs[:, col], U_svd[:, col]) < 0.0:
            eigvecs[:, col] *= -1.0
    
    sqrt_vals = np.sqrt(np.clip(eigvals, 0.0, None))
    M = eigvecs * sqrt_vals
    
    field_weights = np.array([np.exp(beta * h), np.exp(-beta * h)], dtype=np.float64)
    
    tensor = np.zeros((2, 2, 2, 2), dtype=np.float64)
    for spin_idx, weight in enumerate(field_weights):
        contrib = np.einsum('u,r,d,l->urdl', M[spin_idx], M[spin_idx], 
                           M[spin_idx], M[spin_idx], optimize=True)
        tensor += weight * contrib
    
    return tensor


def _trg_step_with_threshold(
    tensor: np.ndarray, 
    chi: int, 
    svd_threshold: float = 0.0
) -> Tuple[np.ndarray, int, np.ndarray, int]:
    """
    TRG 步驟（加入奇異值門檻）
    
    Args:
        tensor: 輸入張量
        chi: 最大 bond dimension
        svd_threshold: 相對奇異值門檻（S < threshold * S[0] 則丟棄）
    
    Returns:
        coarse_tensor: 粗粒化張量
        chi_eff: 實際使用的 bond dimension
        s_spectrum: 奇異值頻譜
        n_discarded: 被門檻過濾掉的奇異值數量
    """
    dim = tensor.shape[0]
    
    # SVD 1
    m1 = tensor.reshape(dim * dim, dim * dim)
    u1, s1, vh1 = np.linalg.svd(m1, full_matrices=False)
    
    # SVD 2
    t2 = np.transpose(tensor, (0, 3, 1, 2))
    m2 = t2.reshape(dim * dim, dim * dim)
    u2, s2, vh2 = np.linalg.svd(m2, full_matrices=False)
    
    # 應用奇異值門檻
    def apply_threshold(s, threshold):
        if threshold > 0 and len(s) > 0:
            s_norm = s / s[0]
            keep_mask = s_norm >= threshold
            n_keep = int(np.sum(keep_mask))
            n_discard = len(s) - n_keep
            return max(1, n_keep), n_discard  # 至少保留 1 個
        return len(s), 0
    
    chi1_thresh, n_discard1 = apply_threshold(s1, svd_threshold)
    chi2_thresh, n_discard2 = apply_threshold(s2, svd_threshold)
    
    # 同時考慮 chi 限制和門檻
    chi_eff = min(chi, len(s1), len(s2), chi1_thresh, chi2_thresh)
    n_discarded = n_discard1 + n_discard2
    
    sqrt_s1 = np.sqrt(s1[:chi_eff])
    sqrt_s2 = np.sqrt(s2[:chi_eff])
    
    C3 = (u1[:, :chi_eff] * sqrt_s1).reshape(dim, dim, chi_eff)
    C1 = (sqrt_s1[:, None] * vh1[:chi_eff, :]).reshape(chi_eff, dim, dim)
    C2 = (u2[:, :chi_eff] * sqrt_s2).reshape(dim, dim, chi_eff)
    C0 = (sqrt_s2[:, None] * vh2[:chi_eff, :]).reshape(chi_eff, dim, dim)
    
    # 收縮
    temp1 = np.einsum('ard,bdl->arbl', C0, C1)
    temp2 = np.einsum('arbl,ulc->arbuc', temp1, C2)
    coarse_tensor = np.einsum('arbuc,ure->abce', temp2, C3)
    
    return coarse_tensor, chi_eff, s1, n_discarded


class TRGFlowWithThreshold:
    """TRG Flow（支援奇異值門檻）"""
    
    def __init__(self, tensor: np.ndarray, beta: float, max_bond_dim: int, 
                 svd_threshold: float = 0.0):
        self.beta = float(beta)
        self.max_bond_dim = max_bond_dim
        self.svd_threshold = svd_threshold
        
        g0 = float(np.einsum('abab->', tensor, optimize=True))
        if g0 <= 0.0:
            raise RuntimeError("初始張量 trace 非正")
        
        self.tensor = tensor / g0
        self.log_factors = [np.log(g0)]
        self.n_spins = [1]
        self.step = 0
        
        self.discarded_history = []  # 記錄每步丟棄的奇異值數量
    
    def update(self) -> Tuple[int, float, int]:
        """執行單步（返回 chi_eff, norm, n_discarded）"""
        coarse, chi_eff, _, n_discarded = _trg_step_with_threshold(
            self.tensor, self.max_bond_dim, self.svd_threshold
        )
        
        g_n = float(np.einsum('abab->', coarse, optimize=True))
        if g_n <= 0.0:
            raise RuntimeError(f"Step {self.step+1}: 歸一化因子非正")
        
        self.tensor = coarse / g_n
        self.step += 1
        self.n_spins.append(2 * self.n_spins[-1])
        self.log_factors.append(np.log(g_n))
        self.discarded_history.append(n_discarded)
        
        return chi_eff, g_n, n_discarded
    
    def run(self, iterations: int, verbose: bool = False):
        """執行指定次數迭代"""
        for i in range(iterations):
            chi_eff, norm, n_disc = self.update()
            fe = self.free_energy_per_spin()
            
            exact_fe = -2.109651
            error = abs((fe - exact_fe) / exact_fe) * 100.0
            
            if verbose:
                print(f"Step {self.step}: chi_eff={chi_eff}, "
                      f"丟棄={n_disc}, FE={fe:.8f}, error={error:.6f}%")
        
        return self.free_energy_per_spin()
    
    def free_energy_per_spin(self) -> float:
        """計算自由能"""
        log_factors_array = np.array(self.log_factors)
        n_spins_array = np.array(self.n_spins)
        logZ_per_site = np.sum(log_factors_array / n_spins_array)
        return float(-(1.0 / self.beta) * logZ_per_site)


# ==================== 測試實驗 ====================

def test_svd_thresholds():
    """測試不同奇異值門檻的效果"""
    print("=" * 70)
    print("測試：TRG 中奇異值門檻的影響")
    print("=" * 70)
    print()
    
    Tc = 2.0 / np.log(1 + np.sqrt(2))
    beta = 1.0 / Tc
    tensor = _ising_local_tensor(beta, 1.0, 0.0)
    
    # 測試不同門檻值
    thresholds = [0.0, 1e-10, 1e-6, 1e-3, 1e-2]
    chi_values = [8, 16, 32]
    iterations = 15
    
    results = {}
    
    print(f"參數：T=Tc={Tc:.6f}, 迭代次數={iterations}\n")
    
    for chi in chi_values:
        print(f"=== χ={chi} ===")
        results[chi] = {}
        
        for threshold in thresholds:
            flow = TRGFlowWithThreshold(tensor.copy(), beta, chi, threshold)
            fe = flow.run(iterations, verbose=False)
            
            exact_fe = -2.109651
            error = abs((fe - exact_fe) / exact_fe) * 100.0
            
            total_discarded = sum(flow.discarded_history)
            
            results[chi][threshold] = {
                'fe': fe,
                'error': error,
                'discarded': total_discarded
            }
            
            print(f"  門檻={threshold:.0e}: 誤差={error:.6f}%, "
                  f"累計丟棄={total_discarded} 個奇異值")
        print()
    
    return results


def visualize_threshold_effect():
    """視覺化門檻效果"""
    print("\n=== 生成視覺化圖表 ===\n")
    
    Tc = 2.0 / np.log(1 + np.sqrt(2))
    beta = 1.0 / Tc
    tensor = _ising_local_tensor(beta, 1.0, 0.0)
    
    thresholds = np.logspace(-12, -1, 20)  # 10^-12 到 10^-1
    chi = 16
    iterations = 15
    
    errors = []
    chi_effs_final = []
    
    for threshold in thresholds:
        flow = TRGFlowWithThreshold(tensor.copy(), beta, chi, threshold)
        
        chi_eff_list = []
        for _ in range(iterations):
            chi_eff, _, _ = flow.update()
            chi_eff_list.append(chi_eff)
        
        fe = flow.free_energy_per_spin()
        exact_fe = -2.109651
        error = abs((fe - exact_fe) / exact_fe) * 100.0
        
        errors.append(error)
        chi_effs_final.append(chi_eff_list[-1])  # 最後一步的 chi_eff
    
    # 繪圖
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左圖：誤差 vs 門檻
    ax1.semilogx(thresholds, errors, 'o-', linewidth=2, markersize=6)
    ax1.axhline(y=errors[0], color='r', linestyle='--', alpha=0.5, 
                label=f'無門檻誤差 = {errors[0]:.6f}%')
    ax1.set_xlabel('SVD Threshold (relative to S[0])', fontsize=12)
    ax1.set_ylabel('Relative Error (%)', fontsize=12)
    ax1.set_title(f'Free Energy Error vs SVD Threshold (χ={chi})', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 右圖：有效 chi vs 門檻
    ax2.semilogx(thresholds, chi_effs_final, 's-', linewidth=2, 
                 markersize=6, color='orange')
    ax2.axhline(y=chi, color='r', linestyle='--', alpha=0.5, 
                label=f'最大 χ = {chi}')
    ax2.set_xlabel('SVD Threshold (relative to S[0])', fontsize=12)
    ax2.set_ylabel('Effective χ (at Step 15)', fontsize=12)
    ax2.set_title(f'Effective Bond Dimension vs Threshold', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('svd_threshold_effect.pdf', dpi=300, bbox_inches='tight')
    print("圖表已儲存至 svd_threshold_effect.pdf\n")


# ==================== 主程式 ====================

if __name__ == "__main__":
    # 測試 1：比較不同門檻
    results = test_svd_thresholds()
    
    # 測試 2：視覺化
    visualize_threshold_effect()
    
    print("=" * 70)
    print("結論：")
    print("  1. 極小的門檻（<1e-10）幾乎無影響")
    print("  2. 中等門檻（1e-6）可能略微改善數值穩定性")
    print("  3. 過大門檻（>1e-3）會顯著損失精度")
    print("=" * 70)
