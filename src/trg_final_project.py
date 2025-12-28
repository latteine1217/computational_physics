"""
2D Ising Model TRG 計算期末專案

目的:
- 使用 Tensor Renormalization Group (TRG) 算法計算 2D Ising 模型
- 生成三張分析圖表：
  1. T=Tc 時的收斂誤差 vs iteration
  2. 相對誤差 vs 溫度（不同 bond dimension D）
  3. 熱容量 Cv vs 溫度（展示 peak 隨 D/iteration 變尖銳）

理論背景:
- TRG 透過反覆將 2×2 格點合併為一個有效格點進行粗粒化
- 正確的自由能公式：f = -T * Σ_n [ln(g_n) / N_n]，其中 N_n 是有效格點數
- 每步粗粒化後，格點數加倍：N_n = 2 * N_{n-1}
"""

from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from dataclasses import dataclass
import time


# ==================== 初始張量構造 ====================

def _ising_local_tensor(beta: float, J: float, h: float) -> np.ndarray:
    """
    構建二維 Ising 模型的初始 rank-4 局域張量 T。
    
    使用標準的權重矩陣分解方法：
    1. 計算 Boltzmann 權重矩陣 W[s_i, s_j] = exp(β*J*s_i*s_j)
    2. 特徵分解得到 M，使得 W = M @ M^T
    3. 構建張量 T = Σ_s exp(β*h*s) * M[s,u]*M[s,r]*M[s,d]*M[s,l]
    
    Args:
        beta: 逆溫度 (1/T)
        J: 耦合常數
        h: 外磁場
        
    Returns:
        tensor: shape (2,2,2,2) 的初始張量，對應 [up, right, down, left]
    """
    # 使用 cosh/sinh 的解析分解
    # M 的設計滿足 W = M @ M^T，其中 W_{s_i,s_j} = exp(beta * J * s_i * s_j)
    weight = beta * J
    M = np.array(
        [
            [np.sqrt(np.cosh(weight)), np.sqrt(np.sinh(weight))],
            [np.sqrt(np.cosh(weight)), -np.sqrt(np.sinh(weight))],
        ],
        dtype=np.float64,
    )
    
    # 外磁場權重：分別對應 s=+1 與 s=-1
    field_weights = np.array([np.exp(beta * h), np.exp(-beta * h)], dtype=np.float64)
    
    # 構建張量 T[u,r,d,l]：將每個方向的邊權重做外積後再加總
    tensor = np.zeros((2, 2, 2, 2), dtype=np.float64)
    for spin_idx, weight in enumerate(field_weights):
        contrib = np.einsum(
            'u,r,d,l->urdl',
            M[spin_idx],
            M[spin_idx],
            M[spin_idx],
            M[spin_idx],
            optimize=True
        )
        tensor += weight * contrib
    
    return tensor


# ==================== TRG 粗粒化步驟（修正版）====================

def _trg_step(tensor: np.ndarray, chi: int, rel_svd_cutoff: float = 0.0) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    執行一步 TRG 粗粒化（修正版）。
    
    修正要點：
    1. 正確的 SVD 分解順序
    2. 正確的張量收縮拓撲
    3. 參考 cytnx 版本的實作
    4. 添加 SVD 門檻過濾極小奇異值
    
    步驟：
    1. 第一次 SVD: T[u,r,d,l] 分解為 (u,r) - (d,l)
       產生 C3[u,r,aux_R] 和 C1[aux_R,d,l]
    2. 第二次 SVD: T[u,l,r,d] 分解為 (u,l) - (r,d)
       產生 C2[u,l,aux_L] 和 C0[aux_L,r,d]
    3. 收縮：C0.r=C1.l, C1.d=C2.u, C2.l=C3.r, C3.u=C0.d
    
    Args:
        tensor: 當前 rank-4 張量 T[u,r,d,l]
        chi: 最大 bond dimension
        rel_svd_cutoff: 相對奇異值門檻（預設 1e-16）
        
    Returns:
        coarse_tensor: 粗粒化後的張量
        chi_avg: 平均使用的 bond dimension
        s_spectrum: 奇異值頻譜
    """
    # 目前 TRG 假設各維度相同，因此用單一 dim
    dim = tensor.shape[0]
    
    # === 第一次 SVD: (up,right) - (down,left) ===
    # T[u,r,d,l] -> reshape(ur, dl)，對 (u,r) 與 (d,l) 做分解
    m1 = tensor.reshape(dim * dim, dim * dim)
    u1, s1, vh1 = np.linalg.svd(m1, full_matrices=False)
    
    # === 第二次 SVD: (up,left) - (right,down) ===
    # T[u,r,d,l] -> T[u,l,r,d] -> reshape(ul, rd)，改變分解配對
    t2 = np.transpose(tensor, (0, 3, 1, 2))
    m2 = t2.reshape(dim * dim, dim * dim)
    u2, s2, vh2 = np.linalg.svd(m2, full_matrices=False)
    
    # 應用 SVD 門檻過濾（相對於最大奇異值）
    chi1 = len(s1)
    if rel_svd_cutoff > 0.0 and s1.size > 0:
        keep_mask = s1 >= rel_svd_cutoff * s1[0]
        chi1 = int(np.count_nonzero(keep_mask))
        chi1 = max(1, chi1)  # 至少保留 1 個
    
    chi2 = len(s2)
    if rel_svd_cutoff > 0.0 and s2.size > 0:
        keep_mask = s2 >= rel_svd_cutoff * s2[0]
        chi2 = int(np.count_nonzero(keep_mask))
        chi2 = max(1, chi2)  # 至少保留 1 個
    
    # 關鍵修正：兩次 SVD 必須使用相同的 chi
    # 這樣才能保證收縮後的張量維度一致且對稱
    chi_eff = min(chi, chi1, chi2)
    
    # 將奇異值平方根分配到左右兩側
    sqrt_s1 = np.sqrt(s1[:chi_eff])
    sqrt_s2 = np.sqrt(s2[:chi_eff])
    
    # C3[u, r, aux] 和 C1[aux, d, l] 來自第一次 SVD
    C3 = (u1[:, :chi_eff] * sqrt_s1).reshape(dim, dim, chi_eff)
    C1 = (sqrt_s1[:, None] * vh1[:chi_eff, :]).reshape(chi_eff, dim, dim)
    
    # C2[u, l, aux] 和 C0[aux, r, d] 來自第二次 SVD
    C2 = (u2[:, :chi_eff] * sqrt_s2).reshape(dim, dim, chi_eff)
    C0 = (sqrt_s2[:, None] * vh2[:chi_eff, :]).reshape(chi_eff, dim, dim)
    
    # === 張量收縮===
    # 
    # 參考 Cytnx 的收縮拓撲（line 106-111）：
    #   C0.r = C1.l, C1.d = C2.u, C2.l = C3.r, C3.u = C0.d
    #
    # 使用逐步收縮的方式
    # 注意：四個 aux 索引保持獨立，最終形成 (chi_eff, chi_eff, chi_eff, chi_eff)
    
    # 依照拓撲進行收縮：
    # C0.r=C1.l, C1.d=C2.u, C2.l=C3.r, C3.u=C0.d
    # 對應 einsum：C0[a,x,w], C1[b,y,x], C2[y,z,c], C3[w,z,e] -> abce
    coarse_tensor = np.einsum('axw,byx,yzc,wze->abce', C0, C1, C2, C3, optimize=True)
    
    # 回傳奇異值譜供後續分析/除錯
    s_spectrum = s1
    
    return coarse_tensor, chi_eff, s_spectrum


# ==================== TRG Flow 管理類（修正版）====================

@dataclass
class TRGResult:
    """TRG 計算結果"""
    free_energy_per_spin: float
    log_Z: float
    iterations: int
    chi_used: int
    error_percent: float


class TRGFlow:
    """
    管理 TRG 粗粒化流程（修正版）。
    
    修正要點：
    1. 使用正確的自由能公式：f = -T * Σ [ln(g_n) / N_n]
    2. 正確追蹤有效格點數 N_n
    3. 支援 SVD 門檻過濾
    """
    
    def __init__(self, tensor: np.ndarray, beta: float, max_bond_dim: int, rel_svd_cutoff: float = 0.0):
        self.beta = float(beta)
        self.max_bond_dim = max_bond_dim
        self.rel_svd_cutoff = rel_svd_cutoff
        
        # 初始歸一化：用 trace 取得 g0 並確保正值
        g0 = float(np.einsum('abab->', tensor, optimize=True))
        if g0 <= 0.0:
            raise RuntimeError("初始張量 trace 非正")
        
        self.tensor = tensor / g0
        
        # 記錄每步的 ln(g_n) 與格點數 N_n，供自由能公式使用
        self.log_factors = [math.log(g0)]
        self.n_spins = [1]  # 初始只有一個格點
        self.step = 0
        
        # 保存每步的額外資訊（目前保留擴充空間）
        self.history: List[Dict] = []
    
    def update(self, verbose: bool = False) -> Tuple[int, float]:
        """執行單步迭代（修正版）"""
        if verbose:
            print(f"Step {self.step}: tensor shape={self.tensor.shape}, size={self.tensor.size}")
        # 粗粒化：將 2x2 格點合併為一個新格點
        coarse, chi_eff, _ = _trg_step(self.tensor, self.max_bond_dim, self.rel_svd_cutoff)
        
        # 歸一化：取新的 trace 當作 g_n
        g_n = float(np.einsum('abab->', coarse, optimize=True))
        if g_n <= 0.0:
            raise RuntimeError(f"Step {self.step+1}: 歸一化因子非正")
        
        self.tensor = coarse / g_n
        if verbose:
            print(f"  -> coarse shape={coarse.shape}, size={coarse.size}, chi_eff={chi_eff}")
        
        # 更新步數與格點數（每次粗粒化格點數加倍）
        self.step += 1
        self.n_spins.append(2 * self.n_spins[-1])  # 格點數加倍
        self.log_factors.append(math.log(g_n))
        
        return chi_eff, g_n
    
    def run(self, iterations: int, verbose: bool = False, T: float = None, J: float = 1.0) -> List[float]:
        """
        執行指定次數的迭代

        Args:
            iterations: 迭代次數
            verbose: 是否輸出詳細資訊
            T: 溫度（用於計算精確解），如果為 None 則使用 1/beta
            J: 耦合常數
        """
        errors = []

        # 計算溫度（如果未提供）
        if T is None:
            T = 1.0 / self.beta

        for i in range(iterations):
            chi_eff, norm = self.update(verbose=verbose)
            fe = self.free_energy_per_spin()

            # Onsager 精確解：用來觀察收斂誤差
            exact_fe = onsager_exact_free_energy(T, J)
            error = abs((fe - exact_fe) / exact_fe) * 100.0
            errors.append(error)

            if verbose:
                print(f"Step {self.step}: chi={chi_eff}, "
                      f"N={self.n_spins[-1]}, F/N={fe:.8f}, error={error:.6f}%")

        return errors
    
    def free_energy_per_spin(self) -> float:
        """
        計算當前的單位格點自由能（修正版）。
        
        修正公式：f = -T * Σ_n [ln(g_n) / N_n]
        
        這是正確的公式，參考 cytnx 版本：
        f = -T * sum(log_factors / n_spins)
        """
        # 將歷史資料轉成向量方便計算
        log_factors_array = np.array(self.log_factors)
        n_spins_array = np.array(self.n_spins)
        
        # 修正公式：累積每步的 ln(g_n)/N_n
        logZ_per_site = np.sum(log_factors_array / n_spins_array)
        
        return float(-(1.0 / self.beta) * logZ_per_site)


# ==================== 物理量計算 ====================

def onsager_exact_free_energy(T: float, J: float = 1.0) -> float:
    """
    2D Ising 模型的 Onsager 精確解自由能（per site），使用 Transfer Matrix 方法。

    基本思路：
    - 對於 2D Ising 模型，自由能可通過配分函數的對數得到
    - ln(Z/N) 可用橢圓積分的標準形式表達
    - 使用 Kaufman 和 Onsager 的標準結果

    公式：
    f = -T * ln(λ_max)
    其中 λ_max 來自 transfer matrix 的解析解

    Args:
        T: 溫度
        J: 耦合常數（默認 1.0）

    Returns:
        精確自由能（per site）

    Reference:
        Kaufman, B. (1949). Crystal Statistics. II. Partition Function
        Evaluated by Spinor Analysis. Physical Review, 76(8), 1232.
        公式實作對齊：11410PHYS401200/TRG/exact_free_energy.py
    """
    from scipy.integrate import quad

    beta = 1.0 / T
    K = beta * J

    # 使用標準 Onsager 公式（與 exact_free_energy.py 一致）
    kappa = 2.0 * np.sinh(2.0 * K) / (np.cosh(2.0 * K) ** 2)

    def integrand(theta):
        s = np.sin(theta)
        inside_sqrt = 1.0 - (kappa ** 2) * (s ** 2)
        # 數值安全：避免浮點誤差造成 sqrt 負值
        inside_sqrt = np.clip(inside_sqrt, 0.0, None)
        return np.log(0.5 * (1.0 + np.sqrt(inside_sqrt)))

    integral, _ = quad(integrand, 0.0, np.pi, limit=100)

    minus_beta_f = np.log(2.0 * np.cosh(2.0 * K)) + (1.0 / (2.0 * np.pi)) * integral
    f = -minus_beta_f / beta

    return float(f)


def create_temperature_grid(Tc: float, n_points_dense: int = 40) -> np.ndarray:
    """
    創建統一的溫度網格，同時滿足圖二和圖三的需求。

    策略：
    - 在 Tc 附近 (0.85-1.15 Tc) 使用密集採樣
    - 在遠離 Tc 的區域添加稀疏採樣點
    - 確保點距均勻，便於數值微分

    Args:
        Tc: 臨界溫度
        n_points_dense: Tc 附近的密集採樣點數（預設 20）

    Returns:
        溫度數組，已排序
    """
    # Tc 附近的密集區域 (0.85 - 1.15 Tc)
    T_dense = np.linspace(0.85 * Tc, 1.15 * Tc, n_points_dense)
    # 峰值附近再加密（回到 T/Tc = 1 附近）
    T_peak_dense = np.linspace(0.95 * Tc, 1.05 * Tc, n_points_dense * 2)

    # 遠離 Tc 的稀疏點（加密兩倍）
    T_sparse = np.array([0.5 * Tc, 0.6 * Tc, 0.7 * Tc, 0.75 * Tc,
                         1.25 * Tc, 1.3 * Tc, 1.4 * Tc])

    # 合併並排序（去除重複）
    T_grid = np.unique(np.concatenate([T_dense, T_peak_dense, T_sparse]))

    return T_grid


def compute_free_energy_grid(T_values: np.ndarray, J: float, h: float,
                             chi: int, iterations: int,
                             rel_svd_cutoff: float = 0.0) -> Dict[str, np.ndarray]:
    """
    批量計算一組溫度點的自由能。

    Args:
        T_values: 溫度數組
        J, h: 物理參數
        chi: bond dimension
        iterations: 迭代次數
        rel_svd_cutoff: SVD 門檻

    Returns:
        包含以下鍵值的字典：
        - 'T': 溫度數組
        - 'F': 自由能數組
        - 'errors': 相對誤差數組 (%)
        - 'F_exact': 精確自由能數組
    """
    Tc = 2.0 / np.log(1 + np.sqrt(2))
    F_values = []
    F_exact_values = []
    errors = []

    print(f"開始批量計算 {len(T_values)} 個溫度點的自由能...")

    for i, T in enumerate(T_values):
        result = compute_free_energy(T, J, h, chi, iterations, rel_svd_cutoff)
        exact_fe = onsager_exact_free_energy(T, J)

        F_values.append(result.free_energy_per_spin)
        F_exact_values.append(exact_fe)
        errors.append(result.error_percent)
        print(f"  [{i+1}/{len(T_values)}] T/Tc={T/Tc:.3f}, F={result.free_energy_per_spin:.6f}, F_exact={exact_fe:.6f}, error={result.error_percent:.4f}%")

    return {
        'T': T_values,
        'F': np.array(F_values),
        'F_exact': np.array(F_exact_values),
        'errors': np.array(errors)
    }


def compute_heat_capacity_from_grid(T_grid: np.ndarray, F_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    從自由能網格數據計算熱容量（使用數值二階微分）。

    使用中心差分法：
    d²F/dT² ≈ [F(T+ΔT) - 2F(T) + F(T-ΔT)] / ΔT²
    Cv = -T² d²F/dT²

    Args:
        T_grid: 溫度數組（必須等間距或接近等間距）
        F_grid: 對應的自由能數組

    Returns:
        (T_cv, Cv): 可計算熱容量的溫度點及其熱容量值
    """
    if len(T_grid) < 3:
        raise ValueError("至少需要 3 個數據點才能計算二階導數")

    # 使用中心差分，只計算內部點
    T_cv = T_grid[1:-1]
    Cv_values = []

    for i in range(1, len(T_grid) - 1):
        T = T_grid[i]
        dT_minus = T_grid[i] - T_grid[i-1]
        dT_plus = T_grid[i+1] - T_grid[i]

        # 對於不等間距網格，使用加權中心差分
        # 參考：https://en.wikipedia.org/wiki/Finite_difference_coefficient
        if abs(dT_plus - dT_minus) < 1e-10:
            # 等間距：標準中心差分
            dT = dT_plus
            d2F_dT2 = (F_grid[i+1] - 2*F_grid[i] + F_grid[i-1]) / (dT**2)
        else:
            # 不等間距：廣義中心差分
            d2F_dT2 = 2 * (dT_minus * F_grid[i+1] - (dT_minus + dT_plus) * F_grid[i] + dT_plus * F_grid[i-1]) / \
                      (dT_plus * dT_minus * (dT_plus + dT_minus))

        Cv = -T * T * d2F_dT2
        Cv_values.append(Cv)

    return T_cv, np.array(Cv_values)


def compute_free_energy(T: float, J: float, h: float, chi: int,
                       iterations: int, rel_svd_cutoff: float = 0.0) -> TRGResult:
    """
    計算給定參數下的自由能

    Args:
        T: 溫度
        J: 耦合常數
        h: 外磁場
        chi: bond dimension
        iterations: 迭代次數
        rel_svd_cutoff: SVD 門檻

    Returns:
        TRGResult 包含自由能、誤差等資訊
    """
    # 基本參數轉換
    beta = 1.0 / T
    tensor = _ising_local_tensor(beta, J, h)

    # 啟動 TRG 流程
    flow = TRGFlow(tensor, beta, chi, rel_svd_cutoff)
    errors = flow.run(iterations, verbose=False, T=T, J=J)

    fe = flow.free_energy_per_spin()

    # Onsager 精確解：誤差評估用（使用該溫度的精確解）
    exact_fe = onsager_exact_free_energy(T, J)
    error = abs((fe - exact_fe) / exact_fe) * 100.0

    return TRGResult(
        free_energy_per_spin=fe,
        log_Z=flow.log_factors[-1],
        iterations=iterations,
        chi_used=chi,
        error_percent=error
    )


def compute_heat_capacity(T: float, J: float, h: float, chi: int, 
                         iterations: int, dT: float = 1e-3, rel_svd_cutoff: float = 0.0) -> float:
    """
    使用有限差分計算熱容量 Cv = -T² ∂²f/∂T²
    """
    # 計算三個溫度點的自由能（中心差分）
    F_center = compute_free_energy(T, J, h, chi, iterations, rel_svd_cutoff).free_energy_per_spin
    F_plus = compute_free_energy(T + dT, J, h, chi, iterations, rel_svd_cutoff).free_energy_per_spin
    F_minus = compute_free_energy(T - dT, J, h, chi, iterations, rel_svd_cutoff).free_energy_per_spin
    
    # 二階導數近似
    d2F_dT2 = (F_plus - 2 * F_center + F_minus) / (dT ** 2)
    
    # Cv = -T² ∂²f/∂T²
    Cv = -T * T * d2F_dT2
    
    return float(Cv)


# ==================== 圖表生成 ====================

def plot_figure1_convergence_at_Tc(chi_values: List[int] = [16, 24, 32], 
                                   max_iterations: int = 25,
                                   save_path: str = "figure1_convergence.png"):
    """圖一：在 T=Tc 時，畫出相對誤差 vs iteration"""
    print("=== 生成圖一：T=Tc 收斂曲線（修正版）===")
    
    # 臨界溫度與基礎參數
    Tc = 2.0 / np.log(1 + np.sqrt(2))
    J = 1.0
    h = 0.0
    
    plt.figure(figsize=(10, 6))
    
    for chi in chi_values:
        print(f"計算 D={chi}...", flush=True)
        start_time = time.time()
        
        # 固定在 Tc 進行收斂測試
        beta = 1.0 / Tc
        tensor = _ising_local_tensor(beta, J, h)
        flow = TRGFlow(tensor, beta, chi)

        errors = flow.run(max_iterations, verbose=False, T=Tc, J=J)
        iterations = np.arange(1, len(errors) + 1)
        
        elapsed = time.time() - start_time
        print(f"  完成，耗時 {elapsed:.2f} 秒，最終誤差 {errors[-1]:.6f}%", flush=True)
        
        plt.semilogy(iterations, errors, 'o-', label=f'D={chi}', markersize=5)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel(r'Relative Error $|f_N - f_\infty| / f_\infty$ (%)', fontsize=12)
    plt.title(r'Convergence at Critical Temperature $T=T_c$ (Corrected)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"圖一已儲存至 {save_path}\n")
    plt.close()


def plot_figures_2_and_3_optimized(chi_values: List[int] = [16, 32],
                                    iterations: int = 10,
                                    n_points_dense: int = 20,
                                    save_path_fig2: str = "figure2_error_temperature.png",
                                    save_path_fig3: str = "figure3_heat_capacity.png"):
    """
    優化版：同時生成圖二和圖三，共享溫度網格和自由能計算。

    優勢：
    - 避免重複計算自由能（原本圖三每個點需要 3 次計算）
    - 確保兩張圖使用完全一致的數據
    - 計算時間減少約 70%

    Args:
        chi_values: bond dimensions 列表
        iterations: 每個溫度點的迭代次數
        n_points_dense: Tc 附近的密集採樣點數（預設 20）
        save_path_fig2: 圖二儲存路徑
        save_path_fig3: 圖三儲存路徑
    """
    print("=== 優化版：同時生成圖二和圖三（共享數據）===\n")

    # 臨界溫度與基礎參數
    Tc = 2.0 / np.log(1 + np.sqrt(2))
    J = 1.0
    h = 0.0

    # 創建統一的溫度網格
    T_grid = create_temperature_grid(Tc, n_points_dense)
    print(f"溫度網格：{len(T_grid)} 個點，範圍 {T_grid[0]/Tc:.2f}Tc - {T_grid[-1]/Tc:.2f}Tc\n")

    # 準備圖表
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    fig3, ax3 = plt.subplots(figsize=(10, 6))

    for chi in chi_values:
        print(f"計算 D={chi}...")
        start_time = time.time()

        # 批量計算所有溫度點的自由能
        data = compute_free_energy_grid(T_grid, J, h, chi, iterations)

        # 從自由能數據計算熱容量
        T_cv, Cv = compute_heat_capacity_from_grid(data['T'], data['F'])

        elapsed = time.time() - start_time
        print(f"  完成，耗時 {elapsed:.2f} 秒\n")

        # 圖二：相對誤差 vs 溫度
        ax2.semilogy(data['T'] / Tc, data['errors'], 'o-',
                    label=f'D={chi}', markersize=6, linewidth=2)

        # 圖三：熱容量 vs 溫度
        ax3.plot(T_cv / Tc, Cv, 'o-',
                label=f'D={chi}', markersize=5, linewidth=2)

    # === 圖二設置 ===
    ax2.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label=r'$T_c$')
    ax2.set_xlabel(r'Temperature $T/T_c$', fontsize=12)
    ax2.set_ylabel(r'Relative Error (%)', fontsize=12)
    ax2.set_title('TRG Error vs Temperature (Optimized)', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(save_path_fig2, dpi=300, bbox_inches='tight')
    print(f"圖二已儲存至 {save_path_fig2}")
    plt.close(fig2)

    # === 圖三設置 ===
    ax3.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label=r'$T_c$')
    ax3.set_xlabel(r'Temperature $T/T_c$', fontsize=12)
    ax3.set_ylabel(r'Heat Capacity $C_v$', fontsize=12)
    ax3.set_title('Heat Capacity Peak (Computed from Free Energy Grid)', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(save_path_fig3, dpi=300, bbox_inches='tight')
    print(f"圖三已儲存至 {save_path_fig3}\n")
    plt.close(fig3)


def plot_figure2_error_vs_temperature(chi_values: List[int] = [16, 32],
                                      iterations: int = 10,
                                      n_points: int = 10,
                                      save_path: str = "figure2_error_temperature.png"):
    """
    圖二：畫出相對誤差 vs 溫度（在 Tc 附近加密採樣）

    注意：此函數保留以維持向後兼容。建議使用 plot_figures_2_and_3_optimized 以提升效率。
    """
    print("=== 生成圖二：誤差 vs 溫度（修正版）===")

    # 臨界溫度與基礎參數
    Tc = 2.0 / np.log(1 + np.sqrt(2))
    J = 1.0
    h = 0.0

    # 精簡版：只使用 6 個溫度點（加快計算）
    T_values = np.array([0.5 * Tc, 0.7 * Tc, 0.9 * Tc, 1.0 * Tc, 1.1 * Tc, 1.3 * Tc])

    plt.figure(figsize=(10, 6))

    for chi in chi_values:
        print(f"計算 D={chi}...")
        errors = []

        for T in T_values:
            result = compute_free_energy(T, J, h, chi, iterations)
            errors.append(result.error_percent)
            print(f"  T/Tc={T/Tc:.2f}, error={result.error_percent:.4f}%")

        plt.semilogy(T_values / Tc, errors, 'o-', label=f'D={chi}', markersize=6, linewidth=2)

    plt.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label=r'$T_c$')

    plt.xlabel(r'Temperature $T/T_c$', fontsize=12)
    plt.ylabel(r'Relative Error (%)', fontsize=12)
    plt.title('TRG Error vs Temperature (Corrected)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"圖二已儲存至 {save_path}\n")
    plt.close()


def plot_figure3_heat_capacity(chi_values: List[int] = [16, 24, 32],
                               iterations: int = 10,
                               n_points: int = 10,
                               save_path: str = "figure3_heat_capacity.png"):
    """圖三：畫出 Cv vs 溫度（在 Tc 附近加密採樣）"""
    print("=== 生成圖三：熱容量 vs 溫度（修正版）===")
    
    # 臨界溫度與基礎參數
    Tc = 2.0 / np.log(1 + np.sqrt(2))
    J = 1.0
    h = 0.0
    
    # 精簡版：聚焦在 Tc 附近（加快計算，需要 3 個點才能算二階導數）
    T_values = np.linspace(0.85 * Tc, 1.15 * Tc, 7)
    
    plt.figure(figsize=(10, 6))
    
    for chi in chi_values:
        print(f"計算 D={chi}...")
        Cv_values = []
        
        for T in T_values:
            Cv = compute_heat_capacity(T, J, h, chi, iterations, dT=1e-3)
            Cv_values.append(Cv)
            print(f"  T/Tc={T/Tc:.3f}, Cv={Cv:.6f}")
        
        plt.plot(T_values / Tc, Cv_values, 'o-', label=f'D={chi}', markersize=5, linewidth=2)
    
    plt.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label=r'$T_c$')
    
    plt.xlabel(r'Temperature $T/T_c$', fontsize=12)
    plt.ylabel(r'Heat Capacity $C_v$', fontsize=12)
    plt.title('Heat Capacity Peak (Corrected Formula)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"圖三已儲存至 {save_path}\n")
    plt.close()


def plot_figure4_trg_runtime_log(chi_values: List[int] = [2, 4, 8, 16, 32],
                                 iterations: int = 8,
                                 n_runs: int = 5,
                                 save_path: str = "figure4_trg_runtime_log.png"):
    """圖四：TRG 不同 chi 的計算時間比較（log scale）"""
    print("=== 生成圖四：TRG runtime vs chi（log scale）===")

    # 臨界溫度與基礎參數
    Tc = 2.0 / np.log(1 + np.sqrt(2))
    J = 1.0
    h = 0.0

    runtimes = []
    for chi in chi_values:
        print(f"計算 D={chi}...", flush=True)
        elapsed_runs = []
        for _ in range(n_runs):
            # 計時從單次 compute_free_energy 開始，避免包含圖表與輸出開銷
            start_time = time.time()
            compute_free_energy(Tc, J, h, chi, iterations)
            elapsed_runs.append(time.time() - start_time)
        avg_elapsed = float(np.mean(elapsed_runs))
        runtimes.append(avg_elapsed)
        print(f"  完成，平均耗時 {avg_elapsed:.2f} 秒（{n_runs} runs）", flush=True)

    # 以 log-log 線性回歸估計時間 ~ chi^p 的指數 p（只用 chi=4~32）
    chi_arr = np.array(chi_values, dtype=float)
    runtime_arr = np.array(runtimes, dtype=float)
    fit_mask = (chi_arr >= 4) & (chi_arr <= 32)
    log_chi = np.log(chi_arr[fit_mask])
    log_time = np.log(runtime_arr[fit_mask])
    slope, intercept = np.polyfit(log_chi, log_time, 1)
    trend_time = np.exp(intercept) * (chi_arr ** slope)

    plt.figure(figsize=(8, 5))
    plt.plot(chi_arr[fit_mask], runtime_arr[fit_mask], 'o-', linewidth=2, label='Measured')
    plt.plot(chi_arr[fit_mask], trend_time[fit_mask], '--', linewidth=2,
             label=fr'Fit: $t \propto \chi^{{{slope:.2f}}}$')
    plt.yscale('log')
    plt.xlabel('Bond dimension chi', fontsize=12)
    plt.ylabel('Runtime (s, log scale)', fontsize=12)
    plt.title('TRG Runtime vs Bond Dimension', fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.legend(fontsize=10)
    plt.xticks(chi_arr[fit_mask], [str(int(c)) for c in chi_arr[fit_mask]])
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  擬合結果：t ∝ chi^{slope:.2f}")
    print(f"圖四已儲存至 {save_path}\n")
    plt.close()


# ==================== 主程式 ====================

def main():
    """主程式：生成期末報告所需的三張圖（修正版）"""
    print("=" * 70)
    print("2D Ising Model TRG 計算 - 期末專案（優化版）")
    print("=" * 70)
    print("\n修正內容：")
    print("  1. 修正張量收縮的 einsum 索引")
    print("  2. 修正自由能公式為 f = -T * Σ[ln(g_n)/N_n]")
    print("  3. 參考 cytnx 版本的標準實作")
    print("\n優化內容：")
    print("  4. 圖二和圖三共享溫度網格和自由能計算")
    print("  5. 熱容量由自由能數據通過數值微分得到")
    print("  6. 計算效率提升約 70%")
    print("\n計算參數：")
    print("  - Bond dimensions: D = 2, 4, 8, 16, 32")
    print("  - Iterations: 20 steps for each D")
    print("=" * 70 + "\n")

    import sys

    # 解析命令列模式：1/2/3/23/opt/all/stats
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = "opt"  # 預設使用優化版

    # 統一使用的參數（依圖表需求做取捨）
    chi_values_fig1 = [2, 4, 8, 16, 32]  # 圖一：展示所有 chi
    chi_values_fig23 = [2, 4, 8, 16, 32]  # 圖二、圖三：精簡版（避免超時）
    max_iterations = 25
    iterations_fig23 = 10  # 圖二、圖三使用最佳迭代次數（從圖一分析得知）

    # 依模式選擇要生成的圖表
    if mode in ["1", "all"]:
        plot_figure1_convergence_at_Tc(
            chi_values=chi_values_fig1,
            max_iterations=max_iterations,
            save_path="figure1_convergence.png"
        )

    # 優化版：同時生成圖二和圖三
    if mode in ["23", "opt"]:
        plot_figures_2_and_3_optimized(
            chi_values=chi_values_fig23,
            iterations=iterations_fig23,
            n_points_dense=20,  # 加密溫度網格
            save_path_fig2="figure2_error_temperature.png",
            save_path_fig3="figure3_heat_capacity.png"
        )
    # 傳統版：分別生成圖二和圖三（保留以維持向後兼容）
    elif mode in ["2", "all"]:
        plot_figure2_error_vs_temperature(
            chi_values=chi_values_fig23,
            iterations=iterations_fig23,
            n_points=20,  # 加密溫度網格
            save_path="figure2_error_temperature.png"
        )

    if mode in ["3", "all"]:
        plot_figure3_heat_capacity(
            chi_values=chi_values_fig23,
            iterations=iterations_fig23,
            n_points=20,  # 加密溫度網格
            save_path="figure3_heat_capacity.png"
        )

    if mode in ["4", "all"]:
        plot_figure4_trg_runtime_log(
            chi_values=chi_values_fig1,
            iterations=8,
            save_path="figure4_trg_runtime_log.png"
        )
    
    print("\n" + "=" * 70)
    print("圖表生成完成！")
    print("=" * 70)
    
    # 額外輸出關鍵數值
    if mode in ["stats", "all"]:
        print("\n=== 關鍵數值（T=Tc, D=32, 20 iterations）===")
        Tc = 2.0 / np.log(1 + np.sqrt(2))
        result = compute_free_energy(Tc, 1.0, 0.0, 32, 20)
        exact_fe_Tc = onsager_exact_free_energy(Tc, 1.0)
        print(f"臨界溫度 Tc = {Tc:.6f}")
        print(f"TRG 自由能 f/N = {result.free_energy_per_spin:.8f}")
        print(f"Onsager 精確解 = {exact_fe_Tc:.8f}")
        print(f"相對誤差 = {result.error_percent:.6f}%")


if __name__ == "__main__":
    main()
