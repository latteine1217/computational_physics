import numpy as np
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import time
import math
import warnings
from dataclasses import dataclass, field

# -------------------------
# 基本工具：bit <-> spin
# -------------------------
def bit_to_spin(bits: int, idx: int) -> int:
    """bit -> spin in {-1,+1}；idx 是線性索引（0..N-1）"""
    return 1 if ((bits >> idx) & 1) else -1

def idx2d(i: int, j: int, Lx: int) -> int:
    """把 2D 索引 (i,j) 映到線性索引"""
    return i * Lx + j

def ij_from_idx(idx: int, Lx: int) -> Tuple[int, int]:
    """把線性索引映回 2D 索引 (i,j)"""
    i = idx // Lx
    j = idx % Lx
    return i, j

# -------------------------
# 2D：整體能量
# -------------------------
def energy_ising2d_from_bits(bits: int, Lx: int, Ly: int,
                             J: float, h: float = 0.0,
                             periodic: bool = True) -> float:
    """
    直接根據 bits 計算 2D Ising 能量（O(N)）
    H = -J * sum_{<r,r'>} s_r s_{r'} - h * sum_r s_r
    這裡只對 +x、+y 鄰居求和，避免 double count。
    """
    E = 0.0
    M = 0
    # 場項（磁化）
    N = Lx * Ly
    for t in range(N):
        M += bit_to_spin(bits, t)
    E += -h * M

    # 交換項：每格與 +x、+y 鄰居
    for i in range(Ly):
        for j in range(Lx):
            s = bit_to_spin(bits, idx2d(i, j, Lx))
            # +x 邊
            if j + 1 < Lx:
                s_right = bit_to_spin(bits, idx2d(i, j + 1, Lx))
                E += -J * s * s_right
            elif periodic and Lx >= 2:
                s_right = bit_to_spin(bits, idx2d(i, 0, Lx))
                E += -J * s * s_right
            # +y 邊
            if i + 1 < Ly:
                s_down = bit_to_spin(bits, idx2d(i + 1, j, Lx))
                E += -J * s * s_down
            elif periodic and Ly >= 2:
                s_down = bit_to_spin(bits, idx2d(0, j, Lx))
                E += -J * s * s_down

    return E

# -------------------------
# 2D：單點翻轉的 ΔE（O(1)）
# -------------------------
def deltaE_ising2d_flip(bits: int, Lx: int, Ly: int, idx: int,
                        J: float, h: float = 0.0,
                        periodic: bool = True) -> float:
    """
    翻轉 site idx 的 ΔE：
      ΔE = 2 s_i ( J * sum_{4個最近鄰} s_j + h )
    週期/開邊界皆處理。
    """
    i, j = ij_from_idx(idx, Lx)
    si = bit_to_spin(bits, idx)

    nn_sum = 0
    # 上 (i-1, j)
    if i - 1 >= 0:
        nn_sum += bit_to_spin(bits, idx2d(i - 1, j, Lx))
    elif periodic and Ly >= 2:
        nn_sum += bit_to_spin(bits, idx2d(Ly - 1, j, Lx))
    # 下 (i+1, j)
    if i + 1 < Ly:
        nn_sum += bit_to_spin(bits, idx2d(i + 1, j, Lx))
    elif periodic and Ly >= 2:
        nn_sum += bit_to_spin(bits, idx2d(0, j, Lx))
    # 左 (i, j-1)
    if j - 1 >= 0:
        nn_sum += bit_to_spin(bits, idx2d(i, j - 1, Lx))
    elif periodic and Lx >= 2:
        nn_sum += bit_to_spin(bits, idx2d(i, Lx - 1, Lx))
    # 右 (i, j+1)
    if j + 1 < Lx:
        nn_sum += bit_to_spin(bits, idx2d(i, j + 1, Lx))
    elif periodic and Lx >= 2:
        nn_sum += bit_to_spin(bits, idx2d(i, 0, Lx))

    return 2.0 * si * (J * nn_sum + h)

# -------------------------
# Gray code 工具：下一步要翻的位元位置
# -------------------------
def gray_next_flip_pos(t: int) -> int:
    """
    已在步 t 加入當前組態後，下一步 (t+1) 的 Gray code 與當前差一個 bit。
    回傳要翻轉的 bit 位置（最右側 set bit 的索引）。
    """
    g  = t ^ (t >> 1)
    g1 = (t + 1) ^ ((t + 1) >> 1)
    diff = g ^ g1
    pos = 0
    while ((diff >> pos) & 1) == 0:
        pos += 1
    return pos

# -------------------------
# 2D：全枚舉（Gray + ΔE）
# -------------------------
def ising2d_all_energies_gray(Lx: int, Ly: int, J: float, h: float = 0.0, periodic: bool = True):
    """
    遍歷所有 2^(Lx*Ly) 組態，回傳：
      energies: np.ndarray shape (2^N,)
      mags:     np.ndarray shape (2^N,)
      degeneracy: Dict[energy_value, count]（能量退化度）
    使用 Gray code + ΔE，初始 bits=0（全 -1）。
    """
    if Lx <= 0 or Ly <= 0:
        raise ValueError("Lx, Ly 必須 >= 1")
    N = Lx * Ly
    total = 1 << N  # 2^N

    energies = np.empty(total, dtype=np.float64)
    mags     = np.empty(total, dtype=np.int32)

    bits = 0
    E = energy_ising2d_from_bits(bits, Lx, Ly, J, h, periodic)  # 初始能量
    M = -N  # 全 -1

    for t in range(total):
        energies[t] = E
        mags[t]     = M

        if t == total - 1:
            break

        # 取得下一步要翻的線性索引（0..N-1）
        flip_idx = gray_next_flip_pos(t)

        # ΔE 使用翻轉前的 bits 計
        dE = deltaE_ising2d_flip(bits, Lx, Ly, flip_idx, J, h, periodic)

        # 進入下一個狀態：翻 bit
        bits ^= (1 << flip_idx)

        # 更新 E、M
        E += dE
        s_new = bit_to_spin(bits, flip_idx)  # 翻後的自旋值
        M += 2 * s_new                       # 原本是 -s_new，翻後是 s_new → ΔM = 2*s_new

    # 能量退化度
    uniqE, counts = np.unique(energies, return_counts=True)
    degeneracy = {float(e): int(c) for e, c in zip(uniqE, counts)}
    return energies, mags, degeneracy

# -------------------------
# 熱力學量（Z、<E>、<M>、C_v）：以 log-sum-exp 穩定化
# -------------------------
def partition_stats(energies: np.ndarray, mags: np.ndarray, beta: float):
    """
    給定所有能量與磁化，計算 Z、<E>、<M>、<M^2>、<E^2>、C_v（每自旋可自行除以 N）。
    以 log-sum-exp 避免 overflow。
    """
    a = -beta * energies
    amax = np.max(a)
    wa = np.exp(a - amax)
    norm = wa.sum()
    Z = norm * np.exp(amax)
    Ew = (energies * wa).sum() / norm
    Mw = (mags * wa).sum() / norm
    M2w = ((mags**2) * wa).sum() / norm
    E2w = ((energies**2) * wa).sum() / norm
    Cv = beta**2 * (E2w - Ew**2)
    return Z, Ew, Mw, M2w, Cv

# ---------- 小工具：row-wise 的 logsumexp，避免 under/overflow ----------
def logsumexp_rows(x: np.ndarray) -> np.ndarray:
    """針對每一列套用 log-sum-exp，善用列最大值避免浮點溢位。"""
    m = np.max(x, axis=1, keepdims=True)
    return (m.squeeze() + np.log(np.sum(np.exp(x - m), axis=1)))


@dataclass(frozen=True)
class MethodResult:
    """整合各演算法輸出的宏觀物理量與額外元資料。"""
    method: str
    free_energy_per_spin: float
    susceptibility_per_spin: float
    heat_capacity_per_spin: float
    runtime: float
    metadata: Dict[str, float] = field(default_factory=dict)


def enumeration_observables_2d(Lx: int, Ly: int, T: float, J: float, h: float = 0.0,
                               periodic: bool = True) -> MethodResult:
    """以枚舉方式計算 2D Ising 在單一溫度下的統計量。"""
    start = time.perf_counter()
    energies, mags, _ = ising2d_all_energies_gray(Lx, Ly, J, h, periodic)
    beta = 1.0 / T
    Z, E_mean, M_mean, M2_mean, Cv = partition_stats(energies, mags, beta)
    runtime = time.perf_counter() - start

    N = Lx * Ly
    free_energy_total = -(1.0 / beta) * math.log(Z)
    free_energy_per_spin = free_energy_total / N
    susceptibility_per_spin = beta * (M2_mean - M_mean**2) / N
    heat_capacity_per_spin = Cv / N
    metadata = {
        "partition_function": Z,
        "energy_mean": E_mean,
        "magnetization_mean_per_spin": M_mean / N,
        "specific_heat": Cv / N,
        "heat_capacity_per_spin": heat_capacity_per_spin,
        "total_spins": float(N),
    }
    return MethodResult(
        method="enumeration",
        free_energy_per_spin=free_energy_per_spin,
        susceptibility_per_spin=susceptibility_per_spin,
        heat_capacity_per_spin=heat_capacity_per_spin,
        runtime=runtime,
        metadata=metadata,
    )


def _row_spin_array(bits: int, Lx: int) -> np.ndarray:
    """將單列 bitmask 轉為 {-1,+1} 自旋陣列，供列轉移矩陣重用。"""
    arr = np.empty(Lx, dtype=np.int8)
    for x in range(Lx):
        arr[x] = bit_to_spin(bits, x)
    return arr


def _row_internal_coupling(spins: np.ndarray, periodic: bool) -> int:
    """計算單列自旋的水平耦合 Σ s_i s_{i+1}，考慮是否套用 PBC。"""
    total = 0
    Lx = spins.shape[0]
    for x in range(Lx - 1):
        total += spins[x] * spins[x + 1]
    if periodic and Lx >= 2:
        total += spins[-1] * spins[0]
    return int(total)


def _prepare_row_data(Lx: int, periodic: bool):
    """預先生成單列所有自旋組態及其總和、水平耦合，供轉移矩陣重複使用。"""
    dim = 1 << Lx
    row_spins = np.empty((dim, Lx), dtype=np.int8)
    row_spin_sum = np.empty(dim, dtype=np.int16)
    row_horizontal = np.empty(dim, dtype=np.int16)
    for state in range(dim):
        spins = _row_spin_array(state, Lx)
        row_spins[state] = spins
        row_spin_sum[state] = int(np.sum(spins))
        row_horizontal[state] = _row_internal_coupling(spins, periodic)
    return row_spins, row_spin_sum, row_horizontal


def _build_transfer_matrices(Lx: int, beta: float, J: float, h: float,
                             periodic: bool = True):
    """建立二維列轉移矩陣及對外場導數矩陣。"""
    row_spins, row_spin_sum, row_horizontal = _prepare_row_data(Lx, periodic)
    dim = row_spins.shape[0]
    T = np.empty((dim, dim), dtype=np.float64)
    dT = np.empty((dim, dim), dtype=np.float64)
    d2T = np.empty((dim, dim), dtype=np.float64)
    half_beta = 0.5 * beta

    max_exponent = -math.inf

    # 先儲存指數與導數 pref，以便後續一次性平移；避免直接取指數造成溢位
    for a in range(dim):
        spins_a = row_spins[a]
        spin_sum_a = row_spin_sum[a]
        horiz_a = row_horizontal[a]
        for b in range(dim):
            spins_b = row_spins[b]
            spin_sum_b = row_spin_sum[b]
            horiz_b = row_horizontal[b]
            vertical = int(np.dot(spins_a, spins_b))
            exponent = beta * (
                0.5 * J * (horiz_a + horiz_b)
                + J * vertical
                + 0.5 * h * (spin_sum_a + spin_sum_b)
            )
            if exponent > max_exponent:
                max_exponent = exponent
            T[a, b] = exponent
            sum_field = spin_sum_a + spin_sum_b
            pref = half_beta * sum_field
            dT[a, b] = pref
            d2T[a, b] = pref * pref

    # 以最大指數作平移，確保矩陣元素維持在數值穩定範圍
    np.subtract(T, max_exponent, out=T)
    np.exp(T, out=T)
    np.multiply(dT, T, out=dT)
    np.multiply(d2T, T, out=d2T)

    return T, dT, d2T, max_exponent


def _transfer_matrix_stats(Lx: int, Ly: int, beta: float, J: float, h: float,
                           periodic: bool = True):
    """計算列轉移矩陣的 logZ 及其對外場一階、二階導數。"""
    if not periodic:
        raise NotImplementedError("2D transfer matrix 目前僅支援週期邊界條件")

    T, dT, d2T, log_scale = _build_transfer_matrices(Lx, beta, J, h, periodic)
    dim = T.shape[0]
    identity = np.eye(dim, dtype=np.float64)
    prefix = [identity]
    for _ in range(Ly):
        prefix.append(prefix[-1] @ T)

    Z = np.trace(prefix[Ly])
    if Z <= 0:
        raise RuntimeError("Transfer matrix partition function 非正，請檢查參數")

    T_power_minus1 = prefix[Ly - 1]
    dZ = Ly * float(np.trace(dT @ T_power_minus1))
    sum_term = 0.0
    if Ly >= 2:
        for k in range(Ly - 1):
            sum_term += float(np.trace((dT @ prefix[k]) @ dT @ prefix[Ly - 2 - k]))
    d2Z = Ly * float(np.trace(d2T @ T_power_minus1)) + Ly * sum_term

    logZ = math.log(Z) + Ly * log_scale
    d_logZ = dZ / Z
    d2_logZ = d2Z / Z - (dZ / Z) ** 2
    return logZ, d_logZ, d2_logZ, dim


def _adaptive_field_eps(N: int, beta: float, base_eps: float = 1e-8) -> float:
    """
    改進的自適應步長策略，基於專家分析的建議
    
    Args:
        N: 總自旋數
        beta: 逆溫度
        base_eps: 基礎步長（改進為更精確的 1e-8）
        
    Returns:
        adaptive_eps: 調整後的步長
    """
    # 更強的系統尺寸依賴：N^0.6 而非 N^0.5
    size_factor = 1.0 / (N ** 0.6)
    
    # 低溫保護：避免極低溫時步長過大
    temp_factor = min(1.0, 2.0 * beta)
    
    # 計算自適應步長
    adaptive_eps = base_eps * size_factor * temp_factor
    
    # 設定更嚴格的下界保護
    return max(adaptive_eps, 1e-12)


def transfer_matrix_observables_2d(Lx: int, Ly: int, T: float, J: float, h: float = 0.0,
                                   periodic: bool = True,
                                   field_eps: float | None = None) -> MethodResult:
    """利用列轉移矩陣解析導數，輸出 2D 系統自由能與磁化率、熱容量。"""
    if not periodic:
        raise NotImplementedError("transfer matrix 方法目前僅支援週期邊界條件")
    beta = 1.0 / T
    delta_beta = max(1e-5, 1e-3 * beta)
    if beta - delta_beta <= 0.0:
        delta_beta = 0.5 * beta

    start = time.perf_counter()

    N = Lx * Ly
    logZ, d_logZ, d2_logZ, dim = _transfer_matrix_stats(Lx, Ly, beta, J, h, periodic)
    logZ_beta_plus, _, _, _ = _transfer_matrix_stats(Lx, Ly, beta + delta_beta, J, h, periodic)
    logZ_beta_minus, _, _, _ = _transfer_matrix_stats(Lx, Ly, beta - delta_beta, J, h, periodic)
    runtime = time.perf_counter() - start

    free_energy_per_spin = -(logZ / beta) / N
    magnetization_per_spin = d_logZ / (beta * N)
    susceptibility_per_spin = d2_logZ / (beta * N)
    d2_logZ_dbeta2 = (logZ_beta_plus - 2.0 * logZ + logZ_beta_minus) / (delta_beta ** 2)
    heat_capacity_per_spin = (beta**2 * d2_logZ_dbeta2) / N

    metadata = {
        "log_partition_function": logZ,
        "magnetization_per_spin": magnetization_per_spin,
        "derivative_mode": "analytic",
        "heat_capacity_per_spin": heat_capacity_per_spin,
        "row_dimension": float(dim),
        "total_spins": float(N),
    }

    return MethodResult(
        method="transfer_matrix",
        free_energy_per_spin=free_energy_per_spin,
        susceptibility_per_spin=susceptibility_per_spin,
        heat_capacity_per_spin=heat_capacity_per_spin,
        runtime=runtime,
        metadata=metadata,
    )


def _is_power_of_two(n: int) -> bool:
    """檢查正整數是否為 2 的冪次，TRG 粗粒化需求。"""
    return n > 0 and (n & (n - 1) == 0)


def _trg_supports_lattice(Lx: int, Ly: int, periodic: bool) -> bool:
    """TRG 需要正方形且邊長為 2 的冪次，並假設週期邊界。"""
    if not periodic:
        return False
    if Lx != Ly:
        return False
    return _is_power_of_two(Lx) and _is_power_of_two(Ly)


def _build_trg_initial_tensor(beta: float, J: float, h: float) -> np.ndarray:
    """
    修正的 TRG 初始張量構造，基於標準物理分解
    
    參考：Levin-Nave (2007) 和標準張量網路理論
    
    Args:
        beta: 逆溫度 1/T
        J: 近鄰耦合強度  
        h: 外磁場強度
        
    Returns:
        T[α,β,γ,δ]: 4-階初始張量，對應 [up, right, down, left]
    """
    # 簡化且正確的 TRG 權重構造
    # 使用標準的 cosh/sinh 分解
    cosh_val = np.cosh(beta * J)
    sinh_val = np.sinh(beta * J)
    
    # 權重矩陣：w[α, s] 其中 α∈{0,1} 是鍵索引，s∈{0,1} 是自旋索引 (-1,+1)
    w = np.array([
        [np.sqrt(cosh_val), np.sqrt(cosh_val)],     # α=0: 平行自旋權重
        [np.sqrt(sinh_val), -np.sqrt(sinh_val)]     # α=1: 反平行自旋權重  
    ], dtype=np.float64)
    
    # 初始化張量 T[up, right, down, left]
    T = np.zeros((2, 2, 2, 2), dtype=np.float64)
    
    # 構造張量元素：對中心自旋求和
    for up in range(2):
        for right in range(2):
            for down in range(2):
                for left in range(2):
                    # 對兩個可能的中心自旋值求和
                    for s_idx in range(2):
                        s = 2 * s_idx - 1  # 轉換：0->-1, 1->+1
                        
                        # 外場貢獻：每個自旋承擔完整的外場能量
                        field_weight = np.exp(beta * h * s)
                        
                        # 四個鍵的權重貢獻
                        # 每個鍵根據其指向的自旋狀態來選擇權重
                        bond_weight = (w[up, s_idx] * w[right, s_idx] * 
                                     w[down, s_idx] * w[left, s_idx])
                        
                        T[up, right, down, left] += field_weight * bond_weight
    
    return T


def _trg_step(T: np.ndarray, chi: int) -> Tuple[np.ndarray, int]:
    """
    標準 TRG 步驟：2×2 → 1×1 粗粒化
    參考：Levin & Nave (2007)
    """
    dim = T.shape[0]
    
    # SVD 分解：將張量重組為矩陣進行分解
    T_perm = np.transpose(T, (0, 3, 1, 2))  # (up,left,right,down)
    mat = T_perm.reshape(dim * dim, dim * dim)
    U, S, Vh = np.linalg.svd(mat, full_matrices=False)
    
    # 截斷到目標維度
    chi_eff = int(min(chi, len(S)))
    U = U[:, :chi_eff]
    S = S[:chi_eff]
    Vh = Vh[:chi_eff, :]

    # 平均分配奇異值
    sqrtS = np.sqrt(S)
    U = U * sqrtS
    Vh = (sqrtS[:, None] * Vh)

    # 重構為張量
    S1 = U.reshape(dim, dim, chi_eff)          # (up, left, α)
    S2 = Vh.reshape(chi_eff, dim, dim)         # (α, right, down)

    # 標準 TRG 收縮公式
    T_new = np.einsum('api,ibj,cjq,qda->abcd', S1, S2, S1, S2)
    return T_new, chi_eff


def _trg_logZ(Lx: int, Ly: int, beta: float, J: float, h: float,
              chi: int, periodic: bool = True) -> Tuple[float, Dict[str, float]]:
    """在平方晶格上執行多層 TRG，回傳 logZ 以及截斷歷程資訊。"""
    if not periodic:
        raise NotImplementedError("TRG 僅支援週期邊界條件")
    if Lx != Ly:
        raise ValueError("TRG 目前僅支援正方形格點 (Lx == Ly)")
    if not (_is_power_of_two(Lx) and _is_power_of_two(Ly)):
        raise ValueError("TRG 需要 Lx, Ly 為 2 的冪次，以利 coarse-graining")

    iterations = int(round(math.log2(Lx)))
    T = _build_trg_initial_tensor(beta, J, h)
    n_sites = Lx * Ly
    log_scale = 0.0
    chi_history = []

    for level in range(iterations):
        T, chi_eff = _trg_step(T, chi)
        chi_history.append(float(chi_eff))
        norm = np.max(np.abs(T))
        if norm <= 0.0:
            raise RuntimeError("TRG tensor 規範為零，數值崩潰")
        T = T / norm
        log_scale += math.log(norm)
        n_sites //= 4

    final_scalar = np.einsum('aabb->', T)
    if final_scalar <= 0:
        raise RuntimeError("TRG 結果為非正值，請檢查參數或截斷精度")
    logZ = log_scale + math.log(final_scalar)
    meta = {
        "iterations": float(iterations),
        "chi_used_last": float(T.shape[0]),
        "chi_history_mean": float(np.mean(chi_history)) if chi_history else 0.0,
    }
    return logZ, meta


def trg_observables_2d(Lx: int, Ly: int, T: float, J: float, h: float = 0.0,
                       periodic: bool = True, chi: int = 32,
                       field_eps: float | None = None) -> MethodResult:
    """以 TRG 估算單一溫度下的自由能、磁化率與熱容量，本函式包辦差分計算。"""
    beta = 1.0 / T
    N = Lx * Ly

    if field_eps is not None:
        eps = field_eps
    else:
        eps = max(2e-3, 2e-2 / np.sqrt(N))

    delta_beta = max(5e-4, 5e-3 / np.sqrt(N))
    if beta - delta_beta <= 0.0:
        delta_beta = 0.5 * beta

    start = time.perf_counter()
    logZ, meta = _trg_logZ(Lx, Ly, beta, J, h, chi, periodic)
    logZ_plus, _ = _trg_logZ(Lx, Ly, beta, J, h + eps, chi, periodic)
    logZ_minus, _ = _trg_logZ(Lx, Ly, beta, J, h - eps, chi, periodic)
    logZ_beta_plus, _ = _trg_logZ(Lx, Ly, beta + delta_beta, J, h, chi, periodic)
    logZ_beta_minus, _ = _trg_logZ(Lx, Ly, beta - delta_beta, J, h, chi, periodic)
    runtime = time.perf_counter() - start

    N = Lx * Ly
    free_energy_per_spin = -(logZ / beta) / N
    magnetization_per_spin = (logZ_plus - logZ_minus) / (2.0 * eps * beta * N)
    susceptibility_per_spin = (logZ_plus - 2.0 * logZ + logZ_minus) / (eps ** 2 * beta * N)
    d2_logZ_dbeta2 = (logZ_beta_plus - 2.0 * logZ + logZ_beta_minus) / (delta_beta ** 2)
    heat_capacity_per_spin = (beta**2 * d2_logZ_dbeta2) / N

    meta.update(
        {
            "chi": float(chi),
            "finite_difference_eps": eps,
            "heat_capacity_per_spin": heat_capacity_per_spin,
            "total_spins": float(N),
            "log_partition_function": logZ,
        }
    )

    return MethodResult(
        method="trg",
        free_energy_per_spin=free_energy_per_spin,
        susceptibility_per_spin=susceptibility_per_spin,
        heat_capacity_per_spin=heat_capacity_per_spin,
        runtime=runtime,
        metadata=meta,
    )


def run_2d_methods(Lx: int, Ly: int, T: float, J: float = 1.0, h: float = 0.0,
                   periodic: bool = True,
                   methods: Tuple[str, ...] = ("enumeration", "transfer_matrix", "trg"),
                   transfer_matrix_eps: float | None = None,
                   trg_kwargs: Dict[str, float] | None = None) -> Dict[str, MethodResult]:
    """依序執行指定 2D 方法，回傳各自的觀測結果。"""
    results: Dict[str, MethodResult] = {}
    trg_kwargs = {} if trg_kwargs is None else dict(trg_kwargs)
    for method in methods:
        if method == "enumeration":
            results[method] = enumeration_observables_2d(Lx, Ly, T, J, h, periodic)
        elif method == "transfer_matrix":
            results[method] = transfer_matrix_observables_2d(
                Lx, Ly, T, J, h, periodic, field_eps=transfer_matrix_eps
            )
        elif method == "trg":
            if not _trg_supports_lattice(Lx, Ly, periodic):
                warnings.warn(
                    f"TRG 僅支援週期邊界的正方形且邊長為 2 的冪次，本次 Lx={Lx}, Ly={Ly} 已自動跳過。",
                    RuntimeWarning,
                )
                continue
            results[method] = trg_observables_2d(
                Lx, Ly, T, J, h, periodic,
                chi=int(trg_kwargs.get("chi", 32)),
                field_eps=trg_kwargs.get("field_eps")
            )
        else:
            raise ValueError(f"未知方法：{method}")
    return results


# ---------- 2D 自由能計算（枚舉法） ----------
def free_energy_2d(Lx: int, Ly: int, T: np.ndarray, J: float, h: float = 0.0, periodic: bool = True) -> np.ndarray:
    """
    回傳 F(T)（shape = (len(T),)） for 2D Ising model using enumeration.
    """
    T = np.asarray(T, dtype=np.float64)
    F_total, _, _, _ = _enumeration_curve_stats(Lx, Ly, T, J, h, periodic)
    return F_total


def _enumeration_curve_stats(Lx: int, Ly: int, T_vals: np.ndarray, J: float, h: float,
                             periodic: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """回傳枚舉法對一組溫度點的自由能、磁化率、熱容量曲線與耗時。"""
    start = time.perf_counter()
    T_vals = np.asarray(T_vals, dtype=np.float64)
    if np.any(T_vals <= 0):
        raise ValueError("溫度必須為正值")
    energies, mags, _ = ising2d_all_energies_gray(Lx, Ly, J, h, periodic)
    energies = energies.astype(np.float64)
    mags = mags.astype(np.float64)
    N = Lx * Ly

    beta_vals = 1.0 / T_vals
    x = -np.outer(beta_vals, energies)
    x_max = np.max(x, axis=1, keepdims=True)
    weights = np.exp(x - x_max)
    norm = weights.sum(axis=1)

    F_total = -T_vals * (np.log(norm) + x_max.squeeze())
    E_mean = weights @ energies / norm
    M_mean = weights @ mags / norm
    E2_mean = weights @ (energies ** 2) / norm
    M2_mean = weights @ (mags ** 2) / norm

    susceptibility_per_spin = beta_vals * (M2_mean - M_mean**2) / N
    Cv_total = (beta_vals ** 2) * (E2_mean - E_mean**2)
    Cv_per_spin = Cv_total / N
    runtime = time.perf_counter() - start
    return (np.atleast_1d(F_total),
            np.atleast_1d(susceptibility_per_spin),
            np.atleast_1d(Cv_per_spin),
            runtime)


def _transfer_matrix_curve_stats(Lx: int, Ly: int, T_vals: np.ndarray, J: float, h: float,
                                 periodic: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """回傳轉移矩陣法的曲線結果與總耗時。"""
    N = Lx * Ly
    T_vals = np.asarray(T_vals, dtype=np.float64)
    F_total = np.empty_like(T_vals)
    chi_arr = np.empty_like(T_vals)
    cv_arr = np.empty_like(T_vals)
    runtime = 0.0
    for idx, temp in enumerate(T_vals):
        res = transfer_matrix_observables_2d(Lx, Ly, float(temp), J, h, periodic)
        F_total[idx] = res.free_energy_per_spin * N
        chi_arr[idx] = res.susceptibility_per_spin
        cv_arr[idx] = res.heat_capacity_per_spin
        runtime += res.runtime
    return np.atleast_1d(F_total), np.atleast_1d(chi_arr), np.atleast_1d(cv_arr), runtime


def _trg_curve_stats(Lx: int, Ly: int, T_vals: np.ndarray, J: float, h: float,
                     periodic: bool, chi: int, field_eps: float | None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """回傳 TRG 法在多個溫度下的自由能、磁化率、熱容量與耗時。"""
    N = Lx * Ly
    T_vals = np.asarray(T_vals, dtype=np.float64)
    F_total = np.empty_like(T_vals)
    chi_arr = np.empty_like(T_vals)
    cv_arr = np.empty_like(T_vals)
    runtime = 0.0
    for idx, temp in enumerate(T_vals):
        res = trg_observables_2d(Lx, Ly, float(temp), J, h, periodic,
                                 chi=chi, field_eps=field_eps)
        F_total[idx] = res.free_energy_per_spin * N
        chi_arr[idx] = res.susceptibility_per_spin
        cv_arr[idx] = res.heat_capacity_per_spin
        runtime += res.runtime
    return np.atleast_1d(F_total), np.atleast_1d(chi_arr), np.atleast_1d(cv_arr), runtime

# ---------- 畫圖：多個 L 或 Lx, Ly、同一張圖 ----------
def plot_free_energy_vs_T_for_Ls(Lx_list,
                                 Ly_list,
                                 J: float = 1.0,
                                 h: float = 0.0,
                                 periodic: bool = True,
                                 T_min: float = 0.1,
                                 T_max: float = 3.0,
                                 nT: int = 200,
                                 per_spin: bool = True,
                                 methods: Tuple[str, ...] = ("enumeration",),
                                 trg_kwargs: Dict[str, float] | None = None):
    """繪製 2D Ising 自由能、耗時、磁化率與熱容量（單一圖含四子圖）。"""
    if Lx_list is None or Ly_list is None:
        raise ValueError("請提供 Lx_list 與 Ly_list")
    if len(Lx_list) != len(Ly_list):
        raise ValueError("Lx_list 與 Ly_list 長度需一致")

    trg_kwargs = {} if trg_kwargs is None else dict(trg_kwargs)
    T_vals = np.linspace(T_min, T_max, nT, dtype=np.float64)
    if np.any(T_vals <= 0):
        raise ValueError("溫度必須為正值")

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    linestyles = {
        "enumeration": "-",
        "transfer_matrix": "--",
        "trg": ":",
    }
    label_alias = {
        "enumeration": "enum",
        "transfer_matrix": "tm",
        "trg": "trg",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax_F, ax_runtime, ax_chi, ax_cv = axes.ravel()

    runtime_records: Dict[str, list[Tuple[int, float]]] = {m: [] for m in methods}

    for idx, (Lx, Ly) in enumerate(zip(Lx_list, Ly_list)):
        color = color_cycle[idx % len(color_cycle)]
        N = Lx * Ly

        for method in methods:
            if method == "enumeration":
                F_total, chi_arr, cv_arr, elapsed = _enumeration_curve_stats(Lx, Ly, T_vals, J, h, periodic)
            elif method == "transfer_matrix":
                F_total, chi_arr, cv_arr, elapsed = _transfer_matrix_curve_stats(Lx, Ly, T_vals, J, h, periodic)
            elif method == "trg":
                if not _trg_supports_lattice(Lx, Ly, periodic):
                    warnings.warn(
                        f"TRG 僅支援週期邊界的正方形且邊長為 2 的冪次，本次 Lx={Lx}, Ly={Ly} 已自動跳過。",
                        RuntimeWarning,
                    )
                    continue
                chi_val = int(trg_kwargs.get("chi", 32))
                field_eps = trg_kwargs.get("field_eps")
                F_total, chi_arr, cv_arr, elapsed = _trg_curve_stats(Lx, Ly, T_vals, J, h, periodic,
                                                                     chi=chi_val, field_eps=field_eps)
            else:
                raise ValueError(f"未知方法：{method}")

            elapsed = max(elapsed, 1e-12)
            runtime_records.setdefault(method, []).append((N, elapsed))
            linestyle = linestyles.get(method, "-")
            label = f"{Lx}x{Ly}-{label_alias.get(method, method)}"
            F_plot = F_total / N if per_spin else F_total

            ax_F.plot(T_vals, F_plot, color=color, linestyle=linestyle, label=label)
            ax_chi.plot(T_vals, chi_arr, color=color, linestyle=linestyle)
            ax_cv.plot(T_vals, cv_arr, color=color, linestyle=linestyle)

    for method, data in runtime_records.items():
        if not data:
            continue
        data.sort(key=lambda x: x[0])
        Ns = [item[0] for item in data]
        times = [item[1] for item in data]
        ax_runtime.plot(Ns, times, marker="o", label=method)

    ylabel_energy = "Free energy per spin F/N" if per_spin else "Free energy F"
    ax_F.set_title("Free Energy")
    ax_F.set_xlabel("Temperature T")
    ax_F.set_ylabel(ylabel_energy)
    ax_F.legend()
    ax_F.grid(True, linestyle="--", alpha=0.3)

    ax_runtime.set_title("Runtime vs N")
    ax_runtime.set_xlabel("Number of spins N")
    ax_runtime.set_ylabel("Total runtime (s)")
    ax_runtime.set_yscale("log")
    ax_runtime.grid(True, which="both", linestyle="--", alpha=0.3)
    if runtime_records:
        ax_runtime.legend()

    ax_chi.set_title("Susceptibility per spin")
    ax_chi.set_xlabel("Temperature T")
    ax_chi.set_ylabel("Susceptibility")
    ax_chi.grid(True, linestyle="--", alpha=0.3)

    ax_cv.set_title("Heat Capacity per spin")
    ax_cv.set_xlabel("Temperature T")
    ax_cv.set_ylabel("Heat capacity")
    ax_cv.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle(f"2D Ising Comparison (J={J}, h={h})", fontsize=14)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()


# -------------------------
# 使用示例（請從小 N 開始）
# -------------------------
if __name__ == "__main__":
    Lx, Ly = 4, 4
    J = 1.0
    h = 0.1
    T = 2.5
    periodic = True

    print("=== 2D Ising observables (small lattice) ===")
    results = run_2d_methods(
        Lx, Ly, T, J=J, h=h, periodic=periodic,
        methods=("enumeration", "transfer_matrix", "trg"),
        trg_kwargs={"chi": 32}
    )
    for name, res in results.items():
        print(f"[{name}] F/N = {res.free_energy_per_spin:.8f}")
        print(f"        chi/N = {res.susceptibility_per_spin:.8f}, C_v/N = {res.heat_capacity_per_spin:.8f}, time = {res.runtime*1e3:.3f} ms")
        for key, value in res.metadata.items():
            print(f"        {key}: {value}")

    energies, mags, _ = ising2d_all_energies_gray(Lx, Ly, J, h, periodic)
    beta = 1.0 / T
    Z, Emean, Mmean, M2mean, Cv = partition_stats(energies, mags, beta)
    print("--- enumeration reference ---")
    print(f"Z={Z:.6e}, <E>={Emean:.6f}, <M>={Mmean:.6f}, <M^2>={M2mean:.6f}, C_v={Cv:.6f}")

    plot_free_energy_vs_T_for_Ls(
        Lx_list=[2, 3, 4],
        Ly_list=[2, 3, 4],
        J=J,
        h=0.0,
        periodic=periodic,
        T_min=0.1,
        T_max=3.0,
        nT=120,
        per_spin=True,
        methods=("enumeration", "transfer_matrix", "trg"),
        trg_kwargs={"chi": 32},
    )
