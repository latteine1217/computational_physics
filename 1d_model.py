import numpy as np
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import math
import time
from dataclasses import dataclass, field
from itertools import cycle

def bit_to_spin(bits: int, idx: int) -> int:
    """bit -> spin in {-1,+1}；idx 從 0..L-1"""
    return 1 if ((bits >> idx) & 1) else -1

def energy_ising1d_from_bits(bits: int, L: int, J: float, h: float = 0.0, periodic: bool = True) -> float:
    """直接根據 bits 計算整體能量（作為 Gray code 初始 E）。O(L)"""
    E = 0.0
    M = 0
    # 場項
    for i in range(L):
        si = bit_to_spin(bits, i)
        M += si
    E += -h * M
    # 交換項
    last = L - 1
    for i in range(L - 1):
        si = bit_to_spin(bits, i)
        sj = bit_to_spin(bits, i + 1)
        E += -J * si * sj
    if periodic and L >= 2:
        s0 = bit_to_spin(bits, 0)
        sl = bit_to_spin(bits, last)
        E += -J * s0 * sl
    return E

def deltaE_ising1d_flip(bits: int, L: int, i: int, J: float, h: float = 0.0, periodic: bool = True) -> float:
    """翻轉 site i 的 ΔE。使用 bits 讀取 s_i 與鄰居；O(1)"""
    si = bit_to_spin(bits, i)
    nn_sum = 0
    # 左鄰
    if i - 1 >= 0:
        nn_sum += bit_to_spin(bits, i - 1)
    elif periodic and L >= 2:
        nn_sum += bit_to_spin(bits, L - 1)
    # 右鄰
    if i + 1 < L:
        nn_sum += bit_to_spin(bits, i + 1)
    elif periodic and L >= 2:
        nn_sum += bit_to_spin(bits, 0)
    # ΔE = 2 s_i (J * sum_nn + h)
    return 2.0 * si * (J * nn_sum + h)

def gray_next_flip_pos(t: int) -> int:
    """
    已在步 t 加入當前組態後，下一步 (t+1) 的 Gray code 與當前差一個 bit。
    回傳要翻轉的 bit 位置 i（最右側 set bit 的索引）。
    """
    g  = t ^ (t >> 1)
    g1 = (t + 1) ^ ((t + 1) >> 1)
    diff = g ^ g1
    # 找 diff 中最右側 1 的位置
    pos = 0
    while ((diff >> pos) & 1) == 0:
        pos += 1
    return pos

def ising1d_all_energies_gray(L: int, J: float, h: float = 0.0, periodic: bool = True):
    """
    遍歷所有 2^L 組態，回傳：
      energies: np.ndarray shape (2^L,)
      mags:     np.ndarray shape (2^L,)
      degeneracy: Dict[energy_value, count]（能量退化度）
    使用 Gray code + ΔE，初始 bits=0（全 -1）。
    """
    if L <= 0:
        raise ValueError("L 必須 >= 1")
    total = 1 << L  # 2^L
    energies = np.empty(total, dtype=np.float64)
    mags     = np.empty(total, dtype=np.int32)

    bits = 0                # 全 -1
    E = energy_ising1d_from_bits(bits, L, J, h, periodic)  # 初始能量
    M = -L                  # 全 -1 的磁化

    for t in range(total):
        energies[t] = E
        mags[t]     = M

        if t == total - 1:
            break
        # 下一步要翻的位元位置
        i = gray_next_flip_pos(t)
        # 先計 ΔE（基於翻轉前的 bits）
        dE = deltaE_ising1d_flip(bits, L, i, J, h, periodic)
        # 翻轉位元
        bits ^= (1 << i)
        # 更新 E、M
        E += dE
        si_new = bit_to_spin(bits, i)   # 翻後 spin 值（+1 或 -1）
        M += 2 * si_new                 # 原本是 -si_new，翻後變 si_new → ΔM = 2 * si_new

    # 能量退化度（同一能量的出現次數）
    uniqE, counts = np.unique(energies, return_counts=True)
    degeneracy = {float(e): int(c) for e, c in zip(uniqE, counts)}
    return energies, mags, degeneracy

def partition_stats(energies: np.ndarray, mags: np.ndarray, beta: float):
    """
    給定所有能量與磁化，計算 Z、<E>、<M>、<M^2>、<E^2>、C_v（每自旋可自行除 L）。
    以 log-sum-exp 技巧避免 overflow。
    """
    a = -beta * energies
    amax = np.max(a)
    wa = np.exp(a - amax)        # normalized weights
    norm = wa.sum()
    Z = norm * np.exp(amax)
    # 期望值（注意需用相同權重）
    Ew = (energies * wa).sum() / norm
    Mw = (mags * wa).sum() / norm
    M2w = ((mags**2) * wa).sum() / norm
    E2w = ((energies**2) * wa).sum() / norm
    Cv = beta**2 * (E2w - Ew**2)   # 熱容（未除以 L）
    return Z, Ew, Mw, M2w, Cv


@dataclass(frozen=True)
class MethodResult:
    """封裝單一計算方法的觀測量摘要，方便統一後處理。"""
    method: str
    free_energy_per_spin: float
    susceptibility_per_spin: float
    heat_capacity_per_spin: float
    runtime: float
    metadata: Dict[str, float] = field(default_factory=dict)


def enumeration_observables(L: int, T: float, J: float, h: float = 0.0,
                            periodic: bool = True) -> MethodResult:
    """以 Gray code 窮舉 2^L 組態，回傳自由能、磁化率與熱容量等量測。"""
    start = time.perf_counter()
    energies, mags, _ = ising1d_all_energies_gray(L, J, h, periodic)
    beta = 1.0 / T
    Z, E_mean, M_mean, M2_mean, Cv = partition_stats(energies, mags, beta)
    runtime = time.perf_counter() - start
    free_energy_total = - (1.0 / beta) * np.log(Z)
    free_energy_per_spin = free_energy_total / L
    susceptibility_per_spin = beta * (M2_mean - M_mean**2) / L
    heat_capacity_per_spin = Cv / L
    metadata = {
        "partition_function": Z,
        "energy_mean": E_mean,
        "magnetization_mean_per_spin": M_mean / L,
        "specific_heat": Cv / L,
        "heat_capacity_per_spin": heat_capacity_per_spin,
        "total_spins": float(L),
    }
    return MethodResult(
        method="enumeration",
        free_energy_per_spin=free_energy_per_spin,
        susceptibility_per_spin=susceptibility_per_spin,
        heat_capacity_per_spin=heat_capacity_per_spin,
        runtime=runtime,
        metadata=metadata,
    )


def _transfer_matrix_stats_1d(L: int, beta: float, J: float, h: float,
                              periodic: bool) -> Tuple[float, float, float]:
    """回傳 1D 轉移矩陣的 logZ、對外場一階與二階導數。"""
    if L <= 0:
        raise ValueError("L 必須 >= 1")
    if not periodic:
        raise NotImplementedError("transfer matrix 方法目前僅支援週期邊界條件")

    spins = np.array([-1.0, 1.0], dtype=np.float64)
    T = np.empty((2, 2), dtype=np.float64)
    dT = np.empty((2, 2), dtype=np.float64)
    d2T = np.empty((2, 2), dtype=np.float64)
    half_beta = 0.5 * beta

    for i, si in enumerate(spins):
        for j, sj in enumerate(spins):
            exponent = beta * (J * si * sj + 0.5 * h * (si + sj))
            weight = math.exp(exponent)
            T[i, j] = weight
            pref = half_beta * (si + sj)
            dT[i, j] = pref * weight
            d2T[i, j] = (pref ** 2) * weight

    prefix = [np.eye(2, dtype=np.float64)]
    for _ in range(L):
        prefix.append(prefix[-1] @ T)

    Z = np.trace(prefix[L])
    if Z <= 0:
        raise RuntimeError("Transfer matrix partition function 非正，請檢查參數")

    T_power_minus1 = prefix[L - 1]
    dZ = L * float(np.trace(dT @ T_power_minus1))
    sum_term = 0.0
    if L >= 2:
        for k in range(L - 1):
            sum_term += float(np.trace((dT @ prefix[k]) @ dT @ prefix[L - 2 - k]))
    d2Z = L * float(np.trace(d2T @ T_power_minus1)) + L * sum_term

    logZ = math.log(Z)
    d_logZ = dZ / Z
    d2_logZ = d2Z / Z - (dZ / Z) ** 2
    return logZ, d_logZ, d2_logZ


def transfer_matrix_observables(L: int, T: float, J: float, h: float = 0.0,
                                periodic: bool = True,
                                field_eps: float | None = None) -> MethodResult:
    """使用解析導數計算 1D 轉移矩陣的自由能、磁化率與熱容量。"""
    if not periodic:
        raise NotImplementedError("transfer matrix 方法目前僅支援週期邊界條件")
    beta = 1.0 / T
    delta_beta = max(1e-5, 1e-3 * beta)
    if beta - delta_beta <= 0.0:
        delta_beta = 0.5 * beta

    start = time.perf_counter()
    logZ, d_logZ, d2_logZ = _transfer_matrix_stats_1d(L, beta, J, h, periodic)
    logZ_beta_plus, _, _ = _transfer_matrix_stats_1d(L, beta + delta_beta, J, h, periodic)
    logZ_beta_minus, _, _ = _transfer_matrix_stats_1d(L, beta - delta_beta, J, h, periodic)
    runtime = time.perf_counter() - start

    free_energy_per_spin = - (logZ / beta) / L
    magnetization_per_spin = d_logZ / (beta * L)
    susceptibility_per_spin = d2_logZ / (beta * L)
    d2_logZ_dbeta2 = (logZ_beta_plus - 2.0 * logZ + logZ_beta_minus) / (delta_beta ** 2)
    heat_capacity_per_spin = (beta**2 * d2_logZ_dbeta2) / L

    metadata = {
        "log_partition_function": logZ,
        "magnetization_per_spin": magnetization_per_spin,
        "finite_difference_eps": None,
        "heat_capacity_per_spin": heat_capacity_per_spin,
        "total_spins": float(L),
    }

    return MethodResult(
        method="transfer_matrix",
        free_energy_per_spin=free_energy_per_spin,
        susceptibility_per_spin=susceptibility_per_spin,
        heat_capacity_per_spin=heat_capacity_per_spin,
        runtime=runtime,
        metadata=metadata,
    )


def run_1d_methods(L: int, T: float, J: float = 1.0, h: float = 0.0,
                   periodic: bool = True,
                   methods: tuple[str, ...] = ("enumeration", "transfer_matrix")) -> Dict[str, MethodResult]:
    """依序執行指定方法，回傳 method -> result。"""
    results: Dict[str, MethodResult] = {}
    for method in methods:
        if method == "enumeration":
            results[method] = enumeration_observables(L, T, J, h, periodic)
        elif method == "transfer_matrix":
            results[method] = transfer_matrix_observables(L, T, J, h, periodic)
        else:
            raise ValueError(f"未知方法：{method}")
    return results

# ---------- 小工具：row-wise 的 logsumexp，避免 under/overflow ----------
def logsumexp_rows(x: np.ndarray) -> np.ndarray:
    """逐列套用 log-sum-exp，減去列最大值避免浮點溢位。"""
    m = np.max(x, axis=1, keepdims=True)
    return (m.squeeze() + np.log(np.sum(np.exp(x - m), axis=1)))

# ---------- 1D 自由能計算（兩種方法） ----------
def free_energy_1d(L: int, T: np.ndarray, J: float, h: float = 0.0,
                   periodic: bool = True, method: str = "auto") -> np.ndarray:
    """
    回傳 F(T)（shape = (len(T),)）。
    method:
      - "auto": h==0 用解析解，否則用枚舉（L 不宜過大）
      - "theory": 解析解（僅限 h==0）
      - "enum":   Gray code 全枚舉
    """
    T = np.asarray(T, dtype=np.float64)
    beta = 1.0 / T

    if method == "auto":
        method = "theory" if h == 0.0 else "enum"

    if method == "theory":
        if h != 0.0:
            raise ValueError("theory 模式僅支援 h=0 的一維 Ising。")
        # 1D h=0 的解析解（PBC）
        # Z = lam_plus^L + lam_minus^L, 其中 lam_± = e^{βJ} ± e^{-βJ} = 2(cosh,sinh)
        lam_plus  = np.exp(beta * J) + np.exp(-beta * J)
        lam_minus = np.exp(beta * J) - np.exp(-beta * J)
        Z = lam_plus**L + lam_minus**L if periodic else 2.0 * (lam_plus**(L-1))
        F = -T * np.log(Z)
        return F

    elif method == "enum":
        energies, _, _ = ising1d_all_energies_gray(L, J, h, periodic)
        # x_{m,n} = -E_n / T_m
        x = -np.outer(1.0 / T, energies)   # shape (M, 2^L)
        logZ = logsumexp_rows(x)           # shape (M,)
        F = -T * logZ
        return F

    else:
        raise ValueError("method 必須是 {'auto','theory','enum'} 之一。")

# ---------- 畫圖：多個 L、同一張圖 ----------
def _free_energy_curve_enumeration(L: int, T: np.ndarray, J: float, h: float,
                                   periodic: bool) -> np.ndarray:
    """回傳枚舉法在多個溫度下的總自由能。"""
    energies, _, _ = ising1d_all_energies_gray(L, J, h, periodic)
    inv_T = 1.0 / T
    x = -np.outer(inv_T, energies)
    logZ = logsumexp_rows(x)
    return -T * logZ


def _free_energy_curve_transfer_matrix(L: int, T: np.ndarray, J: float, h: float,
                                       periodic: bool) -> np.ndarray:
    """對一系列溫度使用轉移矩陣求得總自由能。"""
    F = np.empty_like(T)
    for idx, temp in enumerate(T):
        beta = 1.0 / temp
        logZ, _, _ = _transfer_matrix_stats_1d(L, beta, J, h, periodic)
        F[idx] = -(1.0 / beta) * logZ
    return F


def plot_free_energy_vs_T_for_Ls(L_list, J=1.0, h=0.0, periodic=True,
                                 T_min=0.05, T_max=5.0, nT=200,
                                 per_spin=True,
                                 methods: tuple[str, ...] = ("enumeration", "transfer_matrix")):
    """
    比較多種方法在不同 L 下的自由能曲線，並同時繪製總計算時間 vs L。
    per_spin=True：畫 F/N，方便不同 L 比較；False 則畫總自由能。
    """
    # h ≠ 0 時排除理論解，避免使用者誤觸發錯誤
    filtered_methods: list[str] = []
    for method in methods:
        if method == "theory" and not math.isclose(h, 0.0, rel_tol=1e-12, abs_tol=1e-12):
            continue
        filtered_methods.append(method)

    if len(filtered_methods) == 0:
        raise ValueError("methods 至少需包含一種演算法")

    methods = tuple(filtered_methods)

    T = np.linspace(T_min, T_max, nT, dtype=np.float64)
    method_funcs = {
        "enumeration": _free_energy_curve_enumeration,
        "transfer_matrix": _free_energy_curve_transfer_matrix,
        "theory": lambda L, T_vals, J_val, h_val, periodic_val: free_energy_1d(
            L, T_vals, J_val, h_val, periodic_val, method="theory"
        ),
    }
    runtime_data: Dict[str, list[float]] = {m: [] for m in methods}

    plt.figure(figsize=(7, 5))
    color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    color_for_L: Dict[int, str] = {}
    # 預設以線型區分演算法，顏色用來識別 L
    linestyles = {
        "enumeration": "-",
        "transfer_matrix": "--",
        "theory": ":",
    }
    for L in L_list:
        if L not in color_for_L:
            color_for_L[L] = next(color_cycle)
        color = color_for_L[L]
        for method in methods:
            if method not in method_funcs:
                raise ValueError(f"未知方法：{method}")
            start = time.perf_counter()
            F_total = method_funcs[method](L, T, J, h, periodic)
            # 以對數刻度繪圖需確保耗時為正值
            elapsed = max(time.perf_counter() - start, 1e-12)
            runtime_data[method].append(elapsed)

            y = F_total / L if per_spin else F_total
            label = f"{method} L={L}" if len(methods) > 1 else f"L={L}"
            linestyle = linestyles.get(method, "-")
            plt.plot(T, y, label=label, color=color, linestyle=linestyle)

    ylabel = "Free energy per spin F/N" if per_spin else "Free energy F"
    plt.xlabel("Temperature T")
    plt.ylabel(ylabel)
    plt.title(f"1D Ising Free Energy (J={J}, h={h})")
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(7, 5))
    for method, timings in runtime_data.items():
        plt.plot(L_list, timings, marker="o", label=method)
    plt.xlabel("System size L")
    plt.ylabel("Total runtime (s)")
    plt.title(f"Computation Time vs L (J={J}, h={h})")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.show()

# -------------------------
# 使用示例（請從小 L 開始）
# -------------------------
if __name__ == "__main__":
    L = 8
    J = 1.0
    h = 0.0
    T = 2.5
    periodic = True

    results = run_1d_methods(L, T, J=J, h=h, periodic=periodic)
    for name, res in results.items():
        print(f"[{name}] F/N = {res.free_energy_per_spin:.8f}, ")
        print(f"        chi/N = {res.susceptibility_per_spin:.8f}, C_v/N = {res.heat_capacity_per_spin:.8f}, time = {res.runtime*1e3:.3f} ms")
        for key, value in res.metadata.items():
            print(f"        {key}: {value}")

    L_list = [2, 3, 4, 5, 10, 15, 20]
    plot_free_energy_vs_T_for_Ls(L_list, J=J, h=h, periodic=periodic,
                                 T_min=0.1, T_max=3.0, nT=100,
                                 per_spin=True,
                                 methods=("enumeration", "transfer_matrix", "theory"))
