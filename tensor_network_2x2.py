import numpy as np
import time


def _tensor_network_local_tensor(beta: float, J: float, h: float) -> np.ndarray:
    """
    依照 TRG 常用的 cosh/sinh 分解構造 rank-4 局域張量。
    
    步驟說明：
        1. 將最近鄰耦合權重 W_{ss'} = exp(βJ s s') 拆解為矩陣乘積 M M^T。
        2. 對於每個中心自旋值 s，取出 M 的對應列，四條鍵各自採用同一列元素。
        3. 外場以 exp(β h s) 作為額外權重乘上該自旋的貢獻。
    生成的張量 T[up, right, down, left] 即為單站點對四條虛擬鍵的映射。
    """
    cosh_val = np.cosh(beta * J)
    sinh_val = np.sinh(beta * J)
    sqrt_cosh = np.sqrt(cosh_val)
    sqrt_sinh = np.sqrt(sinh_val)
    # M[自旋索引, 虛擬鍵]；自旋索引 0->+1, 1->-1
    M = np.array(
        [
            [sqrt_cosh, sqrt_sinh],
            [sqrt_cosh, -sqrt_sinh],
        ],
        dtype=np.float64,
    )
    field_weights = np.array(
        [np.exp(beta * h), np.exp(-beta * h)],
        dtype=np.float64,
    )
    tensor = np.zeros((2, 2, 2, 2), dtype=np.float64)
    for spin_idx, weight in enumerate(field_weights):
        # 收集中心自旋為 s 時四條鍵的權重，形成 rank-4 外積
        contrib = np.einsum(
            "i,j,k,l->ijkl",
            M[spin_idx],
            M[spin_idx],
            M[spin_idx],
            M[spin_idx],
            optimize=True,
        )
        tensor += weight * contrib
    return tensor


def _tensor_network_2x2_logZ(beta: float, J: float, h: float) -> float:
    """
    直接收縮 2x2 週期晶格張量網路並回傳 logZ。
    
    愛因斯坦求和式的指標安排如下：
        - 第一個張量 a,b,c,d 對應上、右、下、左。
        - e,d,f,b 等指標代表鄰近張量共享的邊，符合 new_method.md 範例配線。
    收縮完得到完整配分函數 Z，最後取對數以供熱力學計算使用。
    """
    local_tensor = _tensor_network_local_tensor(beta, J, h)
    Z = np.einsum(
        "abcd,edfb,cgah,fheg->",
        local_tensor,
        local_tensor,
        local_tensor,
        local_tensor,
        optimize=True,
    )
    if Z <= 0.0:
        raise RuntimeError("張量網路收縮結果非正值，請檢查輸入參數")
    return float(np.log(Z))


def _adaptive_field_eps(N: int, beta: float, base_eps: float = 1e-8) -> float:
    """
    提供針對外場差分的自適應步長。
    
    考量：
        - 晶格較大時，磁化率對外場更敏感，需縮小步長 (∝ N^{-0.6})。
        - 低溫 (β 大) 時磁化曲線變尖銳，因此也需縮小步長。
    """
    size_factor = 1.0 / (N ** 0.6)
    temp_factor = min(1.0, 2.0 * beta)
    adaptive_eps = base_eps * size_factor * temp_factor
    return max(adaptive_eps, 1e-12)


def tensor_network_task02_tensors(beta: float, J: float, h: float = 0.0):
    """
    依 Task-02 要求回傳 rank-2 張量 M、rank-4 張量 T 以及直接收縮四個 T 的配分函數。
    
    詳細流程：
        1. 透過 cosh/sinh 分解建立二階矩陣 M，使得鄰接權重 W = M M^T。
        2. 對中心自旋求和，以外積組合四個鍵的 M 元素，形成 rank-4 張量 T。
        3. 以 2x2 週期邊界拓撲收縮四個 T，取得 Z_{2x2}。
    
    Args:
        beta: 逆溫度 1/T
        J: 交換參數
        h: 外場
    
    Returns:
        M: rank-2 張量
        T: rank-4 張量（shape=(2,2,2,2)）
        Z: 2x2 晶格的配分函數
    """
    cosh_val = np.cosh(beta * J)
    sinh_val = np.sinh(beta * J)
    sqrt_cosh = np.sqrt(cosh_val)
    sqrt_sinh = np.sqrt(sinh_val)
    M = np.array(
        [
            [sqrt_cosh, sqrt_sinh],
            [sqrt_cosh, -sqrt_sinh],
        ],
        dtype=np.float64,
    )
    field_weights = np.array(
        [np.exp(beta * h), np.exp(-beta * h)],
        dtype=np.float64,
    )
    T = np.zeros((2, 2, 2, 2), dtype=np.float64)
    for spin_idx, weight in enumerate(field_weights):
        # 將四個方向上的虛擬鍵與同一中心自旋連結（四個 M 列向量外積）
        contrib = np.einsum(
            "i,j,k,l->ijkl",
            M[spin_idx],
            M[spin_idx],
            M[spin_idx],
            M[spin_idx],
            optimize=True,
        )
        T += weight * contrib
    # 依 2x2 PBC 對應的指標連接順序收縮四個張量
    Z = np.einsum(
        "abcd,edfb,cgah,fheg->",
        T,
        T,
        T,
        T,
        optimize=True,
    )
    return M, T, float(Z)


def tensor_network_2x2_observables(Lx: int, Ly: int, T: float, J: float, h: float,
                                   periodic: bool = True,
                                   beta_eps: float | None = None,
                                   field_eps: float | None = None):
    """
    針對 2x2 週期性 Ising 模型，考慮 tensor_method.md 所述張量分解，計算熱力學量。
    
    流程摘要：
        1. 以張量收縮求得 logZ。
        2. 對 β、h 施以對稱有限差分，分別取得能量與磁化相關導數。
        3. 將導數換算為自由能、磁化率與熱容量（均以每自旋表示）。
    
    Returns:
        free_energy_per_spin, susceptibility_per_spin, heat_capacity_per_spin,
        runtime, metadata
    """
    if (Lx, Ly) != (2, 2):
        raise ValueError("tensor_network_2x2_observables 僅支援 2x2 晶格")
    if not periodic:
        raise NotImplementedError("當前張量網路實作僅支援週期邊界條件")
    if T <= 0.0:
        raise ValueError("溫度必須為正值")

    beta = 1.0 / T
    delta_beta = beta_eps if beta_eps is not None else max(1e-6, 1e-4 * beta)
    if beta - delta_beta <= 0.0:
        delta_beta = 0.5 * beta

    N = 4
    adaptive_field = _adaptive_field_eps(N, beta) if field_eps is None else field_eps
    delta_h = max(adaptive_field, 1e-4)

    start = time.perf_counter()
    logZ = _tensor_network_2x2_logZ(beta, J, h)
    logZ_beta_plus = _tensor_network_2x2_logZ(beta + delta_beta, J, h)
    logZ_beta_minus = _tensor_network_2x2_logZ(beta - delta_beta, J, h)
    logZ_h_plus = _tensor_network_2x2_logZ(beta, J, h + delta_h)
    logZ_h_minus = _tensor_network_2x2_logZ(beta, J, h - delta_h)
    runtime = time.perf_counter() - start

    # 對數配分函數對 β、h 的一階與二階導數（對稱差分可降低截斷誤差）
    d_logZ_dbeta = (logZ_beta_plus - logZ_beta_minus) / (2.0 * delta_beta)
    d2_logZ_dbeta2 = (logZ_beta_plus - 2.0 * logZ + logZ_beta_minus) / (delta_beta ** 2)
    d_logZ_dh = (logZ_h_plus - logZ_h_minus) / (2.0 * delta_h)
    d2_logZ_dh2 = (logZ_h_plus - 2.0 * logZ + logZ_h_minus) / (delta_h ** 2)

    free_energy_per_spin = -(logZ / beta) / N
    energy_mean = -d_logZ_dbeta
    heat_capacity_per_spin = (beta ** 2 / N) * d2_logZ_dbeta2
    magnetization_per_spin = (d_logZ_dh / beta) / N
    susceptibility_per_spin = d2_logZ_dh2 / (beta * N)

    metadata = {
        "log_partition_function": logZ,
        "energy_mean_per_spin": energy_mean / N,
        "magnetization_per_spin": magnetization_per_spin,
        "delta_beta": delta_beta,
        "delta_h": delta_h,
    }

    return (
        free_energy_per_spin,
        susceptibility_per_spin,
        heat_capacity_per_spin,
        runtime,
        metadata,
    )


if __name__ == "__main__":
    temp = 1.0
    beta = 1.0/temp
    J = 1.0
    h = 0.0
    M, T, Z = tensor_network_task02_tensors(beta, J, h)
    print("Rank-2 tensor M:")
    print(M)
    print("\nRank-4 tensor T (reshape to 4x4 for展示):")
    print(T.reshape(4, 4))
    print(f"\nPartition function Z_2x2 = {Z:.8f}")
