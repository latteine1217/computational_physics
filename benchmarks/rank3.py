"""簡易 Rank-3 tensor 乘法效能測試腳本。

使用方式：
    1. 視硬體調整 N 以控制張量大小，避免純 Python 迴圈耗時過長。
    2. 直接執行檔案即可列出各種張量乘法方法的耗時與是否一致。
"""

import time
import statistics
import timeit

import numpy as np
import matplotlib

from rank2 import loop as rank2_loop

matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = 30
# 產生可重現的測試張量，避免將隨機生成時間納入效能評估
A = np.arange(N**3, dtype=float).reshape(N, N, N)
B = np.arange(N**3, dtype=float).reshape(N, N, N)


def loop(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """根據 task.md 的張量收縮公式實作純 Python 迴圈。

    對於 Rank-2 輸入保留原本 2D 矩陣乘法的行為；當輸入為 Rank-3
    張量時，依 task.md 公式自動選擇：
        C_{ijlm} = Σ_k A_{ijk} B_{klm}   或
        C_{ij}   = Σ_{αβ} S_{iαβ} B_{βαj}"""
    if a.ndim == 2 and b.ndim == 2:
        if a.shape[1] != b.shape[0]:
            raise ValueError("內積維度不相符，無法進行矩陣乘法。")

        result = np.zeros(
            (a.shape[0], b.shape[1]), dtype=np.result_type(a.dtype, b.dtype)
        )
        for i in range(a.shape[0]):
            for k in range(a.shape[1]):
                aik = a[i, k]
                if aik == 0:
                    continue  # 稀疏時可少做一次內層迴圈的乘法與加法
                for j in range(b.shape[1]):
                    result[i, j] += aik * b[k, j]
        return result

    if a.ndim == 3 and b.ndim == 3:
        if a.shape[2] != b.shape[0]:
            raise ValueError("A 的最後一維需等於 B 的第一維，才能依 task.md 公式收縮。")

        # 依 b 的第二維是否等於 a 的第二維，判斷應套用哪一項張量收縮公式
        if b.shape[1] == a.shape[1]:
            # C_{ij} = sum_{alpha beta} S_{i alpha beta} B_{beta alpha j}
            result = np.zeros(
                (a.shape[0], b.shape[2]), dtype=np.result_type(a.dtype, b.dtype)
            )
            for i in range(a.shape[0]):
                for alpha in range(a.shape[1]):
                    for beta in range(a.shape[2]):
                        s_val = a[i, alpha, beta]
                        if s_val == 0:
                            continue
                        for j in range(b.shape[2]):
                            result[i, j] += s_val * b[beta, alpha, j]
            return result

        # C_{ijlm} = sum_k A_{ijk} B_{klm}
        result = np.zeros(
            (a.shape[0], a.shape[1], b.shape[1], b.shape[2]),
            dtype=np.result_type(a.dtype, b.dtype),
        )
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                for k in range(a.shape[2]):
                    aijk = a[i, j, k]
                    if aijk == 0:
                        continue
                    for l in range(b.shape[1]):
                        for m in range(b.shape[2]):
                            result[i, j, l, m] += aijk * b[k, l, m]
        return result

    raise ValueError("loop 方法僅支援 Rank-2 或 Rank-3 的輸入。")


def einsum_taskmd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """以 np.einsum 對應 task.md 兩種張量乘法公式，作為迴圈版的對照。

    參考 loop() 中的條件分支，確保張量形狀要求一致。"""
    if a.ndim == 2 and b.ndim == 2:
        return np.einsum("ik,kj->ij", a, b)

    if a.ndim == 3 and b.ndim == 3:
        if a.shape[2] != b.shape[0]:
            raise ValueError("A 的最後一維需等於 B 的第一維。")
        if b.shape[1] == a.shape[1]:
            # C_{ij} = Σ_{αβ} S_{iαβ} B_{βαj}
            return np.einsum("iab,baj->ij", a, b)
        # C_{ijlm} = Σ_k A_{ijk} B_{klm}
        return np.einsum("ijk,klm->ijlm", a, b)

    raise ValueError("einsum_taskmd 僅支援 Rank-2 或 Rank-3 的輸入。")


def contract_rank3_using_rank2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """以 rank2.loop 為基礎，透過張量 reshape 將 Rank-3 收縮化為矩陣乘法。

    主要針對 task.md 描述的兩種收縮：
        1. C_{ijlm} = Σ_k A_{ijk} B_{klm}
        2. C_{ij}   = Σ_{αβ} S_{iαβ} B_{βαj}

    將相關維度攤平成 Rank-2 矩陣後，呼叫 rank2.loop 完成乘法，再 reshape 回目標形狀。
    """
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError("僅支援三階張量輸入。")
    if a.shape[2] != b.shape[0]:
        raise ValueError("A 的第三軸需與 B 的第一軸一致，才能依 task.md 收縮。")

    if b.shape[1] == a.shape[1]:
        # C_{ij} = Σ_{αβ} S_{iαβ} B_{βαj}
        i_dim, alpha_dim, beta_dim = a.shape
        beta_b, alpha_b, j_dim = b.shape
        if beta_b != beta_dim or alpha_b != alpha_dim:
            raise ValueError("B 的第零與第一軸必須分別等於 A 的第二與第三軸。")

        # 先將 S_{iαβ} 攤平成 (i, αβ)，以便與 B 的 (αβ, j) 相乘
        a_matrix = a.reshape(i_dim, alpha_dim * beta_dim)
        # 轉置 B 為 (α, β, j) 再攤平，確保索引順序與 a_matrix 對應
        b_matrix = b.transpose(1, 0, 2).reshape(alpha_dim * beta_dim, j_dim)
        return rank2_loop(a_matrix, b_matrix)

    # C_{ijlm} = Σ_k A_{ijk} B_{klm}
    i_dim, j_dim, k_dim = a.shape
    k_b, l_dim, m_dim = b.shape
    if k_b != k_dim:
        raise ValueError("B 的第一軸需等於 A 的第三軸，才能收縮 Σ_k。")

    # 將 (i, j) 併為矩陣列，(l, m) 併為矩陣欄，完成 Rank-2 乘法後再還原形狀
    a_matrix = a.reshape(i_dim * j_dim, k_dim)
    b_matrix = b.reshape(k_dim, l_dim * m_dim)
    result_matrix = rank2_loop(a_matrix, b_matrix)
    return result_matrix.reshape(i_dim, j_dim, l_dim, m_dim)


def loop_rank3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """針對 Rank-3 張量逐切片使用 2D 乘法，模擬簡化版張量收縮。"""
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError("loop_rank3 僅支援三維張量。")
    if a.shape[1] != b.shape[0]:
        raise ValueError("第二維與第一維必須一致才可進行收縮。")
    if a.shape[2] != b.shape[2]:
        raise ValueError("第三維大小需一致以逐切片相乘。")

    result = np.zeros(
        (a.shape[0], b.shape[1], a.shape[2]), dtype=np.result_type(a.dtype, b.dtype)
    )
    # 針對第三維每個切片做一次矩陣乘法，再堆疊成 Rank-3 張量
    for k in range(a.shape[2]):
        result[:, :, k] = loop(a[:, :, k], b[:, :, k])
    return result


def einsum_rank3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """使用 np.einsum 實作相同的 Rank-3 張量收縮。"""
    return np.einsum("ilk,ljk->ijk", a, b)


def matmul_rank3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """利用 np.matmul 的批次矩陣乘法功能加速 Rank-3 張量收縮。"""
    tmp = np.matmul(a.transpose(2, 0, 1), b.transpose(2, 0, 1))
    return tmp.transpose(1, 2, 0)


loop_rank3(A, B)  # 預熱一次，避免第一次呼叫初始化開銷
loop_timer = timeit.Timer(lambda: loop_rank3(A, B))
loop_samples = loop_timer.repeat(repeat=5, number=1)
loop_time = statistics.median(loop_samples)
C_loop = loop_rank3(A, B)

einsum_rank3(A, B)  # 預熱
einsum_timer = timeit.Timer(lambda: einsum_rank3(A, B))
einsum_samples = einsum_timer.repeat(repeat=5, number=1)
einsum_time = statistics.median(einsum_samples)
einsum_result = einsum_rank3(A, B)

matmul_rank3(A, B)  # 預熱
matmul_timer = timeit.Timer(lambda: matmul_rank3(A, B))
matmul_samples = matmul_timer.repeat(repeat=5, number=1)
matmul_time = statistics.median(matmul_samples)
matmul_result = matmul_rank3(A, B)

# 統一檢查結果一致性（也可確保預熱與正式量測的矩陣一致）
loop_consistent = np.allclose(C_loop, einsum_result)
einsum_consistent = np.allclose(einsum_result, matmul_result, rtol=1e-4, atol=1e-6)

# 組裝表格資料
records = [
    ("Loop", loop_time, loop_consistent),
    ("np.einsum", einsum_time, einsum_consistent),
    ("np.matmul", matmul_time, einsum_consistent),
]

header = f"{'Method':<12}{'Time [s]':>12}{'Allclose':>12}"
separator = "-" * len(header)
print(header)
print(separator)
for name, duration, ok in records:
    print(f"{name:<12}{duration:>12.6f}{str(ok):>12}")

# 針對 task.md 定義的張量收縮公式，分別比較迴圈與 np.einsum 實作
def evaluate_taskmd_methods(a: np.ndarray, b: np.ndarray, label: str) -> list[tuple[str, float, bool]]:
    """回傳指定張量對的 loop 與 np.einsum 兩種實作耗時與一致性訊息。"""
    loop(a, b)  # 預熱
    loop_timer = timeit.Timer(lambda: loop(a, b))
    loop_time = statistics.median(loop_timer.repeat(repeat=3, number=1))
    loop_result = loop(a, b)

    einsum_taskmd(a, b)  # 預熱
    einsum_timer = timeit.Timer(lambda: einsum_taskmd(a, b))
    einsum_time = statistics.median(einsum_timer.repeat(repeat=3, number=1))
    einsum_result = einsum_taskmd(a, b)

    baseline = loop_result
    return [
        (f"{label} Loop", loop_time, True),
        (f"{label} np.einsum", einsum_time, np.allclose(baseline, einsum_result)),
    ]


def prepare_formula1_tensors(size: int) -> tuple[np.ndarray, np.ndarray]:
    """生成符合 C_{ijlm} = Σ_k A_{ijk} B_{klm} 的測試張量。"""
    i_dim = size
    j_dim = max(2, size // 2 + 1)
    k_dim = size
    l_dim = j_dim + 1  # 確保 b.shape[1] != a.shape[1]
    m_dim = max(2, size // 2 + 2)
    a = np.arange(i_dim * j_dim * k_dim, dtype=float).reshape(i_dim, j_dim, k_dim)
    b = np.arange(k_dim * l_dim * m_dim, dtype=float).reshape(k_dim, l_dim, m_dim)
    return a, b


def prepare_formula2_tensors(size: int) -> tuple[np.ndarray, np.ndarray]:
    """生成符合 C_{ij} = Σ_{αβ} S_{iαβ} B_{βαj} 的測試張量。"""
    i_dim = size
    alpha_dim = size
    beta_dim = max(2, size // 2 + 1)
    j_dim = max(2, size // 2 + 2)
    a = np.arange(i_dim * alpha_dim * beta_dim, dtype=float).reshape(i_dim, alpha_dim, beta_dim)
    b = np.arange(beta_dim * alpha_dim * j_dim, dtype=float).reshape(beta_dim, alpha_dim, j_dim)
    return a, b


task_size = min(10, N)
formula1_tensors = prepare_formula1_tensors(task_size)
formula2_tensors = prepare_formula2_tensors(task_size)

formula1_records = evaluate_taskmd_methods(*formula1_tensors, label="Formula1")
formula2_records = evaluate_taskmd_methods(*formula2_tensors, label="Formula2")

task_header = f"{'Task.md Method':<28}{'Time [s]':>12}{'Allclose':>12}"
task_separator = "-" * len(task_header)
print(task_separator)
print(task_header)
print(task_separator)
for name, duration, ok in formula1_records + formula2_records:
    print(f"{name:<28}{duration:>12.6f}{str(ok):>12}")


# 針對多組 N 進行效能測試並繪圖
def benchmark(ns: list[int], loop_cutoff: int = 60) -> None:
    """針對多個張量大小評估三種乘法方法的效能，採 timeit.repeat() 取中位數。"""
    loop_times: list[float] = []
    einsum_times: list[float] = []
    matmul_times: list[float] = []

    for size in ns:
        # 使用簡單等差矩陣避免隨機生成帶來的額外時間成本
        a = np.arange(size**3, dtype=float).reshape(size, size, size)
        b = np.arange(size**3, dtype=float).reshape(size, size, size)

        if size <= loop_cutoff:
            loop_rank3(a, b)  # 預熱
            timer = timeit.Timer(lambda: loop_rank3(a, b))
            samples = timer.repeat(repeat=5, number=1)
            loop_times.append(statistics.median(samples))
        else:
            loop_times.append(np.nan)

        einsum_rank3(a, b)  # 預熱
        timer = timeit.Timer(lambda: einsum_rank3(a, b))
        samples = timer.repeat(repeat=5, number=1)
        einsum_times.append(statistics.median(samples))

        matmul_rank3(a, b)  # 預熱
        timer = timeit.Timer(lambda: matmul_rank3(a, b))
        samples = timer.repeat(repeat=5, number=1)
        matmul_times.append(statistics.median(samples))

    plt.figure(figsize=(8, 5))
    plt.plot(ns, loop_times, marker="o", label="Loop")
    plt.plot(ns, einsum_times, marker="o", label="np.einsum")
    plt.plot(ns, matmul_times, marker="o", label="np.matmul")
    plt.title("Rank-3 Tensor Contraction Timing Comparison")
    plt.xlabel("Tensor size N")
    plt.xscale("log")
    plt.ylabel("Time [s]")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("matrix_multiplication_timings.png", dpi=150)


def benchmark_task_methods(ns: list[int]) -> None:
    """同時評估 task.md 兩種收縮公式的 loop 與 np.einsum 時間並繪於單一圖表。"""
    f1_loop: list[float] = []
    f1_einsum: list[float] = []
    f2_loop: list[float] = []
    f2_einsum: list[float] = []

    for size in ns:
        # Formula 1: C_{ijlm} = Σ_k A_{ijk} B_{klm}
        tensor_a1, tensor_b1 = prepare_formula1_tensors(size)
        loop(tensor_a1, tensor_b1)  # 預熱
        timer = timeit.Timer(lambda: loop(tensor_a1, tensor_b1))
        f1_loop.append(statistics.median(timer.repeat(repeat=5, number=1)))

        einsum_taskmd(tensor_a1, tensor_b1)  # 預熱
        timer = timeit.Timer(lambda: einsum_taskmd(tensor_a1, tensor_b1))
        f1_einsum.append(statistics.median(timer.repeat(repeat=5, number=1)))

        # Formula 2: C_{ij} = Σ_{αβ} S_{iαβ} B_{βαj}
        tensor_a2, tensor_b2 = prepare_formula2_tensors(size)
        loop(tensor_a2, tensor_b2)  # 預熱
        timer = timeit.Timer(lambda: loop(tensor_a2, tensor_b2))
        f2_loop.append(statistics.median(timer.repeat(repeat=5, number=1)))

        einsum_taskmd(tensor_a2, tensor_b2)  # 預熱
        timer = timeit.Timer(lambda: einsum_taskmd(tensor_a2, tensor_b2))
        f2_einsum.append(statistics.median(timer.repeat(repeat=5, number=1)))

    plt.figure(figsize=(8, 5))
    plt.plot(ns, f1_loop, marker="o", label="Formula 1 Loop")
    plt.plot(ns, f1_einsum, marker="o", label="Formula 1 np.einsum")
    plt.plot(ns, f2_loop, marker="o", label="Formula 2 Loop")
    plt.plot(ns, f2_einsum, marker="o", label="Formula 2 np.einsum")
    plt.title("Task.md Formulas Timing Comparison")
    plt.xlabel("Tensor size N")
    plt.xscale("log")
    plt.ylabel("Time [s]")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("taskmd_formula_comparison.png", dpi=150)


if __name__ == "__main__":
    sizes = [10, 20, 40, 60, 80, 100]
    benchmark(sizes)
    task_sizes = [4, 6, 8, 10]
    benchmark_task_methods(task_sizes)
