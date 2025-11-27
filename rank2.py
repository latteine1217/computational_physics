"""簡易矩陣乘法效能測試腳本。

使用方式：
    1. 視硬體調整 N 以控制矩陣大小，避免純 Python 迴圈耗時過長。
    2. 直接執行檔案即可列出各種乘法方法的耗時與是否一致。
"""

import time
import statistics
import timeit

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = 100
# 產生可重現的測試矩陣，避免將隨機生成時間納入效能評估
A = np.arange(N * N, dtype=float).reshape(N, N)
B = np.arange(N * N, dtype=float).reshape(N, N)


def loop(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """以純 Python 迴圈實作 2D 矩陣乘法，作為後續比較基準。"""
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("loop 方法僅支援二維矩陣相乘。")
    if a.shape[1] != b.shape[0]:
        raise ValueError("內積維度不相符，無法進行矩陣乘法。")

    result = np.zeros((a.shape[0], b.shape[1]), dtype=np.result_type(a.dtype, b.dtype))
    for i in range(a.shape[0]):
        for k in range(a.shape[1]):
            aik = a[i, k]
            if aik == 0:
                continue  # 稀疏時可少做一次內層迴圈的乘法與加法
            for j in range(b.shape[1]):
                result[i, j] += aik * b[k, j]
    return result


loop_start = time.perf_counter()
# 以純 Python 迴圈實作矩陣乘法 (O(N^3) 複雜度)，作為基準比較
loop(A, B)  # 預熱一次，避免第一次呼叫初始化開銷
loop_timer = timeit.Timer(lambda: loop(A, B))
loop_samples = loop_timer.repeat(repeat=5, number=1)
loop_time = statistics.median(loop_samples)
C_loop = loop(A, B)

np.matmul(A, B)  # 預熱
matmul_timer = timeit.Timer(lambda: np.matmul(A, B))
matmul_samples = matmul_timer.repeat(repeat=5, number=1)
matmul_time = statistics.median(matmul_samples)
d = np.matmul(A, B)

A @ B  # 預熱
at_timer = timeit.Timer(lambda: A @ B)
at_samples = at_timer.repeat(repeat=5, number=1)
at_time = statistics.median(at_samples)
e = A @ B

# 統一檢查結果一致性（也可確保預熱與正式量測的矩陣一致）
loop_consistent = np.allclose(C_loop, d)
matmul_consistent = np.allclose(d, e, rtol=1e-4, atol=1e-6)

# 組裝表格資料
records = [
    ("Loop", loop_time, loop_consistent),
    ("np.matmul", matmul_time, matmul_consistent),
    ("@ operator", at_time, matmul_consistent),
]

header = f"{'Method':<12}{'Time [s]':>12}{'Allclose':>12}"
separator = "-" * len(header)
print(header)
print(separator)
for name, duration, ok in records:
    print(f"{name:<12}{duration:>12.6f}{str(ok):>12}")


# 針對多組 N 進行效能測試並繪圖
def benchmark(ns: list[int], loop_cutoff: int = 300) -> None:
    """針對多個矩陣大小評估三種乘法方法的效能，採 timeit.repeat() 取中位數。"""
    loop_times: list[float] = []
    matmul_times: list[float] = []
    at_times: list[float] = []

    for size in ns:
        # 使用簡單等差矩陣避免隨機生成帶來的額外時間成本
        a = np.arange(size * size, dtype=float).reshape(size, size)
        b = np.arange(size * size, dtype=float).reshape(size, size)

        if size <= loop_cutoff:
            loop(a, b)  # 預熱
            timer = timeit.Timer(lambda: loop(a, b))
            samples = timer.repeat(repeat=5, number=1)
            loop_times.append(statistics.median(samples))
        else:
            loop_times.append(np.nan)

        np.matmul(a, b)  # 預熱
        timer = timeit.Timer(lambda: np.matmul(a, b))
        samples = timer.repeat(repeat=5, number=1)
        matmul_times.append(statistics.median(samples))

        a @ b  # 預熱
        timer = timeit.Timer(lambda: a @ b)
        samples = timer.repeat(repeat=5, number=1)
        at_times.append(statistics.median(samples))

    plt.figure(figsize=(8, 5))
    plt.plot(ns, loop_times, marker="o", label="Loop")
    plt.plot(ns, matmul_times, marker="o", label="np.matmul")
    plt.plot(ns, at_times, marker="o", label="@ operator")
    plt.title("Matrix Multiplication Timing Comparison")
    plt.xlabel("Matrix size N")
    plt.xscale("log")
    plt.ylabel("Time [s]")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("matrix_multiplication_timings.png", dpi=150)


if __name__ == "__main__":
    sizes = [10, 20, 40, 60, 80, 100, 200, 300, 400, 500, 1000, 1500, 2000, 3000, 5000]
    benchmark(sizes)
