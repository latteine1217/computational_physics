"""
簡易版 TRG (Tensor Renormalization Group) 流程與視覺化範例

目的:
- 示範如何使用 Levin-Nave TRG 算法計算二維 Ising 模型的單位格點自由能。
- 提供視覺化功能，幫助理解 TRG 算法的內部幾何結構與數值特性，包括：
    - TRG 張量收縮的幾何拓撲示意圖 (ASCII Art)。
    - 迭代過程中奇異值頻譜 (Singular Value Spectrum) 的演化，這反映了系統的糾纏結構與截斷誤差。

理論背景:
- TRG 是一種基於張量網路的數值方法，用於處理經典統計力學模型。
- 它通過反覆將晶格粗粒化 (Coarse-graining)，將 $2 \times 2$ 的格點合併為一個有效格點，從而以對數複雜度縮減系統尺寸。
- 自由能的計算依賴於在每一步提取歸一化因子，避免數值溢出。

說明:
- 僅使用 numpy 實現核心算法，自包含、可直接執行。
- 使用 `matplotlib` 繪製奇異值頻譜圖。
- 預設在二維 Ising 模型的臨界溫度 (Tc) 進行計算，以觀察相變點附近的行為。
"""

from __future__ import annotations  # 啟用未來版本的特性，例如類型提示的延遲評估

import argparse  # 用於解析命令列參數
import math  # 提供數學函數，如 log, exp
import matplotlib  # 用於繪圖
import numpy as np  # 用於數值計算，尤其是矩陣 SVD 和張量收縮

# 使用無頭 backend (Agg)，避免在沒有顯示器的環境下 (如遠端伺服器) 執行時報錯
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def visualize_trg_topology():
    """
    以 ASCII Art 視覺化 TRG 的張量收縮拓撲結構。
    這個函數不進行計算，僅在終端機印出圖示，解釋在 `_trg_step` 函數中，
    四個半張量 (S1, S2, S3, S4) 是如何連接並收縮形成一個新的粗粒化張量的。
    """
    diagram = r"""
    [TRG 張量收縮幾何拓撲示意圖 - Plaquette Contraction]

    這示意圖展示了 TRG 算法的核心步驟：四個半張量 (S1, S2, S3, S4)
    構成一個閉合迴路 (Plaquette)，並收縮形成一個新的粗粒化張量 T_new。

    拓撲結構 (2D 視圖)：
    (內部連接 x, y, z, w 被求和，外部連接 U, R, D, L 形成新張量)

              U (新 Up-bond, 來自 S2)
              |
            [ S2 ]---y---[ S3 ]--- R (新 Right-bond, 來自 S3)
              |             |
              x             z
              |             |
       L ---[ S1 ]---w---[ S4 ]--- D (新 Down-bond, 來自 S4)
    (新 Left-bond,
     來自 S1)

    連接說明 (Internal Summation):
    - x: 連接 S1 與 S2
    - y: 連接 S2 與 S3
    - z: 連接 S3 與 S4
    - w: 連接 S4 與 S1

    外部指標 (Output Indices of T_new[U,R,D,L]):
    - U: 來自 S2 (上方)
    - R: 來自 S3 (右方)
    - D: 來自 S4 (下方)
    - L: 來自 S1 (左方)

    對應程式碼中的 einsum 字符串: "xwl,xyu,ryz,dzw->urdl"
    """
    print(diagram)


def _ising_bond_decomposition(beta: float, J: float) -> tuple[np.ndarray, np.ndarray]:
    """
    計算二維 Ising 模型單鍵的 Boltzmann 權重矩陣 W，並透過 SVD 和特徵分解
    找到其平方根分解 M，使得 W = M M^T。

    這是構建張量網路的第一步：將兩個自旋之間的相互作用 $e^{\beta J s_i s_j}$ 分解
    到兩個自旋各自的張量上。

    Args:
        beta: 逆溫度 (1/T)。
        J: 耦合常數。

    Returns:
        W: 2x2 的權重矩陣。
        M: 2x2 的分解矩陣，用於構建局域張量。
    """
    spins = np.array([1.0, -1.0], dtype=np.float64)
    # W_{s_i, s_j} = exp(beta * J * s_i * s_j)
    W = np.exp(beta * J * np.outer(spins, spins))

    # 透過 SVD 獲取參考的正交基底，確保分解的數值穩定性
    U_svd, singular_vals, Vh_svd = np.linalg.svd(W, full_matrices=False)

    # 對稱矩陣的特徵分解：W = V * diag(lambda) * V^T
    eigvals, eigvecs = np.linalg.eigh(W)

    # 確保特徵值按降序排列 (eigh 預設是升序)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # 調整特徵向量的相位，使其與 SVD 的左奇異向量一致，避免符號模糊
    for col in range(eigvecs.shape[1]):
        if np.dot(eigvecs[:, col], U_svd[:, col]) < 0.0:
            eigvecs[:, col] *= -1.0

    # 計算 M = V * sqrt(lambda)。這樣 M M^T = V * lambda * V^T = W
    sqrt_vals = np.sqrt(np.clip(eigvals, 0.0, None))
    M = eigvecs * sqrt_vals
    return W, M


def _tensor_network_local_tensor(beta: float, J: float, h: float) -> np.ndarray:
    """
    依照 TRG 算法的標準構建方式，創建二維 Ising 模型的初始 rank-4 局域張量 T。

    T_{ijkl} 代表一個格點，連接上下左右四個鄰居。
    指標 i, j, k, l 對應於分解矩陣 M 的虛擬鍵。

    Args:
        beta: 逆溫度。
        J: 耦合常數。
        h: 外磁場。

    Returns:
        tensor: shape (2,2,2,2) 的初始張量。
    """
    _, M = _ising_bond_decomposition(beta, J)

    # 考慮外磁場對自旋的權重: exp(beta * h * s)
    field_weights = np.array(
        [np.exp(beta * h), np.exp(-beta * h)],
        dtype=np.float64,
    )

    # T = sum_s (w_s * M_{s,i} * M_{s,j} * M_{s,k} * M_{s,l})
    # 這表示中心自旋 s 通過矩陣 M 連接到四個方向。
    tensor = np.zeros((2, 2, 2, 2), dtype=np.float64)
    for spin_idx, weight in enumerate(field_weights):
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


def _trg_step(
    tensor: np.ndarray, chi: int | None, rel_svd_cutoff: float = 0.0
) -> tuple[np.ndarray, int, np.ndarray]:
    """
    執行 Levin–Nave TRG 算法的一個粗粒化步驟 (Coarse-graining Step)。

    步驟：
    1. 將 rank-4 張量分解為兩對 rank-3 張量 (S1, S3) 和 (S2, S4)。這相當於將正方形格點沿對角線切開。
    2. 進行 SVD 分解並根據 `chi` (最大鍵維度) 進行截斷，保留最重要的物理資訊。
    3. 將這四個 S 張量按照新的拓撲結構收縮 (Contraction)，形成一個新的、代表 2x2 原始格點的粗粒化張量。

    Args:
        tensor: 當前步驟的 rank-4 張量 T[u, r, d, l]。
        chi: 最大鍵維度 (Bond Dimension)，控制截斷精度。若為 None 則不截斷 (記憶體會爆炸)。
        rel_svd_cutoff: 相對奇異值截斷門檻。

    Returns:
        coarse_tensor: 下一步的 rank-4 張量。
        chi_avg: 平均鍵維度 (用於統計)。
        s_spectrum: 用於視覺化的奇異值頻譜 (取自第一次分解)。
    """
    dim = tensor.shape[0]  # 當前張量的維度

    # --- Decomposition 1: 切分 (Up, Left) - (Right, Down) ---
    # 將張量變形為矩陣 M1，行指標為 (u, l)，列指標為 (r, d)
    t1_permuted = np.transpose(tensor, (0, 3, 1, 2))  # (u,r,d,l) -> (u,l,r,d)
    m1 = t1_permuted.reshape(dim * dim, dim * dim)
    u1, s1, vh1 = np.linalg.svd(m1, full_matrices=False)

    # 決定保留多少個奇異值 (chi1)
    chi1 = s1.size
    if rel_svd_cutoff > 0.0 and s1.size > 0:
        keep_mask = s1 >= rel_svd_cutoff * s1[0]
        chi1 = int(np.count_nonzero(keep_mask))
    if chi is not None:
        chi1 = min(chi1, chi)
    chi1 = max(1, chi1)

    # 截斷並分配奇異值 sqrt(S) 到兩邊，構建半張量 S1, S3
    u1_trunc = u1[:, :chi1] * np.sqrt(s1[:chi1])
    vh1_trunc = vh1[:chi1, :] * np.sqrt(s1[:chi1])[:, None]
    S1 = u1_trunc.reshape(dim, dim, chi1)  # S1(u, l, new_bond)
    S3 = vh1_trunc.reshape(chi1, dim, dim)  # S3(new_bond, r, d)

    # --- Decomposition 2: 切分 (Up, Right) - (Down, Left) ---
    # 將張量變形為矩陣 M2，行指標為 (u, r)，列指標為 (d, l)
    # 原始張量順序 (u, r, d, l) 已經符合，直接 reshape
    m2 = tensor.reshape(dim * dim, dim * dim)
    u2, s2, vh2 = np.linalg.svd(m2, full_matrices=False)

    # 決定保留多少個奇異值 (chi2)
    chi2 = s2.size
    if rel_svd_cutoff > 0.0 and s2.size > 0:
        keep_mask = s2 >= rel_svd_cutoff * s2[0]
        chi2 = int(np.count_nonzero(keep_mask))
    if chi is not None:
        chi2 = min(chi2, chi)
    chi2 = max(1, chi2)

    # 截斷並構建半張量 S2, S4
    u2_trunc = u2[:, :chi2] * np.sqrt(s2[:chi2])
    vh2_trunc = vh2[:chi2, :] * np.sqrt(s2[:chi2])[:, None]
    S2 = u2_trunc.reshape(dim, dim, chi2)  # S2(u, r, new_bond)
    S4 = vh2_trunc.reshape(chi2, dim, dim)  # S4(new_bond, d, l)

    # --- Contraction: 收縮形成新張量 ---
    # 根據 visualize_trg_topology 中的圖示，將 S1, S2, S3, S4 連接起來。
    # einsum: "xwl,xyu,ryz,dzw->urdl"
    # x, y, z, w 是原始格點的指標，被求和。
    # u, r, d, l 是新的虛擬鍵指標。
    coarse_tensor = np.einsum(
        "xwl,xyu,ryz,dzw->urdl",
        S1,
        S2,
        S3,
        S4,
        optimize=True,
    )

    chi_avg = int((chi1 + chi2) / 2)

    # 回傳奇異值頻譜供視覺化 (這裡我們取第一次分解的 s1 作為代表)
    s_spectrum = s1

    return coarse_tensor, chi_avg, s_spectrum


# === SimpleTRGFlow 類定義 ===
class SimpleTRGFlow:
    """
    管理 TRG 粗粒化流程的類別。

    功能：
    1. 儲存當前的張量狀態。
    2. 執行迭代更新 (update)。
    3. 計算並累積自由能密度。
    4. 記錄和繪製奇異值頻譜。
    """

    def __init__(
        self,
        tensor: np.ndarray,
        beta: float,
        *,
        max_bond_dim: int | None = None,
        rel_svd_cutoff: float = 0.0,
    ) -> None:
        # 初始化參數
        self.tensor = self._ensure_rank4_square_tensor(tensor)
        self.beta = float(beta)
        self.max_bond_dim = max_bond_dim
        self.rel_svd_cutoff = rel_svd_cutoff

        # 用於累積自由能的變數
        # 公式: f = -T * sum( ln(g_n) / 2^n )
        self.free_energy_sum_ln_gn_weighted = 0.0
        self.step = 0  # 當前迭代步數
        self.initial_num_sites = 1  # 初始只有一個格點
        self.current_scale_factor = 1.0  # 當前步的權重 (1/2^n)

        # 記錄奇異值頻譜的歷史列表
        self.spectrum_history: list[np.ndarray] = []

    def _ensure_rank4_square_tensor(self, tensor: np.ndarray) -> np.ndarray:
        """輔助方法：驗證輸入張量的格式。"""
        arr = np.asarray(tensor, dtype=np.float64)
        if arr.ndim != 4:
            raise ValueError("TRG 僅支援 rank-4 張量")
        if not (arr.shape[0] == arr.shape[1] == arr.shape[2] == arr.shape[3]):
            raise ValueError("張量必須各向同性 (所有維度相等)")
        return arr

    def update(self) -> tuple[int, float]:
        """
        執行單步迭代：
        1. 調用 _trg_step 進行張量收縮。
        2. 提取最大元素作為歸一化因子 (norm)。
        3. 更新自由能累積和。
        """
        coarse, chi_eff, s_spectrum = _trg_step(
            self.tensor, self.max_bond_dim, rel_svd_cutoff=self.rel_svd_cutoff
        )

        # 記錄本次的奇異值頻譜 (截取前 max_bond_dim 個以便繪圖)
        max_s_to_store = self.max_bond_dim if self.max_bond_dim is not None else chi_eff
        self.spectrum_history.append(s_spectrum[:max_s_to_store])

        # 歸一化：找出張量中的最大值 g_n
        norm = float(np.max(np.abs(coarse)))
        if norm <= 0.0:
            raise RuntimeError("TRG 歸一化因子為零 (數值不穩定)")

        # 將張量除以 norm，保持數值在可控範圍內
        self.tensor = coarse / norm

        self.step += 1
        # 更新權重因子：每一步系統尺寸縮小 2 倍 (或者說格點數合併為一半)，所以權重除以 2
        self.current_scale_factor /= 2.0

        # 累積自由能貢獻: (1/2^n) * ln(g_n)
        self.free_energy_sum_ln_gn_weighted += self.current_scale_factor * math.log(
            norm
        )

        return chi_eff, norm

    def run(self, iterations: int) -> None:
        """執行指定次數的迭代，並在結束後繪圖。"""
        if iterations < 0:
            raise ValueError("迭代次數需為非負整數")

        visualize_trg_topology()  # 在開始前顯示 ASCII 拓撲圖

        print(f"[step 0] 初始張量值 (shape={self.tensor.shape}):\n{self.tensor}\n")

        for _ in range(iterations):
            chi_eff, norm = self.update()

            # 計算當前有效格點數 (初始 * 2^step)
            current_effective_sites = self.initial_num_sites * (2**self.step)

            # 計算當前估計的自由能
            current_fe = self.free_energy_per_spin()

            print(
                f"[step {self.step}] chi={chi_eff}, norm={norm:.6e}, "
                f"sites={current_effective_sites}, FE/spin={current_fe:.8f}"
            )

        if iterations > 0:
            self.plot_spectrum_evolution()

    def free_energy_per_spin(self) -> float:
        """
        計算當前的單位格點自由能。
        公式: f = -T * [ sum(ln g_n / 2^n) + ln(Tr(T_final)) / (N_total) ]
        """
        # 將最終張量收縮為一個標量 (Trace)
        final_scalar = float(np.einsum("abab->", self.tensor, optimize=True))
        if final_scalar <= 0.0:
            raise RuntimeError("TRG 最終收縮結果非正值")

        # 總有效格點數 N
        total_sites = self.initial_num_sites * (2**self.step)

        # 總 ln Z / N = 累積因子 + 剩餘項 / N
        logZ_per_site = self.free_energy_sum_ln_gn_weighted + (
            math.log(final_scalar) / total_sites
        )

        return -(1.0 / self.beta) * logZ_per_site

    def plot_spectrum_evolution(self):
        """
        繪製奇異值頻譜演化圖。
        這張圖展示了：
        1. 奇異值衰減的速度 (Entanglement Spectrum)：衰減越快，TRG 效果越好。
        2. 截斷的影響：可以看到在 k > chi 處的截斷。
        3. 隨迭代的變化：觀察是否流向不動點 (Fixed Point)。
        """
        if not self.spectrum_history:
            return

        fig, ax = plt.subplots(figsize=(8, 6))

        total_steps = len(self.spectrum_history)
        # 選擇幾個關鍵步驟來繪圖，避免圖表過於擁擠
        steps_to_plot_indices = sorted(
            list(set([0, 1, 2, total_steps // 2, total_steps - 1]))
        )
        steps_to_plot_indices = [
            idx for idx in steps_to_plot_indices if idx < total_steps
        ]

        for i in steps_to_plot_indices:
            s = self.spectrum_history[i]
            # 歸一化奇異值 (除以最大值)，方便比較形狀
            s_norm = s / s[0] if s.size > 0 and s[0] != 0 else np.zeros_like(s)
            if s_norm.size > 0:
                ax.plot(
                    np.arange(1, s_norm.size + 1),
                    s_norm,
                    "o-",
                    label=f"Step {i+1}",
                    alpha=0.7,
                    markersize=4,
                )

        ax.set_yscale("log")  # 使用對數坐標，因為奇異值通常是指數衰減的
        ax.set_xlabel("Singular Value Index (k)")
        ax.set_ylabel("Normalized Singular Value (S_k / S_0)")
        ax.set_title("TRG Singular Value Spectrum Evolution")
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.legend()

        if self.max_bond_dim and self.max_bond_dim > 0:
            ax.axvline(
                x=self.max_bond_dim + 0.5,
                color="r",
                linestyle="--",
                alpha=0.7,
                label=f"Cutoff (chi={self.max_bond_dim})",
            )
            ax.legend()

        out_path = "trg_spectrum_evolution.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"\n[視覺化] 奇異值頻譜演化圖已儲存: {out_path}")


# === main 函數定義 ===
def main() -> None:
    parser = argparse.ArgumentParser(description="TRG二維Ising模型演示與視覺化")
    parser.add_argument("--step", type=int, default=0, help="TRG 迭代次數")
    parser.add_argument("--chi", type=int, default=16, help="TRG 最大鍵維度 (chi)")
    parser.add_argument("--rel-cutoff", type=float, default=0.0, help="相對截斷門檻")
    args = parser.parse_args()

    # 計算 Ising 模型物理參數 (臨界點)
    Tc = 2 / np.log(1 + np.sqrt(2))
    beta = 1.0 / Tc
    J = 1.0
    h = 0.0
    print(f"物理參數: T={Tc:.6f}, beta={beta:.6f}, J={J}, h={h}")
    print("預期自由能 (Onsager solution) ≈ -2.109651\n")

    # 1. 構建初始張量
    tensor = _tensor_network_local_tensor(beta, J, h)

    # 2. 執行 TRG
    print("=== 開始 TRG 迭代 ===")
    trg_flow = SimpleTRGFlow(
        tensor, beta, max_bond_dim=args.chi, rel_svd_cutoff=args.rel_cutoff
    )
    trg_flow.run(args.step)

    # 3. 輸出最終結果
    final_fe = trg_flow.free_energy_per_spin()
    print(f"\n最終結果 (Step {args.step}):")
    print(f"有效格點數: {trg_flow.initial_num_sites * (2**trg_flow.step)}")
    print(f"自由能/自旋: {final_fe:.8f}")


if __name__ == "__main__":
    main()

