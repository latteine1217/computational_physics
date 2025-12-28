import numpy as np


def build_local_tensor(beta: float, J: float, h: float) -> np.ndarray:
    """
    建立 2D Ising 單站點 rank-4 張量。
    
    基本流程：
        1. 以 cosh/sinh 分解最近鄰權重 W_{ss'} = exp(βJ s s')。
        2. 蒐集同一中心自旋對四條虛擬鍵的貢獻，形成 rank-4 張量。
        3. 外場權重以 exp(β h s) 乘入。
    回傳張量指標順序為 (up, right, down, left)。
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


def trg_single_coarse_grain(
    tensor: np.ndarray,
    *,
    orientation: str = "horizontal",
    max_bond_dim: int | None = None,
    rel_svd_cutoff: float = 0.0,
    normalize: bool = True,
) -> tuple[np.ndarray, float, int]:
    """
    將 2x2 張量方塊收縮成新的粗粒化張量。
    
    作法：
        1. 以四份輸入張量組成 2x2 方塊，內部共用鍵直接收縮。
        2. 外圍四組邊 (上下左右) 保留下來，對應新張量的四個指標。
        3. 選擇性地對新張量做正規化以避免數值爆炸，回傳正規化係數。
    
    Args:
        tensor: Rank-4 張量，指標順序必須為 (up, right, down, left) 且各向度相同。
        orientation: 'horizontal' 代表以原始 (up, right, down, left) 配線收縮；
            'vertical' 則先旋轉張量，再套用相同拆分流程。
        max_bond_dim: 允許的最大虛擬鍵維度；若為 None 則不截斷。
        rel_svd_cutoff: 依據最大奇異值的比例篩選保留奇異值，避免留下過小權重。
        normalize: 是否以 2-範數重新縮放回傳張量。

    Returns:
        renormalized_tensor: 收縮後的 rank-4 張量，外部指標對應到 (up, right, down, left)。
        norm_factor: 若 normalize=True，為用於縮放的 2-範數；否則為 1.0。
        chi_effective: 本次截斷後保留的虛擬鍵維度。
    
    注意：此函式採用 Levin–Nave (2007) TRG 的奇偶拆分/重組流程。
    """
    if tensor.ndim != 4:
        raise ValueError("輸入張量必須為 rank-4")
    if not (tensor.shape[0] == tensor.shape[1] == tensor.shape[2] == tensor.shape[3]):
        raise ValueError("四個指標的維度必須一致")
    if orientation not in {"horizontal", "vertical"}:
        raise ValueError("orientation 必須為 'horizontal' 或 'vertical'")

    if orientation == "vertical":
        rotated = np.transpose(tensor, (1, 2, 3, 0))
        renormalized, chi_eff = _levin_nave_trg_step(
            rotated,
            max_bond_dim=max_bond_dim,
            rel_svd_cutoff=rel_svd_cutoff,
        )
        renormalized = np.transpose(renormalized, (3, 0, 1, 2))
    else:
        renormalized, chi_eff = _levin_nave_trg_step(
            tensor,
            max_bond_dim=max_bond_dim,
            rel_svd_cutoff=rel_svd_cutoff,
        )

    if normalize:
        norm = np.linalg.norm(renormalized)
        if norm == 0.0:
            norm = 1.0
        renormalized = renormalized / norm
        return renormalized, norm, chi_eff
    return renormalized, 1.0, chi_eff


def _levin_nave_trg_step(
    tensor: np.ndarray,
    *,
    max_bond_dim: int | None,
    rel_svd_cutoff: float,
) -> tuple[np.ndarray, int]:
    """
    以 Levin–Nave TRG 標準步驟粗粒化 2x2 張量方塊。
    """
    dim = tensor.shape[0]
    permuted = np.transpose(tensor, (0, 3, 1, 2))  # (up, left, right, down)
    matrix = permuted.reshape(dim * dim, dim * dim)
    U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
    if rel_svd_cutoff > 0.0 and S.size > 0:
        sigma_max = S[0]
        keep_mask = S >= rel_svd_cutoff * sigma_max
        keep = int(np.count_nonzero(keep_mask))
    else:
        keep = S.size
    if max_bond_dim is not None:
        keep = min(keep, max_bond_dim)
    keep = max(1, keep)

    U = U[:, :keep]
    S = S[:keep]
    Vh = Vh[:keep, :]
    sqrt_S = np.sqrt(S)
    U = U * sqrt_S
    Vh = sqrt_S[:, None] * Vh

    S1 = U.reshape(dim, dim, keep)          # (up, left, α)
    S2 = Vh.reshape(keep, dim, dim)         # (α, right, down)

    coarse = np.einsum(
        "api,ibj,cjq,qda->abcd",
        S1,
        S2,
        S1,
        S2,
        optimize=True,
    )
    return coarse, keep


def trg_iterative_flow(
    tensor: np.ndarray,
    num_steps: int,
    *,
    max_bond_dim: int | None = None,
    rel_svd_cutoff: float = 0.0,
    normalize: bool = True,
) -> tuple[np.ndarray, float, dict[str, list[float]]]:
    """
    重複執行 TRG 粗粒化步驟，並在水平/垂直配線間交替。
    
    每次粗粒化的正規化係數以對數累積，可做為最終 logZ 的重建資訊之一。
    
    Returns:
        tensor: 最後一次粗粒化後的張量。
        log_scale: 歷次正規化係數之對數總和。
        history: 包含 'scales' 與 'bond_dims' 的截斷紀錄。
    """
    current = tensor
    log_scale = 0.0
    scales: list[float] = []
    chi_history: list[float] = []
    orientation = "horizontal"

    for _ in range(num_steps):
        current, scale, chi_eff = trg_single_coarse_grain(
            current,
            orientation=orientation,
            max_bond_dim=max_bond_dim,
            rel_svd_cutoff=rel_svd_cutoff,
            normalize=normalize,
        )
        if normalize and scale <= 0.0:
            raise RuntimeError("出現非正的正規化係數，請檢查張量內容")
        if normalize:
            log_scale += float(np.log(scale))
            scales.append(float(scale))
        else:
            scales.append(1.0)
        chi_history.append(float(chi_eff))
        orientation = "vertical" if orientation == "horizontal" else "horizontal"

    history = {
        "scales": scales,
        "bond_dims": chi_history,
    }
    return current, log_scale, history


def trg_log_partition_estimate(
    tensor: np.ndarray,
    log_scale: float,
) -> float:
    """
    依據最終張量與縮放係數累積，估算對數配分函數。
    
    若最終張量尚未完全收縮為純量，將所有指標以 trace 方式收縮成單一數值。
    """
    if tensor.ndim != 4:
        raise ValueError("最終張量必須為 rank-4")
    # 將 (up, right, down, left) 依 PBC 連成 (up=down, right=left) 以完成收縮
    scalar = np.einsum("abab->", tensor, optimize=True)
    if scalar <= 0.0:
        raise RuntimeError("最終收縮結果非正值，無法取得 logZ")
    return log_scale + float(np.log(scalar))


if __name__ == "__main__":
    # 示範：建立局域張量後，交替進行水平與垂直粗粒化
    temp = 1.0
    beta = 1.0 / temp
    J = 1.0
    h = 0.0
    local_tensor = build_local_tensor(beta, J, h)
    final_tensor, log_scale, history = trg_iterative_flow(
        local_tensor,
        num_steps=4,
        max_bond_dim=3,
        rel_svd_cutoff=1e-8,
    )
    print("Final tensor shape:", final_tensor.shape)
    print("Log scale accumulation:", log_scale)
    print("Per-step scales:", history["scales"])
    print("Effective bond dims:", history["bond_dims"])
    logZ_est = trg_log_partition_estimate(final_tensor, log_scale)
    print("Estimated logZ:", logZ_est)
