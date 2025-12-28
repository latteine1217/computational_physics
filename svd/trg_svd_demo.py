"""
TRG 中 SVD 應用的互動式示範

本程式展示：
1. 如何將張量重組為矩陣
2. SVD 分解的詳細過程
3. 截斷策略的影響
4. 奇異值分佈與能量保留
5. 與您的 truncated_svd.py 整合
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


def build_ising_local_tensor(beta: float, J: float = 1.0, h: float = 0.0) -> np.ndarray:
    """
    建立 2D Ising 模型的局域 rank-4 張量
    
    使用 cosh/sinh 分解方法
    """
    cosh_val = np.cosh(beta * J)
    sinh_val = np.sinh(beta * J)
    
    # M[自旋索引, 虛擬鍵]: 平方根分解
    M = np.array([
        [np.sqrt(cosh_val), np.sqrt(sinh_val)],
        [np.sqrt(cosh_val), -np.sqrt(sinh_val)]
    ], dtype=np.float64)
    
    # 外場權重
    field_weights = np.array([np.exp(beta * h), np.exp(-beta * h)])
    
    # 構建 rank-4 張量 T[up, right, down, left]
    tensor = np.zeros((2, 2, 2, 2), dtype=np.float64)
    for spin_idx, weight in enumerate(field_weights):
        # 四個方向的虛擬鍵外積
        contrib = np.einsum('i,j,k,l->ijkl',
                           M[spin_idx], M[spin_idx],
                           M[spin_idx], M[spin_idx],
                           optimize=True)
        tensor += weight * contrib
    
    return tensor


def visualize_tensor_to_matrix_reshaping(tensor: np.ndarray, save_path: Optional[str] = None):
    """
    視覺化張量到矩陣的重組過程
    """
    dim = tensor.shape[0]
    
    # 重組為矩陣
    permuted = np.transpose(tensor, (0, 3, 1, 2))  # (up, left, right, down)
    matrix = permuted.reshape(dim * dim, dim * dim)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 原始張量（展示為 4D 切片）
    axes[0].imshow(tensor[:, :, 0, 0], cmap='RdBu', aspect='auto')
    axes[0].set_title(f'Original Tensor T[u,r,d=0,l=0]\nShape: {tensor.shape}')
    axes[0].set_xlabel('Right index')
    axes[0].set_ylabel('Up index')
    
    # 重排後的張量
    axes[1].imshow(permuted[:, :, 0, 0], cmap='RdBu', aspect='auto')
    axes[1].set_title(f'Permuted T[u,l,r=0,d=0]\nShape: {permuted.shape}')
    axes[1].set_xlabel('Left index')
    axes[1].set_ylabel('Up index')
    
    # 重組為矩陣
    im = axes[2].imshow(matrix, cmap='RdBu', aspect='auto')
    axes[2].set_title(f'Matrix M[(u,l), (r,d)]\nShape: {matrix.shape}')
    axes[2].set_xlabel('Column: (right, down)')
    axes[2].set_ylabel('Row: (up, left)')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已儲存圖片至: {save_path}")
    
    plt.show()


def analyze_svd_spectrum(tensor: np.ndarray, save_path: Optional[str] = None):
    """
    分析張量 SVD 的奇異值譜
    """
    dim = tensor.shape[0]
    
    # 重組為矩陣
    permuted = np.transpose(tensor, (0, 3, 1, 2))
    matrix = permuted.reshape(dim * dim, dim * dim)
    
    # SVD 分解
    U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
    
    # 計算累積能量
    cumulative_energy = np.cumsum(S**2) / np.sum(S**2)
    
    # 繪圖
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 奇異值分佈
    axes[0].semilogy(range(1, len(S) + 1), S, 'bo-', markersize=6)
    axes[0].set_xlabel('Index k')
    axes[0].set_ylabel('Singular Value σ_k')
    axes[0].set_title('Singular Value Spectrum')
    axes[0].grid(True, alpha=0.3)
    
    # 歸一化奇異值
    axes[1].plot(range(1, len(S) + 1), S / S[0], 'ro-', markersize=6)
    axes[1].set_xlabel('Index k')
    axes[1].set_ylabel('σ_k / σ_1')
    axes[1].set_title('Normalized Singular Values')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    # 累積能量
    axes[2].plot(range(1, len(S) + 1), cumulative_energy * 100, 'go-', markersize=6)
    axes[2].axhline(y=90, color='orange', linestyle='--', label='90%')
    axes[2].axhline(y=95, color='red', linestyle='--', label='95%')
    axes[2].axhline(y=99, color='purple', linestyle='--', label='99%')
    axes[2].set_xlabel('Bond Dimension χ')
    axes[2].set_ylabel('Cumulative Energy (%)')
    axes[2].set_title('Energy Retention')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已儲存圖片至: {save_path}")
    
    plt.show()
    
    # 輸出統計信息
    print("=" * 60)
    print("奇異值統計")
    print("=" * 60)
    print(f"總奇異值數量: {len(S)}")
    print(f"最大奇異值: {S[0]:.6e}")
    print(f"最小奇異值: {S[-1]:.6e}")
    print(f"條件數: {S[0] / S[-1]:.2e}")
    print()
    
    for threshold in [0.90, 0.95, 0.99, 0.999]:
        rank = np.argmax(cumulative_energy >= threshold) + 1
        print(f"保留 {threshold*100:.1f}% 能量所需秩數: {rank}")
    
    print("=" * 60)
    
    return S, cumulative_energy


def compare_truncation_effects(
    tensor: np.ndarray,
    chi_values: list[int],
    save_path: Optional[str] = None
):
    """
    比較不同截斷秩數的效果
    """
    dim = tensor.shape[0]
    
    # 重組為矩陣
    permuted = np.transpose(tensor, (0, 3, 1, 2))
    matrix = permuted.reshape(dim * dim, dim * dim)
    
    # 完整 SVD（參考）
    U_full, S_full, Vh_full = np.linalg.svd(matrix, full_matrices=False)
    matrix_ref = U_full @ np.diag(S_full) @ Vh_full
    
    results = []
    
    for chi in chi_values:
        # 截斷 SVD
        keep = min(chi, len(S_full))
        U = U_full[:, :keep]
        S = S_full[:keep]
        Vh = Vh_full[:keep, :]
        
        # 重建矩陣
        matrix_approx = U @ np.diag(S) @ Vh
        
        # 計算誤差
        rel_error = np.linalg.norm(matrix - matrix_approx, 'fro') / np.linalg.norm(matrix, 'fro')
        
        # 能量保留
        energy_kept = np.sum(S**2) / np.sum(S_full**2)
        
        # 壓縮率
        original_params = matrix.size
        compressed_params = U.size + S.size + Vh.size
        compression_ratio = compressed_params / original_params
        
        results.append({
            'chi': chi,
            'rel_error': rel_error,
            'energy_kept': energy_kept,
            'compression_ratio': compression_ratio
        })
        
        print(f"χ = {chi:3d} | 誤差: {rel_error:.6e} | 能量: {energy_kept*100:.2f}% | 壓縮率: {compression_ratio*100:.1f}%")
    
    # 繪圖
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    chi_arr = [r['chi'] for r in results]
    
    # 相對誤差
    axes[0].semilogy(chi_arr, [r['rel_error'] for r in results], 'bo-', markersize=8)
    axes[0].set_xlabel('Bond Dimension χ')
    axes[0].set_ylabel('Relative Error')
    axes[0].set_title('Truncation Error')
    axes[0].grid(True, alpha=0.3)
    
    # 能量保留
    axes[1].plot(chi_arr, [r['energy_kept']*100 for r in results], 'ro-', markersize=8)
    axes[1].axhline(y=99, color='purple', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Bond Dimension χ')
    axes[1].set_ylabel('Energy Retained (%)')
    axes[1].set_title('Energy Retention')
    axes[1].grid(True, alpha=0.3)
    
    # 壓縮率
    axes[2].plot(chi_arr, [r['compression_ratio']*100 for r in results], 'go-', markersize=8)
    axes[2].set_xlabel('Bond Dimension χ')
    axes[2].set_ylabel('Compression Ratio (%)')
    axes[2].set_title('Storage Efficiency')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n已儲存圖片至: {save_path}")
    
    plt.show()
    
    return results


def trg_step_with_visualization(
    tensor: np.ndarray,
    max_bond_dim: int = 8,
    show_details: bool = True
) -> Tuple[np.ndarray, int]:
    """
    執行一次 TRG 步驟，並提供詳細的視覺化輸出
    """
    dim = tensor.shape[0]
    
    if show_details:
        print("\n" + "=" * 60)
        print("TRG 單步分解 - 詳細過程")
        print("=" * 60)
        print(f"\n原始張量形狀: {tensor.shape}")
        print(f"虛擬鍵維度: {dim}")
    
    # 步驟 1: 重組
    permuted = np.transpose(tensor, (0, 3, 1, 2))
    matrix = permuted.reshape(dim * dim, dim * dim)
    
    if show_details:
        print(f"\n[步驟 1] 張量 → 矩陣")
        print(f"  重排指標: (u,r,d,l) → (u,l,r,d)")
        print(f"  矩陣形狀: {matrix.shape}")
    
    # 步驟 2: SVD
    U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
    
    if show_details:
        print(f"\n[步驟 2] SVD 分解")
        print(f"  U 形狀: {U.shape}")
        print(f"  奇異值數量: {len(S)}")
        print(f"  前 5 個奇異值: {S[:min(5, len(S))]}")
    
    # 步驟 3: 截斷
    keep = min(max_bond_dim, len(S))
    
    if show_details:
        cumulative_energy = np.cumsum(S**2) / np.sum(S**2)
        print(f"\n[步驟 3] 截斷")
        print(f"  原始秩: {len(S)}")
        print(f"  截斷秩: {keep}")
        print(f"  保留能量: {cumulative_energy[keep-1]*100:.2f}%")
    
    U = U[:, :keep]
    S = S[:keep]
    Vh = Vh[:keep, :]
    
    # 步驟 4: 平方根分配
    sqrt_S = np.sqrt(S)
    U = U * sqrt_S
    Vh = sqrt_S[:, None] * Vh
    
    if show_details:
        print(f"\n[步驟 4] 平方根分配奇異值")
        print(f"  U 吸收 √S 後形狀: {U.shape}")
        print(f"  Vh 吸收 √S 後形狀: {Vh.shape}")
    
    # 步驟 5: 重組
    S1 = U.reshape(dim, dim, keep)
    S2 = Vh.reshape(keep, dim, dim)
    
    if show_details:
        print(f"\n[步驟 5] 矩陣 → rank-3 張量")
        print(f"  S1 (左張量) 形狀: {S1.shape}")
        print(f"  S2 (右張量) 形狀: {S2.shape}")
    
    # 步驟 6: 收縮
    coarse = np.einsum(
        'api,ibj,cjq,qda->abcd',
        S1, S2, S1, S2,
        optimize=True
    )
    
    if show_details:
        print(f"\n[步驟 6] 收縮成粗粒化張量")
        print(f"  新張量形狀: {coarse.shape}")
        print(f"  虛擬鍵維度: {dim} → {keep}")
        print("=" * 60 + "\n")
    
    return coarse, keep


# ============================================================
# 主示範程式
# ============================================================

def main():
    """
    完整示範 SVD 在 TRG 中的應用
    """
    print("=" * 60)
    print("SVD 在 TRG 算法中的應用 - 互動式示範")
    print("=" * 60)
    
    # 參數設定
    beta = 0.5  # 逆溫度（高溫）
    J = 1.0
    h = 0.0
    
    print(f"\n參數設定:")
    print(f"  溫度 T = 1/β = {1/beta:.2f}")
    print(f"  交換參數 J = {J}")
    print(f"  外場 h = {h}")
    
    # 建立局域張量
    print("\n正在建立 Ising 局域張量...")
    tensor = build_ising_local_tensor(beta, J, h)
    print(f"張量形狀: {tensor.shape}")
    print(f"張量範數: {np.linalg.norm(tensor):.6f}")
    
    # 示範 1: 張量到矩陣的重組
    print("\n" + "=" * 60)
    print("示範 1: 張量 → 矩陣重組")
    print("=" * 60)
    visualize_tensor_to_matrix_reshaping(tensor)
    
    # 示範 2: 奇異值譜分析
    print("\n" + "=" * 60)
    print("示範 2: 奇異值譜分析")
    print("=" * 60)
    S, cumulative_energy = analyze_svd_spectrum(tensor)
    
    # 示範 3: 不同截斷秩的比較
    print("\n" + "=" * 60)
    print("示範 3: 截斷策略比較")
    print("=" * 60)
    chi_values = [1, 2, 3, 4]
    results = compare_truncation_effects(tensor, chi_values)
    
    # 示範 4: 完整 TRG 步驟
    print("\n" + "=" * 60)
    print("示範 4: 執行完整 TRG 步驟")
    print("=" * 60)
    coarse_tensor, chi_eff = trg_step_with_visualization(
        tensor, 
        max_bond_dim=3,
        show_details=True
    )
    
    print("\n完成！所有示範已執行。")
    print("\n建議:")
    print("  1. 調整 beta 參數，觀察不同溫度下的奇異值譜")
    print("  2. 嘗試更大的截斷秩 max_bond_dim")
    print("  3. 查看 SVD_in_TRG_tutorial.md 了解更多理論細節")


if __name__ == "__main__":
    main()
