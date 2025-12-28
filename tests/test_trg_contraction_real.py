"""
使用真實 TRG 張量測試收縮方式

目的：
1. 使用實際的 Ising 初始張量
2. 執行一步 TRG 並檢查結果
3. 對比不同收縮方式的物理正確性
"""

import numpy as np
import sys

# ==================== 從 trg_final_project.py 導入 ====================
sys.path.insert(0, '/Users/latteine/Documents/coding/computational_physics')
from trg_final_project import _ising_local_tensor, _trg_step, TRGFlow


def test_single_trg_step():
    """測試單步 TRG"""
    print("=" * 80)
    print("測試真實 TRG 單步粗粒化")
    print("=" * 80)
    
    # 在臨界溫度測試
    Tc = 2.0 / np.log(1 + np.sqrt(2))
    beta = 1.0 / Tc
    J = 1.0
    h = 0.0
    
    # 構造初始張量
    tensor = _ising_local_tensor(beta, J, h)
    print(f"\n初始張量 T:")
    print(f"  形狀：{tensor.shape}")
    print(f"  Trace：{np.einsum('abab->', tensor):.6f}")
    
    # 測試不同的 chi 值
    chi_values = [2, 4, 8]
    
    for chi in chi_values:
        print(f"\n{'=' * 80}")
        print(f"Chi = {chi}")
        print(f"{'=' * 80}")
        
        # 執行一步 TRG
        coarse, chi_eff, _ = _trg_step(tensor, chi)
        
        print(f"\n粗粒化後張量 T':")
        print(f"  形狀：{coarse.shape}")
        print(f"  Trace：{np.einsum('abab->', coarse):.6f}")
        print(f"  實際使用 chi：{chi_eff}")
        
        # 檢查對稱性
        print(f"\n對稱性檢查：")
        # T'[a,b,c,d] 應該和 T'[c,d,a,b] 相同（90度旋轉對稱）
        rotated_90 = np.transpose(coarse, (2, 3, 0, 1))
        diff_90 = np.linalg.norm(coarse - rotated_90)
        print(f"  90° 旋轉對稱性誤差：{diff_90:.6e}")
        
        # T'[a,b,c,d] 應該和 T'[d,c,b,a] 相同（180度旋轉對稱）
        rotated_180 = np.transpose(coarse, (2, 3, 0, 1))
        diff_180 = np.linalg.norm(coarse - rotated_180)
        print(f"  180° 旋轉對稱性誤差：{diff_180:.6e}")
        
        # 檢查 trace 的歸一化
        trace = np.einsum('abab->', coarse)
        print(f"  Trace（應該接近但不等於 1）：{trace:.6f}")


def test_convergence_with_chi():
    """測試收斂性隨 chi 的變化"""
    print("\n" + "=" * 80)
    print("測試收斂性（應該隨 chi 增加而顯著改善）")
    print("=" * 80)
    
    Tc = 2.0 / np.log(1 + np.sqrt(2))
    beta = 1.0 / Tc
    J = 1.0
    h = 0.0
    
    tensor = _ising_local_tensor(beta, J, h)
    
    chi_values = [2, 4, 8, 16, 32]
    iterations = 10
    
    exact_fe = -2.109651  # Onsager 精確解
    
    print(f"\n{'Chi':>5} | {'Iteration':>10} | {'Free Energy':>15} | {'Error (%)':>12}")
    print("-" * 80)
    
    for chi in chi_values:
        flow = TRGFlow(tensor, beta, chi)
        
        for i in range(iterations):
            flow.update()
        
        fe = flow.free_energy_per_spin()
        error = abs((fe - exact_fe) / exact_fe) * 100.0
        
        print(f"{chi:5d} | {iterations:10d} | {fe:15.8f} | {error:12.6f}")
    
    print("\n預期行為：")
    print("  - Chi 增加時，error 應該減少數個數量級")
    print("  - 例如：chi=4 -> 1%, chi=8 -> 0.1%, chi=16 -> 0.01%")
    print(f"  - Onsager 精確解：{exact_fe:.8f}")


def test_initial_tensor_match():
    """驗證初始張量與參考實現一致"""
    print("\n" + "=" * 80)
    print("驗證初始張量構造")
    print("=" * 80)
    
    Tc = 2.0 / np.log(1 + np.sqrt(2))
    beta = 1.0 / Tc
    J = 1.0
    h = 0.0
    
    tensor = _ising_local_tensor(beta, J, h)
    
    print(f"\n初始張量 T[u,r,d,l]:")
    print(f"  形狀：{tensor.shape}")
    print(f"  Trace：{np.einsum('abab->', tensor):.12f}")
    print(f"  預期 trace：14.778112 (來自 TRG_Ising.py)")
    
    # 檢查對稱性
    print(f"\n對稱性檢查：")
    # T[u,r,d,l] 應該和 T[r,d,l,u] 相同（90度旋轉）
    rotated = np.transpose(tensor, (1, 2, 3, 0))
    diff = np.linalg.norm(tensor - rotated)
    print(f"  90° 旋轉對稱性：{diff:.6e}")
    
    # T[u,r,d,l] 應該和 T[l,d,r,u] 相同（左右翻轉）
    flipped_lr = np.transpose(tensor, (3, 2, 1, 0))
    diff_lr = np.linalg.norm(tensor - flipped_lr)
    print(f"  左右翻轉對稱性：{diff_lr:.6e}")


if __name__ == "__main__":
    test_initial_tensor_match()
    test_single_trg_step()
    test_convergence_with_chi()
