# SVD 在 TRG 算法中的應用

## 目錄
1. [TRG 算法概述](#1-trg-算法概述)
2. [SVD 在 TRG 中的三大角色](#2-svd-在-trg-中的三大角色)
3. [數學原理詳解](#3-數學原理詳解)
4. [完整代碼示範](#4-完整代碼示範)
5. [視覺化理解](#5-視覺化理解)
6. [實際應用案例](#6-實際應用案例)

---

## 1. TRG 算法概述

**Tensor Renormalization Group (TRG)** 是由 Levin 和 Nave 在 2007 年提出的張量網路重整化方法，用於計算二維統計物理系統的配分函數。

### 核心思想
將二維格子上的配分函數表示為張量網路的收縮，通過**迭代粗粒化**將格子尺寸減半，直到可以直接計算。

### 基本流程
```
初始格子 (L×L)  →  粗粒化  →  (L/2 × L/2)  →  粗粒化  →  ...  →  (2×2)  →  收縮得到 Z
```

---

## 2. SVD 在 TRG 中的三大角色

### 角色 1: 構建局域張量（初始化階段）

**目的**: 將 Boltzmann 權重分解為張量形式

對於 2D Ising 模型，鄰接自旋的相互作用權重為：
```
W_{ss'} = exp(βJ s s')
```

**SVD 的作用**: 將 2×2 矩陣 W 分解為平方根形式
```python
W, M = _ising_bond_decomposition(beta, J)
# W = M @ M.T  (通過對稱特徵分解)
```

**物理意義**: 
- W 描述兩個自旋的關聯
- M 將這個關聯"平方根化"，每個自旋各佔一個 M
- 四個方向的 M 外積形成 rank-4 張量

```python
# 來自您的代碼 (tensor_network_2x2.py 第 5-32 行)
def _ising_bond_decomposition(beta: float, J: float):
    spins = np.array([1.0, -1.0])
    W = np.exp(beta * J * np.outer(spins, spins))
    
    # 使用 SVD 檢查數值穩定性
    U_svd, singular_vals, Vh_svd = np.linalg.svd(W, full_matrices=False)
    
    # 使用對稱特徵分解得到 M，使 W = M M^T
    eigvals, eigvecs = np.linalg.eigh(W)
    sqrt_vals = np.sqrt(eigvals)
    M = eigvecs * sqrt_vals
    
    return W, M
```

---

### 角色 2: 張量粗粒化（核心步驟）⭐

**這是 SVD 在 TRG 中最關鍵的應用！**

#### 問題設定
給定一個 rank-4 張量 T[u,r,d,l]（上、右、下、左四個指標），要將 2×2 的四個張量收縮成一個新張量。

#### SVD 解決方案

**步驟 1: 重組張量為矩陣**
```python
# 從您的代碼 (tensor_trg_step.py 第 115-117 行)
dim = tensor.shape[0]
permuted = np.transpose(tensor, (0, 3, 1, 2))  # (up, left, right, down)
matrix = permuted.reshape(dim * dim, dim * dim)
```

將四個指標分為兩組：
- 行指標: (up, left)
- 列指標: (right, down)

形成矩陣: M_{(u,l), (r,d)}

**步驟 2: 執行 SVD**
```python
U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
```

得到:
```
M = U @ diag(S) @ Vh
```

其中:
- U: 左奇異向量 (dim² × rank)
- S: 奇異值 (rank,)
- Vh: 右奇異向量 (rank × dim²)

**步驟 3: 截斷到最大鍵維度 χ**
```python
# 從您的代碼 (tensor_trg_step.py 第 119-127 行)
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
```

**步驟 4: 分配奇異值到兩側**
```python
# 從您的代碼 (tensor_trg_step.py 第 132-134 行)
sqrt_S = np.sqrt(S)
U = U * sqrt_S        # 左邊吸收 √S
Vh = sqrt_S[:, None] * Vh  # 右邊吸收 √S
```

這樣做的原因：
- 保持對稱性
- U 和 Vh 成為"等價"的張量
- 可以在水平和垂直方向重複使用

**步驟 5: 重組為新的 rank-3 張量**
```python
# 從您的代碼 (tensor_trg_step.py 第 136-137 行)
S1 = U.reshape(dim, dim, keep)     # (up, left, α)
S2 = Vh.reshape(keep, dim, dim)    # (α, right, down)
```

**步驟 6: 收縮成粗粒化張量**
```python
# 從您的代碼 (tensor_trg_step.py 第 139-146 行)
coarse = np.einsum(
    "api,ibj,cjq,qda->abcd",
    S1,  # 左上
    S2,  # 右上
    S1,  # 左下
    S2,  # 右下
    optimize=True,
)
```

---

### 角色 3: 截斷控制（數值穩定性）

SVD 提供了**最優截斷策略**：

#### Eckart-Young-Mirsky 定理
保留前 k 個奇異值，是秩-k 近似的最佳選擇（在 Frobenius 範數意義下）。

```python
# 兩種截斷策略

# 策略 1: 固定最大鍵維度
max_bond_dim = 16
keep = min(len(S), max_bond_dim)

# 策略 2: 相對奇異值閾值
rel_svd_cutoff = 1e-8
keep_mask = S >= rel_svd_cutoff * S[0]
keep = np.count_nonzero(keep_mask)
```

**物理意義**:
- 小的奇異值 → 弱關聯
- 大的奇異值 → 強關聯
- 截斷 = 忽略弱關聯，保留主要物理信息

---

## 3. 數學原理詳解

### 3.1 為什麼要將張量重組為矩陣？

**張量收縮的計算複雜度**問題：
- 直接收縮 4 個 rank-4 張量: O(d^8)
- 通過 SVD 分解: O(d^3) + O(d^6) ≈ O(d^6)

**關鍵洞察**: 矩陣 SVD 是高效且數值穩定的

### 3.2 SVD 分解的幾何意義

對於矩陣 M:
```
M = U @ diag(S) @ V^T
```

可以理解為：
1. V^T: 輸入空間的旋轉（新基底）
2. diag(S): 沿各基底方向的縮放
3. U: 輸出空間的旋轉

**在 TRG 中**:
- U: 提取"左側"（up, left）的主要模式
- V^T: 提取"右側"（right, down）的主要模式
- S: 這些模式的重要性權重

### 3.3 為什麼要平方根分配奇異值？

```python
sqrt_S = np.sqrt(S)
U_tilde = U @ diag(sqrt_S)
V_tilde = diag(sqrt_S) @ V^T
```

**原因**:
1. **對稱性**: 左右兩個張量地位相同
2. **可重用性**: 同一個張量可用於多個收縮
3. **數值穩定**: 避免奇異值過大或過小

**數學驗證**:
```
M = U @ diag(S) @ V^T
  = (U @ diag(√S)) @ (diag(√S) @ V^T)
  = U_tilde @ V_tilde
```

---

## 4. 完整代碼示範

### 4.1 基本 TRG 步驟（帶詳細註解）

```python
import numpy as np
from truncated_svd import truncated_svd, reconstruct_matrix

def trg_svd_step_detailed(tensor, max_bond_dim=None):
    """
    TRG 單步分解 - 詳細版本
    
    輸入: tensor[u,r,d,l] - rank-4 張量
    輸出: coarse_tensor[u,r,d,l] - 粗粒化後的張量
    """
    dim = tensor.shape[0]
    
    print("=" * 60)
    print("TRG SVD 步驟詳解")
    print("=" * 60)
    
    # 步驟 1: 重組為矩陣
    print("\n[步驟 1] 張量 → 矩陣重組")
    print(f"  原始張量形狀: {tensor.shape}")
    
    # 將指標重排為 (up, left, right, down)
    permuted = np.transpose(tensor, (0, 3, 1, 2))
    print(f"  重排後形狀: {permuted.shape}")
    
    # 合併為矩陣: 行=(up,left), 列=(right,down)
    matrix = permuted.reshape(dim * dim, dim * dim)
    print(f"  矩陣形狀: {matrix.shape}")
    print(f"  矩陣秩: {np.linalg.matrix_rank(matrix)}")
    
    # 步驟 2: SVD 分解
    print("\n[步驟 2] 執行 SVD 分解")
    U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
    print(f"  U 形狀: {U.shape}")
    print(f"  奇異值數量: {len(S)}")
    print(f"  前 5 個奇異值: {S[:5]}")
    print(f"  Vh 形狀: {Vh.shape}")
    
    # 計算能量保留比例
    cumulative_energy = np.cumsum(S**2) / np.sum(S**2)
    print(f"  前 90% 能量所需秩數: {np.argmax(cumulative_energy >= 0.9) + 1}")
    print(f"  前 99% 能量所需秩數: {np.argmax(cumulative_energy >= 0.99) + 1}")
    
    # 步驟 3: 截斷
    print("\n[步驟 3] 截斷到最大鍵維度")
    if max_bond_dim is None:
        keep = len(S)
    else:
        keep = min(max_bond_dim, len(S))
    
    print(f"  原始秩: {len(S)}")
    print(f"  截斷秩: {keep}")
    print(f"  保留能量: {cumulative_energy[keep-1]*100:.2f}%")
    
    U = U[:, :keep]
    S = S[:keep]
    Vh = Vh[:keep, :]
    
    # 步驟 4: 平方根分配
    print("\n[步驟 4] 平方根分配奇異值")
    sqrt_S = np.sqrt(S)
    print(f"  奇異值範圍: [{S.min():.2e}, {S.max():.2e}]")
    print(f"  √奇異值範圍: [{sqrt_S.min():.2e}, {sqrt_S.max():.2e}]")
    
    U = U * sqrt_S
    Vh = sqrt_S[:, None] * Vh
    
    # 步驟 5: 重組為 rank-3 張量
    print("\n[步驟 5] 矩陣 → rank-3 張量")
    S1 = U.reshape(dim, dim, keep)
    S2 = Vh.reshape(keep, dim, dim)
    print(f"  S1 (左張量) 形狀: {S1.shape}")
    print(f"  S2 (右張量) 形狀: {S2.shape}")
    
    # 步驟 6: 收縮成新張量
    print("\n[步驟 6] 收縮成粗粒化張量")
    # 2×2 方塊收縮示意:
    #   S1 -- S2
    #   |     |
    #   S1 -- S2
    coarse = np.einsum(
        "api,ibj,cjq,qda->abcd",
        S1, S2, S1, S2,
        optimize=True
    )
    print(f"  粗粒化張量形狀: {coarse.shape}")
    print(f"  虛擬鍵維度: {dim} → {keep}")
    
    print("\n" + "=" * 60)
    
    return coarse, keep


def demonstrate_trg_svd():
    """
    完整示範 TRG 中的 SVD 應用
    """
    # 建立 2D Ising 局域張量
    beta = 1.0  # 逆溫度
    J = 1.0     # 交換參數
    h = 0.0     # 外場
    
    # 使用 cosh/sinh 分解
    cosh_val = np.cosh(beta * J)
    sinh_val = np.sinh(beta * J)
    
    M = np.array([
        [np.sqrt(cosh_val), np.sqrt(sinh_val)],
        [np.sqrt(cosh_val), -np.sqrt(sinh_val)]
    ])
    
    # 外場權重
    field_weights = np.array([np.exp(beta * h), np.exp(-beta * h)])
    
    # 構建 rank-4 張量
    tensor = np.zeros((2, 2, 2, 2))
    for spin_idx, weight in enumerate(field_weights):
        contrib = np.einsum('i,j,k,l->ijkl', 
                           M[spin_idx], M[spin_idx], 
                           M[spin_idx], M[spin_idx])
        tensor += weight * contrib
    
    print("\n初始局域張量")
    print(f"形狀: {tensor.shape}")
    print(f"範數: {np.linalg.norm(tensor):.4f}")
    
    # 執行一次 TRG 步驟
    coarse_tensor, chi = trg_svd_step_detailed(tensor, max_bond_dim=4)
    
    return tensor, coarse_tensor, chi
```

---

## 5. 視覺化理解

### 5.1 張量網路圖示

```
初始狀態 (2×2 方塊):

    u1        u2
    |         |
    T1 --r1-- T2
    |         |
   d1        d2
    |         |
    T3 --r3-- T4
    |         |
   l3        l4
```

### 5.2 SVD 分解示意

```
步驟 1: 合併張量對
    T1 -- T2  →  組合矩陣 M_top
    T3 -- T4  →  組合矩陣 M_bottom

步驟 2: SVD 分解
    M_top = U_top @ S_top @ Vh_top

步驟 3: 截斷 + 平方根分配
    U_top' = U_top[:, :χ] @ sqrt(S_top[:χ])
    Vh_top' = sqrt(S_top[:χ]) @ Vh_top[:χ, :]

步驟 4: 新的粗粒化張量
         u
         |
    l -- T_new -- r
         |
         d
```

---

## 6. 實際應用案例

### 6.1 使用您的 truncated_svd.py

```python
from truncated_svd import truncated_svd, visualize_singular_values

def trg_with_truncated_svd(tensor, max_bond_dim=16):
    """
    使用 truncated_svd.py 中的函數實現 TRG
    """
    dim = tensor.shape[0]
    
    # 重組為矩陣
    permuted = np.transpose(tensor, (0, 3, 1, 2))
    matrix = permuted.reshape(dim * dim, dim * dim)
    
    # 視覺化奇異值（分析截斷效果）
    visualize_singular_values(matrix, max_rank=50)
    
    # 使用 truncated_svd
    U, S, V = truncated_svd(matrix, rank=max_bond_dim, 
                           return_singular_values=True)
    
    # 平方根分配
    sqrt_S = np.sqrt(S)
    U = U * sqrt_S
    V = V * sqrt_S
    
    # 重組並收縮
    keep = len(S)
    S1 = U.reshape(dim, dim, keep)
    S2 = V.T.reshape(keep, dim, dim)
    
    coarse = np.einsum('api,ibj,cjq,qda->abcd', 
                      S1, S2, S1, S2, optimize=True)
    
    return coarse
```

### 6.2 比較不同截斷策略

```python
def compare_truncation_strategies(tensor):
    """
    比較不同 χ 值的截斷效果
    """
    import matplotlib.pyplot as plt
    
    chi_values = [2, 4, 8, 16, 32]
    errors = []
    
    # 參考：不截斷的結果
    coarse_ref, _ = trg_svd_step_detailed(tensor, max_bond_dim=None)
    
    for chi in chi_values:
        coarse, _ = trg_svd_step_detailed(tensor, max_bond_dim=chi)
        error = np.linalg.norm(coarse - coarse_ref) / np.linalg.norm(coarse_ref)
        errors.append(error)
        print(f"χ = {chi:2d} | 相對誤差: {error:.6f}")
    
    # 繪圖
    plt.figure(figsize=(8, 5))
    plt.semilogy(chi_values, errors, 'bo-', markersize=8)
    plt.xlabel('Bond Dimension χ')
    plt.ylabel('Relative Error')
    plt.title('TRG Truncation Error vs Bond Dimension')
    plt.grid(True, alpha=0.3)
    plt.show()
```

---

## 總結

### SVD 在 TRG 中的核心價值

1. **降維**: 將高維張量收縮問題轉化為矩陣 SVD
2. **最優截斷**: 保證截斷誤差最小（Eckart-Young 定理）
3. **數值穩定**: 正交分解保證良好的條件數
4. **物理直觀**: 奇異值 = 關聯強度

### 關鍵參數選擇

| 參數 | 典型值 | 影響 |
|------|--------|------|
| `max_bond_dim` | 8-32 | 精度 vs 計算成本 |
| `rel_svd_cutoff` | 1e-8 ~ 1e-12 | 自動截斷閾值 |
| 迭代次數 | log₂(L) | 格子尺寸 L |

### 延伸閱讀

- Levin & Nave, "Tensor renormalization group approach to two-dimensional classical lattice models", PRL 99, 120601 (2007)
- Gu & Wen, "Tensor-entanglement-filtering renormalization approach", PRB 80, 155131 (2009)
- Evenbly & Vidal, "Tensor network renormalization", PRL 115, 180405 (2015)
