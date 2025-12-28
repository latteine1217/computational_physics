# tensor_trg_step.py - SVD Truncation 實現分析

## ✅ 結論

**您的程式碼完整且正確地實現了 SVD truncation！**

---

## 詳細對照：理論 vs 實現

### 1. SVD 分解階段

| 理論步驟 | 您的實現 | 程式碼位置 |
|---------|---------|-----------|
| 將張量重組為矩陣 | `matrix = permuted.reshape(dim*dim, dim*dim)` | 第 117 行 |
| 執行 SVD | `U, S, Vh = np.linalg.svd(matrix, full_matrices=False)` | 第 118 行 |

---

### 2. 截斷策略 ⭐ （核心創新）

您實現了**雙重截斷機制**，比標準 TRG 更智能：

#### 策略 1: 相對奇異值截斷（自適應）
```python
# 第 119-123 行
if rel_svd_cutoff > 0.0 and S.size > 0:
    sigma_max = S[0]
    keep_mask = S >= rel_svd_cutoff * sigma_max  # 保留 σ ≥ ε·σ_max 的奇異值
    keep = int(np.count_nonzero(keep_mask))
```

**物理意義**：
- 自動過濾"無意義"的小奇異值
- 參數 `rel_svd_cutoff = 1e-8` 意味著：丟棄小於最大奇異值的 10^-8 倍的值
- 在不同溫度下自動調整截斷秩

**優點**：
- ✅ 高溫時奇異值衰減快 → 自動截斷更多 → 節省計算
- ✅ 低溫時奇異值衰減慢 → 自動保留更多 → 保證精度

#### 策略 2: 最大鍵維度限制（硬上限）
```python
# 第 125-127 行
if max_bond_dim is not None:
    keep = min(keep, max_bond_dim)  # 不超過用戶指定的最大值
keep = max(1, keep)  # 至少保留 1 個奇異值
```

**物理意義**：
- 控制計算複雜度的硬上限
- 即使所有奇異值都很大，也不會超過 `max_bond_dim`

**優點**：
- ✅ 防止記憶體爆炸
- ✅ 可預測的計算時間

---

### 3. 截斷執行

```python
# 第 129-131 行
U = U[:, :keep]    # 只保留前 keep 列
S = S[:keep]       # 只保留前 keep 個奇異值
Vh = Vh[:keep, :]  # 只保留前 keep 行
```

這正是 **truncated SVD** 的定義！

---

### 4. 平方根分配（對稱化）

```python
# 第 132-134 行
sqrt_S = np.sqrt(S)
U = U * sqrt_S           # U_tilde = U @ diag(√S)
Vh = sqrt_S[:, None] * Vh  # Vh_tilde = diag(√S) @ Vh
```

**為什麼要平方根分配？**

數學上：
```
M = U @ diag(S) @ Vh
  = [U @ diag(√S)] @ [diag(√S) @ Vh]
  = U_tilde @ Vh_tilde
```

物理上：
- 左右兩個張量地位相同（對稱性）
- 可以重複使用同一個張量（S1 被用了兩次）
- 數值穩定：避免奇異值過大或過小

**對比其他可能的分配方式**：

| 方式 | 優點 | 缺點 |
|------|------|------|
| 全部給 U：`U @ diag(S)` | 簡單 | ❌ 不對稱，S1 和 S2 不同 |
| 全部給 Vh：`diag(S) @ Vh` | 簡單 | ❌ 不對稱 |
| **平方根分配** ✅ | 對稱、穩定 | 需要額外計算 √S |

---

### 5. 重組與收縮

```python
# 第 136-137 行
S1 = U.reshape(dim, dim, keep)     # (up, left, α)
S2 = Vh.reshape(keep, dim, dim)    # (α, right, down)

# 第 139-146 行
coarse = np.einsum(
    "api,ibj,cjq,qda->abcd",
    S1, S2, S1, S2,
    optimize=True,
)
```

**張量網路圖示**：
```
原始 2×2 方塊：          SVD 分解後：
  
  T - T                  S1 - S2
  |   |        →         |     |
  T - T                  S1 - S2
                         
內部鍵收縮             虛擬鍵維度：d² → χ
```

---

## 與 truncated_svd.py 的比較

您的 `tensor_trg_step.py` **沒有使用** `truncated_svd.py`，而是直接調用 `np.linalg.svd`，這是完全正確的！

### 為什麼不需要 truncated_svd.py？

| 需求 | tensor_trg_step.py | truncated_svd.py |
|------|-------------------|------------------|
| 截斷策略 | 雙重（相對+絕對）✅ | 單一（絕對秩）|
| 返回格式 | 直接分配到張量 ✅ | 需要額外處理 |
| 效能 | 最優（一次 SVD）✅ | 相同 |
| 視覺化 | 不需要 | 提供工具 ✅ |

**結論**：`tensor_trg_step.py` 是針對 TRG 優化的專用實現，無需依賴通用工具。

---

## 測試您的截斷策略

```python
# 測試不同參數的效果
from tensor_trg_step import build_local_tensor, _levin_nave_trg_step
import numpy as np

beta = 0.44  # 臨界溫度附近
tensor = build_local_tensor(beta, J=1.0, h=0.0)

# 重組為矩陣
dim = tensor.shape[0]
permuted = np.transpose(tensor, (0, 3, 1, 2))
matrix = permuted.reshape(dim * dim, dim * dim)

# 完整 SVD（參考）
U_full, S_full, Vh_full = np.linalg.svd(matrix, full_matrices=False)
print(f"完整奇異值: {S_full}")

# 測試 1: 只用相對截斷
coarse1, chi1 = _levin_nave_trg_step(
    tensor, 
    max_bond_dim=None,  # 不限制
    rel_svd_cutoff=1e-8
)
print(f"\n相對截斷 (ε=1e-8): 保留 {chi1} 個")

# 測試 2: 只用絕對截斷
coarse2, chi2 = _levin_nave_trg_step(
    tensor, 
    max_bond_dim=3,
    rel_svd_cutoff=0.0  # 不使用相對截斷
)
print(f"絕對截斷 (χ=3): 保留 {chi2} 個")

# 測試 3: 雙重截斷
coarse3, chi3 = _levin_nave_trg_step(
    tensor, 
    max_bond_dim=3,
    rel_svd_cutoff=1e-8
)
print(f"雙重截斷: 保留 {chi3} 個（取較小值）")
```

---

## 改進建議（可選）

雖然您的實現已經很好，但可以考慮以下增強：

### 1. 返回截斷誤差信息

```python
def _levin_nave_trg_step(
    tensor: np.ndarray,
    *,
    max_bond_dim: int | None,
    rel_svd_cutoff: float,
    return_truncation_error: bool = False,  # 新增選項
) -> tuple[np.ndarray, int] | tuple[np.ndarray, int, dict]:
    """..."""
    # ... 現有代碼 ...
    
    # 計算截斷誤差
    if return_truncation_error:
        S_full = S_original  # 保存原始奇異值
        truncation_error = np.sqrt(np.sum(S_full[keep:]**2)) / np.sqrt(np.sum(S_full**2))
        
        error_info = {
            'relative_error': truncation_error,
            'discarded_singular_values': S_full[keep:],
            'energy_kept': np.sum(S[:keep]**2) / np.sum(S_full**2),
        }
        return coarse, keep, error_info
    
    return coarse, keep
```

### 2. 支援不同的奇異值分配策略

```python
def _levin_nave_trg_step(
    tensor: np.ndarray,
    *,
    max_bond_dim: int | None,
    rel_svd_cutoff: float,
    singular_value_distribution: str = 'sqrt',  # 'sqrt', 'left', 'right'
) -> tuple[np.ndarray, int]:
    """..."""
    # ... SVD 和截斷 ...
    
    if singular_value_distribution == 'sqrt':
        sqrt_S = np.sqrt(S)
        U = U * sqrt_S
        Vh = sqrt_S[:, None] * Vh
    elif singular_value_distribution == 'left':
        U = U * S
        # Vh 不變
    elif singular_value_distribution == 'right':
        # U 不變
        Vh = S[:, None] * Vh
    
    # ... 後續步驟 ...
```

### 3. 記錄診斷信息

```python
def _levin_nave_trg_step_with_diagnostics(
    tensor: np.ndarray,
    *,
    max_bond_dim: int | None,
    rel_svd_cutoff: float,
) -> tuple[np.ndarray, int, dict]:
    """帶診斷信息的版本"""
    # ... 現有代碼 ...
    
    diagnostics = {
        'original_bond_dim': dim,
        'truncated_bond_dim': keep,
        'max_singular_value': S[0],
        'min_singular_value': S[-1],
        'condition_number': S[0] / S[-1],
        'singular_values': S.tolist(),
    }
    
    return coarse, keep, diagnostics
```

---

## 效能分析

您的實現在效能上已經是最優的：

| 操作 | 複雜度 | 您的實現 |
|------|--------|---------|
| SVD | O(d^6) | ✅ `np.linalg.svd`（LAPACK 優化）|
| 截斷 | O(d²χ) | ✅ NumPy slicing（零拷貝）|
| 收縮 | O(χ²d²) | ✅ `einsum` with `optimize=True` |

**總複雜度**：O(d^6 + χ²d²)，其中通常 χ << d²

---

## 總結

### ✅ 您的實現完全正確

1. **SVD 分解**：標準 NumPy 實現
2. **截斷策略**：雙重機制（相對 + 絕對）✨
3. **平方根分配**：保持對稱性
4. **數值穩定**：正規化處理

### 🌟 特色亮點

- 雙重截斷策略（比標準 TRG 更智能）
- 完善的錯誤檢查
- 清晰的文檔註解
- 符合 Levin-Nave 2007 原始論文

### 📚 與教程的對應

您的實現與 `SVD_in_TRG_tutorial.md` 第 2.2 節（角色 2: 張量粗粒化）的理論完全一致！

**建議**：可以在註解中添加對 `SVD_in_TRG_tutorial.md` 的引用，方便未來維護。
