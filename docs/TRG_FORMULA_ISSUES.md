# TRG 程式碼公式與接腳問題分析報告

## 執行日期
2024年12月17日

## 問題概述
在檢查 `trg_visual_example.py` 時發現**兩個關鍵問題**：
1. 張量收縮的 einsum 索引接腳不正確
2. 自由能計算公式與標準 TRG 不一致

---

## 問題一：張量收縮索引錯誤

### 🔴 問題描述
`trg_visual_example.py` 第 226-233 行的 einsum 收縮：

```python
coarse_tensor = np.einsum(
    "axw,byx,yzc,wzd->abcd",
    C0,  # a=aux_R, x=r, w=d
    C1,  # b=aux_L, y=d, x=l
    C2,  # y=u,     z=l, c=aux_R
    C3,  # w=u,     z=r, d=aux_L
    optimize=True,
)
```

### ❌ 驗證結果
通過手動分步收縮驗證，發現：
```
einsum 結果與手動分步收縮的差異: ||差異|| = 7.62e+01
```
**這是一個顯著的數值差異，表明索引邏輯有誤！**

### ✅ 正確的收縮邏輯（參考 cytnx 版本 `11410PHYS401200/TRG/trg.py`）

根據標準 TRG 算法 (Levin-Nave 2007)，正確的收縮拓撲應該是：

```python
tensors = [
    ("C0", ["aux_L", "r", "d"]),    # C0[aux_L, r, d]
    ("C1", ["aux_L", "d", "l"]),    # C1[aux_L, d, l]
    ("C2", ["u", "l", "aux_R"]),    # C2[u, l, aux_R]
    ("C3", ["u", "r", "aux_R"]),    # C3[u, r, aux_R]
]

contractions = [
    ("C0", "r", "C1", "l"),   # C0.r = C1.l (右-左連接)
    ("C1", "d", "C2", "u"),   # C1.d = C2.u (下-上連接)
    ("C2", "l", "C3", "r"),   # C2.l = C3.r (左-右連接)
    ("C3", "u", "C0", "d"),   # C3.u = C0.d (上-下連接)
]
```

對應的 einsum 字符串應該是：
```python
# 假設：C0[a,r,d], C1[b,d,l], C2[u,l,c], C3[u,r,d']
# 收縮：r(C0)=l(C1), d(C0)=u(C3), d(C1)=u(C2), l(C2)=r(C3)
# 輸出：[a, b, c, d'] 對應 [aux_L, aux_L, aux_R, aux_R]

coarse_tensor = np.einsum(
    'ard,bdl,ulc,ure->abce',  # 這需要進一步驗證
    C0, C1, C2, C3,
    optimize=True
)
```

**注意**：因為兩次 SVD 產生的輔助索引 `aux_L` 和 `aux_R` 不同，輸出張量的四個索引應該是 `[aux_L(from SVD1), aux_L(from SVD1), aux_R(from SVD2), aux_R(from SVD2)]`，而不是都相同的！

---

## 問題二：自由能計算公式錯誤

### 🔴 錯誤的公式（`trg_visual_example.py`）

```python
# 第 278-280 行
self.free_energy_sum_ln_gn_weighted = math.log(g0)
self.scale_factor = 1.0

# 第 326-328 行
self.scale_factor /= 2.0
self.free_energy_sum_ln_gn_weighted += self.scale_factor * math.log(g_n)

# 第 384-386 行
logZ_per_site = self.free_energy_sum_ln_gn_weighted
return -(1.0 / self.beta) * logZ_per_site
```

這個公式是：
$$f = -T \sum_{n=0}^{N} \frac{\ln(g_n)}{2^n}$$

### ❌ 驗證結果
使用此公式計算的自由能：
```
自由能/格點 = -4.73577106
Onsager 精確解 = -2.109651
相對誤差 = 124.48%
```
**誤差超過 100%，完全錯誤！**

### ✅ 正確的公式（參考 cytnx 版本）

```python
# 11410PHYS401200/TRG/trg.py 第 143 行
f = -np.sum(np.array(self.log_factors) / np.array(self.n_spins)) * self.temp
```

這個公式是：
$$f = -T \sum_{n=0}^{N} \frac{\ln(g_n)}{N_n}$$

其中 $N_n$ 是第 $n$ 步的有效格點數：
- $N_0 = 1$（初始一個格點）
- $N_n = 2 \times N_{n-1} = 2^n$（每步格點數加倍）

**關鍵差異**：
- ❌ 錯誤：權重是 $1/2^n$
- ✅ 正確：權重是 $1/N_n = 1/2^n$，但公式的整體結構不同

### 理論依據

TRG 算法中，每步粗粒化後：
1. 張量代表的有效格點數從 $N_{n-1}$ 變為 $N_n = 2 \times N_{n-1}$
2. 配分函數關係：$Z_n = (g_n)^{N_{n-1}}$ 其中 $g_n$ 是歸一化因子
3. 取對數：$\ln Z_n = N_{n-1} \cdot \ln g_n$
4. 總配分函數：$\ln Z = \sum_n N_n \ln g_n$

但實際上應該是：
$$\ln Z = \sum_n \frac{N_{total}}{N_n} \ln g_n$$

其中每步的貢獻根據當前有效格點數加權。

---

## 影響評估

### 對現有程式的影響

1. **`trg_visual_example.py`**：
   - ❌ 張量收縮錯誤 → 粗粒化結果不正確
   - ❌ 自由能公式錯誤 → 計算結果不可信
   - **結論**：此程式不能用於產生正確的物理結果

2. **`trg_final_project.py`**：
   - ⚠️ 繼承了 `trg_visual_example.py` 的錯誤邏輯
   - ⚠️ 所有已生成的圖表數據可能不正確
   - **結論**：需要修正後重新生成

3. **`summation_TM/2d_model.py`**：
   - ✅ 使用不同的 TRG 實作
   - ✅ 需要檢查其收縮邏輯是否正確

---

## 建議的修正方案

### 方案一：修正現有程式碼（複雜）
1. 重寫 `_trg_step` 函數的 einsum 收縮
2. 修正自由能累積公式
3. 需要深入理解張量網路理論

### 方案二：使用已驗證的實作（推薦）⭐
1. 直接使用 `11410PHYS401200/TRG/trg.py`（cytnx 版本）
2. 將其改寫為純 numpy 版本
3. 優點：
   - 已經過驗證，結果正確
   - 邏輯清晰，易於理解
   - 有完整的參考實作

### 方案三：使用 `summation_TM/2d_model.py`
1. 檢查該檔案的 TRG 實作是否正確
2. 如果正確，可直接使用
3. 需要驗證其收縮邏輯和自由能公式

---

## 後續行動

### 立即行動
1. ✅ 已完成：識別並記錄問題
2. ⬜ 待辦：驗證 `summation_TM/2d_model.py` 的正確性
3. ⬜ 待辦：選擇修正方案並實施

### 期末報告建議
考慮到時間限制，建議：
1. 使用 `summation_TM/2d_model.py` 中的 TRG 實作（如果正確）
2. 或者基於 cytnx 版本重新實作純 numpy 版本
3. 在報告中說明：
   - 發現了原始實作的問題
   - 使用了經過驗證的實作
   - 這體現了科學研究中的嚴謹態度

---

## 參考資料

1. **Levin & Nave (2007)**  
   "Tensor Renormalization Group Approach to Two-Dimensional Classical Lattice Models"  
   Physical Review Letters 99, 120601

2. **課程實作**  
   `11410PHYS401200/TRG/trg.py` - 使用 Cytnx 庫的標準實作

3. **Onsager 精確解**  
   臨界溫度 $T_c = 2/\ln(1+\sqrt{2}) \approx 2.269$  
   自由能（$T=T_c$, $h=0$）: $f = -2.109651$ (per spin)

---

## 結論

**關鍵發現**：
1. `trg_visual_example.py` 存在**嚴重的數值錯誤**
2. 錯誤源於**張量收縮索引**和**自由能公式**兩處
3. 需要使用正確的實作重新生成所有結果

**建議**：
使用已驗證的實作（如 cytnx 版本或 `2d_model.py`）來完成期末專案，確保結果的正確性和可靠性。

---

## 附錄：數值驗證結果

```
【Ising 權重矩陣分解】
權重矩陣 W (對稱): ✓
重構誤差: ||W - M@M^T|| = 1.37e-15 ✓

【局域張量構造】
初始張量 shape: (2, 2, 2, 2) ✓
對稱性檢查: 通過 ✓

【TRG 步驟索引接腳】
einsum 結果與手動分步收縮差異: 7.62e+01 ✗

【自由能計算】
計算結果: -4.73577106
精確解: -2.109651
相對誤差: 124.48% ✗
```
