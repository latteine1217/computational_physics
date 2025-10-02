# 📊 1D Ising Model 完整程式解析文檔

## 🎯 程式總體架構

`1d_model.py` 是一個高效實現一維 Ising 模型的完整計算框架，提供三種不同的計算方法：
1. **窮舉法 (Enumeration)** - 使用 Gray Code 優化遍歷所有可能組態
2. **轉移矩陣法 (Transfer Matrix)** - 解析求解配分函數
3. **理論解析解 (Theory)** - 零外場條件下的精確解

---

## 🔧 核心數據結構與轉換

### 位元-自旋映射系統

```python
def bit_to_spin(bits: int, idx: int) -> int:
    """bit -> spin in {-1,+1}；idx 從 0..L-1"""
    return 1 if ((bits >> idx) & 1) else -1
```

**🎯 設計理念**：
- **記憶體效率**：用單一整數表示整個自旋組態，而非自旋陣列
- **位元操作優勢**：直接利用 CPU 位元運算，比陣列存取更快
- **編碼規則**：
  - 位元 `0` → 自旋 `-1`
  - 位元 `1` → 自旋 `+1`

**🌟 實例解析**：
```
L = 4, bits = 9 (二進位: 1001)
→ 自旋組態: [+1, -1, -1, +1]
   位置:      0   1   2   3
```

---

## ⚡ 能量計算核心

### 完整能量計算 (`energy_ising1d_from_bits`)

**📐 物理模型**：
```
H = -J Σᵢ sᵢsᵢ₊₁ - h Σᵢ sᵢ
```
其中：
- `J` = 交換耦合常數
- `h` = 外磁場強度
- `sᵢ` = 第 i 個自旋 (±1)

**🔄 計算流程**：

#### 1️⃣ 磁場項計算
```python
M = 0
for i in range(L):
    si = bit_to_spin(bits, i)
    M += si
E += -h * M
```
- **物理意義**：每個自旋在外磁場中的能量貢獻
- **優化技巧**：先計算總磁化 M，再乘以 -h

#### 2️⃣ 交換項計算 (最近鄰耦合)
```python
for i in range(L - 1):
    si = bit_to_spin(bits, i)
    sj = bit_to_spin(bits, i + 1)
    E += -J * si * sj
```
- **物理意義**：相鄰自旋間的交換作用能
- **耦合規則**：J > 0 鐵磁性；J < 0 反鐵磁性

#### 3️⃣ 週期邊界條件
```python
if periodic and L >= 2:
    s0 = bit_to_spin(bits, 0)
    sl = bit_to_spin(bits, last)
    E += -J * s0 * sl
```
- **物理意義**：消除邊界效應，模擬無限長鏈
- **實現方式**：首尾自旋耦合

---

## 🚀 增量能量計算 (`deltaE_ising1d_flip`)

### 核心優化原理

**🎯 問題**：重新計算完整能量需要 O(L) 時間
**✨ 解決方案**：只計算翻轉造成的能量變化 O(1)

**📊 數學推導**：

當翻轉位置 `i` 的自旋時：
```
翻轉前：sᵢ 對能量貢獻 = -J×sᵢ×(左鄰+右鄰) - h×sᵢ
翻轉後：-sᵢ 對能量貢獻 = -J×(-sᵢ)×(左鄰+右鄰) - h×(-sᵢ)
────────────────────────────────────────────────────
ΔE = 2×sᵢ×[J×(左鄰+右鄰) + h]
```

**🔧 實現細節**：

```python
def deltaE_ising1d_flip(bits: int, L: int, i: int, J: float, h: float = 0.0, periodic: bool = True) -> float:
    si = bit_to_spin(bits, i)  # 當前自旋值
    nn_sum = 0  # 鄰居自旋總和
    
    # 🔍 左鄰居處理
    if i - 1 >= 0:
        nn_sum += bit_to_spin(bits, i - 1)
    elif periodic and L >= 2:
        nn_sum += bit_to_spin(bits, L - 1)  # 週期邊界
    
    # 🔍 右鄰居處理
    if i + 1 < L:
        nn_sum += bit_to_spin(bits, i + 1)
    elif periodic and L >= 2:
        nn_sum += bit_to_spin(bits, 0)      # 週期邊界
    
    return 2.0 * si * (J * nn_sum + h)
```

**⚡ 性能優勢**：
- 時間複雜度：O(L) → O(1)
- 總遍歷效率：O(L×2^L) → O(2^L)

---

## 🎨 Gray Code 遍歷策略

### 智能組態遍歷

**🎯 核心思想**：確保相鄰組態只差一個自旋，最大化增量計算效益

```python
def gray_next_flip_pos(t: int) -> int:
    """
    已在步 t，下一步 (t+1) 的 Gray code 與當前差一個 bit。
    回傳要翻轉的 bit 位置。
    """
    g  = t ^ (t >> 1)        # 當前步的 Gray code
    g1 = (t + 1) ^ ((t + 1) >> 1)  # 下一步的 Gray code
    diff = g ^ g1            # 找出差異位元
    
    # 🔍 找最右側 1 的位置
    pos = 0
    while ((diff >> pos) & 1) == 0:
        pos += 1
    return pos
```

**📊 Gray Code 序列示例**：
```
步驟 | 二進位 | Gray Code | 翻轉位置
-----|--------|-----------|----------
  0  |  000   |    000    |    -
  1  |  001   |    001    |    0
  2  |  010   |    011    |    1
  3  |  011   |    010    |    0
  4  |  100   |    110    |    2
  5  |  101   |    111    |    0
  6  |  110   |    101    |    1
  7  |  111   |    100    |    0
```

**🌟 優勢分析**：
- **連續性**：每步只翻轉一個自旋
- **完整性**：遍歷所有 2^L 種組態
- **效率性**：完美搭配增量能量計算

---

## 🔄 完整遍歷引擎 (`ising1d_all_energies_gray`)

### 主要計算流程

```python
def ising1d_all_energies_gray(L: int, J: float, h: float = 0.0, periodic: bool = True):
    total = 1 << L  # 2^L 總組態數
    energies = np.empty(total, dtype=np.float64)  # 能量陣列
    mags     = np.empty(total, dtype=np.int32)    # 磁化陣列

    # 🎯 初始化：全 -1 組態
    bits = 0                # 二進位 000...0
    E = energy_ising1d_from_bits(bits, L, J, h, periodic)
    M = -L                  # 全 -1 的總磁化

    for t in range(total):
        # 📊 記錄當前組態
        energies[t] = E
        mags[t]     = M

        if t == total - 1:
            break

        # 🔄 Gray Code 下一步
        i = gray_next_flip_pos(t)           # 確定翻轉位置
        dE = deltaE_ising1d_flip(bits, L, i, J, h, periodic)  # 計算 ΔE
        
        # ⚡ 執行翻轉更新
        bits ^= (1 << i)                   # 翻轉位元 i
        E += dE                             # 更新能量
        si_new = bit_to_spin(bits, i)       # 翻轉後的自旋值
        M += 2 * si_new                     # 磁化變化: Δ M = 2×s_new
```

**🎯 重要細節**：

#### 磁化更新邏輯
```
翻轉前：sᵢ = -si_new
翻轉後：sᵢ = si_new
磁化變化：ΔM = si_new - (-si_new) = 2×si_new
```

#### 退化度統計
```python
uniqE, counts = np.unique(energies, return_counts=True)
degeneracy = {float(e): int(c) for e, c in zip(uniqE, counts)}
```
- **物理意義**：統計相同能量的組態數量
- **重要性**：計算熵與配分函數的關鍵資訊

---

## 📈 統計力學計算引擎

### 配分函數與熱力學量 (`partition_stats`)

**🎯 核心任務**：從所有組態的能量分布計算宏觀熱力學量

```python
def partition_stats(energies: np.ndarray, mags: np.ndarray, beta: float):
    # 🔥 Log-Sum-Exp 技巧避免數值溢位
    a = -beta * energies        # 玻爾茲曼因子的指數
    amax = np.max(a)            # 找最大值避免 overflow
    wa = np.exp(a - amax)       # 標準化權重
    norm = wa.sum()             # 標準化常數
    Z = norm * np.exp(amax)     # 配分函數
    
    # 📊 熱力學期望值計算
    Ew = (energies * wa).sum() / norm     # ⟨E⟩
    Mw = (mags * wa).sum() / norm         # ⟨M⟩
    M2w = ((mags**2) * wa).sum() / norm   # ⟨M²⟩
    E2w = ((energies**2) * wa).sum() / norm # ⟨E²⟩
    
    # 🌡️ 熱容量
    Cv = beta**2 * (E2w - Ew**2)         # C_v = β²(⟨E²⟩ - ⟨E⟩²)
    
    return Z, Ew, Mw, M2w, Cv
```

**🔬 數值穩定性技巧**：

#### Log-Sum-Exp 方法
```
原始計算：Z = Σᵢ exp(-βEᵢ)  ← 可能 overflow
安全計算：Z = exp(E_max) × Σᵢ exp(-β(Eᵢ - E_max))
```

**📊 物理量對應**：
- `Z` = 配分函數
- `Ew` = 平均能量 ⟨E⟩
- `Mw` = 平均磁化 ⟨M⟩
- `Cv` = 熱容量 C_v

---

## 🌡️ 熱容量計算專題

### 熱容量的物理意義與計算方法

**🎯 物理背景**：
熱容量 (Heat Capacity) 描述系統溫度變化時吸收或釋放熱量的能力，是研究相變和臨界行為的重要物理量。

#### 定義與關係式
```
C_v = ∂⟨E⟩/∂T |_V = k_B β² (⟨E²⟩ - ⟨E⟩²)
```
其中：
- `C_v` = 定容熱容量
- `⟨E⟩` = 平均能量
- `⟨E²⟩` = 能量平方的期望值
- `β = 1/(k_B T)` = 逆溫度

**🔧 替代計算方法**：
```
C_v = -T ∂²F/∂T² = -k_B β² ∂²(βF)/∂β²
```

### 數值微分實現 (`calculate_cv_from_free_energy`)

**📐 理論基礎**：
使用自由能的二階溫度導數計算熱容量：

```python
def calculate_cv_from_free_energy(L, T_values, J=1.0, h=0.0, periodic=True, method="auto"):
    """
    使用自由能二階微分計算熱容量：Cv = -T ∂²F/∂T²
    支援方法：{"auto", "theory", "enum", "enumeration", "transfer_matrix"}
    """
    # 🔍 數值微分參數選擇
    for i, temp in enumerate(T_values):
        delta_T = delta_T_factor * temp  # 自適應步長
        
        # 📊 三點數值微分計算 ∂²F/∂T²
        if method == "transfer_matrix":
            # 🔧 特殊處理：使用轉移矩陣觀測量函數
            F_plus = transfer_matrix_observables(L, T_plus, J, h, periodic).free_energy_per_spin * L
            F_center = transfer_matrix_observables(L, temp, J, h, periodic).free_energy_per_spin * L
            F_minus = transfer_matrix_observables(L, T_minus, J, h, periodic).free_energy_per_spin * L
        else:
            # 🚀 使用統一的自由能函數 (auto, theory, enum)
            F_plus = free_energy_1d(L, [T_plus], J, h, periodic, method)[0]
            F_center = free_energy_1d(L, [temp], J, h, periodic, method)[0]
            F_minus = free_energy_1d(L, [T_minus], J, h, periodic, method)[0]
        
        # 🧮 二階導數數值計算
        d2F_dT2 = (F_plus - 2 * F_center + F_minus) / (delta_T ** 2)
        
        # 🌡️ 單自旋熱容量計算
        cv_per_spin = -temp * d2F_dT2 / L
```

**🔬 數值微分精度控制**：

#### 自適應步長策略
```python
delta_T = delta_T_factor * temp  # 通常 delta_T_factor = 1e-4
```
- **比例控制**：相對誤差控制，避免低溫時步長過大
- **穩定性**：確保微分近似的有效性

#### 三點差分公式
```
∂²F/∂T² ≈ [F(T+ΔT) - 2F(T) + F(T-ΔT)] / (ΔT)²
```
**優勢**：相比兩點差分具有更高精度 O(ΔT²)

### 多方法熱容量計算 (`calculate_cv_multiple_methods`)

**🎯 功能特色**：同時使用統計漲落公式與數值微分方法計算熱容量

```python
def calculate_cv_multiple_methods(L, T_values, J=1.0, h=0.0, periodic=True, 
                                  methods=["enumeration", "transfer_matrix"]):
    """
    多方法熱容量計算與比較
    返回格式：{method: cv_array}
    """
    results = {}
    
    for method in methods:
        # 🔥 方法一：統計漲落公式 C_v = β²(⟨E²⟩ - ⟨E⟩²)
        cv_fluctuation = []
        for T in T_values:
            if method == "enumeration":
                result = enumeration_observables(L, T, J, h, periodic)
                cv_fluctuation.append(result.heat_capacity_per_spin)
            elif method == "transfer_matrix":
                result = transfer_matrix_observables(L, T, J, h, periodic)
                cv_fluctuation.append(result.heat_capacity_per_spin)
        
        results[method] = np.array(cv_fluctuation)
    
    return results
```

**🔍 方法比較意義**：
- **一致性驗證**：不同方法應給出相同結果
- **數值穩定性**：檢驗微分步長選擇的合理性  
- **理論驗證**：確保熱力學關係式的正確實現

### 熱容量可視化 (`plot_heat_capacity_vs_T_for_Ls`)

**📊 多尺寸熱容量繪圖**：

```python
def plot_heat_capacity_vs_T_for_Ls(L_list, J=1.0, h=0.0, periodic=True,
                                   T_min=0.1, T_max=5.0, nT=200,
                                   methods=["enumeration", "transfer_matrix", "theory"]):
    """
    繪製多系統尺寸的熱容量-溫度關係圖
    支援方法自動過濾：h≠0時排除theory方法
    """
    # 🎨 視覺化設計
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"] 
    linestyles = {
        "enumeration": "-",
        "transfer_matrix": "--", 
        "theory": ":",
    }
    
    for L in L_list:
        for method in methods:
            # 📈 計算熱容量 (自動調用相應方法)
            cv_values = calculate_cv_from_free_energy(L, T_values, J, h, periodic, method)
            
            # 🎯 繪製：熱容量曲線
            label = f'{method} L={L}'
            plt.plot(T_values, cv_values, label=label, linestyle=linestyles[method])
```

**🔬 物理現象觀察**：
- **低溫行為**：Cv → 0 (量子統計效應)
- **高溫行為**：Cv → 常數 (經典極限)
- **尺寸效應**：有限尺寸如何影響熱容量峰值
- **相變信號**：Cv 峰值可能指示相變溫度

### 自由能與熱容量對比圖 (`plot_comparison_F_and_Cv`)

**📊 雙物理量關聯分析**：

```python
def plot_comparison_F_and_Cv(L_list, J=1.0, h=0.0, periodic=True,
                             T_min=0.1, T_max=5.0, nT=200, methods=["enumeration", "transfer_matrix", "theory"]):
    """
    自由能與熱容量的溫度依賴性四圖對比
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    for L in L_list:
        # 📈 計算物理量
        F_values = np.array([free_energy_1d(L, [T], J, h, periodic, "auto")[0] for T in T_values])
        Cv_values = calculate_cv_from_free_energy(L, T_values, J, h, periodic, "auto")
        
        # 🎯 四個子圖分析
        ax1.plot(T_values, F_values/L, label=f'L={L}')         # F vs T
        ax2.plot(T_values, Cv_values, label=f'L={L}')          # Cv vs T
        ax3.plot(F_values/L, Cv_values, label=f'L={L}')        # Cv vs F
        ax4.semilogy(T_values, np.abs(Cv_values), label=f'L={L}') # log(|Cv|)
    
    # 🏷️ 軸標籤與標題
    ax1.set_title('Free Energy per Spin vs Temperature')
    ax2.set_title('Heat Capacity per Spin vs Temperature') 
    ax3.set_title('Heat Capacity vs Free Energy')
    ax4.set_title('Log Scale: |Heat Capacity| vs Temperature')
```

**🔍 物理洞察**：
- **圖1 (F vs T)**：自由能的溫度依賴性
- **圖2 (Cv vs T)**：熱容量的溫度依賴性
- **圖3 (Cv vs F)**：熱容量與自由能的關聯
- **圖4 (log|Cv|)**：低溫區域熱容量的指數行為

### 熱容量的數值挑戰與解決方案

**⚠️ 常見問題**：

#### 1️⃣ 方法不一致性問題
```python
# 🔧 問題：不同計算方法需要不同的函數接口
# ✅ 解決：在 calculate_cv_from_free_energy 中統一處理

if method == "transfer_matrix":
    # 使用轉移矩陣專用函數
    F_center = transfer_matrix_observables(L, temp, J, h, periodic).free_energy_per_spin * L
elif method in ["auto", "theory", "enum", "enumeration"]:
    # 使用統一的自由能函數
    method_name = "enum" if method == "enumeration" else method
    F_center = free_energy_1d(L, [temp], J, h, periodic, method_name)[0]
```

#### 2️⃣ 低溫數值不穩定
```python
# 🔧 問題：T → 0 時，∂²F/∂T² 數值微分不穩定
# ✅ 解決：自適應步長 + 合理的溫度下限

T_minus = max(temp - delta_T, 0.01)  # 避免負溫度或過小溫度
```

#### 3️⃣ 高溫數值精度損失
```python
# 🔧 問題：高溫時自由能變化小，微分誤差放大
# ✅ 解決：優先使用統計漲落公式

if method in ["enumeration", "transfer_matrix"]:
    # 直接從觀測量函數獲取 Cv = β²(⟨E²⟩ - ⟨E⟩²)
    cv_direct = method_observables(L, T, J, h, periodic).heat_capacity_per_spin
```

**🎯 最佳實踐建議**：

1. **方法選擇**：
   - 小系統 (L ≤ 15)：優先使用窮舉法的統計漲落公式
   - 大系統 (L > 15)：使用轉移矩陣法 + 數值微分
   - 零外場 (h = 0)：理論解提供最高精度

2. **參數調節**：
   - 微分步長：`delta_T_factor = 1e-4` 通常是良好選擇
   - 溫度範圍：避免極低溫 (T < 0.05) 的數值問題

3. **結果驗證**：
   - 對比不同方法的計算結果
   - 檢查熱容量的物理合理性 (非負性、有界性)
   - 驗證高溫極限的正確性

---

## 🏭 計算方法整合

### 窮舉法接口 (`enumeration_observables`)

```python
def enumeration_observables(L: int, T: float, J: float, h: float = 0.0,
                            periodic: bool = True) -> MethodResult:
    start = time.perf_counter()
    
    # 🔄 執行完整遍歷
    energies, mags, _ = ising1d_all_energies_gray(L, J, h, periodic)
    
    # 📊 統計力學計算
    beta = 1.0 / T
    Z, E_mean, M_mean, M2_mean, Cv = partition_stats(energies, mags, beta)
    
    runtime = time.perf_counter() - start
    
    # 🎯 關鍵物理量
    free_energy_per_spin = -(1.0 / beta) * np.log(Z) / L
    susceptibility_per_spin = beta * (M2_mean - M_mean**2) / L
    heat_capacity_per_spin = Cv / L
```

**📋 輸出封裝**：使用 `MethodResult` 數據類統一格式，包含：
- 方法名稱
- 單自旋自由能
- 單自旋磁化率  
- 單自旋熱容量
- 計算時間
- 詳細元數據

---

## 🔬 轉移矩陣方法

### 解析方法核心 (`_transfer_matrix_stats_1d`)

**🎯 理論基礎**：
- 配分函數：Z = Tr(T^L)
- 轉移矩陣：T[i,j] = exp(β(Js_is_j + 0.5h(s_i + s_j)))

```python
def _transfer_matrix_stats_1d(L: int, beta: float, J: float, h: float, periodic: bool):
    spins = np.array([-1.0, 1.0], dtype=np.float64)
    T = np.empty((2, 2), dtype=np.float64)
    dT = np.empty((2, 2), dtype=np.float64)    # 對 h 的一階導數
    d2T = np.empty((2, 2), dtype=np.float64)   # 對 h 的二階導數
    
    # 🔧 構建轉移矩陣及其導數
    for i, si in enumerate(spins):
        for j, sj in enumerate(spins):
            exponent = beta * (J * si * sj + 0.5 * h * (si + sj))
            weight = math.exp(exponent)
            T[i, j] = weight
            
            pref = 0.5 * beta * (si + sj)
            dT[i, j] = pref * weight
            d2T[i, j] = (pref ** 2) * weight
```

**⚡ 高效矩陣冪運算**：
```python
# 🚀 累積計算 T^k，避免重複矩陣乘法
prefix = [np.eye(2, dtype=np.float64)]
for _ in range(L):
    prefix.append(prefix[-1] @ T)

Z = np.trace(prefix[L])  # Tr(T^L)
```

**📊 磁化率計算**：使用鏈式法則計算對外場的導數
```
χ = ∂²ln Z/∂h² = (∂²Z/∂h²)/Z - (∂Z/∂h)²/Z²
```

---

## 🎨 理論解析解

### 零外場精確解 (`free_energy_1d`)

**📐 數學背景**：h=0 時的 1D Ising 模型有精確解

```python
if method == "theory":
    if h != 0.0:
        raise ValueError("theory 模式僅支援 h=0 的一維 Ising。")
    
    # 🎯 特徵值計算
    lam_plus  = np.exp(beta * J) + np.exp(-beta * J)  # = 2cosh(βJ)
    lam_minus = np.exp(beta * J) - np.exp(-beta * J)  # = 2sinh(βJ)
    
    # 📊 配分函數
    Z = lam_plus**L + lam_minus**L  # 週期邊界
    # Z = 2.0 * (lam_plus**(L-1))  # 開放邊界
    
    F = -T * np.log(Z)
```

**🔬 物理解釋**：
- `lam_plus`：主要貢獻項，決定基態性質
- `lam_minus`：激發態貢獻，高溫時重要
- 週期邊界條件使配分函數包含兩個特徵值

---

## 📊 性能分析與可視化

### 多方法比較框架 (`plot_free_energy_vs_T_for_Ls`)

```python
def plot_free_energy_vs_T_for_Ls(L_list, J=1.0, h=0.0, periodic=True,
                                  T_min=0.05, T_max=5.0, nT=200,
                                  per_spin=True,
                                  methods=("enumeration", "transfer_matrix")):
```

**🎯 功能特色**：

#### 1️⃣ 自動方法過濾
```python
filtered_methods = []
for method in methods:
    if method == "theory" and not math.isclose(h, 0.0):
        continue  # h≠0 時自動排除理論解
    filtered_methods.append(method)
```

#### 2️⃣ 計算時間監控
```python
runtime_data = {m: [] for m in methods}
for L in L_list:
    for method in methods:
        start = time.perf_counter()
        F_total = method_funcs[method](L, T, J, h, periodic)
        elapsed = max(time.perf_counter() - start, 1e-12)
        runtime_data[method].append(elapsed)
```

#### 3️⃣ 雙圖表輸出
- **自由能曲線**：F(T) 或 F/N(T)
- **性能曲線**：計算時間 vs 系統尺寸

**🎨 視覺化設計**：
- 顏色：區分不同系統尺寸 L
- 線型：區分不同計算方法
- 對數軸：展示計算時間的指數增長

---

## 🔧 實用工具函數

### 數值穩定性工具

```python
def logsumexp_rows(x: np.ndarray) -> np.ndarray:
    """逐列套用 log-sum-exp，減去列最大值避免浮點溢位。"""
    m = np.max(x, axis=1, keepdims=True)
    return (m.squeeze() + np.log(np.sum(np.exp(x - m), axis=1)))
```

**🎯 應用場景**：處理多溫度同時計算的配分函數

### 統一接口設計

```python
@dataclass(frozen=True)
class MethodResult:
    """封裝單一計算方法的觀測量摘要"""
    method: str
    free_energy_per_spin: float
    susceptibility_per_spin: float
    heat_capacity_per_spin: float
    runtime: float
    metadata: Dict[str, float] = field(default_factory=dict)
```

**🌟 設計優勢**：
- **不可變性**：`frozen=True` 防止意外修改
- **類型安全**：明確指定所有字段類型
- **可擴展性**：`metadata` 字典允許額外資訊
- **統一格式**：所有方法返回相同數據結構

---

## ⚡ 性能優化總結

### 計算複雜度分析

| 方法 | 時間複雜度 | 空間複雜度 | 適用範圍 |
|------|------------|------------|----------|
| **窮舉法** | O(2^L) | O(2^L) | L ≤ 20 |
| **轉移矩陣** | O(L) | O(1) | 任意 L |
| **理論解** | O(1) | O(1) | h=0 僅限 |

### 關鍵優化技術

1. **位元表示**：自旋組態壓縮儲存
2. **Gray Code**：最小化翻轉次數
3. **增量計算**：O(1) 能量更新
4. **Log-Sum-Exp**：數值穩定配分函數
5. **矩陣冪累積**：避免重複計算
6. **記憶體預分配**：減少動態分配開銷

### 數值穩定性保證

- **浮點溢位防護**：Log-Sum-Exp 技巧
- **精度控制**：統一使用 `float64`
- **邊界條件檢查**：防止無效參數
- **異常處理**：配分函數非正檢測

---

## 🧮 Algorithm 實現邏輯

### Algorithm 1: Gray Code 窮舉法 (Enumeration)
```
Algorithm: 1D_Ising_Enumeration_Gray_Code
Input: L (系統尺寸), J (耦合常數), h (外磁場), T (溫度), periodic (邊界條件)
Output: 自由能, 磁化率, 熱容量

1. INITIALIZATION:
   total ← 2^L                          // 總組態數
   bits ← 0                             // 初始組態 (全 -1)
   E ← compute_full_energy(bits, L, J, h, periodic)  // 初始能量
   M ← -L                               // 初始磁化
   energies[0] ← E                      // 儲存首個能量
   mags[0] ← M                          // 儲存首個磁化

2. GRAY_CODE_TRAVERSAL:
   for t = 0 to total-2 do:
       // 🔍 確定下一步翻轉位置
       flip_pos ← gray_next_flip_position(t)
       
       // ⚡ 計算增量能量 (O(1) 操作)
       dE ← compute_delta_energy(bits, L, flip_pos, J, h, periodic)
       
       // 🔄 執行翻轉更新
       bits ← bits XOR (1 << flip_pos)   // 翻轉指定位元
       E ← E + dE                        // 更新能量
       si_new ← bit_to_spin(bits, flip_pos) // 翻轉後自旋值
       M ← M + 2 * si_new                // 更新磁化
       
       // 📊 記錄結果
       energies[t+1] ← E
       mags[t+1] ← M
   end for

3. STATISTICAL_MECHANICS:
   beta ← 1/T
   // 🔥 使用 Log-Sum-Exp 計算配分函數
   a ← -beta * energies
   a_max ← max(a)
   weights ← exp(a - a_max)
   Z ← sum(weights) * exp(a_max)
   
   // 📈 計算熱力學量
   <E> ← sum(energies * weights) / sum(weights)
   <M> ← sum(mags * weights) / sum(weights)
   <M²> ← sum(mags² * weights) / sum(weights)
   <E²> ← sum(energies² * weights) / sum(weights)
   
   // 🎯 最終物理量
   F_per_spin ← -ln(Z) / (beta * L)
   χ_per_spin ← beta * (<M²> - <M>²) / L
   C_v_per_spin ← beta² * (<E²> - <E>²) / L

4. RETURN: {F_per_spin, χ_per_spin, C_v_per_spin, runtime}

// 🔧 輔助函數
function gray_next_flip_position(t):
    g ← t XOR (t >> 1)                  // 當前 Gray code
    g_next ← (t+1) XOR ((t+1) >> 1)     // 下一個 Gray code
    diff ← g XOR g_next                 // 找差異位元
    return position_of_rightmost_set_bit(diff)

function compute_delta_energy(bits, L, pos, J, h, periodic):
    si ← bit_to_spin(bits, pos)         // 當前自旋值
    neighbor_sum ← 0
    
    // 左鄰居
    if pos > 0:
        neighbor_sum += bit_to_spin(bits, pos-1)
    else if periodic:
        neighbor_sum += bit_to_spin(bits, L-1)
    
    // 右鄰居  
    if pos < L-1:
        neighbor_sum += bit_to_spin(bits, pos+1)
    else if periodic:
        neighbor_sum += bit_to_spin(bits, 0)
    
    return 2 * si * (J * neighbor_sum + h)
```

**⏱️ 複雜度分析**：
- **時間複雜度**: O(2^L) - 遍歷所有組態，每步 O(1)
- **空間複雜度**: O(2^L) - 儲存所有能量與磁化
- **適用範圍**: L ≤ 20 (受記憶體限制)

---

### Algorithm 2: 轉移矩陣法 (Transfer Matrix)
```
Algorithm: 1D_Ising_Transfer_Matrix
Input: L (系統尺寸), J (耦合常數), h (外磁場), T (溫度)
Output: 自由能, 磁化率, 熱容量

1. MATRIX_CONSTRUCTION:
   beta ← 1/T
   spins ← [-1, +1]
   T_matrix ← 2×2 matrix
   dT_matrix ← 2×2 matrix                // 對 h 的一階導數
   d2T_matrix ← 2×2 matrix               // 對 h 的二階導數
   
   for i = 0 to 1 do:
       for j = 0 to 1 do:
           si ← spins[i]
           sj ← spins[j]
           exponent ← beta * (J*si*sj + 0.5*h*(si + sj))
           weight ← exp(exponent)
           T_matrix[i,j] ← weight
           
           pref ← 0.5 * beta * (si + sj)
           dT_matrix[i,j] ← pref * weight
           d2T_matrix[i,j] ← pref² * weight
       end for
   end for

2. MATRIX_POWER_COMPUTATION:
   // 🚀 累積計算避免重複矩陣乘法
   powers[0] ← Identity_2×2
   for k = 1 to L do:
       powers[k] ← powers[k-1] @ T_matrix  // 計算 T^k
   end for

3. PARTITION_FUNCTION_AND_DERIVATIVES:
   Z ← trace(powers[L])                   // Tr(T^L)
   
   // 📊 一階導數: ∂Z/∂h
   T_power_L_minus_1 ← powers[L-1]
   dZ ← L * trace(dT_matrix @ T_power_L_minus_1)
   
   // 📈 二階導數: ∂²Z/∂h²
   sum_term ← 0
   if L ≥ 2:
       for k = 0 to L-2 do:
           sum_term += trace((dT_matrix @ powers[k]) @ dT_matrix @ powers[L-2-k])
       end for
   d2Z ← L * trace(d2T_matrix @ T_power_L_minus_1) + L * sum_term

4. THERMODYNAMIC_QUANTITIES:
   ln_Z ← ln(Z)
   d_ln_Z ← dZ / Z                       // ∂ln Z/∂h
   d2_ln_Z ← d2Z/Z - (dZ/Z)²            // ∂²ln Z/∂h²
   
   // 🎯 物理量計算
   F_per_spin ← -ln_Z / (beta * L)
   M_per_spin ← d_ln_Z / (beta * L)
   χ_per_spin ← d2_ln_Z / (beta * L)
   
   // 🌡️ 熱容量 (數值微分)
   delta_beta ← max(1e-5, 1e-3 * beta)
   ln_Z_plus ← ln(trace(compute_T_power(beta + delta_beta, L)))
   ln_Z_minus ← ln(trace(compute_T_power(beta - delta_beta, L)))
   d2_ln_Z_dbeta2 ← (ln_Z_plus - 2*ln_Z + ln_Z_minus) / delta_beta²
   C_v_per_spin ← beta² * d2_ln_Z_dbeta2 / L

5. RETURN: {F_per_spin, χ_per_spin, C_v_per_spin, runtime}
```

**⏱️ 複雜度分析**：
- **時間複雜度**: O(L) - 主要為矩陣乘法次數
- **空間複雜度**: O(L) - 儲存矩陣冪次
- **適用範圍**: 任意 L (僅限週期邊界條件)

---

### Algorithm 3: 理論解析解 (Theory - h=0 only)
```
Algorithm: 1D_Ising_Analytical_Solution
Input: L (系統尺寸), J (耦合常數), T (溫度), periodic (邊界條件)
Output: 自由能
Constraint: h = 0 (零外磁場)

1. PARAMETER_VALIDATION:
   if h ≠ 0:
       raise Error("理論解僅適用於零外磁場")

2. EIGENVALUE_CALCULATION:
   beta ← 1/T
   // 🎯 轉移矩陣特徵值
   λ_plus ← exp(beta*J) + exp(-beta*J)   // = 2*cosh(beta*J)
   λ_minus ← exp(beta*J) - exp(-beta*J)  // = 2*sinh(beta*J)

3. PARTITION_FUNCTION:
   if periodic:
       Z ← λ_plus^L + λ_minus^L          // 週期邊界
   else:
       Z ← 2 * λ_plus^(L-1)              // 開放邊界

4. FREE_ENERGY:
   F ← -T * ln(Z)                        // 總自由能

5. RETURN: F
```

**⏱️ 複雜度分析**：
- **時間複雜度**: O(1) - 僅需簡單數學運算
- **空間複雜度**: O(1) - 常數空間
- **適用範圍**: 僅限 h=0，任意 L

---

### Algorithm 4: 多溫度自由能計算
```
Algorithm: Multi_Temperature_Free_Energy
Input: L, T_array (溫度陣列), J, h, method
Output: F_array (對應溫度的自由能陣列)

1. METHOD_SELECTION:
   if method == "auto":
       method ← "theory" if h == 0 else "enum"

2. TEMPERATURE_LOOP:
   if method == "theory":
       // 🚀 向量化計算
       beta ← 1 / T_array
       λ_plus ← exp(beta*J) + exp(-beta*J)
       λ_minus ← exp(beta*J) - exp(-beta*J)
       Z ← λ_plus^L + λ_minus^L
       F_array ← -T_array * ln(Z)
       
   else if method == "enum":
       // 🔄 一次性計算所有組態能量
       energies, _, _ ← ising1d_all_energies_gray(L, J, h, periodic)
       
       // 📊 批量處理多溫度
       x ← -outer(1/T_array, energies)    // shape: (n_temp, 2^L)
       ln_Z_array ← logsumexp_rows(x)     // 每行做 log-sum-exp
       F_array ← -T_array * ln_Z_array

3. RETURN: F_array
```

**🌟 批量處理優勢**：
- **窮舉法**: 組態計算只需一次，多溫度重複利用
- **理論解**: 完全向量化，同時計算所有溫度
- **記憶體效率**: 避免重複儲存中間結果

---

### Algorithm 5: 性能基準測試流程
```
Algorithm: Performance_Benchmark
Input: L_list (系統尺寸列表), methods (方法列表), 物理參數
Output: 效能分析圖表

1. INITIALIZATION:
   T_range ← linspace(T_min, T_max, n_points)
   runtime_data ← empty_dict for each method
   
2. NESTED_BENCHMARKING:
   for L in L_list do:
       for method in methods do:
           // 🕐 計時開始
           start_time ← current_time()
           
           // 📊 執行計算
           if method == "enumeration":
               F_data ← enumeration_free_energy(L, T_range, J, h)
           else if method == "transfer_matrix":
               F_data ← transfer_matrix_free_energy(L, T_range, J, h)
           else if method == "theory":
               F_data ← theoretical_free_energy(L, T_range, J)
           
           // ⏱️ 記錄時間
           elapsed ← current_time() - start_time
           runtime_data[method].append(elapsed)
       end for
   end for

3. VISUALIZATION:
   // 📈 雙圖表輸出
   plot_free_energy_curves(L_list, methods, F_data)
   plot_runtime_scaling(L_list, runtime_data)

4. RETURN: {runtime_data, performance_metrics}
```

**📊 性能評估指標**：
- **計算時間**: 絕對執行時間
- **擴展性**: 時間複雜度驗證 (O(2^L) vs O(L))
- **精度**: 與理論解的偏差分析
- **穩定性**: 數值精度與收斂性檢查

---

## 🎯 總結

`1d_model.py` 是一個設計精良的計算物理框架，展現了以下特色：

### 🏆 技術亮點
- **多方法整合**：窮舉、轉移矩陣、理論解三管齊下
- **算法優化**：Gray Code + 增量計算的高效組合
- **數值穩定**：Log-Sum-Exp 等專業技巧
- **接口統一**：MethodResult 數據類設計

### 🎓 學習價值
- **計算物理方法**：經典算法的現代實現
- **性能優化**：位元操作與算法設計的完美結合
- **軟件架構**：模組化設計與錯誤處理
- **科學計算**：數值穩定性與精度控制

### 🚀 實用性
- **可擴展性**：易於添加新的計算方法
- **可維護性**：清晰的代碼結構與文檔
- **可重現性**：統一的接口與結果格式
- **教育性**：適合物理學習與算法研究

這個實現不僅解決了 1D Ising 模型的計算問題，更提供了一個高質量的計算物理代碼範例，值得深入學習和參考。