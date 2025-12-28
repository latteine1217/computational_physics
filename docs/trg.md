# TRG 演算法完整文檔（2D Ising Model）

## 1. 專案目標與背景

TRG 是一種對格點模型的張量網路粗粒化方法。以 2D Ising 模型為例，我們將配分函數寫成局域 rank-4 張量的收縮，並透過反覆「將 2×2 格點合併為一個有效格點」進行重整化。TRG 的核心優點是：

- 可在較大的系統尺寸下近似計算熱力學量
- 以截斷 bond dimension（\(\chi\)）控制精度與計算成本
- 可追蹤自由能的收斂與誤差

本專案對應的主要實作檔：`trg_final_project.py`。

---

## 2. 理論基礎（2D Ising 與張量表示）

### 2.1 2D Ising 模型

哈密頓量：

\[
H = -J \sum_{\langle i,j \rangle} s_i s_j - h \sum_i s_i, \quad s_i \in \{+1, -1\}
\]

配分函數：

\[
Z = \sum_{\{s\}} e^{-\beta H}
\]

其中 \(\beta = 1/T\)。

### 2.2 張量化（局域張量 T）

將每條鍵的 Boltzmann 權重寫成矩陣：

\[
W_{s_i s_j} = e^{\beta J s_i s_j}
\]

用特徵分解（對稱矩陣）：

\[
W = M M^T
\]

其中 \(M = V \sqrt{\Lambda}\)。

接著將每個格點的四條邊權重乘起來，得到 rank-4 張量：

\[
T_{u r d l} = \sum_s e^{\beta h s}
M_{s u} M_{s r} M_{s d} M_{s l}
\]

此張量即為 TRG 的起點。

---

## 3. TRG 核心流程（詳細步驟）

TRG 的單步粗粒化核心思想：將 2×2 的四個格點（四個 rank-4 張量）合併成一個新的 rank-4 張量。

**整體流程概述：**

1. 對同一個張量 T 做兩種不同配對的 SVD 分解
2. 用截斷參數 \(\chi\) 保留主要奇異值
3. 將四個分解出的因子張量按固定拓撲收縮成新張量
4. 記錄歸一化因子並更新有效格點數

---

### 3.1 初始狀態：2×2 格點配置

假設我們有一個 2×2 的格點系統，每個格點都是相同的 rank-4 張量 T[u,r,d,l]：

```text
     格點配置（2×2）：

        T ---- T
        |      |
        |      |
        T ---- T

每個 T 有四個接腳：
  - u (up): 向上連結
  - r (right): 向右連結
  - d (down): 向下連結
  - l (left): 向左連結
```

**目標：** 將這 2×2 的四個張量合併成一個新的有效張量 T'，它仍然是 rank-4，代表粗粒化後的單一格點。

---

### 3.2 核心策略：SVD 分解

由於直接收縮四個張量的計算量過大，TRG 採用「先分解、再重組」的策略：

1. 將 T 做兩次不同的 SVD 分解，產生四個較小的因子張量
2. 這四個因子張量恰好對應 2×2 配置的四個位置
3. 收縮這四個因子張量，得到新的粗粒化張量

---

### 3.3 第一次 SVD：垂直分解（產生 C3 與 C1）

#### 步驟 1：重排索引

將原始張量 T[u,r,d,l] 的索引重排為兩組：
- 上半部：(u, r)
- 下半部：(d, l)

```text
原始張量 T[u,r,d,l]：

         u
         |
    l ---+--- r
         |
         d

重排成矩陣形式：
  行索引：(u,r)
  列索引：(d,l)
```

#### 步驟 2：Reshape 成矩陣

\[
T[u,r,d,l] \rightarrow T_{\text{matrix}}[(ur), (dl)]
\]

其中 (ur) 表示將 u 和 r 兩個索引合併成一個複合索引。

#### 步驟 3：SVD 分解

\[
T_{\text{matrix}}[(ur), (dl)] = U_{(ur), a} \cdot S_a \cdot V_{a, (dl)}^{\dagger}
\]

其中：
- U: 左奇異向量矩陣，形狀 [(ur), a]
- S: 奇異值對角矩陣，形狀 [a]
- V†: 右奇異向量矩陣，形狀 [a, (dl)]
- a: 內部鍵維度（bond dimension），最大為 min(dim(ur), dim(dl))

**截斷處理：** 只保留前 \(\chi\) 個最大的奇異值：

\[
a_{\text{eff}} = \min(\chi, a_{\text{max}})
\]

#### 步驟 4：分配奇異值（對稱分解）

為了保持數值穩定性，將奇異值的平方根分別吸收到 U 和 V 中：

\[
C3_{u,r,a} = U_{(ur),a} \cdot \sqrt{S_a}
\]

\[
C1_{a,d,l} = \sqrt{S_a} \cdot V_{a,(dl)}
\]

reshape 回 rank-3 張量：
- C3[u, r, a]：上半部因子
- C1[a, d, l]：下半部因子

```text
第一次 SVD 結果：

    u
    |
    C3[u,r,a]
    |\
    | r
    |
    a (內部鍵)
    |
    C1[a,d,l]
    |\
    | l
    d
```

---

### 3.4 第二次 SVD：水平分解（產生 C2 與 C0）

#### 步驟 1：重排索引

將原始張量 T[u,r,d,l] 重排為：
- 左半部：(u, l)
- 右半部：(r, d)

```text
原始張量 T[u,r,d,l]：

         u
         |
    l ---+--- r
         |
         d

重排成矩陣形式：
  行索引：(u,l)
  列索引：(r,d)
```

#### 步驟 2：Reshape 成矩陣

\[
T[u,l,r,d] \rightarrow T_{\text{matrix}}[(ul), (rd)]
\]

#### 步驟 3：SVD 分解

\[
T_{\text{matrix}}[(ul), (rd)] = U'_{(ul), b} \cdot S'_b \cdot V'_{b, (rd)}^{\dagger}
\]

**截斷處理：** 保留前 \(\chi\) 個奇異值。

#### 步驟 4：分配奇異值

\[
C2_{u,l,b} = U'_{(ul),b} \cdot \sqrt{S'_b}
\]

\[
C0_{b,r,d} = \sqrt{S'_b} \cdot V'_{b,(rd)}
\]

reshape 回 rank-3 張量：
- C2[u, l, b]：左半部因子
- C0[b, r, d]：右半部因子

```text
第二次 SVD 結果：

         u
         |
    l ---C2[u,l,b]
         |
         b (內部鍵)
         |
         C0[b,r,d]--- r
         |
         d
```

---

### 3.5 四個因子的幾何配置

現在我們有四個 rank-3 張量：C0, C1, C2, C3，它們在 2×2 配置中的位置：

```text
位置示意（格點視角）：

     C3 ---- C2
     |       |
     |       |
     C1 ---- C0

每個因子的接腳：
  C3[u, r, a]: 左上
  C2[u, l, b]: 右上
  C1[a, d, l]: 左下
  C0[b, r, d]: 右下

內部連線（需要被收縮）：
  - C3.a = C1.a (垂直連結)
  - C2.b = C0.b (垂直連結)
  - C3.r ←→ C2.l (水平連結，將在最後收縮)
  - C1.l ←→ C0.r (水平連結，將在最後收縮)
```

---

### 3.6 收縮步驟（逐步詳解）

#### 收縮策略

我們需要將內部的接腳收縮掉，只保留外部的四個接腳形成新的 T'。

**內部接腳（需消去）：**
- 垂直內部鍵：a, b
- 水平連結：C3.r, C2.l, C1.l, C0.r
- 上下連結：C3.u, C2.u (將成為外部接腳)
- 左右連結：C1.d, C0.d (將成為外部接腳)

**外部接腳（保留）：**
- 新的 u: 從 C3, C2 的 u 連出
- 新的 r: 從 C0 的 r 連出
- 新的 d: 從 C0, C1 的 d 連出
- 新的 l: 從 C2 的 l 連出

#### 詳細收縮步驟

**步驟 1：收縮 C0 與 C1（消去 d）**

```text
C0[b,r,d] 與 C1[a,d,l] 共享索引 d

收縮圖示：
         C1[a,d,l]
              |
              d (收縮)
              |
         C0[b,r,d]

結果：temp1[a,r,b,l]
```

數學表達：
\[
\text{temp1}_{a,r,b,l} = \sum_d C0_{b,r,d} \cdot C1_{a,d,l}
\]

形狀變化：
- C0: [b, r, d] → d 被消去
- C1: [a, d, l] → d 被消去
- temp1: [a, r, b, l]

**步驟 2：收縮 temp1 與 C2（消去 l）**

```text
temp1[a,r,b,l] 與 C2[u,l,c] 共享索引 l

收縮圖示：
         C2[u,l,c]
              |
              l (收縮)
              |
    temp1[a,r,b,l]

結果：temp2[a,r,b,u,c]
```

數學表達：
\[
\text{temp2}_{a,r,b,u,c} = \sum_l \text{temp1}_{a,r,b,l} \cdot C2_{u,l,c}
\]

形狀變化：
- temp1: [a, r, b, l] → l 被消去
- C2: [u, l, c] → l 被消去
- temp2: [a, r, b, u, c]

**步驟 3：收縮 temp2 與 C3（消去 u 和 r）**

```text
temp2[a,r,b,u,c] 與 C3[u,r,e] 共享索引 u, r

收縮圖示：
      C3[u,r,e]
       | |
      u r (收縮)
       | |
  temp2[a,r,b,u,c]

結果：T'[a,b,c,e]
```

數學表達：
\[
T'_{a,b,c,e} = \sum_{u,r} \text{temp2}_{a,r,b,u,c} \cdot C3_{u,r,e}
\]

形狀變化：
- temp2: [a, r, b, u, c] → u, r 被消去
- C3: [u, r, e] → u, r 被消去
- T': [a, b, c, e]

---

### 3.7 最終結果：新的粗粒化張量

經過三步收縮，我們得到新的 rank-4 張量 T'[a,b,c,e]：

```text
最終張量 T'[a,b,c,e]：

         e (新 u)
         |
    c ---+--- a (新 r)
         |
         b (新 d)

(l 方向為 c)
```

**索引重命名（映射到標準方向）：**

為了下一次迭代，我們將新張量的索引重命名為標準形式：

\[
T'[u_{\text{new}}, r_{\text{new}}, d_{\text{new}}, l_{\text{new}}] = T'[e, a, b, c]
\]

即：
- 新 u ← e
- 新 r ← a
- 新 d ← b
- 新 l ← c

---

### 3.8 歸一化處理

為避免數值溢出或下溢，每步都對張量進行歸一化：

\[
g_n = \text{Trace}(T') = \sum_{i} T'_{i,i,i,i}
\]

\[
T' \rightarrow \frac{T'}{g_n^{1/4}}
\]

歸一化因子 \(g_n\) 會被記錄下來，用於最終計算自由能。

---

### 3.9 完整流程圖（ASCII 示意）

```text
========================================
TRG 單步完整流程
========================================

輸入：T[u,r,d,l]

       ┌─────────────┐
       │  原始張量 T  │
       │  [u,r,d,l]  │
       └──────┬──────┘
              │
       ┌──────┴──────┐
       │             │
   SVD1 (ur|dl)  SVD2 (ul|rd)
       │             │
   ┌───┴───┐     ┌───┴───┐
   │       │     │       │
  C3[u,r,a] C1[a,d,l]  C2[u,l,b] C0[b,r,d]
   │       │     │       │
   └───┬───┴─────┴───┬───┘
       │             │
       │   配置成2×2  │
       │             │
       │   C3 ── C2  │
       │   |     |   │
       │   C1 ── C0  │
       │             │
       └──────┬──────┘
              │
        逐步收縮：
        1. C0 × C1 (消 d)
        2. temp1 × C2 (消 l)
        3. temp2 × C3 (消 u,r)
              │
       ┌──────┴──────┐
       │  新張量 T'   │
       │  [a,b,c,e]  │
       └──────┬──────┘
              │
         重命名為
       T'[u',r',d',l']
              │
          歸一化
              │
         輸出 T', g_n
========================================
```

---

### 3.10 索引追蹤表（完整對照）

| 步驟 | 張量 | 形狀 | 說明 |
|------|------|------|------|
| 初始 | T | [u,r,d,l] | 原始 rank-4 張量 |
| SVD1 | C3 | [u,r,a] | 第一次分解上半部 |
| SVD1 | C1 | [a,d,l] | 第一次分解下半部 |
| SVD2 | C2 | [u,l,b] | 第二次分解左半部 |
| SVD2 | C0 | [b,r,d] | 第二次分解右半部 |
| 收縮1 | temp1 | [a,r,b,l] | C0 × C1，消去 d |
| 收縮2 | temp2 | [a,r,b,u,c] | temp1 × C2，消去 l |
| 收縮3 | T' | [a,b,c,e] | temp2 × C3，消去 u,r |
| 重命名 | T' | [u',r',d',l'] | 標準化索引 |

---

### 3.11 實作中的 einsum 表達式

對應程式碼中的實際收縮：

```python
# 步驟 1
temp1 = np.einsum('ard,bdl->arbl', C0, C1)

# 步驟 2
temp2 = np.einsum('arbl,ulc->arbuc', temp1, C2)

# 步驟 3
T_new = np.einsum('arbuc,ure->abce', temp2, C3)
```

**einsum 語法解讀：**
- `'ard,bdl->arbl'`：將 C0 的 d 與 C1 的 d 收縮
- `'arbl,ulc->arbuc'`：將 temp1 的 l 與 C2 的 l 收縮
- `'arbuc,ure->abce'`：將 temp2 的 u,r 與 C3 的 u,r 收縮

---

### 3.12 關鍵注意事項

1. **一致性截斷**：兩次 SVD 必須使用相同的有效 \(\chi\)：
   \[
   \chi_{\text{eff}} = \min(\chi, \chi_1, \chi_2)
   \]

2. **對稱分解**：奇異值的平方根分別吸收到 U 和 V，保持數值穩定

3. **索引一致性**：所有步驟的接腳方向（u,r,d,l）必須一致

4. **歸一化時機**：在每步收縮後立即歸一化，避免累積誤差

---

## 4. 自由能與歸一化公式（重要）

TRG 每步會對張量做歸一化，避免數值爆炸。若第 \(n\) 步的歸一化因子為 \(g_n\)，有效格點數為 \(N_n\)，則自由能密度（每格點）：

\[
\boxed{f = -T \sum_n \frac{\ln g_n}{N_n}}
\]

這是目前實作中使用的正確公式，並與 cytnx 版本一致。

---

## 5. 數值穩定性與收斂性

### 5.1 SVD 門檻

在 SVD 時使用相對門檻 `rel_svd_cutoff`：

\[
\sigma_i \ge \text{cutoff} \cdot \sigma_0
\]

避免極小奇異值導致數值誤差放大。

### 5.2 一致的 \(\chi\) 截斷

兩次 SVD 必須使用相同的 \(\chi\)（取兩者最小值），保證新張量維度一致：

\[
\chi_{\text{eff}} = \min(\chi, \chi_1, \chi_2)
\]

---

## 6. 實作對應（程式結構）

| 功能 | 函式 | 說明 |
|---|---|---|
| 初始張量 | `_ising_local_tensor` | 建立 Ising 局域張量 \(T\) |
| 單步 TRG | `_trg_step` | SVD 分解 + 收縮 + 截斷 |
| 流程控制 | `TRGFlow` | 更新、歸一化、累積 log factors |
| 自由能 | `compute_free_energy` | 以 TRG 計算 \(f\) |
| 熱容量 | `compute_heat_capacity` | 以有限差分估算 \(C_v\) |
| 繪圖 | `plot_figure1/2/3` | 產生三張報告圖 |

---

## 7. 圖表與數值輸出

程式會輸出三張圖（英文標題與 label）：

1. **Convergence at Critical Temperature**
2. **TRG Error vs Temperature**
3. **Heat Capacity Peak**

主要輸出檔：

- `figure1_convergence.pdf`
- `figure2_error_temperature.pdf`
- `figure3_heat_capacity.pdf`

---

## 8. 使用方式與重現流程

### 8.1 執行模式

```bash
python trg_final_project.py all
python trg_final_project.py 1
python trg_final_project.py 2
python trg_final_project.py 3
python trg_final_project.py stats
```

- `all`：產生所有圖表
- `1/2/3`：分別生成圖 1/2/3
- `stats`：輸出關鍵數值（T=Tc）

### 8.2 參數建議

- bond dimension \(\chi\)：越大精度越高、計算越慢
- iterations：圖 2/3 可使用較少迭代以節省時間
- `rel_svd_cutoff`：避免過小奇異值導致不穩定

---

## 9. 效能與擴展性

- 時間成本主要來自 SVD，約 \(O(\chi^6)\)
- 可透過降低 \(\chi\) 或減少 iterations 來控制成本
- 若需更高精度，可延伸到 HOTRG 或 SRG

---

## 10. 可驗證性與再現性

- 固定參數（T, J, h, \(\chi\), iterations）即可重現
- 自由能可與 Onsager 精確解比較（目前以 \(-2.109651\) 為基準）
- 若需更嚴格驗證，可加入誤差曲線或對比 transfer matrix 結果

---

## 11. 後續擴充建議

1. **磁化率**：可由 \(\partial M / \partial h\) 或二階差分近似
2. **計算時間**：將每次 TRG iteration 的時間累積輸出
3. **自動化測試**：加入單元測試檢查自由能收斂與對稱性

---

## 12. 附錄：TRG 單步偽碼

```text
Input: T (rank-4), chi
SVD1: T(u,r,d,l) -> reshape(ur, dl)
SVD2: T(u,l,r,d) -> reshape(ul, rd)
chi_eff = min(chi, chi1, chi2)
Construct C0,C1,C2,C3 with sqrt(sigma)
Contract C0,C1,C2,C3 -> T'
Normalize by g_n = Tr(T')
Return: T', g_n
```

---

完成。
