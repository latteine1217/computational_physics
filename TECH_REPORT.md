# Ising 模型數值研究技術報告

作者：科研代理人（Codex CLI）  
版本：2024-XX-XX

---

## 1. 研究動機與專案概覽

本專案旨在針對 Ising 模型在一維／二維晶格上的統計力學性質，整合且比較下列數值演算法：

1. **枚舉（Enumeration）**：以 Gray code 完整走訪所有自旋態，提供最高精度的基準資料。
2. **轉移矩陣（Transfer Matrix, TM）**：將自旋相互作用線性化為矩陣運算，並解析求得對外場的導數，以高效率計算巨觀量。
3. **張量重正化群（Tensor Renormalization Group, TRG）**：透過張量網路重整 (coarse-graining) 近似大型二維晶格的配分函數。

所有演算法會輸出自由能、磁化率、熱容量與計算耗時，藉此評估演算法之精確度與效能。程式碼主要統整於 `1d_model.py` 與 `2d_model.py` 中，並搭配 `test_1d_model.py`、`test_2d_model.py` 進行數值驗證。

---

## 2. 模型假設與物理量定義

### 2.1 晶格與邊界條件

- **一維 (1D)**：長度為 $L$ 的自旋鏈，週期邊界條件 (Periodic Boundary Condition, PBC)，每個自旋與兩個鄰居作用。
- **二維 (2D)**：尺寸為 $L_x \times L_y$ 的方形晶格，同樣採用 PBC，確保能量項在所有方向都一致。
- 除非另行指定，耦合常數 $J>0$（鐵磁）且外場為均勻常數 $h$。

### 2.2 哈密頓量與配分函數

考慮標準 Ising 模型：

\[
\mathcal{H}(\{\sigma_i\}) = -J \sum_{\langle i,j \rangle} \sigma_i \sigma_j - h \sum_i \sigma_i,\qquad \sigma_i \in \{-1,+1\}.
\]

令 $\beta = 1/T$，配分函數為 $Z = \sum_{\{\sigma\}} e^{-\beta \mathcal{H}}$，本專案關注下列熱力學量：

- **自由能**：\( F = -\beta^{-1} \ln Z \)
- **磁化**：\( \langle M \rangle = \langle \sum_i \sigma_i \rangle \)
- **磁化率**：\( \chi = \frac{\beta}{N} \left(\langle M^2 \rangle - \langle M \rangle^2 \right) \)
- **熱容量**：\( C_v = \frac{\beta^2}{N} \left(\langle E^2 \rangle - \langle E \rangle^2 \right) \)

以上 $N$ 為總自旋數，一維為 $N=L$，二維為 $N=L_x L_y$，並且所有輸出皆以「每自旋」形式呈現，方便不同尺寸間比較。

---

## 3. 枚舉演算法

### 3.1 Gray Code 列舉

為避免逐一翻轉自旋時重新計算整體能量，採用 Gray code 產生自旋態序列。Gray code 保證相鄰編碼僅有一位元不同，因此每次只需計算單個自旋翻轉造成的能量增量：

\[
\Delta E_i = 2 \sigma_i \left(J \sum_{j \in \mathrm{NN}(i)} \sigma_j + h \right).
\]

實作細節：
- 以 `gray_next_flip_pos` 決定下一個翻轉的位元位置。
- 透過 `deltaE_ising*d_flip` 快速更新能量與磁化量。
- 為所有組態存取能量與磁化陣列，最後使用 `log-sum-exp` 技巧計算配分函數，以避免數值溢位。

### 3.2 時間複雜度與適用範圍

複雜度為 $\mathcal{O}(2^N)$，對 1D 最多能處理 $L \approx 20$，2D 只適用於 $L_x L_y \lesssim 16$ 的微小晶格。儘管耗時，枚舉結果可作為其他方法的**黃金標準**，於測試中驗證演算法正確性。

---

## 4. 轉移矩陣方法

### 4.1 一維解析實作

對 1D 模型，轉移矩陣 $T$ 為 $2\times2$：

\[
T_{\sigma, \sigma'} = \exp\big[\beta (J \sigma \sigma' + \tfrac{h}{2}(\sigma + \sigma'))\big], \quad \sigma,\sigma' \in \{-1,+1\}.
\]

使用者可藉由特徵值求得：

\[
\lambda_{\pm} = e^{\beta J} \cosh(\beta h) \pm \sqrt{e^{2\beta J} \sinh^2(\beta h) + e^{-2\beta J}}.
\]

為避免數值誤差，實作採用**矩陣冪法**而非直接使用解析式：
1. 建構 $T$ 及其對 $h$ 的導數矩陣 $\partial T / \partial h$、$\partial^2 T / \partial h^2$。
2. 透過累乘 $T$ 得到 $T^L$，同時利用 trace 計算 $Z=\operatorname{Tr}(T^L)$。
3. 將導數矩陣插入適當位置，採用以下恆等式求得導數（以 $T_1, T_2$ 表示導數）：

\[
\frac{\partial Z}{\partial h} = \sum_{k=0}^{L-1} \operatorname{Tr} (T^k T_1 T^{L-1-k}),
\]

\[
\frac{\partial^2 Z}{\partial h^2} = \sum_{k=0}^{L-1} \operatorname{Tr} (T^k T_2 T^{L-1-k}) + \sum_{0 \leq i < j \leq L-1} \operatorname{Tr}(T^i T_1 T^{j-i-1} T_1 T^{L-1-j}).
\]

將 $Z$ 的導數除以 $Z$ 便可得到 $\partial \log Z / \partial h$ 與 $\partial^2 \log Z / \partial h^2$，進而計算磁化與磁化率。熱容量則可由對 $\beta$ 的二階導數推得。

### 4.2 二維列轉移矩陣

將橫向一列視為超自旋，建立維度 $2^{L_x}$ 的轉移矩陣。矩陣元包含：

- 水平方向耦合（上、下列各取一半）
- 垂直方向耦合
- 場項（同樣拆半分布於兩列）

使用 `prefix` 陣列儲存累乘結果，以矩陣相乘於 trace 上插入導數張量。演算法流程與 1D 類似，但矩陣尺寸成長迅速，運算複雜度為 $\mathcal{O}(2^{3L_x})$，適合「窄條晶格 + 導數解析」的問題。

### 4.3 優點與限制

- **優點**：在適用範圍內具高準確度，並能提供磁化率、熱容量等導數精準值；不再需要有限差分。
- **限制**：矩陣維度隨橫向自旋數指數成長，故 2D 需控制在 $L_x \lesssim 8$；若外場或耦合非常大，仍須注意數值穩定性。

---

## 5. Tensor Renormalization Group (TRG)

### 5.1 張量化表示

每個晶格點對應一個四階張量 $T_{ijkl}$，四個索引代表上下左右連結。張量元素為相應自旋組態的 Boltzmann 權重：

\[
T_{ijkl} = \sum_{\sigma = \pm1} W_{i\sigma} W_{j\sigma} W_{k\sigma} W_{l\sigma} e^{\beta h \sigma},
\]

其中 $W$ 為拆解水平耦合的二階張量；實作中直接以解析式生成 $2 \times 2 \times 2 \times 2$ 張量。

### 5.2 Coarse-graining 流程

1. **張量重組**：將張量重新排列成矩陣，以方便做奇異值分解 (SVD)。
2. **SVD 與截斷**：針對矩陣進行 SVD，保留前 $\chi$ 個最大奇異值及對應向量。
3. **張量重組線縮合**：利用保留的向量重建新的張量，縮小晶格尺寸（每步驟縮小一半）。
4. **正規化**：為避免數值爆炸，以最大張量值正規化，並記錄縮放常數，最終加總於 $\log Z$。

重複以上步驟直至系統縮減為 $1 \times 1$，此時張量 trace 即為配分函數。

### 5.3 導數估計與參數

- 磁化率與熱容量透過有限差分估計：
  - $\partial h$ 方向使用 $\chi$ 的對稱差分。
  - $\partial \beta$ 方向估算 $C_v$。
- 截斷維度 $\chi$ 是主要誤差來源；預設為 32，可視需要調整。
- 目前只支援 $L_x = L_y$ 且為 $2^n$ 的格點。

---

## 6. 數值驗證與測試

### 6.1 枚舉 vs. 轉移矩陣

透過 `test_1d_model.py` 和 `test_2d_model.py`，分別驗證：

- 自由能的相對誤差 $< 10^{-10}$。
- 磁化率誤差落在 $10^{-5}$（1D）與 $5\times10^{-5}$（2D）內。
- 熱容量在 $10^{-6}$ 至 $5\times10^{-4}$ 的誤差範圍內。

### 6.2 TRG 精度

與枚舉結果比較，以 $4\times4$ 晶格為例，在 $\chi=32$ 時自由能可逼近 $5\times10^{-4}$，磁化率落在 $10^{-2}$ 等級。由於 TRG 本質為近似法，誤差會隨 $\chi$ 增大而下降。

### 6.3 測試與重現流程

1. 安裝 `numpy`、`matplotlib`。
2. 執行單元測試：`python3 -m unittest`。
3. 若要對比數值結果，可執行：
   - `python3 1d_model.py`
   - `python3 2d_model.py`

程式會輸出各演算法的自由能、磁化率、熱容量與耗時，同時生成比較圖。

---

## 7. 結論與建議

1. **枚舉法** 提供高精度基準，適於驗證小系統與調試演算法。
2. **轉移矩陣法** 經過解析導數後，能夠在維持高精度下取得磁化率與熱容量，且大幅減少有限差分所帶來的誤差。
3. **TRG** 提供跨越現有枚舉／轉移矩陣可行範圍的能力，是處理較大二維晶格的重要工具；需透過截斷參數控制精度。

未來可以進一步研究：

- 引入自動微分或高階差分以改善 TRG 的導數估計。
- 加入 Monte Carlo 或訊息傳遞類演算法以進行交叉驗證。
- 建立命令列介面與結果儲存模組，方便進行大量參數掃描與資料分析。

---

## 8. 參考資料

1. R. J. Baxter, *Exactly Solved Models in Statistical Mechanics*, Academic Press (1982).
2. M. Levin, C. P. Nave, “Tensor renormalization group approach to two-dimensional classical lattice models”, *Phys. Rev. Lett.*, **99**, 120601 (2007).
3. Stanley, H. E., *Introduction to Phase Transitions and Critical Phenomena*, Oxford University Press (1971).

---

本文件為專案技術與理論之正式紀錄，後續有新演算法或改進時，建議同步更新此報告。
