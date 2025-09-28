# Algorithm 1  Numerical workflow for Ising-model observables

1. **設定模型參數**：輸入晶格尺寸、耦合常數與外場參數
   \[
   \Theta = \{L, L_x, L_y, J, h, T, \text{PBC}, \chi, M\},
   \]
   其中 $\chi$ 為 TRG 的截斷維度，$M$ 為 TRG temporal blocking 段數；當只考慮 1D 時使用 $L$，2D 時使用 $(L_x, L_y)$。

2. **初始化資料結構**：建立可重複使用的工具函式
   - `gray_next_flip_pos(t)`、`deltaE_ising*d_flip(...)`：供枚舉法快速更新能量與磁化。
   - `_transfer_matrix_stats_1d`、`_transfer_matrix_stats`：回傳 $\log Z$ 及其對外場與溫度導數。
   - `_build_trg_initial_tensor`、`_trg_step`：構建與重整張量網路。

3. **枚舉（Enumeration）階段**  
   for 目標維度 in \{1D, 2D\}
   1. Initialize `bits ← 0`（全為 $-1$ 自旋），計算初始能量與磁化。
   2. for $t = 0, 1, \ldots, 2^N-1$
      1. 儲存目前能量 $E_t$ 與磁化 $M_t$。
      2. 使用 `gray_next_flip_pos` 找到下一個翻轉位置 $i$，並利用
         \[
         \Delta E_i = 2\,\sigma_i\left(J\sum_{j \in NN(i)} \sigma_j + h\right)
         \]
         更新 $E$ 與 $M$。
   3. 利用 log-sum-exp 計算配分函數
      \[
      Z = \sum_t e^{-\beta E_t},\qquad F = -\beta^{-1} \log Z,
      \]
      並依據 $E_t, M_t$ 求得磁化率與熱容量：
      \[
      \chi = \frac{\beta}{N}\Big(\langle M^2 \rangle - \langle M \rangle^2\Big),\quad
      C_v = \frac{\beta^2}{N}\Big(\langle E^2 \rangle - \langle E \rangle^2\Big).
      \]

4. **轉移矩陣（Transfer Matrix）階段**
   1. 建立單列或單格的轉移矩陣 $T$ 及其導數：
      \[
      T_{ab} = e^{\beta(\mathcal{J}_{ab} + h\mathcal{H}_{ab}/2)},\quad
      \partial_h T_{ab},\quad \partial_h^2 T_{ab}.
      \]
   2. 透過 prefix 累積矩陣冪，計算
      \[
      Z = \operatorname{Tr}(T^L),\quad
      \frac{\partial Z}{\partial h} = \sum_{k=0}^{L-1} \operatorname{Tr}(T^k T_h T^{L-1-k}),
      \]
      \[
      \frac{\partial^2 Z}{\partial h^2} = \sum_{k=0}^{L-1} \operatorname{Tr}(T^k T_{hh} T^{L-1-k}) + \sum_{0\leq i<j<L} \operatorname{Tr}(T^i T_h T^{j-i-1} T_h T^{L-1-j}).
      \]
   3. 推得磁化率、熱容量：
      \[
      \chi = \frac{1}{\beta N}\left(\frac{\partial^2 \log Z}{\partial h^2}\right),\qquad
      C_v = \frac{\beta^2}{N}\left(\frac{\partial^2 \log Z}{\partial \beta^2}\right).
      \]

5. **TRG 階段**（僅二維）
   1. 建立局部張量 $T^{(0)}$（見 ` _build_trg_initial_tensor`）。
   2. for coarse-graining level $\ell = 1, \ldots, \log_2 L_x$
      1. 重新排列張量索引並執行 SVD，保留前 $\chi$ 個奇異值與向量。
      2. 以保留向量重建縮小後張量 $T^{(\ell)}$，並將最大元素正規化，紀錄縮放量 $s_\ell$。
   3. 得到最終縮減張量 $T^{(\text{final})}$，其 trace 為 $Z_{\text{core}}$，配分函數為
      \[
      \log Z = \sum_{\ell} N_\ell \log s_\ell + \log Z_{\text{core}}.
      \]
   4. 透過有限差分估計磁化與磁化率：
      \[
      \chi \approx \frac{\log Z(h+\varepsilon) - 2\log Z(h) + \log Z(h-\varepsilon)}{\varepsilon^2 \beta N}.
      \]
      熱容量以 $\beta$ 方向差分計算。

6. **統整與輸出**
   - 定義 `MethodResult = (F/N, \chi, C_v, \text{runtime}, \text{metadata})`。
   - 回傳或寫檔列舉 `enumeration`、`transfer_matrix`、`trg` 等方法之結果，用於比較與後續視覺化。

