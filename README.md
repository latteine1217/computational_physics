# 一維／二維 Ising Model 數值模擬工具

本專案旨在比較多種經典統計物理算法於 Ising model 上的表現，涵蓋一維與二維系統。已實作的演算法包含：

- 1D：枚舉（Gray code）、轉移矩陣、解析解（$h=0$）
- 2D：枚舉、列轉移矩陣、Tensor Renormalization Group (TRG)

重點觀察量包括自由能、磁化率與演算法耗時，以便評估理論正確性與數值效率。

---

## 理論背景

### Hamiltonian 與配分函數
Ising model 的作用量為


$$ \mathcal{H}(\{\sigma_i\}) = -J \sum_{\langle i,j \rangle} \sigma_i \sigma_j - h \sum_i \sigma_i,\qquad \sigma_i \in \{-1,+1\} $$

其中 $J$ 為耦合常數、$h$ 為外場、和號遍歷最近鄰。溫度 $T$ 下的配分函數與主要觀察量定義為

$$ Z = \sum_{\{\sigma_i\}} e^{-\beta \mathcal{H}},\quad F = -\frac{1}{\beta} \ln Z,\quad M = \langle \sum_i \sigma_i \rangle, $$

$$ \chi = \frac{\beta}{N}\big( \langle M^2 \rangle - \langle M \rangle^2 \big),\quad C_v = \frac{\beta^2}{N}\big( \langle E^2 \rangle - \langle E \rangle^2 \big),\quad N=L\text{ for 1D}.$$

### 一維解析解
對於週期邊界條件且無外場 ($h=0$) 的一維系統，轉移矩陣為

$$
T = \begin{pmatrix}
e^{\beta J} & e^{-\beta J} \\
e^{-\beta J} & e^{\beta J}
\end{pmatrix},
$$

其特徵值 $\lambda_\pm = e^{\beta J} \pm e^{-\beta J}$，因此配分函數為 $Z = \lambda_+^L + \lambda_-^L$，自由能可直接由 $F = -\beta^{-1} \ln Z$ 得到。當 $h \neq 0$ 時仍可使用 2×2 轉移矩陣，但需數值求解並以有限差分取得磁化率。

---

## 演算法實作

### 1. 完全枚舉（Gray code）
- 透過 Gray code 逐一翻轉單一自旋，計算所有 $2^L$ 組態的能量與磁化。
- 利用 $\Delta E$ 更新避免重複計算能量，並以 log-sum-exp 穩定計算配分函數。
- 適用於小系統（目前建議 $L\lesssim20$），可作為數值真值用於驗證其它方法。

### 2. 轉移矩陣（Transfer Matrix）
- 建立 2×2 轉移矩陣 $T_{s_i,s_{i+1}} = \exp[\beta(J s_i s_{i+1} + \tfrac{h}{2}(s_i+s_{i+1}))]$。
- 週期邊界下以特徵值求得 $Z$，再計算自由能。
- 對於非零外場，對 $h$ 做中心差分近似導數，以獲得磁化與磁化率。
- 此方法在 1D 中為解析等價方案，且計算量隨 $L$ 線性。

### 3. 二維列轉移矩陣
- 將橫向自旋列視為超自旋，建立 $2^{L_x} \times 2^{L_x}$ 的列轉移矩陣。
- 權重同時包含列內、列間耦合與場項（各以 1/2 權重避免重複計數）。
- 以特徵值求得 $Z = \sum_i \lambda_i^{L_y}$，自由能及磁化率同樣以有限差分取得。
- 適用於寬度小（$L_x \lesssim 8$）的矩形晶格，計算成本隨 $2^{L_x}$ 的平方成長。

### 4. Tensor Renormalization Group (TRG)
- 採用 Levin & Nave (2007) 的 TRG 流程，將格點映為四階張量並以 SVD 進行截斷。
- 每回合 coarse-grain 使系統尺寸縮小一半；透過標準化記錄縮放因子，最終得到配分函數。
- 支援外場並以有限差分計算磁化率；截斷 bond dimension `chi` 可由使用者調整。
- 目前限制：需正方形且邊長為 2 的冪次、週期邊界。

---

## 檔案結構

```
├── 1d_model.py                # 1D Ising：枚舉、轉移矩陣、理論解與繪圖
├── 2d_model.py                # 2D Ising：枚舉、列轉移矩陣、TRG 與繪圖
├── tensor_network_2x2.py      # 2×2 張量網路收縮與觀測量計算
├── 1D_MODEL_DETAILED_ANALYSIS.md
├── 2D_MODEL_DETAILED_ANALYSIS.md
├── tensor_method.md
├── TECH_REPORT.md
├── task-01.md
├── README.md
├── AGENTS.md
└── __pycache__/
```

---

## 使用方式

### 環境需求
- Python 3.10 以上
- 套件：`numpy`, `matplotlib`

### 指令範例
1. **執行一維範例與繪圖**
   ```bash
   python3 1d_model.py
   ```
   - 主程式會針對指定的 $L, J, h, T$ 執行窮舉與轉移矩陣，列出自由能、磁化率與耗時。
   - `plot_free_energy_vs_T_for_Ls` 會在同一張圖比較不同方法的自由能，並額外產生對應的 log-scale 時間圖。

2. **自訂參數**
   - 可在互動式環境中匯入 `run_1d_methods`：
     ```python
     from 1d_model import run_1d_methods
     results = run_1d_methods(L=10, T=2.0, J=1.0, h=0.3)
     ```
   - 回傳的 `MethodResult` 物件包含自由能／磁化率／熱容量（每自旋）、耗時與附加統計量。

3. **繪製 1D 自由能與耗時比較**
   ```python
   from 1d_model import plot_free_energy_vs_T_for_Ls
   plot_free_energy_vs_T_for_Ls(L_list=[4,6,8,10], J=1.0, h=0.1,
                                T_min=0.5, T_max=3.0, nT=150,
                                methods=("enumeration", "transfer_matrix"))
   ```
   - 當 $h \neq 0$ 時，函式會自動略過理論解以避免錯誤。

4. **比較 2D 各種演算法**
   ```python
   from 2d_model import run_2d_methods, plot_free_energy_vs_T_for_Ls

   res = run_2d_methods(Lx=4, Ly=4, T=2.5, J=1.0, h=0.1,
                        methods=("enumeration", "transfer_matrix", "trg"),
                        trg_kwargs={"chi": 32})

   plot_free_energy_vs_T_for_Ls(Lx_list=[2, 4], Ly_list=[2, 4],
                                J=1.0, h=0.0,
                                methods=("enumeration", "transfer_matrix", "tensor_network_2x2", "trg"),
                                trg_kwargs={"chi": 32})
   ```
   - 圖像一次呈現自由能、運算時間、磁化率與熱容量四項比較。
   - `tensor_network_2x2` 僅支援 $2\times 2$ 且具有週期邊界的晶格，其結果會自動整合至曲線與耗時比較圖。
   - TRG 目前僅支援正方形且邊長為 $2^n$；若超出條件會拋出例外。

---

- 完全枚舉計算量隨 $2^N$ 成長，1D 建議 $L\lesssim 20$，2D 則僅適合非常小的晶格。
- 列轉移矩陣演算法複雜度隨 $2^{2L_x}$ 成長，宜搭配較小的寬度與適量的溫度點。
- TRG 透過截斷控制複雜度，`chi` 增大可提升精度但成本也上升；對低溫或大外場情況需評估收斂性。
- 磁化率統一透過有限差分近似，建議針對敏感區域調整 `field_eps`。

---

## 後續開發方向
- 加入 2D Monte Carlo / 羅列更多可比較的近似方法。
- 為 TRG 增加自動截斷策略與誤差估計，支援非正方形晶格。
- 提供命令列介面與結果儲存功能，便於批次掃描參數空間。

---

如有建議或疑問，歡迎於專案中提出。祝計算順利！
