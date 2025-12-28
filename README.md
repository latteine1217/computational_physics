# 2D Ising Model Numerical Study via TRG

![Language](https://img.shields.io/badge/language-python-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

本專案為計算物理期末專案，旨在透過 **張量重整化群 (Tensor Renormalization Group, TRG)** 演算法研究二維 Ising 模型的熱力學性質，並與解析解（Onsager Solution）及經典數值方法（枚舉、轉移矩陣）進行深入對照與效能分析。

## 🚀 專案亮點
- **多方法驗證**：實作 1D/2D 枚舉 (Gray code)、轉移矩陣 (Transfer Matrix) 與 TRG。
- **深度物理分析**：探討 TRG 的有效秩瓶頸 (Effective Rank Bottleneck)、有限糾纏標度 (Finite Entanglement Scaling) 與數值穩定性。
- **高效能實作**：優化 `einsum` 收縮路徑，確保 $O(\chi^6)$ 複雜度，並處理對數空間累積以防數值溢位。
- **完整期末報告**：包含專業 LaTeX 撰寫的學術報告與詳細的數據圖表。

## 📂 檔案結構
```text
.
├── src/                # 核心演算法實現
│   ├── trg_final_project.py  # TRG 核心邏輯與流程控制
│   ├── 1d_model.py           # 1D 模型驗證工具
│   └── 2d_model.py           # 2D 模型驗證工具
├── report/             # 期末報告與數據圖表
│   ├── final_report.tex      # LaTeX 報告主檔
│   └── figures/              # 報告使用的所有數據圖表 (.png, .pdf)
├── analysis/           # 物理特性與誤差分析腳本
│   ├── chi_saturation_analysis.py  # 探討 Bond Dimension 飽和效應
│   └── visualize_chi_saturation.py # 視覺化精度與 chi 的關係
├── tests/              # 單元測試與正確性驗證
├── benchmarks/         # 效能測試與矩陣運算優化實驗
├── docs/               # 開發筆記與詳細文檔
└── archive/            # 過時或偵錯用檔案 (已歸檔)
```

## 🛠️ 安裝與執行

### 環境需求
- Python 3.10.12
- 依賴套件：`numpy`, `matplotlib`, `scipy`

### 執行範例
1. **執行 TRG 主模擬並生成數據**：
   ```bash
   python3 src/trg_final_project.py
   ```
2. **運行 1D 基準測試**：
   ```bash
   python3 src/1d_model.py
   ```
3. **進行 Bond Dimension 飽和度分析**：
   ```bash
   python3 analysis/chi_saturation_analysis.py
   ```

## 📝 期末報告
詳細的理論推導、數值結果分析與演算法討論請參閱：
👉 **[report/final_report.tex](./report/final_report.tex)** (LaTeX 源碼)
👉 本專案亦包含編譯後的圖表於 `report/figures/` 目錄中。

## 🔗 相關資源
- **GitHub Repository**: [https://github.com/latteine1217/computational_physics](https://github.com/latteine1217/computational_physics)
- **核心算法參考**: Levin & Nave (2007)

---
*Student ID: 113011527 | JunYi Li*