# LaTeX 期末報告編譯指南

## 🌐 Overleaf 使用指南（推薦）

**Overleaf 是線上 LaTeX 編輯平台，無需安裝任何軟體，特別適合 LaTeX 初學者！**

### 快速開始（Overleaf）

1. **上傳檔案到 Overleaf**
   - 訪問 [https://www.overleaf.com](https://www.overleaf.com)
   - 創建新專案：「New Project」→「Blank Project」
   - 上傳以下檔案：
     - `final_report.tex`
     - `figure1_convergence.png`
     - `figure2_error_temperature.png`
     - `figure3_heat_capacity.png`

2. **設定編譯器**
   - 點擊左上角「Menu」
   - 在「Settings」中找到「Compiler」
   - **選擇「XeLaTeX」**（非常重要！）

3. **修改個人資訊**
   - 打開 `final_report.tex`
   - 找到第 67-69 行
   - 修改學號和姓名

4. **編譯**
   - 點擊右上角綠色的「Recompile」按鈕
   - 或使用快捷鍵：`Ctrl+S`（Windows/Linux）或 `Cmd+S`（macOS）
   - PDF 會自動生成在右側預覽窗口

5. **下載 PDF**
   - 點擊 PDF 預覽上方的「Download PDF」
   - 或使用「Menu」→「Download」→「PDF」

### Overleaf 常見問題

**Q: 編譯時出現中文亂碼或字體錯誤**
- **解決方法**：確認編譯器設為 **XeLaTeX**（不是 pdfLaTeX）
- 如果仍有問題，打開 `final_report.tex`，將第 8-13 行註解掉，啟用第 17-19 行：
  ```latex
  % 註解掉這三行
  % \setCJKmainfont{Noto Sans CJK TC}[...]
  % \setCJKsansfont{Noto Sans CJK TC}
  % \setCJKmonofont{Noto Sans Mono CJK TC}

  % 啟用這三行（刪除開頭的 %）
  \setCJKmainfont{FandolSong}
  \setCJKsansfont{FandolHei}
  \setCJKmonofont{FandolFang}
  ```

**Q: 圖片無法顯示**
- 確認三張 `.png` 圖片已上傳到 Overleaf 專案中
- 圖片檔名必須完全一致（區分大小寫）

**Q: 編譯速度很慢**
- 正常現象，Overleaf 免費版編譯較慢（約 10-30 秒）
- 可升級為付費版以加速（非必需）

**Q: 如何與他人協作**
- 點擊右上角「Share」→ 輸入對方 Email
- 或生成「Anyone with this link can view this project」連結

---

## 📋 文件說明

- **final_report.tex**: 主要 LaTeX 報告文檔
- **figure1_convergence.png**: 圖一（收斂曲線）
- **figure2_error_temperature.png**: 圖二（誤差 vs 溫度）
- **figure3_heat_capacity.png**: 圖三（熱容量）

## ⚙️ 編譯前準備

### 1. 修改個人資訊
打開 `final_report.tex`，修改第 55-57 行：
```latex
\title{\textbf{2D Ising Model 的 Tensor Renormalization Group 數值計算} \\
\large{計算物理期末專案報告}}
\author{學號：[你的學號] \\ 姓名：[你的姓名]}  % <-- 修改這裡
\date{2025年12月22日}
```

### 2. 確認圖片存在
確保以下圖片檔案在同一目錄下：
```bash
ls figure*.png
```
如果缺少圖片，請先執行：
```bash
python trg_final_project.py 1    # 生成圖一
python trg_final_project.py opt  # 生成圖二和圖三
```

### 3. 選擇中文字體（Overleaf 已優化）

**Overleaf 用戶**（推薦，預設已設定好）：
```latex
% 預設使用 Noto Sans CJK TC（已設定）
\setCJKmainfont{Noto Sans CJK TC}
```

如果 Overleaf 編譯時字體出錯，可切換為備用方案：
```latex
% 方案 1: FandolSong（TeX Live 內建，100% 兼容）
\setCJKmainfont{FandolSong}
\setCJKsansfont{FandolHei}
\setCJKmonofont{FandolFang}

% 方案 2: AR PL 系列
\setCJKmainfont{AR PL UMing TW}
```

**本地編譯用戶**：
- **macOS**: `\setCJKmainfont{PingFang TC}`
- **Linux**: `\setCJKmainfont{Noto Sans CJK TC}`
- **Windows**: `\setCJKmainfont{Microsoft JhengHei}`

## 🔨 編譯方法

### 方法一：使用 XeLaTeX（推薦）
```bash
xelatex final_report.tex
xelatex final_report.tex  # 再執行一次以更新目錄和引用
```

### 方法二：使用 Makefile
```bash
make          # 編譯
make clean    # 清理輔助檔案
make view     # 編譯並開啟 PDF（macOS）
```

### 方法三：使用 latexmk（自動處理依賴）
```bash
latexmk -xelatex -interaction=nonstopmode final_report.tex
```

## 📦 所需套件

確保已安裝以下 LaTeX 套件：
- **xeCJK**: 中文支持
- **amsmath, amssymb, amsthm**: 數學符號
- **physics**: 物理符號（如 \dd）
- **graphicx, float, subcaption**: 圖片處理
- **booktabs**: 專業表格
- **listings, xcolor**: 代碼高亮
- **hyperref**: 超連結

**TeX Live 用戶**（通常已包含所有套件）：
```bash
# 檢查安裝
tlmgr list --installed | grep -E 'xecjk|physics|booktabs'
```

**MacTeX 用戶**：預設已安裝所有套件。

## 🔍 常見問題

### 1. 中文無法顯示
**原因**：系統缺少指定字體。

**解決方法**：
```bash
# macOS：查看可用字體
fc-list :lang=zh

# 修改 .tex 文件中的字體設定
\setCJKmainfont{[你系統中的字體名稱]}
```

### 2. 圖片無法加載
**錯誤訊息**：`File 'figure1_convergence.png' not found`

**解決方法**：
1. 確認圖片與 .tex 檔案在同一目錄
2. 或修改圖片路徑為絕對路徑

### 3. 編譯卡住不動
**原因**：LaTeX 遇到錯誤等待輸入。

**解決方法**：
- 按 `Ctrl+C` 中止
- 使用 `-interaction=nonstopmode` 參數：
  ```bash
  xelatex -interaction=nonstopmode final_report.tex
  ```

### 4. 參考文獻格式調整
如需使用 BibTeX 管理文獻，可將參考文獻部分改為：
```latex
\bibliographystyle{plain}
\bibliography{references}  % 需要 references.bib 檔案
```

## 📄 輸出檔案

成功編譯後會生成：
- **final_report.pdf**: 最終報告（這是你要交的檔案）
- **.aux, .log, .toc** 等：輔助檔案（可刪除）

## 🎨 自訂調整

### 修改頁邊距
```latex
\usepackage[margin=2.5cm]{geometry}  % 修改 2.5cm 為你需要的數值
```

### 修改行距
```latex
\onehalfspacing  % 1.5 倍行距
% 或
\doublespacing   % 2 倍行距
```

### 添加更多圖表
```latex
\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{your_figure.png}
\caption{你的圖片說明}
\label{fig:your_label}
\end{figure}
```

## 📚 進階功能

### 添加數學定理環境
```latex
\newtheorem{theorem}{定理}[section]
\newtheorem{lemma}[theorem]{引理}
\newtheorem{proposition}[theorem]{命題}

% 使用
\begin{theorem}
你的定理內容
\end{theorem}
```

### 代碼高亮配色
可調整 `lstset` 參數自訂配色方案（第 23-31 行）。

---

## ✅ 快速檢查清單

編譯前請確認：
- [ ] 已修改作者姓名和學號
- [ ] 三張圖片檔案存在
- [ ] 中文字體設定正確
- [ ] 已安裝 XeLaTeX
- [ ] 已安裝所需 LaTeX 套件

編譯後請檢查：
- [ ] PDF 可正常開啟
- [ ] 中文正確顯示
- [ ] 圖片正確嵌入
- [ ] 目錄和頁碼正確
- [ ] 公式正確渲染
- [ ] 參考文獻格式正確

---

如有任何問題，請參考 [LaTeX Wikibook](https://en.wikibooks.org/wiki/LaTeX) 或使用 [TeX Stack Exchange](https://tex.stackexchange.com/)。
