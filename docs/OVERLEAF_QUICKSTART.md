# 📤 Overleaf 快速上傳指南

## 🎯 5分鐘內完成編譯

### 步驟 1：準備文件 (已完成 ✅)

您需要上傳的文件：
```
✅ final_report.tex              (主文件)
✅ figure1_convergence.png       (圖1)
✅ figure2_error_temperature.png (圖2)
✅ figure3_heat_capacity.png     (圖3)
```

### 步驟 2：前往 Overleaf

1. 打開瀏覽器，前往：https://www.overleaf.com/
2. 登入或註冊帳號（免費版即可）

### 步驟 3：創建新專案

**方法 A：上傳 ZIP 檔案（推薦）**

1. 在本地打包文件：
   ```bash
   cd /Users/latteine/Documents/coding/computational_physics
   zip trg_report.zip final_report.tex figure*.png
   ```

2. 在 Overleaf 點擊：
   - "New Project" → "Upload Project"
   - 選擇 `trg_report.zip`

**方法 B：手動上傳（逐個文件）**

1. 點擊 "New Project" → "Blank Project"
2. 命名專案：`TRG Final Report`
3. 刪除預設的 `main.tex`
4. 點擊 "Upload" 圖標（在左側文件列表上方）
5. 逐個上傳 4 個文件

### 步驟 4：設置編譯器

1. 點擊左上角的 "Menu" 按鈕
2. 找到 "Compiler" 下拉選單
3. 選擇 **pdfLaTeX**
4. 關閉選單

### 步驟 5：編譯

1. 點擊綠色的 "Recompile" 按鈕
2. 等待編譯完成（約10-20秒）
3. PDF 將顯示在右側預覽窗格

### 步驟 6：更新學生資訊

1. 在左側文件列表點擊 `final_report.tex`
2. 找到第 51-52 行：
   ```latex
   \author{Student ID: [Your ID] \\ Name: [Your Name]}
   ```
3. 修改為您的實際資訊，例如：
   ```latex
   \author{Student ID: 11234567 \\ Name: 張三}
   ```
4. 保存（Ctrl+S 或 Cmd+S）
5. 再次點擊 "Recompile"

### 步驟 7：下載 PDF

1. 點擊 PDF 預覽窗格上方的 "Download PDF" 按鈕
2. 或點擊 "Menu" → "Download" → "PDF"

---

## 🔧 常見問題排除

### ❓ 問題：圖片無法顯示

**解決方案：**
1. 檢查文件名是否完全一致（包括大小寫）
2. 確保圖片與 .tex 文件在同一目錄
3. 在 Overleaf 文件列表中查看圖片是否成功上傳

### ❓ 問題：編譯超時

**解決方案：**
1. 點擊 "Menu" → "Compiler" → 選擇 "pdfLaTeX"
2. 清除緩存：Menu → "Logs and output files" → "Clear cached files"
3. 再次編譯

### ❓ 問題：某些符號顯示錯誤

**解決方案：**
1. 確認編譯器是 pdfLaTeX（不是 XeLaTeX 或 LuaLaTeX）
2. 檢查是否遺漏上傳某些文件

### ❓ 問題：參考文獻未顯示

**解決方案：**
- 編譯兩次（第一次生成引用，第二次插入引用）
- Overleaf 通常會自動處理，但手動點擊兩次 "Recompile" 確保萬無一失

---

## 📊 檢查清單

編譯成功後，請檢查以下內容：

### 文檔結構
- [ ] 目錄（Table of Contents）顯示正確
- [ ] 所有 6 個主要章節都存在
- [ ] 所有 3 個附錄都存在

### 圖表
- [ ] Figure 1: Convergence (第 488 頁附近)
- [ ] Figure 2: Error vs Temperature (第 529 頁附近)
- [ ] Figure 3: Heat Capacity (第 552 頁附近)
- [ ] 所有 5 個表格顯示正確

### 數學公式
- [ ] 公式編號正確
- [ ] 沒有「?」符號（表示未定義的引用）
- [ ] 所有希臘字母顯示正常

### 程式碼
- [ ] 所有 5 個程式碼區塊格式正確
- [ ] 語法高亮顯示正常
- [ ] 行號顯示清楚

### 內容完整性
- [ ] Section 2.4: Alternative Methods 顯示完整
- [ ] Section 5.2: Chi Saturation Analysis 有 6 個子章節
- [ ] Appendix B 有完整程式碼（4個函數）

---

## 🎨 優化建議（可選）

### 調整頁邊距（如果內容太緊）
在第 15 行，將 `margin=2.5cm` 改為 `margin=2cm`：
```latex
\usepackage[margin=2cm]{geometry}
```

### 調整行距（如果想要更緊湊）
在第 17 行，將 `\onehalfspacing` 改為 `\singlespacing`：
```latex
\singlespacing
```

### 添加頁碼
如果需要不同的頁碼格式，在 preamble 添加：
```latex
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\rfoot{\thepage}
```

---

## 📥 本地編譯備用方案

如果 Overleaf 遇到問題，可以在 Mac 上本地編譯：

### 安裝 BasicTeX（輕量級，約 80MB）
```bash
brew install --cask basictex
# 安裝後重啟終端
sudo tlmgr update --self
sudo tlmgr install collection-fontsrecommended
sudo tlmgr install booktabs physics
```

### 編譯
```bash
cd /Users/latteine/Documents/coding/computational_physics
pdflatex final_report.tex
pdflatex final_report.tex
open final_report.pdf
```

---

## 🎯 預期結果

成功編譯後，您應該獲得：

- **文件頁數：** 約 40-50 頁
- **檔案大小：** 約 800KB - 1.5MB
- **目錄深度：** 3 層（Section → Subsection → Subsubsection）
- **圖表數量：** 3 張圖 + 5 個表格

---

## ✅ 完成標誌

當您在 Overleaf 看到以下內容時，表示成功：

1. **左側：** 綠色勾號（表示無編譯錯誤）
2. **右側：** 完整的 PDF 預覽
3. **目錄：** 顯示所有章節標題
4. **最後一頁：** 顯示參考文獻

---

**需要幫助？** 如果遇到任何問題，請告訴我具體的錯誤訊息！

祝編譯順利！🎉
