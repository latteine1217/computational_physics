# Makefile for LaTeX compilation
# 使用 XeLaTeX 編譯中文文檔

# 主文件名稱（不含 .tex 擴展名）
MAIN = final_report

# 編譯器
LATEX = xelatex

# 編譯選項
LATEXFLAGS = -interaction=nonstopmode -halt-on-error

# PDF 查看器（根據系統調整）
# macOS
VIEWER = open
# Linux
# VIEWER = evince
# Windows
# VIEWER = start

# ==================== 主要目標 ====================

.PHONY: all clean view help

# 預設目標：編譯 PDF
all: $(MAIN).pdf

# 編譯 PDF（執行兩次以更新目錄和交叉引用）
$(MAIN).pdf: $(MAIN).tex
	@echo "=== 第一次編譯 ==="
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	@echo ""
	@echo "=== 第二次編譯（更新目錄和引用）==="
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	@echo ""
	@echo "✅ 編譯完成！輸出檔案：$(MAIN).pdf"

# 清理輔助檔案
clean:
	@echo "清理輔助檔案..."
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).toc $(MAIN).synctex.gz
	rm -f $(MAIN).fdb_latexmk $(MAIN).fls $(MAIN).xdv
	@echo "✅ 清理完成！"

# 完全清理（包括 PDF）
cleanall: clean
	rm -f $(MAIN).pdf
	@echo "✅ 已刪除所有生成檔案！"

# 編譯並查看
view: $(MAIN).pdf
	$(VIEWER) $(MAIN).pdf &

# 快速重新編譯（不清理）
rebuild: $(MAIN).pdf

# 完全重新編譯（先清理）
fresh: cleanall all

# 檢查圖片是否存在
check-figures:
	@echo "檢查必要的圖片檔案..."
	@test -f figure1_convergence.png && echo "✅ figure1_convergence.png" || echo "❌ 缺少 figure1_convergence.png"
	@test -f figure2_error_temperature.png && echo "✅ figure2_error_temperature.png" || echo "❌ 缺少 figure2_error_temperature.png"
	@test -f figure3_heat_capacity.png && echo "✅ figure3_heat_capacity.png" || echo "❌ 缺少 figure3_heat_capacity.png"

# 生成圖片（需要 Python 程式）
generate-figures:
	@echo "生成期末報告圖表..."
	python trg_final_project.py 1
	python trg_final_project.py opt
	@echo "✅ 圖表生成完成！"

# 使用 latexmk（如果已安裝）
latexmk: $(MAIN).tex
	latexmk -xelatex -interaction=nonstopmode $(MAIN).tex

# 幫助訊息
help:
	@echo "=== LaTeX 編譯 Makefile 使用說明 ==="
	@echo ""
	@echo "可用指令："
	@echo "  make              - 編譯 PDF（預設）"
	@echo "  make all          - 同 make"
	@echo "  make clean        - 刪除輔助檔案（保留 PDF）"
	@echo "  make cleanall     - 刪除所有生成檔案（包括 PDF）"
	@echo "  make view         - 編譯並開啟 PDF"
	@echo "  make rebuild      - 快速重新編譯"
	@echo "  make fresh        - 完全重新編譯（先清理）"
	@echo "  make check-figures - 檢查圖片檔案是否存在"
	@echo "  make generate-figures - 執行 Python 程式生成圖表"
	@echo "  make latexmk      - 使用 latexmk 自動編譯"
	@echo "  make help         - 顯示此幫助訊息"
	@echo ""
	@echo "注意事項："
	@echo "  1. 首次編譯前請修改 $(MAIN).tex 中的個人資訊"
	@echo "  2. 確保已安裝 XeLaTeX 和必要的 LaTeX 套件"
	@echo "  3. 確保圖片檔案存在（可用 'make check-figures' 檢查）"
	@echo "  4. 如系統非 macOS，請修改 VIEWER 變數"
	@echo ""
