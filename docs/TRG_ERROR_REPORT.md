# TRG 圖一誤差異常問題報告

## 一、問題描述
在 `trg_final_project.py` 中，圖一（T=Tc 的誤差 vs iteration）出現以下異常：
- 誤差明顯過大
- 提高 `chi` 時誤差不下降（幾乎無差異）

這代表 TRG 的粗粒化流程可能在結構上就錯誤，使得更高的張量截斷維度也無法改善結果。

---

## 二、理論背景（正確的 TRG 拓撲）
Levin–Nave TRG 的核心步驟是：
1. 將 rank-4 張量沿兩種方向做 SVD 分解
2. 產生四個 rank-3 張量（C0、C1、C2、C3）
3. 依照固定的 plaquette 拓撲進行收縮，形成新的 coarse-grained 張量

正確的收縮拓撲為：
- C0.r = C1.l
- C1.d = C2.u
- C2.l = C3.r
- C3.u = C0.d

這代表每個張量的「物理方向」必須連到正確的鄰邊。一旦連錯，粗粒化就不再對應原本的 Ising 模型，而變成另一個錯誤的張量網路。

---

## 三、原本的實作方式（問題來源）
`trg_final_project.py` 原本在 `_trg_step` 的收縮是使用分步 `einsum`：

```python
# 步驟 1: C0[a,r,d] 與 C1[b,d,l] 收縮 d
temp1 = np.einsum('ard,bdl->arbl', C0, C1)

# 步驟 2: temp1[a,r,b,l] 與 C2[u,l,c] 收縮 l
temp2 = np.einsum('arbl,ulc->arbuc', temp1, C2)

# 步驟 3: temp2[a,r,b,u,c] 與 C3[u,r,e] 收縮 u 和 r
coarse_tensor = np.einsum('arbuc,ure->abce', temp2, C3)
```

這段寫法沒有對齊正確拓撲索引，實際上讓某些方向接到錯的邊。結果是：
- 張量網路「連錯邊」
- 粗粒化步驟不再對應 Ising 模型
- 即使提高 `chi`，也只是近似錯誤的網路，誤差不會下降

---

## 四、修正方式（與 cytnx 版本對齊）
改為一次性 `einsum`，明確對應正確拓撲（替換原本分步收縮）： 

```python
coarse_tensor = np.einsum(
    'axw,byx,yzc,wze->abce',
    C0, C1, C2, C3,
    optimize=True,
)
```

對應關係如下：
- C0[a,x,w]
- C1[b,y,x]
- C2[y,z,c]
- C3[w,z,e]

收縮關係：
- x 對應 C0.r = C1.l
- y 對應 C1.d = C2.u
- z 對應 C2.l = C3.r
- w 對應 C3.u = C0.d

此收縮拓撲與 `11410PHYS401200/TRG/trg.py` 的 cytnx 版本一致。

---

## 五、為什麼這樣改就會改善誤差
TRG 的本質是對「正確拓撲的張量網路」做壓縮與粗粒化。若拓撲接錯，物理結構被破壞，任何截斷都無法改善結果。

修正拓撲後，粗粒化流程才會忠實反映 2D Ising 的有效自由度，因此：
- 誤差才會隨迭代下降
- 提高 `chi` 才會逐步改善自由能誤差

也就是說：
> 錯誤拓撲導致 chi 失效；修正拓撲後 chi 才恢復作用。

---

## 六、驗證與後續建議
1. 重新生成圖一，觀察不同 `chi` 是否開始分離
2. 建議 `iterations >= 3`，因為 `chi` 的效果通常在較深層粗粒化才顯現
3. 可額外輸出每步 `chi_eff` 與奇異值譜衰減，用於量化截斷品質

---

## 七、結論
圖一誤差異常與 `chi` 失效的主要原因是 **TRG 收縮拓撲索引接錯**。修正 `einsum` 收縮後，粗粒化拓撲對齊標準 TRG，理論上誤差應能隨 `chi` 改善。建議重新生成圖一以確認修正效果。
