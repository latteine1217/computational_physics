# TRG method

$$
W_{ss'}=e^{\beta Jss'}=\left[
\begin{array}{cc}
e^{\beta J} & e^{-\beta J} \\
e^{-\beta J} & e^{\beta J} \\
\end{array}
\right]
$$

$$ [W] = [M][M^T] $$

$$ 
[M] =  \left[
\begin{array}{cc}
\sqrt{\cosh{\beta J}} & \sqrt{\sinh{\beta J}} \\
\sqrt{\cosh{\beta J}} & -\sqrt{\sinh{\beta J}} \\
\end{array}
\right] 
$$

$$ T_{ijkl} = \sum_s M_{si}M_{sj}M_{sk}M_{sl} $$

- i : T向上接 
- j : T向左接 
- k : T向下接 
- l : T向右接
(逆時針)

視覺化範例：
 \       i
 \       |
 \      ---
 \ j - | T | - l 
 \      ---
 \       |
 \       k
今天設定一個2x2的ising model，左右為週期性邊界

設定：
T_1 - T_2
 |     |
T_4 - T_3

根據定義，會得到T為：
T_{s1,s2,s3,s4} T_{s_5,s_4,s_6,s_2}
T_{s3,s8,s1,s7} T_{s_6,s_7,s_5,s_8}

# task 
for 2D ising model
- construct rank-2 tensor M
- construct rank-4 tensor T from M
- when temp=1, J=1, elements of T includes
    [[4.762200e+00 0.0e+00 0.0e+00 3.62686e+00]
     [0.0e+00 3.62686e+00 3.62686e+00 0.0e+00]
     [0.0e+00 3.62686e+00 3.62686e+00 0.0e+00]
     [3.62686e+00 0.0e+00 0.0e+00 4.762200e+00]]
- contract four T tensors to get Z_{2x2}
