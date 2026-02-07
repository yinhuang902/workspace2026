snoglode-main-2-LG - 副本 (2)
似乎改的很成功，1e-1顺利 运行，保存一下以修改至1e-2版本


snoglode-main-2-LG - 副本 (3)
改1e-2问题改动中的版本，先保存一下，说要做给 subgradient 做归一化
加一个 μ 的 box / clipping（更粗暴但很稳）
用 Polyak step（需要 UB/LB）


snoglode-main-2-LG 
做了上述东西并符合paper迭代公式的东西（也可能不符合）
不确定，需要再double check

