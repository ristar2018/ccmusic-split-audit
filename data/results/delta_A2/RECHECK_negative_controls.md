# 负对照 A2−A0 单 seed 行的复核（2026-11-06）

`e1_delta.csv` 中 chest_falsetto / erhu_playing_tech / GZ_IsoTech 六行均为 seed0 单 seed，`seed_range_pt` 为空。
其中 **chest_falsetto/swin_t A2−A0 = +5.57 pt [95% CI +0.62, +10.31]（脚本 meets_sesoi=是）不可采信**，理由：

1. n_seeds=1 → 无 seed 平均、无 seed 极差；同表 20 个 3-seed 行的极差 0.87–8.29 pt（中位 2.1），5.57 落在单 seed 波动内。
2. A0 与 A2 不是同一数据池：A0 = e1f_drv（n_train 768 / n_val 190），A2 = e1f2（666 / 83）；Q_test 相同（n=257，y/records 逐元素一致），但 Δ 混入划分差异，非纯臂效应。
3. A0 该组 Q_test F1 0.8261 比自身 val_best 0.8940 低 6.8 pt（同批 gap 中偏负）；A2 gap −0.3 pt。
4. CI 宽 9.7 pt，下界 0.62 刚过 0；A2 不属确认性族（CONFIRM_ARM=A1），无 Holm 支撑。
5. 逐类 ΔF1：+3.55 / +3.98 / +4.70 / +10.06，非单类坍塌，但类 4 recall 0.717→0.804 贡献约一半。

判定：该行记「单 seed 观察值，待复核」，**不写「过 SESOI 门」**。复核动作 = A0/A2 同划分、各 3 seed 重跑封存后重算该行。
其余五行为 |Δ|<1.6 pt，与负对照预期一致。
