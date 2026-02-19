# ITSC 进阶冲刺策略 (Advanced Strategy)

为了将论文录用概率从“有竞争力”提升到“稳中 (Strong Accept)”，我们需要在 **“虚拟分支” (Step 1)** 的基础上，叠加更高阶的创新点和实验设计。

## 1. 理论深度升级 (Theoretical Depth)

### 策略 A: 自适应风险概率 (Adaptive Risk Probability) [性价比最高]
*   **现状**: 鬼探头分支的发生概率 $P_{ghost}$ 是固定的（如 constant 10%）。
*   **改进**: 让 $P_{ghost}$ 成为环境感知的函数。
    $$ P_{ghost} = \sigma(w_1 \cdot Area_{blind} + w_2 \cdot Speed_{ego} + w_3 \cdot Semantics) $$
    *   **语义感知**: 如果地图标注为“School Zone”或“Crosswalk”，$P_{ghost}$ 自动激增。
    *   **动态感知**: 盲区面积越大，$P_{ghost}$ 越高。
*   **Paper Claim**: "Context-Aware Probabilistic Risk Assessment"。

### 策略 B: 引入 CVaR 风险度量 (The MARC Way) [硬核数学]
*   **现状**: 优化器最小化期望代价 (Expected Cost)。
*   **改进**: 使用 CVaR (Conditional Value-at-Risk) 关注尾部风险。
    *   不仅仅是加权平均，而是对“最坏的那 5% 的情况”赋予极高权重。
    *   这需要修改 iLQR 的 Cost Function 定义。
*   **Paper Claim**: "Risk-Averse Planning via CVaR Optimization"。

---

## 2. 实验设计升级 (Experimental Design)

审稿人最看重的是 **Baseline 的对比**。你需要构建一个“打脸”的故事线。

| 方法 (Method) | 表现 (Performance) | 缺陷 (Drawback) |
| :--- | :--- | :--- |
| **Baseline 1: Rule-based** | 极度保守，遇盲区必停 | 通行效率极低 (Low Efficiency) |
| **Baseline 2: Pure MIND** | 极度激进，无视盲区 | 安全性极差 (Unsafe, Collision) |
| **Baseline 3: MIND + Simple Risk Field** | 绕行但不减速 | 虽有意识，但无法保证急刹成功 (Soft Constraint Failure) |
| **Ours: Virtual Branching** | **完美平衡** | **既能预减速备刹，又不至于完全停车** |

**核心图表**:
*   画出 **效率 (Efficiency)** vs **安全性 (Safety)** 的 Pareto 曲线。
*   证明你的方法处于曲线的最优前沿 (Pareto Frontier)。

---

## 3. 执行路线图 (Execution Roadmap)

1.  **Step 1 (必做 - Code)**: 实现 `ScenarioTreeGenerator` 中的 **Virtual Ghost Branching**。这是所有一切的物理基础。
2.  **Step 2 (选做 - Algo)**: 在代码中简单实现 **自适应概率函数**（即使只是基于盲区距离的一个简单线性函数，写在论文里也很好看）。
3.  **Step 3 (必做 - Exp)**: 按照上面的表格设计实验，跑数据，画图。

---

**决策点**:
建议立即开始 **Step 1** 的代码修改。后续的自适应概率可以在此基础上轻松叠加。
