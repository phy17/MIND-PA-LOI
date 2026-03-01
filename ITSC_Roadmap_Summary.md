# ITSC 投稿工作总结与路线图

## 1. 已完成工作 (Work Done)

我们主要完成了从 **"工程实现"** 到 **"学术创新"** 的思维转型与基础建设：

1.  **痛点挖掘 (Problem Formulation)**：
    *   识别出 MIND 算法作为 End-to-End / Data-Driven 方法的致命弱点：**对未知盲区 (OOD) 的不可知论 (Agnosticism)**。如果没有历史观测数据，MIND 无法预测风险，导致潜在的安全隐患。
2.  **初步方案 (Initial Approach)**：
    *   实现了基于几何规则的 `Ghost Probe` 机制 (`utils.py`)，能够识别盲区并计算风险源。
    *   采用了 **Risk Field (Soft Constraint)** 的方式，通过 Cost Function 引导车辆避让。
3.  **理论升维 (Theoretical Upgrade)**：
    *   通过深度分析 MARC 论文，确定了将 Engineering Trick (Cost Function) 升级为 **Knowledge-Driven Explicit Contingency Planning (知识驱动的显式应急规划)** 的战略。
    *   提出了 **"虚拟鬼探头分支 (Virtual Ghost Branching)"** 的核心创新点，旨在实现概率层面上的安全保证 (Probabilistic Safety Guarantee)。

---

## 2. 后续行动计划 (Action Plan)

### Phase 1: 核心实现 (Implementation) - [Critical]
*   **目标**: 让代码具备“生成虚拟分支”的能力。
*   **任务**:
    1.  修改 `ScenarioTreeGenerator`，在 `branch_aime` 阶段注入逻辑。
    2.  实现 `generate_ghost_branches`，人工合成 Ghost Agent 的轨迹与协方差。
    3.  调试优化器，确保 iLQR 能正确处理这个新增的分支。

### Phase 2: 实验验证 (Validation) - [Critical]
*   **目标**: 证明新方法比 Baseline 更好。
*   **实验设计**:
    *   **场景**: 典型的十字路口遮挡场景。
    *   **指标**: 碰撞率 (Collision Rate)、通过时间 (Time to Goal)、最大减速度 (Max Jerk)。
    *   **对比组**:
        1.  **Pure MIND**: 激进，碰撞率高。
        2.  **MIND + Risk Field** (Current): 绕行但可能刹不住，轨迹不平滑。
        3.  **Ours (Virtual Branching)**: 提前预减速 (Pre-braking)，平滑且安全。

### Phase 3: 论文进阶 (Refinement) - [Bonus]
*   **目标**: 提升论文的理论深度，冲击高分录用。
*   **策略**:
    *   **自适应概率 (Adaptive Probability)**: $P_{ghost} = f(Context)$。
    *   **参数敏感性分析**: 分析不同风险容忍度对驾驶行为的影响。
    *   **CVaR**: 引入更高级的风险度量（视时间而定）。

---

**当前状态**: 处于 **Phase 1** 的起步阶段。代码逻辑已设计好，等待落地。
