# MIND 项目代码分析文档

## 1. 项目概览
**MIND** (Multi-modal Integrated PredictioN and Decision-making) 是一个自动驾驶仿真和规划框架。它的核心目标是通过**整合预测和决策**来解决自动驾驶中复杂的交互问题。

与传统的“先预测，后规划”的流程不同，MIND 使用**场景树 (Scenario Tree)** 来表示环境的多模态演变，并通过**轨迹树 (Trajectory Tree)** 生成对应于不同未来分支的最佳驾驶策略。

## 2. 目录结构分析

| 路径 | 描述 |
| :--- | :--- |
| `run_sim.py` | **入口点**。启动仿真。接受 `--config` 参数。 |
| `simulator.py` | **仿真核心**。负责主循环，加载地图/代理，更新状态和渲染。 |
| `agent.py` | **代理定义**。定义了 `NonReactiveAgent` (回放代理) 和 `MINDAgent` (智能代理)。 |
| `planners/mind/` | **MIND 算法实现**。 |
| `planners/mind/planner.py` | **规划器入口**。连接场景树生成和轨迹树优化。 |
| `planners/mind/scenario_tree.py` | **场景树生成器**。使用神经网络 (NN) 预测多模态的未来。 |
| `planners/mind/trajectory_tree.py` | **轨迹树优化器**。使用 iLQR 求解最优控制。 |
| `planners/ilqr/` | **优化器**。基于 Theano (自动微分) 的 iLQR 算法实现。 |
| `configs/` | **仿真配置**。例如 `demo_1.json`，定义场景和代理参数。 |

## 3. 核心工作流程

系统在一个严格的 **感知 -> 规划 -> 控制** 循环中运行：

1.  **初始化**: `Simulator` 加载 Argoverse 2 语义地图 (`SemanticMap`) 和历史轨迹数据 (`ArgoAgentLoader`)。
2.  **循环 (`Simulator.run_sim`)**:
    *   **观察 (`observe`)**: 每个代理收集周围环境信息。
    *   **规划 (`plan`)**: 核心逻辑所在。适用于 `CustomizedAgent` (如 `MINDAgent`)。
    *   **步进 (`step`)**: 更新物理状态 (运动学)。
    *   **渲染 (`render_video`)**: 可视化结果。

## 4. 深度解析：MIND 算法 (“魔法”所在)

核心逻辑位于 `MINDPlanner` (`planners/mind/planner.py`)，分为两个主要阶段：

### 第一阶段：场景树生成 (`ScenarioTreeGenerator`)
这一步解决“未来可能发生什么？”的问题。

*   **输入**: 历史观测 (轨迹、位置、速度) + 地图信息。
*   **模型**: 一个深度神经网络 (在 `configs/networks/` 中引用) 预测周围代理未来轨迹的概率。
*   **分支策略 (`branch_aime`)**:
    *   它不预测单一的未来，而是预测**多种模式** (Modes)。
    *   它构建一棵树，其中每个分支代表一种可能的场景演变。
    *   **剪枝与合并**: 合并相似的场景以保持树的可管理性。
    *   **交互**: 能够处理自车 (Ego) 的未来行为如何影响他人 (尽管在此代码库中，重点似乎在于针对多模态预测进行鲁棒规划)。

### 第二阶段：轨迹树优化 (`TrajectoryTreeOptimizer`)
这一步解决“面对这些未来，我该如何驾驶？”的问题。

*   **转换**: 将 `场景树` 转换为逻辑上的 `代价树 (Cost Tree)`。
*   **优化 (iLQR)**: 使用 **迭代线性二次调节器 (iLQR)**。
    *   **代价函数 (Cost Function)**:
        *   **舒适度**: 惩罚过大的加速度/转向。
        *   **效率**: 鼓励达到目标速度。
        *   **目标车道**: 惩罚偏离参考车道。
        *   **安全 (势场法)**:
            *   **障碍物**: 其他代理被建模为高代价的势场 (`w_exo`)。
            *   **车道边界**: 惩罚驶出道路。
    *   **求解器**: `planners.ilqr.solver.iLQR` 在树结构上求解最小化这些代价的控制序列 ($u$)。
    *   **动力学**: 使用运动学自行车模型 (在 `_get_dynamic_model` 中定义)。

### 第三阶段：选择与执行
*   规划器根据代价评估优化后的轨迹树。
*   选择**最佳分支**。
*   输出下一个时间步的控制命令 (加速度，转向角)。

## 5. 关键原理总结

1.  **树结构规划**: 能处理“如果情况 A 发生，我做 X；如果情况 B 发生，我做 Y”。这比单一轨迹规划更智能。
2.  **学习 + 优化**:
    *   **深度学习** 处理人类行为预测的*不确定性*和*复杂性*。
    *   **控制理论 (iLQR)** 处理车辆动力学的*精确性*和*安全性*。
3.  **以地图为中心**: 严重依赖 `SemanticMap` (车道) 进行引导 (`tgt_lane`)。

## 6. 如何运行

基于 `setup_mac.sh`，该项目是为 Mac M系列芯片配置的：

1.  **环境**: 需要 Python 3.10 和专用库 (`av2`, `Theano`, `torch`)。
2.  **执行**:
    ```bash
    # 激活虚拟环境
    source venv_mind_cpu/bin/activate
    # 运行演示
    python run_sim.py --config configs/demo_1.json
    ```
3.  **输出**: 视频将在 `outputs/` 目录中生成。
