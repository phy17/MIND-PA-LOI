# MIND Fusion 改进方案：动态时空走廊与势场设计

本文档详细描述了为了解决"固定走廊导致的误报/漏报"以及"行为死板（要么无视要么急刹）"问题而提出的改进方案。方案核心在于引入**环境自适应的动态走廊**和**基于速度的势场代价函数**。

## 1. 核心改进一：动态时空走廊 (Dynamic Spatiotemporal Corridor)

为了解决固定宽度走廊在窄路（易误报）和宽路（易漏报）上的局限性，我们提出 **Environment-Adaptive Spatiotemporal Risk Corridor (EA-SRC)**。走廊分为内层和外层，其宽度 $W$ 根据道路几何动态调整。

### 1.1 符号定义
*   $W_{ego}$: 自车宽度。
*   $D_L$: 当前位置到左侧道路边界（如路沿、墙壁）的横向距离。
*   $D_R$: 当前位置到右侧道路边界的横向距离。
*   $\delta_{margin}$: 安全余量（例如 0.5m）。
*   $W_{scan\_max}$: 最大扫描距离（例如 5.0m），防止在空旷广场过度搜索。

### 1.2 宽度计算公式

**内层走廊 (Inner Layer - Absolute Safety)**
内层走廊用于保障物理安全，其边界必须紧贴车身及其安全余量，但在任何情况下都不得超出物理道路边界。

$$
W_{inner}^L = \min(D_L, \frac{W_{ego}}{2} + \delta_{margin})
$$

$$
W_{inner}^R = \min(D_R, \frac{W_{ego}}{2} + \delta_{margin})
$$

**外层走廊 (Outer Layer - Potential Risk)**
外层走廊用于探测潜在的鬼探头风险（幻影），其范围应尽可能覆盖盲区，但同样受到物理道路边界的限制。

$$
W_{outer}^L = \min(D_L, W_{scan\_max})
$$

$$
W_{outer}^R = \min(D_R, W_{scan\_max})
$$

---

## 2. 核心改进二："谨慎但不恐惧"的风险势场 (Cautious Risk Potential Field)

现有的 Cost 函数仅考虑距离，导致车辆对远处的风险无感，而对近处的风险反应过度（急刹）。我们需要引入一个与**速度相关**的势场，使车辆表现出人类司机的"滑行备刹"行为。

### 2.1 双层风险源模型 (Dual-Layer Risk Source)

当检测到视野遮挡产生的鬼探头点时，我们在该位置生成两个同心的风险源：

1.  **核心风险核 (Critical Kernel)**: 代表"绝对危险区"，模拟物理障碍物。
2.  **势能缓冲区 (Potential Field)**: 代表"潜在危险区"，模拟驾驶员的心理压力场。

### 2.2 代价函数设计 (Cost Function)

总风险代价 $J_{risk}$ 由两部分组成：

$$
J_{risk}(s) = J_{crit} + J_{pot}
$$

#### 1. 核心风险代价 (Critical Cost)
该项用于迫使车辆在物理上避开风险点（例如稍微向左打方向盘，Nudging）。

$$
J_{crit} = w_{static} \cdot \exp\left(-\frac{d^2}{2\sigma_{crit}^2}\right)
$$

*   $w_{static}$: 静态权重，取较大值（如 100），确保车辆通过转向避让。
*   $\sigma_{crit}$: 核心半径，取较小值（如 0.8m）。
*   $d$: 车辆到风险点的欧氏距离。

#### 2. 势能缓冲代价 (Potential Cost) - 速度相关
该项用于迫使车辆在接近风险点时降低速度。其权重与车速的平方成正比。

$$
J_{pot} = (k_{v} \cdot v_{ego}^2) \cdot \exp\left(-\frac{d^2}{2\sigma_{pot}^2}\right)
$$

*   $v_{ego}$: 自车当前速度。
*   $k_{v}$: 速度敏感系数。
*   $\sigma_{pot}$: 势场半径，取较大值（如 2.5m）。

### 2.3 行为分析

通过上述公式，我们可以实现以下行为模式：

*   **场景 A：高速接近 (High Speed Approach)**
    *   设 $v_{ego} = 10m/s (36km/h)$。
    *   由于 $v_{ego}^2$ 很大，$J_{pot}$ 变得非常大。
    *   优化器为了最小化总代价，会选择降低 $v_{ego}$（例如降至 5m/s），从而大幅降低 $J_{pot}$。
    *   **结果**：车辆在远处通过**提前松油门**来应对风险，而不是急刹。

*   **场景 B：低速蠕行 (Low Speed Creeping)**
    *   设 $v_{ego} = 2m/s (7.2km/h)$。
    *   此时 $v_{ego}^2$ 很小，$J_{pot}$ 几乎可以忽略。
    *   车辆可以安全地驶入势场范围（$\sigma_{pot}$），直到逼近核心区（$\sigma_{crit}$）。
    *   **结果**：车辆可以贴近盲区观察，不会被"吓停"，实现了在拥挤窄路中的通行能力。

---

## 3. 实现步骤 (Implementation Steps)

1.  **感知层 ([utils.py](file:///Users/phy/Desktop/MIND/planners/mind/utils.py))**:
    *   修改 [get_semantic_risk_sources](file:///Users/phy/Desktop/MIND/planners/mind/utils.py#561-787)，接收地图边界信息 $(D_L, D_R)$。
    *   根据上述公式计算动态的 $W_{outer}$，只保留在走廊内的风险点。
    *   输出包含 `CRITICAL` 和 `POTENTIAL` 两种类型的风险源列表。

2.  **规划层 ([planner.py](file:///Users/phy/Desktop/MIND/planners/mind/planner.py))**:
    *   修改 [evaluate_traj_tree](file:///Users/phy/Desktop/MIND/planners/mind/planner.py#297-379)，解析两种风险源。
    *   计算 $J_{crit}$：只与距离有关。
    *   计算 $J_{pot}$：引入 $v_{ego}^2$ 项。
    *   将两者相加作为最终 $J_{risk}$。
