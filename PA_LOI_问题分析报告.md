# PA-LOI 鬼探头防御系统 — 问题分析报告

> **背景**: 本项目基于 MIND 自动驾驶框架，使用 iLQR 优化器进行轨迹规划。我们开发了 PA-LOI（Proactive Awareness - Lateral Occlusion Inference）模块，用于检测"鬼探头"风险（行人/车辆从路边停放车辆后方突然冲出）。
> 
> **当前问题**: 车辆在**没有真正鬼探头出现**的情况下，仅仅因为靠近路边停放的车辆就反复刹车-加速循环，无法正常通过危险区域。鬼探头（行人）从未被实际生成。

---

## 一、期望行为 vs 实际行为

### 期望行为
1. **经过停放车辆旁边时**：正常或略微谨慎行驶（轻微减速，如 8→5m/s），继续前进
2. **真正有行人/车辆从遮挡物后冲出时**：紧急刹车，直到完全停止
3. 两个阶段应该是**分离的**：没看到真实威胁就不应刹停

### 实际行为（v17b 实验数据）
1. Step 201-370: 车辆从 2.36m/s 缓慢加速至 3.99m/s，方向控制完美（steer≈0.01）
2. **Step 371**: TTA降到 2.89s，系统进入 BRAKE 状态，**猛刹车**（acc=−2.12 m/s²）
3. Step 391: 速度降到 2.36m/s，TTA 回升到 4.77s → 系统退回 OBSERVE 状态
4. Step 400-500: 车辆又开始缓慢加速，重复上述循环
5. **鬼探头从未被生成**（Ghost spawned: False），因为车辆永远到不了触发点

**核心矛盾**: 系统在"潜在风险"阶段就做出了"实际威胁"级别的刹车反应。

---

## 二、系统架构与代码链路

整个 PA-LOI 功能涉及以下文件，按调用顺序排列：

```
ghost_experiment.json (配置)
  → run_ghost_experiment.py (实验框架，负责生成真实的鬼探头 agent)
    → planner.py (规划器入口)
      → utils.py :: get_semantic_risk_sources() (识别路边停车 → 判断鬼探头风险)
        → utils.py :: calculate_phantom_behavior() (状态机：OBSERVE/BRAKE/PASS)
      → trajectory_tree.py :: construct_tree_from_scenario() (将风险注入 iLQR Cost)
        → potential.py :: VelocityAwareRiskPotential (计算 Cost 和梯度)
      → ilqr/cost.py (iLQR 优化器读取梯度，计算最优控制)
    → run_ghost_experiment.py :: should_spawn_ghost() (判断是否注入真实鬼探头)
```

---

## 三、各模块详细分析

### 3.1 状态机逻辑 — `calculate_phantom_behavior()`

**文件**: `planners/mind/utils.py`, 第 682-754 行

```python
def calculate_phantom_behavior(longitudinal_dist, lateral_dist, ego_vel):
    HUMAN_MAX_SPEED = 5.0     # 人类冲刺速度上限
    LOOKAHEAD_TIME = 3.0      # 前瞻时间阈值（秒）

    # 计算 TTA (Time To Arrival)
    tta_ego = longitudinal_dist / ego_vel    # ego 到达遮挡物的时间
    tta_human = lateral_dist / HUMAN_MAX_SPEED  # 行人冲出来的时间

    # 计算鬼需要的速度
    v_required = lateral_dist / tta_ego

    # ====== 状态机判断 ======

    if v_required > HUMAN_MAX_SPEED:
        state = 'OBSERVE'      # 鬼跑不到 → 安全
    elif tta_ego > LOOKAHEAD_TIME:
        state = 'OBSERVE'      # ego 还远 → 先观察
    else:
        state = 'BRAKE'        # ⚠️ 既近又能撞上 → 直接 BRAKE！
        inject_phantom = True
```

**⚠️ 问题所在**:
- 当 `tta_ego < 3.0s` 且 `v_required < 5.0m/s` 时，状态直接跳到 `BRAKE`。
- 但这个条件只说明"**如果有人冲出来**，他物理上能撞到你"——这是一个**概率性风险**，不是**已确认的威胁**。
- 实际上路边的每辆停放车辆都满足这个条件（lateral_dist 通常 1-2m，v_required 很小），所以车一接近就 BRAKE。

### 3.2 权重计算 — `get_semantic_risk_sources()`

**文件**: `planners/mind/utils.py`, 第 962-972 行

```python
# 根据状态决定权重
base_weight = 10.0
if phantom_result['state'] == 'OBSERVE':
    weight = base_weight * 0.5                              # OBSERVE: weight = 5.0
elif phantom_result['state'] == 'BRAKE':
    weight = base_weight * (1.0 + 0.1 * (ego_vel ** 2))    # BRAKE: weight = 10*(1+0.1*v²)
    # 例：v=4m/s → weight = 10*(1+1.6) = 26.0
else:  # PASS
    weight = base_weight * 0.2                              # PASS: weight = 2.0
```

**权重链路**:
- OBSERVE: weight=5.0 → 传入 trajectory_tree → `w_base = 5.0 * 0.7 = 3.5`
- BRAKE: weight=26.0 → 传入 trajectory_tree → `w_base = 26.0 * 1.0 = 26.0`

### 3.3 权重→Cost 映射 — `trajectory_tree.py`

**文件**: `planners/mind/trajectory_tree.py`, 第 122-148 行

```python
for risk in risk_sources:
    phantom_state = risk.get('phantom_state', 'BRAKE')

    # 根据幻影状态再次调整权重
    if phantom_state == 'PASS':
        w_base = risk['weight'] * 0.3
    elif phantom_state == 'OBSERVE':
        w_base = risk['weight'] * 0.7     # OBSERVE: w_base = 5.0 * 0.7 = 3.5
    else:  # BRAKE
        w_base = risk['weight']            # BRAKE: w_base = 26.0

    # 创建 VelocityAwareRiskPotential
    risk_pot = VelocityAwareRiskPotential(
        risk_pos=risk_mean,
        lane_heading=lane_heading,
        ghost_lateral=ghost_lateral,
        w_base=w_base,       # ← 这个值决定了刹车力度
        lambda_v=0.1,
        ego_half_width=1.0,
        k_steep=2.0
    )
```

同时，如果有 BRAKE 状态的风险源，还会**锁死方向盘**（第167-183行）:
```python
if is_brake_state:
    w_ctrl = np.diag([0.1, 200.0])  # 鼓励刹车(0.1)，禁止乱动方向(200.0)
```

### 3.4 Cost 函数 — `VelocityAwareRiskPotential`

**文件**: `planners/ilqr/potential.py`, 第 270-505 行

**Cost 公式**: `Cost = w_kinetic × Sigmoid(clearance) × v²`

```python
def get_potential(self, state):
    sig = self._compute_sigmoid(clearance)

    w_kinetic = 0.0                  # OBSERVE: Cost = 0（完全不影响）
    if self.w_base > 20.0:
        w_kinetic = 50.0             # BRAKE: Cost = 50 * Sigmoid * v²

    forward_vel = max(0.0, v)
    kinetic_energy = forward_vel * forward_vel
    total_cost = w_kinetic * sig * kinetic_energy
    return total_cost
```

**梯度（决定 iLQR 的控制输出）**:
```python
def get_gradient(self, state):
    w_kinetic = 0.0
    if self.w_base > 20.0:
        w_kinetic = 50.0

    sig = self._compute_sigmoid(clearance)

    # 速度梯度: dC/dv = W * S * 2v
    grad_v = w_kinetic * sig * 2.0 * v
    gradient[2] = grad_v

    # 空间梯度: BRAKE 状态下设为 0（防打转）
    if self.w_base > 20.0:
        grad_dist_factor = 0.0     # ← v17修复：BRAKE时不推方向
    else:
        grad_dist_factor = w_kinetic * dsig * (v * v)
```

**关键阈值**: `self.w_base > 20.0` 决定是 OBSERVE 还是 BRAKE 模式。
- OBSERVE (w_base=3.5): `w_kinetic=0`，**完全没有效果**
- BRAKE (w_base=26.0): `w_kinetic=50`，**极强的减速力**

### 3.5 鬼探头生成逻辑 — `should_spawn_ghost()`

**文件**: `experiments/ghost_probe/run_ghost_experiment.py`, 第 349-383 行

```python
def should_spawn_ghost(self, debug=False):
    ego_vel = ego_agent.state[2]

    # 条件1: ego 速度必须 > 2.0 m/s
    if ego_vel < self.min_ego_speed:       # min_ego_speed = 2.0
        return False

    # 条件2: 距离埋伏点 < 15m
    distance = np.linalg.norm(ego_pos - target_pos)

    # 条件3: TTA ≈ 行人穿越时间
    tta = distance / ego_vel
    trigger_threshold = time_to_cross + 0.1  # ≈ 1.15s

    if tta <= trigger_threshold and distance < 15.0:
        return True  # 生成鬼探头！
```

**触发条件**: `TTA ≤ 1.15s` 且 `距离 < 15m`。

**问题**: 由于 PA-LOI 在 TTA=3.0s 时就进入 BRAKE 状态并强制减速，ego 的 TTA 永远不会降到 1.15s，所以鬼永远不会被生成。

---

## 四、数据链路：一次完整的 BRAKE 触发过程

以 v17b 实验的 Step 371 为例：

| 步骤 | 模块 | 输入 | 输出 |
|------|------|------|------|
| 1 | `should_spawn_ghost()` | Dist≈13m, TTA≈3.3s | **不生成鬼** (TTA > 1.15s) |
| 2 | `get_semantic_risk_sources()` | 检测到 Agent 5 (停放车辆) | ghost_lateral≈1.5m, ghost_longitudinal≈12m |
| 3 | `calculate_phantom_behavior()` | long=12m, lat=1.5m, v=3.99m/s | tta_ego=3.0s, v_required=0.5 < 5.0 → **BRAKE** |
| 4 | 权重计算 | BRAKE, v=3.99 | weight = 10*(1+0.1*16) = **26.0** |
| 5 | `trajectory_tree.py` | BRAKE, weight=26 | w_base=26, w_ctrl=[0.1, 200] |
| 6 | `VelocityAwareRiskPotential` | w_base=26 > 20 | w_kinetic=**50**, Cost=50*S*v² |
| 7 | `get_gradient()` | w_kinetic=50, v=3.99 | grad_v = 50*S*2*3.99 ≈ **399** |
| 8 | iLQR | 极大的速度梯度 | acc = **-2.12** m/s² |
| 9 | 减速后 | v降到 2.36m/s | tta_ego = 12/2.36 = 5.1s > 3.0s → **OBSERVE** |
| 10 | OBSERVE | w_base=3.5 < 20 | w_kinetic=**0**, Cost=0 → 开始加速 |
| 11 | 循环回到 Step 3 | - | - |

**这就是"减速-加速无限循环"的完整机制。**

---

## 五、当前各参数设置汇总

| 参数 | 值 | 文件位置 | 说明 |
|------|-----|---------|------|
| LOOKAHEAD_TIME | 3.0s | utils.py:702 | OBSERVE→BRAKE 的 TTA 阈值 |
| HUMAN_MAX_SPEED | 5.0 m/s | utils.py:700 | 假设行人冲刺速度 |
| base_weight | 10.0 | utils.py:963 | 权重基数 |
| OBSERVE weight | 5.0 (=10*0.5) | utils.py:966 | OBSERVE 状态权重 |
| BRAKE weight | 26.0 (v=4m/s时) | utils.py:969 | BRAKE 状态权重（含 v² 项） |
| w_kinetic (OBSERVE) | 0.0 | potential.py:361 | OBSERVE 动能场权重 → **无效果** |
| w_kinetic (BRAKE) | 50.0 | potential.py:365 | BRAKE 动能场权重 → **极强减速** |
| w_base 判断阈值 | 20.0 | potential.py:364 | w_base > 20 → BRAKE模式 |
| w_exo | 10.0 | demo_2.py:81 | 静态障碍物排斥权重（原 200，已改回） |
| w_ctrl (正常) | 5.0*I | demo_2.py:70 | 默认控制权重 |
| w_ctrl (BRAKE) | [0.1, 200] | trajectory_tree.py:183 | BRAKE 时锁方向/放刹车 |
| target_velocity | 8 m/s | ghost_experiment.json:21 | 目标速度 |
| state_upper_bound[v] | 8.0 | demo_2.py:64 | 速度上限 |
| state_lower_bound[v] | 0.0 | demo_2.py:65 | 速度下限（禁止倒车） |
| trigger_distance | 15.0m | run_ghost_experiment.py:54 | 鬼生成触发距离 |
| min_ego_speed | 2.0 m/s | run_ghost_experiment.py:56 | 鬼生成最低速度 |
| pedestrian_speed | 2.5 m/s | run_ghost_experiment.py:55 | 鬼的移动速度 |
| 鬼生成 TTA 阈值 | ≈1.15s | run_ghost_experiment.py:374 | TTA需降到此值才生成鬼 |

---

## 六、v17b 实验关键数据

| Step | vel (m/s) | acc (m/s²) | steer | TTA_ego (s) | Dist (m) | State | 说明 |
|------|-----------|------------|-------|-------------|----------|-------|------|
| 201 | 2.36 | 0.07 | 0.002 | 9.33 | 22.0 | OBSERVE | 正常加速 |
| 260 | 2.62 | 0.55 | -0.006 | 7.33 | 19.6 | OBSERVE | 稳定加速 |
| 300 | 3.11 | 0.48 | -0.005 | 5.49 | 17.3 | OBSERVE | 继续加速 |
| 340 | 3.55 | 0.79 | 0.081 | 4.03 | 14.6 | OBSERVE | 接近触发区 |
| 360 | 3.84 | 0.73 | 0.051 | 3.35 | 13.1 | OBSERVE | 即将触发 |
| **371** | **3.99** | **0.66** | **0.039** | **2.89** | **~12** | **🚨 BRAKE** | **状态机跳转！** |
| 376 | 3.5 | -2.12 | 0.013 | ~3.4 | ~12 | BRAKE→OBSERVE | 猛刹车后TTA回升 |
| 391 | 2.36 | -2.12 | 0.013 | 4.77 | ~11 | OBSERVE | 已减速到安全 |
| 400 | 2.00 | -1.13 | 0.041 | 5.31 | 10.6 | OBSERVE | 继续减速中 |
| 420 | 1.72 | -0.20 | 0.041 | 5.43 | ~10 | OBSERVE | 速度基本稳定 |
| 440 | 1.64 | -0.07 | 0.030 | 5.32 | ~9.5 | OBSERVE | 开始恢复 |
| 460 | 1.70 | 0.41 | -0.008 | 4.69 | ~8.5 | OBSERVE | 又开始加速 |
| 480 | 1.96 | 0.56 | 0.117 | 3.86 | ~7.8 | OBSERVE | 接近又一次触发 |
| 500 | 2.14 | 0.62 | 0.052 | 3.29 | ~7 | OBSERVE | 实验结束 |

**最终结果**:
- Ghost spawned: **False**（鬼从未生成）
- Min Distance to Ghost: **7.279m**
- Collisions: **0**
- 全程 steer 都在 0.002-0.117 范围内（**方向控制完美，无打转**）

---

## 七、问题总结

### 根本原因
**OBSERVE 和 BRAKE 之间的反应是"全有或全无"（0 vs 50），没有中间过渡态。**

| 状态 | w_kinetic | 效果 |
|------|-----------|------|
| OBSERVE | **0.0** | 完全不影响车辆，正常行驶 |
| BRAKE | **50.0** | 极强减速，相当于急刹 |

当 TTA 在 3.0s 边界附近时，系统在 0 和 50 之间反复跳变，导致：
1. BRAKE → 猛刹车 → 速度下降 → TTA回升到>3s
2. OBSERVE → 完全放松 → 重新加速 → TTA降到<3s
3. 回到 1，形成**无限循环**

### 需要解决的核心问题
1. **"谨慎行驶"没有被实现**: OBSERVE 状态下 w_kinetic=0，完全不减速，只有到了 BRAKE 才突然暴力刹车
2. **BRAKE 的触发条件只看 TTA，不看是否真有威胁**: 路边每辆停放的车都会触发 BRAKE
3. **缺少渐进式响应**: 应该有一个从"轻微减速"到"中度减速"到"紧急制动"的连续谱，而不是 0→50 的阶跃

### 可能的解决方向
1. **OBSERVE 状态也给予适当的减速力** (w_kinetic = 5~10)，实现"谨慎驾驶"
2. **BRAKE 状态只在真正检测到碰撞威胁时才触发**（需要区分"潜在风险"和"实际威胁"）
3. **用连续函数替代阶跃函数**: 让 w_kinetic 随着 TTA 平滑变化，而不是非 0 即 50
4. **调整实验的鬼生成时机**: 让鬼在 ego 更远时就生成，形成"真实威胁"场景

---

## 八、相关文件索引

| 文件 | 路径 | 关键代码行 |
|------|------|-----------|
| 实验配置 | `configs/ghost_experiment.json` | 全文 |
| 规划器参数 | `planners/mind/configs/planning/demo_2.py` | 全文(90行) |
| 状态机 | `planners/mind/utils.py` | 682-754 |
| 风险源识别 | `planners/mind/utils.py` | 757-1005 |
| 权重→Cost | `planners/mind/trajectory_tree.py` | 105-198 |
| Cost/梯度计算 | `planners/ilqr/potential.py` | 270-505 |
| iLQR Cost接口 | `planners/ilqr/cost.py` | 380-447 |
| 鬼生成逻辑 | `experiments/ghost_probe/run_ghost_experiment.py` | 349-447 |
| 实验日志 | `output/ghost_experiment_v17b_zero_observe/improved/logs/` | CSV+PNG |
| 实验图表 | `output/ghost_experiment_v17b_zero_observe/improved/logs/*_plot.png` | PNG |
