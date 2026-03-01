# MIND-PA-LOI 代码库完整文档

> **用途**：本文档旨在让任何 AI 或研究者快速、完整地理解当前代码的架构、每个文件的功能、关键算法和数据流。
>
> **项目来源**：基于 HKUST-Aerial-Robotics 的 [MIND](https://github.com/HKUST-Aerial-Robotics/MIND)（IROS 2025），在此基础上增加了 **PA-LOI（Proactive Autonomous Lateral Occlusion Intelligence）** 模块。
>
> **核心改进**：为 MIND 的闭环仿真增加了鬼探头（Ghost Probe）风险检测、速度感知势场、RSS-AEB 安全护盾和碰撞检测。

---

## 1. 项目架构总览

```
MIND-PA-LOI/
├── run_sim.py                    # 入口脚本
├── simulator.py                  # 仿真器主循环
├── agent.py                      # 智能体定义
├── loader.py                     # 数据加载器
├── common/                       # 公共工具库
│   ├── geometry.py               #   几何计算（碰撞检测、投影等）
│   ├── visualization.py          #   3D 可视化渲染
│   ├── semantic_map.py           #   语义地图加载
│   ├── kinematics.py             #   运动学模型
│   ├── bbox.py                   #   包围盒
│   └── data.py                   #   数据结构
├── planners/
│   ├── basic/tree.py             #   树结构基类
│   ├── ilqr/                     #   iLQR 轨迹优化器
│   │   ├── solver.py             #     iLQR 求解器
│   │   ├── cost.py               #     代价树
│   │   ├── potential.py          #     势场函数 ★ 含 VelocityAwareRiskPotential
│   │   ├── dynamics.py           #     自动微分动力学
│   │   ├── autodiff.py           #     Theano 自动微分
│   │   └── utils.py              #     距离场生成
│   ├── mind/
│   │   ├── planner.py            #   ★ 核心规划器（含 AEB + 风险评估）
│   │   ├── scenario_tree.py      #   场景树生成器
│   │   ├── trajectory_tree.py    #   轨迹树优化器 ★ 含 KA-RF 势场集成
│   │   ├── utils.py              #   ★ 工具函数（含鬼探头检测算法）
│   │   ├── data_logger.py        #   实验数据 CSV 记录器
│   │   ├── networks/             #   神经网络
│   │   │   ├── network.py        #     MIND 预测网络（Transformer + RPE）
│   │   │   └── layers.py         #     网络层定义
│   │   └── configs/              #   规划器参数配置
│   └── check_points/             #   模型权重（~50MB×2）
├── configs/                      # 仿真场景配置
├── experiments/ghost_probe/      # 实验脚本和绘图
├── docs/                         # PA-LOI 技术文档
├── misc/                         # README 用的图片/GIF
└── requirements.txt              # 依赖
```

---

## 2. 数据流与调用链

```
run_sim.py
  └─ Simulator(config_path)
       │
       ├─ init_sim()
       │    └─ ArgoAgentLoader → [NonReactiveAgent, CustomizedAgent('AV')]
       │
       ├─ run_sim()  ← 主循环（500 步，dt=0.02s）
       │    │
       │    ├─ agent.observe() → agent_obs（所有智能体的观测）
       │    │
       │    ├─ CustomizedAgent.plan()
       │    │    └─ MINDPlanner.plan(lcl_smp)
       │    │         │
       │    │         ├─ [1] ScenarioTreeGenerator.branch_aime()
       │    │         │     └─ 神经网络推理 → 多模态预测 → 场景树
       │    │         │
       │    │         ├─ [2] get_semantic_risk_sources() ← ★ PA-LOI 鬼探头检测
       │    │         │     ├─ 类型筛选（BUS/VEHICLE + 静止）
       │    │         │     ├─ 动态走廊筛选 (d_critical, d_outer)
       │    │         │     ├─ 目标车道筛选
       │    │         │     ├─ 视线切点算法 → ghost_point
       │    │         │     └─ TTA 状态机 → phantom_state
       │    │         │
       │    │         ├─ [3] TrajectoryTreeOptimizer.init_cost_tree()
       │    │         │     ├─ 原版代价：目标偏差 + 碰撞势场 + 舒适性
       │    │         │     └─ ★ PA-LOI: VelocityAwareRiskPotential 注入 iLQR
       │    │         │
       │    │         ├─ [4] TrajectoryTreeOptimizer.solve()
       │    │         │     └─ iLQR 优化 → 最优轨迹
       │    │         │
       │    │         ├─ [5] evaluate_traj_tree() ← ★ 向量化评估 + Risk Cost
       │    │         │
       │    │         ├─ [6] AEB Safety Shield ← ★ RSS 三级制动
       │    │         │     └─ TTC + RSS → WARNING/DANGER/CRITICAL
       │    │         │
       │    │         └─ [7] 运动学钳位 (v ≥ 0)
       │    │
       │    ├─ Collision Check ← ★ SAT 多边形碰撞检测
       │    │
       │    └─ agent.update_state(dt)
       │
       ├─ render_video() → ffmpeg → .mov
       └─ save_collision_report() → collision_report.json
```

---

## 3. 核心文件详解

### 3.1 `run_sim.py` — 入口脚本（24 行）

```python
# 用法: python run_sim.py --config configs/1.json
parser.add_argument('--config', type=str, required=True)
sim = Simulator(args.config)
sim.run()
```

### 3.2 `simulator.py` — 仿真器（338 行）

**类 `Simulator`**

| 方法 | 行号 | 功能 |
|:---|:---:|:---|
| `__init__(config_path)` | 20-41 | 加载 JSON 配置、语义地图、初始化参数 |
| `init_sim()` | 48-53 | 加载回放智能体 + 闭环 AV 智能体 |
| `run_sim()` | 68-163 | **主循环**：500帧×0.02s=10s，含碰撞检测（★ 新增） |
| `get_agent_polygon(agent)` | 55-66 | ★ 新增：提取 3D 包围盒底面投影为 2D 多边形 |
| `save_collision_report()` | 165-184 | ★ 新增：碰撞事件 JSON 报告 |
| `render_video()` | 186-210 | 逐帧渲染 PNG → ffmpeg 合成视频 |
| `render_frame(frame_idx, ax)` | 226-337 | 3D 可视化：地图 + 场景树 + 轨迹树 + Ghost 点 + GT 轨迹 |

**关键改进**：
- **碰撞检测**（92-117行）：每帧检查 Ego 与所有其他智能体的多边形碰撞（SAT 算法）
- **Ghost 可视化**（296-309行）：渲染鬼探头危险区域
- **GT 轨迹**（283-291行）：渲染原始参考轨迹（红色线）
- **跟随相机**（260-268行）：相机自动跟随 AV

### 3.3 `agent.py` — 智能体系统（303 行）

| 类 | 功能 |
|:---|:---|
| `AgentObservation` | 观测数据容器（位置、速度、航向、类型） |
| `PlainAgent` | 基础智能体（状态 + 包围盒） |
| `NonReactiveAgent` | 回放智能体（按记录轨迹运动，不受 planning 影响） |
| `CustomizedAgent` | ★ AV 闭环智能体（集成 MINDPlanner，执行规划控制循环） |

**`CustomizedAgent` 关键属性**：
```python
self.planner = MINDPlanner(planner_config)   # 规划器
self.state = [x, y, v, heading]               # 自车状态
self.is_enable = False                         # 是否激活（时间触发）
self.rec_interval = 0.1                        # 感知频率 10Hz
self.plan_interval = 0.2                       # 规划频率 5Hz
```

### 3.4 `planners/mind/planner.py` — 核心规划器（675 行）★

**这是改动最大的文件（原版 ~150 行 → 675 行）。**

**全局开关**：
```python
ENABLE_GHOST_PROBE = True      # 鬼探头检测
ENABLE_AEB = True              # AEB 安全护盾
ENABLE_DATA_LOGGING = True     # 数据记录
```

**类 `MINDPlanner`**

| 方法 | 行号 | 功能 |
|:---|:---:|:---|
| `__init__(config_dir)` | 16-52 | 加载网络、初始化场景树/轨迹树、AEB 状态 |
| `init_scen_tree_gen()` | 54-58 | 初始化 `ScenarioTreeGenerator` |
| `init_traj_tree_opt()` | 60-62 | 初始化 `TrajectoryTreeOptimizer` |
| `init_data_logger(...)` | 64-78 | ★ 初始化 CSV 数据记录器 |
| `load_network()` | 80-100 | 加载预训练 Transformer 权重 |
| `resample_target_lane(lcl_smp)` | 102-132 | 以 1.0m 间隔重采样目标车道 |
| `plan(lcl_smp)` | 134-500+ | ★ **核心规划流程**（见下方详解） |
| `evaluate_traj_tree(...)` | 500-600+ | ★ 向量化轨迹评估（含 Risk Cost） |
| `save_experiment_log()` | 末尾 | 保存实验日志 |

**`plan()` 方法详细流程**：

```text
plan(lcl_smp) {
  
  // Step 1: 场景树生成
  scen_trees = scen_tree_gen.branch_aime(lcl_smp, agent_obs)
  
  // Step 2: ★ PA-LOI 鬼探头检测 (如果 ENABLE_GHOST_PROBE)
  risk_sources = get_semantic_risk_sources(
    trajs_pos, trajs_vel, trajs_type, trajs_ang,
    ego_pos, ego_heading, ego_vel,
    lane_width, road_width, target_lane
  )
  
  // Step 3: 为每棵场景树生成/优化轨迹树
  for each scen_tree:
    traj_tree_opt.init_cost_tree(scen_tree, ..., risk_sources)  // ★ 注入风险势场
    traj_tree = traj_tree_opt.solve()
    cost = evaluate_traj_tree(traj_tree, risk_sources)
  
  // Step 4: 选择最优轨迹
  best_traj = argmin(costs)
  ret_ctrl = [acceleration, steering_rate]
  
  // Step 5: ★ AEB Safety Shield (如果 ENABLE_AEB)
  for each agent in front:
    ttc = distance / relative_velocity
    rss_distance = v²/(2*a_max_brake) - v_other²/(2*a_max_brake)
    
    if ttc < TTC_CRITICAL or dist < rss_dist:
      aeb_level = CRITICAL → a = -4.0 m/s²
    elif ttc < TTC_DANGER:
      aeb_level = DANGER   → a = -2.0 m/s²
    elif ttc < TTC_WARNING:
      aeb_level = WARNING  → a = -0.8 m/s²
  
  // Step 6: ★ 运动学钳位
  predicted_v = current_v + acceleration * dt
  if predicted_v < 0:
    acceleration = -current_v / dt  // 刚好刹停，不倒车
  
  // Step 7: ★ 数据记录
  data_logger.log_frame(ego_state, risk_sources, ctrl, ...)
  
  return [success, ctrl, [scen_trees, traj_trees, risk_sources]]
}
```

**AEB 三级安全护盾参数**：

| 级别 | TTC 阈值 | 制动加速度 | 触发条件 |
|:---|:---:|:---:|:---|
| WARNING | < 2.5s | -0.8 m/s² | TTC 进入警告区 |
| DANGER | < 1.5s | -2.0 m/s² | TTC 进入危险区 |
| CRITICAL | < 0.8s 或 RSS | -4.0 m/s² | TTC/RSS 临界 |

**Hysteresis 机制**：AEB 等级只升不降（在同一时间窗口内），3 帧计数器防止抖动。

### 3.5 `planners/mind/utils.py` — 工具函数（1056 行）★

**原版函数（保留不变）**：

| 函数 | 行号 | 功能 |
|:---|:---:|:---|
| `gpu(data, device)` | 9-20 | 递归将 tensor 移到 GPU |
| `from_numpy(data)` | 24-35 | 递归 numpy → torch |
| `padding_traj_nn(traj)` | 38-58 | 最近邻填充缺失轨迹点 |
| `tgt_gather(...)` | 61-71 | 目标节点批量聚合 |
| `graph_gather(...)` | 74-111 | 车道图批量聚合 |
| `actor_gather(...)` | 114-139 | 智能体轨迹批量聚合 |
| `collate_fn(batch)` | 142-168 | DataLoader 整理函数 |
| `get_new_lane_graph(...)` | 171-177 | 坐标系转换 |
| `get_origin_rotation(...)` | 180-190 | 计算局部坐标系原点和旋转 |
| `get_rpe(ctrs, vecs, radius)` | 193-212 | 计算相对位置编码 (RPE) |
| `get_angle(vel)` / `get_cos` / `get_sin` | 215-242 | 角度/余弦/正弦工具 |
| `get_agent_trajectories(...)` | 245-349 | 从观测构建智能体轨迹 |
| `update_lane_graph_from_argo(...)` | 352-490 | 从 Argo2 HD Map 构建车道图 |
| `get_closest_point_on_segment(...)` | 493-506 | 点到线段最近点 |
| `get_distance_to_polyline(...)` | 509-520 | 点到折线最短距离 |
| `get_covariance_matrix(data)` | 523-540 | 构建 2x2 协方差矩阵 |
| `get_max_covariance(data)` | 543-558 | 取最大方差分量 |

**★ PA-LOI 新增函数**：

| 函数 | 行号 | 功能 |
|:---|:---:|:---|
| `calculate_adaptive_corridor(lane_width, road_width, ego_vel)` | 561-597 | **双层动态走廊**：d_critical（禁区）+ d_outer（感知范围），基于车速+路宽，含几何钳位 |
| `is_obstacle_on_target_lane(obs_pos, target_lane, lane_width)` | 600-626 | 检查障碍物是否在目标车道附近（阈值 = lane_width/2 + 4.5m） |
| `project_to_lateral_distance(ego_pos, ghost_point, lane_heading)` | 629-650 | 计算横向投影距离（用于 Sigmoid 势场） |
| `is_separated_by_solid_line(obs_pos, ego_pos, ego_heading, mark_type)` | 653-679 | 检查是否被实线/双黄线分隔 |
| `calculate_phantom_behavior(long_dist, lat_dist, ego_vel)` | 682-756 | **TTA 状态机**：基于 Time-To-Arrival 和物理可达性的三态机（OBSERVE/WARNING/BRAKE） |
| `get_semantic_risk_sources(...)` | 759-1037 | ★ **核心：鬼探头风险源识别算法**（~280行），8级筛选流水线 |
| `get_risk_covariance(sigma, device)` | 1041-1056 | 生成圆形风险区域协方差 |

**`get_semantic_risk_sources()` 算法流程**：

```text
对每个智能体 i (排除 Ego):
  ├─ [1] 类型筛选：仅保留 BUS / VEHICLE（可遮挡行人）
  ├─ [2] 速度筛选：速度 < 0.5 m/s（静止车辆才可能形成遮挡）
  ├─ [3] 纵向筛选：-5m < longitudinal < 50m
  ├─ [4] 横向筛选：lateral < d_outer（动态走廊外边界）
  ├─ [5] 目标车道筛选：距离目标车道 < (lane_width/2 + 4.5m)
  ├─ [6] 视线切点算法：计算遮挡车辆的 4 个角点，选取最近的切线内侧角
  │     └─ ghost_point = 行人最可能冲出的位置
  ├─ [7] Ghost 点验证：ghost_point 必须在 Ego 前方且横向 < d_outer
  └─ [8] TTA 状态机：计算 phantom_state 和 v_safe

输出 risk_sources[]，每个元素包含：
  - pos: ghost_point 位置 [2]
  - cov: 风险协方差 [2,2]
  - weight: 势场权重（0~15，基于 TTA 线性插值）
  - v_safe: 安全速度（2.5 m/s）
  - phantom_state: 'OBSERVE' / 'BRAKE'
  - ghost_lateral, ghost_longitudinal
  - tta_ego, tta_human
```

### 3.6 `planners/mind/trajectory_tree.py` — 轨迹树优化（238 行）★

**类 `TrajectoryTreeOptimizer`**

| 方法 | 行号 | 功能 |
|:---|:---:|:---|
| `__init__(config)` | 13-17 | 初始化 iLQR 求解器 |
| `init_warm_start_cost_tree(...)` | 19-56 | 热启动代价树（仅用目标偏差，无碰撞诊断） |
| `init_cost_tree(...)` | 58-184 | ★ **完整代价树**（含 KA-RF 势场） |
| `warm_start_solve()` | 186-191 | 热启动求解 |
| `solve()` | 193-207 | 完整求解 → 返回轨迹树 |
| `_get_init_state(...)` | 209-211 | 构造初始状态 [x,y,v,θ,a,δ] |
| `_get_dynamic_model(dt, wb)` | 213-237 | 自行车模型（Theano 自动微分） |

**`init_cost_tree()` 的代价组成**（58-184行）：

```text
对场景树中的每个节点：
  总 Cost = w_tgt × 目标偏差²           // 车道保持
          + w_exo × 他车碰撞势场        // 障碍物避让
          + w_ego × 自车预测不确定性     // 自信度
          + w_ctrl × 控制量²            // 舒适性
          + ★ VelocityAwareRiskPotential // PA-LOI 速度感知势场 (仅速度梯度)
```

**关键改进（107-153行）**：
```python
# 原版：cov_dist_field += w_base * sigmoid_field (静态 CostMap)
# PA-LOI：改为 VelocityAwareRiskPotential (动态速度梯度)
#   → 避免了"双重计费"问题
#   → iLQR 通过 ∂C/∂v 梯度实现平滑减速
risk_pot = VelocityAwareRiskPotential(
    risk_pos, lane_heading, ghost_lateral,
    w_base, v_safe=2.5, lambda_v=0.1
)
state_pots = [pot_field, state_pot, state_con] + risk_potentials
```

### 3.7 `planners/ilqr/potential.py` — 势场函数（349 行）

**原版类**：

| 类 | 功能 |
|:---|:---|
| `ControlPotential` | 控制量惩罚 C = u^T W u |
| `StateConstraint` | 状态约束（上下界，违反时二次惩罚） |
| `StatePotential` | 期望状态追踪 C = (x-x_des)^T W (x-x_des) |
| `PotentialField` | 2D 代价场（二次插值 + 梯度/Hessian 解析计算） |

**★ PA-LOI 新增类 `VelocityAwareRiskPotential`**（267-349行）：

```python
class VelocityAwareRiskPotential:
    """
    Cost = W_base × Sigmoid(clearance) × max(0, v - v_safe)²
    
    关键特性：
    - v ≤ v_safe → Cost=0, Gradient=0 (允许保持安全速度通过)
    - v > v_safe → 强减速梯度 ∂C/∂v = 2·W·Sigmoid·(v - v_safe)
    """
    
    def __init__(self, risk_pos, lane_heading, ghost_lateral, w_base,
                 v_safe=0.0, lambda_v=0.1, ego_half_width=1.0, k_steep=2.0):
        # risk_pos: 风险点位置 [2]
        # lane_heading: 车道航向 (rad)
        # ghost_lateral: Ghost 点横向距离 (m)
        # w_base: 基础权重 (0~15)
        # v_safe: 安全速度阈值 (Hinge Loss 零点)
        # k_steep: Sigmoid 陡峭程度
    
    def get_potential(self, state):  # state = [x, y, v, θ, a, δ]
        lateral = |dot(pos - risk_pos, normal)|
        clearance = max(lateral - ego_half_width, 0)
        sigmoid = 1 / (1 + exp(k·(clearance - ghost_lateral)))
        excess_vel = max(0, v - v_safe)
        return w_base × sigmoid × excess_vel²
    
    def get_gradient(self, state):
        # 仅对 state[2] (速度) 有梯度
        gradient[2] = w_base × sigmoid × 2 × excess_vel
    
    def get_hessian(self, state):
        hessian[2,2] = w_base × sigmoid × 2
```

### 3.8 `planners/mind/scenario_tree.py` — 场景树生成器（689 行）

**类 `ScenarioTreeGenerator`**

| 方法 | 行号 | 功能 |
|:---|:---:|:---|
| `branch_aime(lcl_smp, agent_obs)` | 38-62 | **入口**：处理数据 → 预测 → 分支 → 构建场景树 |
| `process_data(lcl_smp, agent_obs)` | 126-210 | 构建模型输入（轨迹 + 车道图 + RPE） |
| `predict_scenes(data)` | 73-75 | 神经网络推理：输入观测 → 输出多模态预测 |
| `decide_branch()` | 86-104 | 基于场景发散度决定分支时间 |
| `get_scenario_tree()` | 212-276 | 从预测构建场景树结构 |
| `prune_merge(data, out)` | 285-416 | 场景树剪枝合并（去除不太可能的分支） |
| `calculate_risk_score(trajs_pos)` | 418-448 | ★ 计算场景风险评分 |
| `update_obser(cur_data)` | 503-603 | 更新观测历史（滑窗） |
| `get_high_level_command(...)` | 649-688 | 高级决策（直行/换道/停车） |

### 3.9 `planners/mind/data_logger.py` — 数据记录器（297 行）★

**类 `PALOIDataLogger`**

每帧记录以下字段到 CSV：

```text
timestamp | frame_idx | ego_x | ego_y | ego_v | ego_heading |
risk_count | risk0_x | risk0_y | risk0_weight | risk0_state |
risk0_tta_ego | risk0_tta_human | risk0_ghost_lat | risk0_ghost_long |
d_critical | d_outer | ctrl_accel | ctrl_steer | is_collision
```

**工具函数**：
- `load_experiment_log(filepath)` — 加载 CSV → pandas DataFrame
- `plot_experiment_results(filepath)` — 生成论文用三子图（速度/横向距离/幻影状态）

### 3.10 `planners/mind/networks/network.py` — 预测网络（607 行）

| 类 | 功能 |
|:---|:---|
| `ActorNet` | Actor 特征提取（Conv1D + FPN） |
| `PointAggregateBlock` | 点聚合（MaxPool + MLP） |
| `LaneNet` | 车道特征提取 |
| `RelaFusionLayer` | 相对位置注意力融合（RPE-MHA） |
| `RelaFusionNet` | 多层融合网络 |
| `AIME` | **顶层网络**：Actor + Lane → Fusion → 多模态预测头 |

**`AIME` 网络结构**：
```
输入: actor_trajs [N, T, 3], lane_graph
  → ActorNet → actor_features [N, 128]
  → LaneNet → lane_features [M, 128]
  → RPE 计算 → edge_features [N+M, N+M, 128]
  → RelaFusionNet (6层 Transformer) → fused_features
  → 预测头 → trajectories [N, K, T_pred, 5]  (pos, cov)
```

### 3.11 `common/geometry.py` — 几何工具（151 行）

| 函数 | 功能 |
|:---|:---|
| `is_inside_ellipse(point, mean, cov)` | 马氏距离椭圆检测 |
| `get_vehicle_vertices(x, y, z, yaw, l, w, h)` | 车辆 3D 包围盒顶点 |
| `project_point_on_polyline(point, polyline)` | 点在折线上的投影（返回位置/航向/累积距离） |
| `is_separating_axis(axis, poly1, poly2)` | SAT 分离轴检测 |
| `check_polygon_intersection(poly1, poly2)` | ★ 凸多边形碰撞检测（SAT 算法） |

### 3.12 `common/visualization.py` — 可视化（331 行）

| 函数 | 功能 |
|:---|:---|
| `draw_map(ax, static_map)` | 绘制 HD Map（车道线、交叉口） |
| `draw_agent(ax, agent)` | 绘制智能体 3D 包围盒 |
| `draw_scen_trees(ax, trees)` | 绘制场景树预测轨迹 |
| `draw_traj_trees(ax, trees)` | 绘制优化轨迹 |
| `draw_ghost_points(ax, ghost_points)` | ★ 绘制鬼探头危险区域（红色圆圈） |
| `draw_polyline(ax, polyline)` | 绘制折线 |

### 3.13 `experiments/ghost_probe/` — 实验脚本

| 文件 | 功能 |
|:---|:---|
| `run_ghost_experiment.py` | 批量运行对比实验（baseline vs improved） |
| `plot_v25_paper.py` | 生成论文用的速度对比图 |
| `plot_v32_control.py` | 绘制控制量（加速度/转向）时序图 |
| `plot_v33.py` | 最新版绘图脚本 |
| `plot_log.py` | CSV 日志快速可视化 |

---

## 4. 配置文件格式

### 4.1 仿真配置 `configs/1.json`

```json
{
  "sim_name": "demo_1",
  "seq_id": "0a0e2f9f-4805-3e53-8798-c6b0a3067c58",
  "output_dir": "./output/demo_1/",
  "num_threads": 1,
  "render": true,
  "render_config": {
    "mode": "follow",
    "camera_position": { "x": 0, "y": 0, "yaw": 0, "elev": 75 }
  },
  "cl_agents": [
    {
      "id": "AV",
      "type": "MINDAgent",
      "planner_config": "planners/mind/configs/demo_1.json",
      "use_cuda": false
    }
  ]
}
```

### 4.2 规划器配置 `planners/mind/configs/demo_1.json`

```json
{
  "planner_type": "MINDPlanner",
  "network_config": "planners/mind/configs/networks/net_cfg.py",
  "planning_config": "planners.mind.configs.planning.demo_1",
  "checkpoint": "planners/check_points/20240121-172745.tar"
}
```

---

## 5. 状态向量定义

| 索引 | 变量 | 单位 | 说明 |
|:---:|:---|:---:|:---|
| 0 | x | m | 全局 X 坐标 |
| 1 | y | m | 全局 Y 坐标 |
| 2 | v | m/s | 纵向速度 |
| 3 | θ | rad | 航向角 |
| 4 | a | m/s² | 纵向加速度 |
| 5 | δ | rad | 前轮转角 |

**控制量**：`[da, dδ]` = 加速度变化率 (jerk) + 转向变化率

**动力学模型**（自行车模型，轮距 wb=2.5m）：
```
x_{t+1} = x_t + v·cos(θ)·dt
y_{t+1} = y_t + v·sin(θ)·dt
v_{t+1} = v_t + a·dt
θ_{t+1} = θ_t + v/wb·tan(δ)·dt
a_{t+1} = a_t + da·dt
δ_{t+1} = δ_t + dδ·dt
```

---

## 6. 依赖环境

- **Python 3.9+**
- **PyTorch 2.0+**（CPU 即可）
- **Theano-PyMC**（iLQR 自动微分）
- **Argoverse 2 SDK**（HD Map 加载）
- **Shapely**（几何计算）
- **FFmpeg**（视频合成）

---

## 7. 运行方式

```bash
# 单场景仿真
python run_sim.py --config configs/1.json

# 批量实验
python experiments/ghost_probe/run_ghost_experiment.py

# 绘制论文图表
python experiments/ghost_probe/plot_v33.py
```

---

## 8. 当前版本号和关键迭代记录

| 版本 | 关键改动 |
|:---|:---|
| v18 | 引入速度感知势场，解决方向盘锁死问题 |
| v25 | TTA 状态机 + 双层走廊 |
| v32 | AEB 三级安全护盾 + Hysteresis |
| v33 | 修复双重计费（Double Counting Fix） |
| v52 | **Hinge-Loss 速度惩罚**（v_safe 阈值），解决 500 帧死锁 |
| v53 | 虚实双轨策略最终版：虚拟场 v_safe=2.5，真实走 AEB |
| v54 | AEB 横向视野修复（行人 5.0m，车辆 1.5m）；搜索距离保底 15m |
| v55 | 智能横向过滤（区分行人/车辆类型）|
| v57 | 预测性运动学钳位最终版 |

---

## 9. 完整参数表

### 9.1 iLQR 轨迹优化器权重（`planners/mind/configs/planning/demo_2.py`）

**代价函数**：`Total = w_tgt × 目标偏差² + w_exo × 障碍势场 + w_ego × 自车不确定性 + w_ctrl × 控制量² + Risk`

| 参数 | 值 | 说明 |
|:---|:---:|:---|
| `w_tgt` | 1.0 | 目标车道偏差权重 |
| `w_exo` | 10.0 | 他车碰撞势场权重 |
| `w_exo_cov_offset` | 2.5 | 他车协方差膨胀（越大越保守） |
| `w_exo_cost_offset` | 10.0 | 他车碰撞区域附加 Cost |
| `w_ego` | 1.0 | 自车预测不确定性权重 |
| `w_ego_cov_offset` | 1.0 | 自车协方差膨胀 |
| `w_ctrl` | 5.0×I₂ | 控制量惩罚（jerk + 转向变化率） |
| `w_des_state[v]` | 0.1 | 期望速度追踪权重 |
| `w_des_state[a]` | 1.0 | 期望加速度为 0 的权重 |
| `w_des_state[δ]` | 10.0 | 期望转角为 0 的权重 |
| `smooth_grid_res` | 0.4m | 平滑势场的网格分辨率 |
| `smooth_grid_size` | 256×256 | 势场网格尺寸 |

**状态约束（硬边界）**：

| 状态 | 下界 | 上界 | 违反惩罚权重 |
|:---|:---:|:---:|:---:|
| x, y | -100000 | 100000 | 0 |
| v (速度) | 0 m/s | 8 m/s | 50 |
| θ (航向) | -10 rad | 10 rad | 0 |
| a (加速度) | -6 m/s² | 4 m/s² | 50 |
| δ (转向角) | -0.2 rad | 0.2 rad | 500 |

### 9.2 `evaluate_traj_tree()` 打分权重（`planner.py` L539-577）

| 分量 | 权重 | 公式 |
|:---|:---:|:---|
| Comfort | acc:0.1, str:5.0 | Σ(0.1·a² + 5.0·δ²) |
| Efficiency | 0.01 | Σ 0.01·(v_target - v)² |
| Target Lane | 0.01 | Σ 0.01·dist_to_lane |
| Risk (PA-LOI) | weight×1.0 | Σ max_over_risks(w · max(0, v-v_safe)²) |

### 9.3 AEB 安全护盾参数（`planner.py` L266-276）

| 参数 | 值 | 说明 |
|:---|:---:|:---|
| `AEB_T_RESPONSE` | 0.2 s | 系统响应延迟 |
| `AEB_A_MAX_BRAKE` | 4.0 m/s² | 最大制动减速度 |
| `AEB_TTC_CRITICAL` | 0.8 s | CRITICAL 阈值 → a=-4.0 |
| `AEB_TTC_DANGER` | 1.6 s | DANGER 阈值 → a=-2.0 |
| `AEB_TTC_WARNING` | 2.6 s | WARNING 阈值 → a=-0.8 |
| `AEB_LAT_STATIC` | 1.2 m | 静态障碍物横向过滤 |
| `AEB_LAT_DYNAMIC` (行人) | 5.0 m | 行人横向搜索范围 |
| `AEB_LAT_DYNAMIC` (车辆) | 1.5 m | 车辆横向搜索范围 |
| `d_safe (RSS)` | v·0.2 + v²/8 | RSS 安全距离（最小 2.0m） |
| 搜索距离 | max(d_safe×1.5, 15.0) | AEB 目标搜索范围 |

### 9.4 鬼探头检测参数（`utils.py`）

| 参数 | 值 | 来源 | 说明 |
|:---|:---:|:---:|:---|
| `STATIC_SPEED_THRES` | 0.5 m/s | L805 | 低于此速度视为静止 |
| `MAX_LONGITUDINAL` | 50.0 m | L806 | 前方最大检测距离 |
| `HUMAN_MAX_SPEED` | 5.0 m/s | L700 | 行人最大冲刺速度 |
| `LOOKAHEAD_TIME` | 1.5 s | L704 | TTA 状态机前瞻时间 |
| `v_safe` | 2.5 m/s | L984 | Hinge-Loss 安全速度阈值 |
| Ghost 点权重范围 | 0 ~ 15 | L989-997 | 基于 TTA 线性插值 |
| 目标车道检测阈值 | lane_width/2 + 4.5m | L624 | is_obstacle_on_target_lane |
| 纵向后方容忍 | -5.0 m | L868 | 允许检测刚经过车头的长车 |

**动态走廊参数**（`calculate_adaptive_corridor`）：

| 参数 | 公式 | 说明 |
|:---|:---|:---|
| `d_critical` | min(0.5 + 0.03·v, lane_w/2 - 0.2) | 内层禁区，几何钳位 |
| `d_outer` | min(7.0, road_w/2) | 外层感知范围 |

**VelocityAwareRiskPotential 参数**（`potential.py`）：

| 参数 | 值 | 说明 |
|:---|:---:|:---|
| `w_base` | 0~15 | 由 TTA 线性插值决定 |
| `v_safe` | 2.5 m/s | Hinge-Loss 零点（低于此免惩罚） |
| `lambda_v` | 0.1 | 已被 v_safe 取代，保留接口兼容 |
| `ego_half_width` | 1.0 m | Ego 车身半宽 |
| `k_steep` | 2.0 | Sigmoid 陡峭程度 |

### 9.5 场景树配置（`ScenTreeCfg`）

| 参数 | 值 | 说明 |
|:---|:---:|:---|
| `max_depth` | 5 | 场景树最大递归深度 |
| `tar_dist_thres` | 10.0 m | 目标距离阈值 |
| `tar_time_ahead` | 5.0 s | 前瞻时间 |
| `seg_length` | 15.0 m | 车道段近似长度 |
| `seg_n_node` | 10 | 每段节点数 |
| `enable_ghost_probe` | True | 鬼探头检测开关 |

### 9.6 神经网络配置（`NetCfg`）

| 参数 | 值 | 说明 |
|:---|:---:|:---|
| `g_num_modes` | 6 | 多模态预测数（6 种未来） |
| `g_obs_len` | 50 | 观测长度（50 帧 = 5s @ 10Hz） |
| `g_pred_len` | 60 | 预测长度（60 帧 = 6s） |
| `d_actor` | 128 | Actor 特征维度 |
| `d_lane` | 128 | Lane 特征维度 |
| `d_rpe` | 128 | 相对位置编码维度 |
| `d_embed` | 128 | 嵌入维度 |
| `n_scene_layer` | 6 | Transformer 层数 |
| `n_scene_head` | 8 | 注意力头数 |
| `dropout` | 0.1 | Dropout 率 |
| `param_out` | 'bezier' | 输出参数化方式（Bézier 曲线） |

### 9.7 仿真器参数

| 参数 | 值 | 说明 |
|:---|:---:|:---|
| `sim_step` | 0.02 s | 物理仿真步长（50Hz） |
| `sim_horizon` | 500 帧 | 仿真时长（500×0.02=10s） |
| `rec_interval` | 0.1 s | 感知更新频率（10Hz） |
| `plan_interval` | 0.2 s | 规划更新频率（5Hz） |
| 碰撞预筛选距离 | 10.0 m | 超过此距离跳过碰撞检测 |

### 9.8 全局开关（`planner.py` 顶部）

```python
ENABLE_GHOST_PROBE = False   # 当前关闭（改为 True 启用）
ENABLE_AEB = True            # AEB 安全护盾
ENABLE_DATA_LOGGING = True   # CSV 数据记录
DEBUG_LOG_ENABLED = True     # 控制台日志
```

> **注意**: `ENABLE_GHOST_PROBE` 当前为 `False`。启用需同时确保场景配置中 `enable_ghost_probe: true`（在 `ScenTreeCfg` 中）。

---

## 10. 设计决策与原因

### 10.1 为什么用 Hinge-Loss 而不是直接的速度惩罚？

**问题**：早期版本（v18~v33）使用 `Cost = w × v²`，导致 Ego 在盲区前**完全刹停后无法恢复**（500 帧死锁）。因为 `∂C/∂v = 2wv`，即使 `v→0`，只要 `v>0` 就有梯度推动继续减速。

**解决方案**（v52）：改为 `Cost = w × max(0, v - v_safe)²`。当 `v ≤ v_safe` 时，Cost=0，Gradient=0，车辆可以匀速 2.5m/s 安全通过盲区。

### 10.2 为什么 VelocityAwareRiskPotential 不产生空间梯度？

**问题**：早期版本同时产生空间梯度（`∂C/∂x, ∂C/∂y`）和速度梯度（`∂C/∂v`），导致 iLQR 在减速的同时猛打方向盘绕行，造成**轨迹偏转碰撞**。

**解决方案**（v33）：VelocityAwareRiskPotential **只对 `state[2]`（速度）产生梯度**，空间上使用 Sigmoid 做衰减权重但不产生空间力。空间避障完全交给原版的 `w_exo` 碰撞势场。

### 10.3 为什么使用"虚实双轨"策略？

- **虚拟风险（盲区遮挡）**：通过 iLQR 势场平滑减速至 v_safe=2.5m/s，由 `get_semantic_risk_sources()` + `VelocityAwareRiskPotential` 负责
- **真实风险（实际碰撞威胁）**：由 AEB 安全护盾硬性覆写控制指令，分三级制动

**原因**：iLQR 是优化器，其输出可以被"说服"增加 Cost 来减速，但无法保证在极端情况下一定能避碰。AEB 作为最后一层安全网，提供硬性保障。

### 10.4 为什么移除了静态 CostMap 叠加？

**问题（双重计费 Double Counting）**：早期版本同时在 `cov_dist_field` 上叠加静态 Sigmoid 场（空间 Cost），并通过 `VelocityAwareRiskPotential` 加速度 Cost。两者在 iLQR 中叠加，导致过度保守。

**修复**（v33, `trajectory_tree.py` L107-153）：完全移除静态 CostMap 叠加，风险 Cost **仅**由 `VelocityAwareRiskPotential` 负责。注释 `# 【已移除】静态 CostMap 叠加 - 避免双重计费`。

### 10.5 AEB Hysteresis 为什么"只升不降"？

在 AEB 级别边界（如 TTC 在 0.8s 附近），TTC 会因为减速而快速变化，导致 AEB 在 CRITICAL/DANGER 之间反复切换（抖动）。Hysteresis 机制确保：
- **升级**：立即响应
- **降级**：需要连续 3 帧确认更低威胁后才降级

### 10.6 运动学钳位的必要性

**问题**：iLQR 使用 `dt=0.2s` 离散化，当 `v=0.3m/s` 且 `a=-4.0m/s²` 时，下一帧 `v' = 0.3 + (-4.0)×0.2 = -0.5`，车辆倒车！

**修复**（`planner.py` L434-451）：
```python
min_acc = -v_curr / dt          # 刚好刹停的加速度
ret_ctrl[0] = max(ret_ctrl[0], min_acc)  # 钳位
```

---

## 11. 已知问题与未实现功能

### 11.1 已知问题

| 编号 | 类别 | 描述 |
|:---:|:---|:---|
| B1 | AEB 释放逻辑 | `planner.py` L393-397 有 TODO：AEB 释放逻辑依赖 `targets` 变量但未定义，当前用简化逻辑替代 |
| B2 | 全局开关不一致 | `ENABLE_GHOST_PROBE` 在 `planner.py`（当前=False）和 `ScenTreeCfg`（=True）中独立控制 |
| B3 | 内存增长 | 不渲染时有 frame 缓存清理（L156），但渲染时 500 帧全保留 |

### 11.2 未实现的核心功能（ITSC 论文所需）

| 编号 | 功能 | 重要性 | 说明 |
|:---:|:---|:---:|:---|
| F1 | **Virtual Ghost Branching** | ⭐⭐⭐ | 在 `ScenarioTreeGenerator` 中显式插入鬼探头应急分支 |
| F2 | **CVaR 风险度量** | ⭐⭐ | 引入条件在险价值，替代简单加权 |
| F3 | **自适应风险概率** | ⭐⭐ | P_ghost = f(遮挡面积, 路口类型, 速度) |
| F4 | **FRS 风险定义** | ⭐ | 用 Forward Reachable Sets 替代单点 Ghost |

### 11.3 代码质量问题

- 根目录的 `ENABLE_GHOST_PROBE = False` 需要手动改为 `True` 才能启用鬼探头功能
- `planner.py` 中 `road_width=14.0` 在 L473 被硬编码
- `evaluate_traj_tree()` 中的权重（comfort/efficiency/target）与 iLQR 中的权重独立，可能不一致
