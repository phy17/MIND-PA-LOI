# PA-LOI 实现方案完整报告

**PA-LOI** (Physics-Aware Latent Obstacle Injection) 是针对 MIND 规划器的风险感知增强模块。

---

## 一、核心公式：KA-RF (Kinematics-Aware Risk Field)

### 1.1 总公式

```
J_risk = W_base × (1 + λ × v²) × 1/(1 + exp(k × (d_lat - d_critical)))
```

**展开形式：**

```
                                        1
J_risk = W_base × (1 + λ × v_ego²) × ─────────────────────────
                                     1 + e^(k × (d_lat - d_crit))
```

### 1.2 参数定义

| 符号 | 名称 | 推荐值 | 说明 |
|------|------|--------|------|
| W_base | 基础权重 | 20.0 | 风险场"底价"，确保优先级高于车道引力 |
| λ (lambda) | 速度系数 | 0.02 | 控制速度对 Cost 的放大效果 |
| v_ego | 自车速度 | 实时变量 | 单位 m/s |
| k | 陡峭因子 | 2.0 | 控制 Sigmoid "悬崖"的陡峭程度 |
| d_lat | 横向净距 | 实时变量 | 车皮到墙角的垂直距离 |
| d_critical | 临界阈值 | 动态计算 | 内外层分界点 |

### 1.3 横向距离计算

```python
# lane_heading: 车道方向角
dx = ego_x - ghost_point_x
dy = ego_y - ghost_point_y

# 投影公式
d_lat = |(-dx × sin(lane_heading) + dy × cos(lane_heading))| - EGO_HALF_WIDTH
```

---

## 二、双层走廊动态宽度

### 2.1 公式设计

```python
def calculate_adaptive_corridor(lane_width, ego_vel):
    EGO_WIDTH = 2.0
    
    # ========= 内层 (d_critical) =========
    # 物理约束
    physical_limit = (lane_width - EGO_WIDTH) / 2.0
    
    # 动力学需求
    dynamic_need = 0.5 + 0.03 × |ego_vel|
    
    # 取较小值
    d_critical = min(physical_limit, dynamic_need)
    d_critical = max(d_critical, 0.2)  # 兜底
    
    # ========= 外层 (d_outer) =========
    d_outer = min(5.0, lane_width)
    
    return d_critical, d_outer
```

### 2.2 效果对照表

| 路宽 | 车速 | d_critical | d_outer | 说明 |
|------|------|------------|---------|------|
| 3.0m | 5m/s | 0.50m | 3.0m | 窄路受限于物理空间 |
| 3.5m | 10m/s | 0.75m | 3.5m | 标准配置 |
| 5.0m | 15m/s | 0.95m | 5.0m | 宽路+高速 |
| 7.0m | 5m/s | 0.65m | 5.0m | 宽路+低速 |

---

## 三、目标车道智能筛选

### 3.1 问题

当前代码使用固定横向阈值 (3.0m)，会误检测旁边车道的静止车辆。

### 3.2 解决方案

```python
def is_obstacle_on_target_lane(obs_pos, target_lane, lane_width=3.5):
    """
    检查障碍物是否在目标车道上
    """
    proj_point, _, proj_dist = project_point_on_polyline(obs_pos, target_lane)
    
    # 阈值 = 半路宽 + 余量
    threshold = (lane_width / 2.0) + 0.5
    
    return proj_dist < threshold
```

---

## 四、实现状态追踪

### ✅ 已完成

| 模块 | 功能 | 文件 |
|------|------|------|
| 遮挡物检测 | 识别静止的 BUS/VEHICLE | utils.py |
| 四角点计算 | 计算障碍物四个角的全局坐标 | utils.py |
| 视线切点选择 | 选择 Ego 会经过一侧的危险角 | utils.py |
| 基础风险 Cost | 圆形高斯场 (待替换) | planner.py |
| 速度惩罚 | 线性 v 权重 (待改为 v²) | planner.py |

### ⏳ 待实现

| 优先级 | 任务 | 说明 |
|--------|------|------|
| **P0** | 横向 Sigmoid 场 | 替换圆形高斯为 KA-RF 公式 |
| **P0** | v² 速度权重 | 将 `2.0 * v` 改为 `1 + λ*v²` |
| **P1** | 目标车道筛选 | 过滤非目标车道的障碍物 |
| **P1** | 动态走廊宽度 | 根据路宽调整 d_critical/d_outer |
| **P2** | 路宽获取 | 从地图 API 获取或使用默认值 |

---

## 五、待讨论问题

### 5.1 路宽信息来源

**问题**：Argoverse2 地图是否提供车道边界线？

**备选方案**：使用固定值 3.5m（中国城市道路标准）

### 5.2 参数调优

**问题**：W_base, λ, k 的最优值需要实验验证

**建议**：先用推荐值跑仿真，再根据效果微调

### 5.3 多风险源聚合

**问题**：当存在多个 Ghost Point 时，Cost 如何聚合？

| 选项 | 公式 | 优缺点 |
|------|------|--------|
| Sum | ΣC_i | 简单但可能过度惩罚 |
| Max | max(C_i) | 只考虑最危险的 |
| LogSumExp | log(Σexp(αC_i))/α | 平滑近似 Max |

---

## 六、代码修改清单

### 6.1 utils.py 修改

```python
# 新增函数
def is_obstacle_on_target_lane(obs_pos, target_lane, lane_width):
    """目标车道筛选"""
    ...

def calculate_adaptive_corridor(lane_width, ego_vel):
    """动态走廊宽度"""
    ...

# 修改 get_semantic_risk_sources
def get_semantic_risk_sources(..., target_lane=None, lane_width=3.5):
    # 新增筛选
    if not is_obstacle_on_target_lane(obs_pos, target_lane, lane_width):
        continue
    
    # 输出增加 lane_heading
    risk_sources.append({
        ...
        'lane_heading': ego_heading,  # 新增
    })
```

### 6.2 planner.py 修改

```python
def evaluate_traj_tree(...):
    # ========= 替换整个 Risk Cost 计算块 =========
    
    W_BASE = 20.0
    LAMBDA_VEL = 0.02
    K_STEEP = 2.0
    
    for risk in risk_sources:
        # 计算横向距离
        heading = risk['lane_heading']
        dx = ego_pos[0] - risk['pos'][0]
        dy = ego_pos[1] - risk['pos'][1]
        d_lat = abs(-dx * sin(heading) + dy * cos(heading)) - EGO_HALF_WIDTH
        
        # Sigmoid 计算
        exponent = K_STEEP * (d_lat - d_critical)
        exponent = np.clip(exponent, -10, 10)
        sigmoid_factor = 1.0 / (1.0 + np.exp(exponent))
        
        # 速度增益
        kinematic_factor = 1.0 + LAMBDA_VEL * (ego_vel ** 2)
        
        # 最终 Cost
        cost += W_BASE * kinematic_factor * sigmoid_factor
```

---

## 七、下一步行动

1. **确认路宽获取方式**（用户决策）
2. **实现 P0 任务**（替换 Cost 公式）
3. **仿真验证**（跑鬼探头场景）
4. **参数调优**（根据效果微调）
