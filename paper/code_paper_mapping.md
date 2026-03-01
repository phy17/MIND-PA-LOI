# PA-LOI 论文与真实代码映射对照表 (Code-Paper Mapping)

为了确保学术论文的严谨性，以及方便后续开源时的代码对齐，以下列出了本次 `main_zh.tex` 重写后，文章中提到的核心防守理论（对应第三章 **System Architecture**）在项目源码中的精确定位。

| **III-A 物理推断与可达性界限**<br>($TTA_{\text{ego}}$ 的衰减防线) | `planners/mind/utils.py` | `calculate_phantom_behavior` 后的 <br>`weight` 计算块 | 若自车物理抵达估算 `tta_ego > 6.5` 秒开外，其 `weight` 强制归零，仅在紧迫时间内建立惩罚权重。 |
| **III-A 防止极度保守的二元宽松法则**<br>($v_{\text{safe}}$ 物理界限) | `planners/mind/utils.py` | 代码中 `v_safe_physics` 的推导演算段 | 从行人的距离 $d_{\text{ped}}$ 和假定冲突速度 $v_{\text{ped}} = 2.0$ 出发，推演其到达时间窗口。借此反推出车辆能以此通过并不足惧的 $v_{\text{safe}}$。这正是彻底解耦并防范幽灵刹车的真实根源！ |
| **III-B 破除排斥阵列的速度感知合页损失项**<br>$\max(0, v_{\text{ego}} - v_{\text{safe}})^2$ | `planners/ilqr/potential.py` | `VelocityAwareRiskPotential` 代价构筑类 <br>`get_potential` 函数之中 | 通过判定 `excess_vel = max(0.0, v - self.v_safe)` 以及平方赋能 `kinetic_energy = excess_vel ** 2` 完成平滑惩罚。构造了一条专对“入局超速”无死角追缉的高压线。 |
| **III-B 横向极度收缩的能量衰减波**<br>$\text{Sigmoid}(d_{\text{lat}}, k)$ | `planners/ilqr/potential.py` | `VelocityAwareRiskPotential` 代价类内部的<br>`_compute_sigmoid` 工具函数 | 根据自车宽度裁剪得出净距离 `clearance = max(lateral - ego_half_width, 0.0)` 后，使用 `k_steep=2.0` 代入得出横断面衰减核 `sig`。<br>建立了距离盲点越近，所遭受推力镇压更汹涌的阻尼侧壁。 |
| **III-B 扑杀预测器抢跑的纵向无界限防守**<br>(无限延伸势能带) | `planners/ilqr/potential.py` | `_compute_lateral_distance` 计算块 | 该公式中完全不处理 $x$ 向距离差，惩罚域因此顺长道延展进深域！它完美锁死了 iLQR “只要看见路尽头无惩罚便提前降维爆发加速（Anticipatory Acceleration）”的不规企图，只能乖乖全程服从车速规训。 |
| **III-B 熔断预测器画龙/死锁的隐态梯度清零**<br>($\text{导数归零机制}$) | `planners/ilqr/potential.py` | `VelocityAwareRiskPotential` 代价求导解析类 <br>`get_gradient` 函数起始段 | 一道绝妙的 `if excess_vel <= 0: return gradient` 彻底扭转战局。它保障了自车遁入安全速值（低于 $v_{safe}$）时，所有反面推挤制动势能和斜侧梯度顷刻瓦解消除，使车辆全速不减震、不偏航地隐身滑过遮挡。 |
| **III-C 坚守红底线的应急防卫绝杀 AEB**<br>($TTC \le 1.4\text{s}, a_{\text{max}}=\SI{-4.0}{\meter\per\second\squared}$) | `planners/mind/planner.py` | 规划调度总集 `MINDPlanner` 主体内调度的 <br>`plan` 核心总成末端裁决处 | 在全段防线崩盘遭受贴脸行刺入侵而诱发 $TTC \le 1.4\text{s}$ 的致命红色警报时，接管强制推翻前方一切输出决策流，抛出 `ret_ctrl = np.array([-4.0, 0.0])` 的斩杀救命死控。 |
