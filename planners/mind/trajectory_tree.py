import numpy as np
import theano.tensor as T
from common.geometry import get_point_mean_distances
from planners.basic.tree import Tree, Node
from planners.ilqr.solver import iLQR
from planners.ilqr.dynamics import AutoDiffDynamics
from planners.ilqr.cost import TreeCost
from planners.ilqr.utils import gen_dist_field
from planners.ilqr.potential import ControlPotential, StatePotential, StateConstraint, PotentialField


class TrajectoryTreeOptimizer:
    def __init__(self, config=None):
        self.config = config
        self.ilqr = iLQR(self._get_dynamic_model(self.config.dt, 2.5))
        self.cost_tree = None
        self.debug = None

    def init_warm_start_cost_tree(self, scen_tree: Tree, init_state, init_ctrl, target_lane, target_vel):
        x0 = self._get_init_state(init_state, init_ctrl)
        res = self.config.w_opt_cfg['smooth_grid_res']
        grid_size = self.config.w_opt_cfg['smooth_grid_size']
        offsets, xx, yy, dist_field = gen_dist_field(x0, target_lane, grid_size, res)
        quad_dist_field = dist_field ** 2

        cost_tree = Tree()
        cost_tree.add_node(Node(-1, None, x0))
        # DFS to convert scenario tree to trajectory tree maintain the last traj node index of each scenario node
        last_traj_node_index = {}
        queue = [scen_tree.get_root()]
        while queue:
            cur_node = queue.pop()
            # [prob] [ego + exo, N, state_dim] [ego + exo, N, state_dim, state_dim]
            prob, trajs, covs, tgt_pts = cur_node.data
            last_index = last_traj_node_index[cur_node.parent_key] if cur_node.parent_key is not None else -1
            duration = trajs.shape[1]
            for i in range(duration):
                if i % 2 == 1:
                    continue
                cur_index = len(cost_tree.nodes) - 1
                quad_cost_field = self.config.w_opt_cfg['w_tgt'] * prob * quad_dist_field
                pot_field = PotentialField(offsets, res, xx, yy, quad_cost_field)
                state_pot = StatePotential(self.config.w_opt_cfg['w_des_state'] * prob,
                                           np.array([0, 0, target_vel, 0.0, 0.0, 0.0]))
                state_con = StateConstraint(self.config.w_opt_cfg['w_state_con'] * prob,
                                            self.config.w_opt_cfg['state_lower_bound'],
                                            self.config.w_opt_cfg['state_upper_bound'])
                ctrl_pot = ControlPotential(self.config.w_opt_cfg['w_ctrl'] * prob)

                cost_tree.add_node(Node(cur_index, last_index, [[pot_field, state_pot, state_con], [ctrl_pot]]))
                last_index = cur_index
            last_traj_node_index[cur_node.key] = len(cost_tree.nodes) - 2
            for child_key in cur_node.children_keys:
                queue.append(scen_tree.get_node(child_key))

        self.cost_tree = TreeCost(cost_tree, self.config.state_size, self.config.action_size)
    #注意这里是计算风险的地方
    def init_cost_tree(self, scen_tree: Tree, init_state, init_ctrl, target_lane, target_vel, risk_sources=None):
        x0 = self._get_init_state(init_state, init_ctrl)
        res = self.config.opt_cfg['smooth_grid_res']
        grid_size = self.config.opt_cfg['smooth_grid_size']
        offsets, xx, yy, dist_field = gen_dist_field(x0, target_lane, grid_size, res)
        centroids = np.vstack([xx.ravel(), yy.ravel()]).T
        quad_dist_field = dist_field ** 2

        cost_tree = Tree()
        cost_tree.add_node(Node(-1, None, x0))
        # DFS to convert scenario tree to trajectory tree maintain the last traj node index of each scenario node
        last_traj_node_index = {}
        queue = [scen_tree.get_root()]
        while len(queue) > 0:
            cur_node = queue.pop()
            # [prob] [ego + exo, N, state_dim] [ego + exo, N, state_dim, state_dim]
            prob, trajs, covs, tgt_pts = cur_node.data
            last_index = last_traj_node_index[cur_node.parent_key] if cur_node.parent_key is not None else -1

            duration = trajs.shape[1]
            for i in range(duration):
                if i % 2 == 1:
                    continue
                cur_index = len(cost_tree.nodes) - 1
                cov_dist_field = dist_field * 0.0

                ego_mean = trajs[0, i]
                ego_cov = covs[0, i] + self.config.opt_cfg['w_ego_cov_offset']
                ego_dist_field = (get_point_mean_distances(centroids, ego_mean) - ego_cov).reshape(cov_dist_field.shape)
                ego_dist_field = np.maximum(ego_dist_field, 0.0)

                # import matplotlib.pyplot as plt
                # _, ax = plt.subplots(figsize=(12, 12))
                # c = ax.pcolormesh(xx_discrete, yy_discrete, ego_dis_field, cmap='viridis', shading='auto')
                # plt.colorbar(c, ax=ax)
                # plt.show()

                for exo_idx in range(1, trajs.shape[0]):
                    exo_mean = trajs[exo_idx, i]
                    # 原版逻辑：不加距离权重
                    exo_cov = covs[exo_idx, i] + self.config.opt_cfg['w_exo_cov_offset']
                    exo_dis_field = (exo_cov - get_point_mean_distances(centroids, exo_mean)).reshape(
                        cov_dist_field.shape)
                    exo_dis_field = np.maximum(exo_dis_field, 0.0)
                    exo_dis_field[exo_dis_field > 0] += self.config.opt_cfg['w_exo_cost_offset']

                    # 原版：直接累加，不加权重
                    cov_dist_field += exo_dis_field

                # --- PA-LOI: KA-RF 各向异性横向屏障 (最终修正版) ---
                # 【修正1】使用真实车道航向计算横向投影
                # 【修正2】使用 VelocityAwareRiskPotential 提供速度梯度
                # 【修正3】移除静态场重复计算，避免双重计费 (Double Counting Fix)
                risk_potentials = []
                if risk_sources:
                    # 从 target_lane 计算车道航向
                    if target_lane is not None and len(target_lane) >= 2:
                        # 取前两个点计算切线方向
                        lane_vec = target_lane[1] - target_lane[0]
                        lane_heading = np.arctan2(lane_vec[1], lane_vec[0])
                    else:
                        # 回退：使用 ego 当前朝向 (从 init_state)
                        lane_heading = init_state[3] if len(init_state) > 3 else 0.0
                    
                    for risk in risk_sources:
                        risk_mean = risk['pos'].cpu().numpy()
                        ghost_lateral = risk.get('ghost_lateral', 1.5)
                        phantom_state = risk.get('phantom_state', 'BRAKE')
                        
                        # [PA-LOI v18] 直接使用 Lerp 权重，不再按状态缩放
                        w_base = risk['weight']
                        
                        # 【关键】创建速度感知势场 (唯一的风险 Cost 来源)
                        # 这个势场的 get_gradient 会返回 ∂C/∂v，让 iLQR 知道减速能降 Cost
                        # ⚠️ 不再往 cov_dist_field 添加，避免双重计费
                        from planners.ilqr.potential import VelocityAwareRiskPotential
                        
                        # [PA-LOI v52] 透传 v_safe 参数
                        v_safe = risk.get('v_safe', 0.0)
                        
                        risk_pot = VelocityAwareRiskPotential(
                            risk_pos=risk_mean,
                            lane_heading=lane_heading,
                            ghost_lateral=ghost_lateral,
                            w_base=w_base,
                            v_safe=v_safe,      # [v52] Hinge Loss 阈值
                            lambda_v=0.1,       # 已被 v_safe 逻辑取代，但保留接口兼容
                            ego_half_width=1.0,
                            k_steep=2.0
                        )
                        risk_potentials.append(risk_pot)
                        
                        # 【已移除】静态 CostMap 叠加 - 避免双重计费
                        # 原来这里有 cov_dist_field += w_base * sigmoid_field
                        # 现在风险完全由 VelocityAwareRiskPotential 独立负责
                # ---------------------------------------------------

                quad_cost_field = (self.config.opt_cfg['w_tgt'] * prob * quad_dist_field +
                                   self.config.opt_cfg['w_exo'] * cov_dist_field +
                                   self.config.opt_cfg['w_ego'] * ego_dist_field)

                pot_field = PotentialField(offsets, res, xx, yy, quad_cost_field)

                state_pot = StatePotential(self.config.opt_cfg['w_des_state'] * prob,
                                           np.array([0, 0, target_vel, 0.0, 0.0, 0.0]))

                state_con = StateConstraint(self.config.opt_cfg['w_state_con'] * prob,
                                            self.config.opt_cfg['state_lower_bound'],
                                            self.config.opt_cfg['state_upper_bound'])
                
                # [PA-LOI v18] 默认控制权重（不再根据状态切换）
                # 空间梯度已被抑制，不需要锁方向盘
                w_ctrl = self.config.opt_cfg['w_ctrl']
                
                ctrl_pot = ControlPotential(w_ctrl * prob)

                # 【PA-LOI】将速度感知风险势场加入 state_pots
                # 这样 iLQR 在优化时会计算 ∂C/∂v，实现减速功能
                state_pots = [pot_field, state_pot, state_con] + risk_potentials
                
                cost_tree.add_node(Node(cur_index, last_index, [state_pots, [ctrl_pot]]))
                last_index = cur_index
            last_traj_node_index[cur_node.key] = len(cost_tree.nodes) - 2
            for child_key in cur_node.children_keys:
                queue.append(scen_tree.get_node(child_key))

        self.cost_tree = TreeCost(cost_tree, self.config.state_size, self.config.action_size)

    def warm_start_solve(self, us_init=None):
        if us_init is None:
            us_init = np.zeros((self.cost_tree.tree.size() - 1, self.config.action_size))

        xs, us = self.ilqr.fit(us_init, self.cost_tree)
        return xs, us

    def solve(self, us_init=None):
        if us_init is None:
            us_init = np.zeros((self.cost_tree.tree.size() - 1, self.config.action_size))


        xs, us = self.ilqr.fit(us_init, self.cost_tree)

        # return traj tree
        traj_tree = Tree()
        for node in self.cost_tree.tree.nodes.values():
            if node.parent_key is None:
                traj_tree.add_node(Node(node.key, None, [node.data, np.zeros(self.config.action_size)]))
            else:
                traj_tree.add_node(Node(node.key, node.parent_key, [xs[node.key], us[node.key]]))
        return traj_tree

    def _get_init_state(self, init_state, init_ctrl):
        return np.array(
            [init_state[0], init_state[1], init_state[2], init_state[3], init_ctrl[0], init_ctrl[1]])

    def _get_dynamic_model(self, dt, wb):
        x_inputs = [
            T.dscalar("x"),
            T.dscalar("y"),
            T.dscalar("v"),
            T.dscalar("q"),
            T.dscalar("a"),
            T.dscalar("theta"),
        ]

        u_inputs = [
            T.dscalar("da"),
            T.dscalar("dtheta"),
        ]

        f = T.stack([
            x_inputs[0] + x_inputs[2] * T.cos(x_inputs[3]) * dt,
            x_inputs[1] + x_inputs[2] * T.sin(x_inputs[3]) * dt,
            x_inputs[2] + x_inputs[4] * dt,
            x_inputs[3] + x_inputs[2] / wb * T.tan(x_inputs[5]) * dt,
            x_inputs[4] + u_inputs[0] * dt,
            x_inputs[5] + u_inputs[1] * dt,
        ])

        return AutoDiffDynamics(f, x_inputs, u_inputs)
