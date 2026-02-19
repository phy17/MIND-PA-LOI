import copy
import numpy as np
from av2.map.map_api import ArgoverseStaticMap
from av2.map.lane_segment import LaneType, LaneMarkType


class SemanticMap:
    def __init__(self):
        self.map_data = None
        self.limits = None
        self.semantic_lanes = None
        self.semantic_lanes_infos = None
        self.exo_agents = []
        self.ego_agent = None

    def load_from_argo2(self, file_dir):
        """ Load the map from the argo2 map """
        self.map_data = ArgoverseStaticMap.from_json(file_dir)
        self.process_argo2_map_data()

    def process_argo2_map_data(self):
        semantic_lane_seq = []
        # first get the lanes that have no predecessors
        for lane_id, lane in self.map_data.vector_lane_segments.items():
            # check the predecessors are not contained in the static_map
            has_predecessors = False
            for predecessor in lane.predecessors:
                if predecessor in self.map_data.vector_lane_segments:
                    has_predecessors = True
                    break

            if not has_predecessors:
                semantic_lane_seq.append([lane_id])

        # extend the semantic lane
        while True:
            has_new_ext = False
            ext_lane_seqs = []
            for lane_seq in semantic_lane_seq:
                ext_lane_seq = []
                for succ in self.map_data.vector_lane_segments[lane_seq[-1]].successors:
                    if succ in self.map_data.vector_lane_segments:
                        has_new_ext = True
                        ext_lane_seq.append(lane_seq + [succ])
                if len(ext_lane_seq) > 0:
                    ext_lane_seqs += ext_lane_seq
                else:
                    ext_lane_seqs += [lane_seq]
            semantic_lane_seq = ext_lane_seqs
            if not has_new_ext:
                break

        self.semantic_lanes = dict()
        self.semantic_lanes_infos = dict()

        # construct the semantic lanes
        pts = []
        for idx, lane_seq in enumerate(semantic_lane_seq):
            #  lane data = [lane points, lane type, left_lane_type, right_lane_type]
            lane_seq_centerline = []
            lane_type, intersect, cross_left, cross_right, left, right = [], [], [], [], [], []
            for lane_id in lane_seq:
                cl_raw = self.map_data.get_lane_segment_centerline(lane_id)[:-1, 0:2]  # use xy
                lane_seq_centerline.append(cl_raw)

                lane = self.map_data.vector_lane_segments[lane_id]
                # ~ lane type
                lane_type_tmp = np.zeros(3, np.float32)
                if lane.lane_type == LaneType.VEHICLE:
                    lane_type_tmp[0] = 1
                elif lane.lane_type == LaneType.BIKE:
                    lane_type_tmp[1] = 1
                elif lane.lane_type == LaneType.BUS:
                    lane_type_tmp[2] = 1
                else:
                    assert False, "[Error] Wrong lane type"
                lane_type.append(np.expand_dims(lane_type_tmp, axis=0).repeat(cl_raw.shape[0], axis=0))

                # ~ intersection
                if lane.is_intersection:
                    intersect.append(np.ones(cl_raw.shape[0], np.float32))
                else:
                    intersect.append(np.zeros(cl_raw.shape[0], np.float32))
                # ~ lane marker type
                cross_left_tmp = np.zeros(3, np.float32)
                if lane.left_mark_type in [LaneMarkType.DASH_SOLID_YELLOW,
                                           LaneMarkType.DASH_SOLID_WHITE,
                                           LaneMarkType.DASHED_WHITE,
                                           LaneMarkType.DASHED_YELLOW,
                                           LaneMarkType.DOUBLE_DASH_YELLOW,
                                           LaneMarkType.DOUBLE_DASH_WHITE]:
                    cross_left_tmp[0] = 1  # crossable
                elif lane.left_mark_type in [LaneMarkType.DOUBLE_SOLID_YELLOW,
                                             LaneMarkType.DOUBLE_SOLID_WHITE,
                                             LaneMarkType.SOLID_YELLOW,
                                             LaneMarkType.SOLID_WHITE,
                                             LaneMarkType.SOLID_DASH_WHITE,
                                             LaneMarkType.SOLID_DASH_YELLOW,
                                             LaneMarkType.SOLID_BLUE]:
                    cross_left_tmp[1] = 1  # not crossable
                else:
                    cross_left_tmp[2] = 1  # unknown/none

                cross_right_tmp = np.zeros(3, np.float32)
                if lane.right_mark_type in [LaneMarkType.DASH_SOLID_YELLOW,
                                            LaneMarkType.DASH_SOLID_WHITE,
                                            LaneMarkType.DASHED_WHITE,
                                            LaneMarkType.DASHED_YELLOW,
                                            LaneMarkType.DOUBLE_DASH_YELLOW,
                                            LaneMarkType.DOUBLE_DASH_WHITE]:
                    cross_right_tmp[0] = 1  # crossable
                elif lane.right_mark_type in [LaneMarkType.DOUBLE_SOLID_YELLOW,
                                              LaneMarkType.DOUBLE_SOLID_WHITE,
                                              LaneMarkType.SOLID_YELLOW,
                                              LaneMarkType.SOLID_WHITE,
                                              LaneMarkType.SOLID_DASH_WHITE,
                                              LaneMarkType.SOLID_DASH_YELLOW,
                                              LaneMarkType.SOLID_BLUE]:
                    cross_right_tmp[1] = 1  # not crossable
                else:
                    cross_right_tmp[2] = 1  # unknown/none

                cross_left.append(np.expand_dims(cross_left_tmp, axis=0).repeat(cl_raw.shape[0], axis=0))
                cross_right.append(np.expand_dims(cross_right_tmp, axis=0).repeat(cl_raw.shape[0], axis=0))

                # ~ has left/right neighbor
                if lane.left_neighbor_id is None:
                    left.append(np.zeros(cl_raw.shape[0], np.float32))  # w/o left neighbor
                else:
                    left.append(np.ones(cl_raw.shape[0], np.float32))
                if lane.right_neighbor_id is None:
                    right.append(np.zeros(cl_raw.shape[0], np.float32))  # w/o right neighbor
                else:
                    right.append(np.ones(cl_raw.shape[0], np.float32))

            lane_seq_centerline = np.concatenate(lane_seq_centerline).astype(np.float32)
            intersect = np.concatenate(intersect, axis=0)
            lane_type = np.concatenate(lane_type, axis=0)
            cross_left = np.concatenate(cross_left, axis=0)
            cross_right = np.concatenate(cross_right, axis=0)
            left = np.concatenate(left, axis=0)
            right = np.concatenate(right, axis=0)
            pts.append(lane_seq_centerline)
            # check whether there is an overlap between points
            segs = lane_seq_centerline[1:] - lane_seq_centerline[:-1]
            assert np.all(np.linalg.norm(segs, axis=1) > 1e-2)
            self.semantic_lanes[idx] = lane_seq_centerline
            self.semantic_lanes_infos[idx] = [intersect, lane_type, cross_left, cross_right, left, right]

        pts = np.concatenate(pts, axis=0)
        # get the map limits
        self.limits = [[np.min(pts[:, 0]), np.max(pts[:, 0])],
                       [np.min(pts[:, 1]), np.max(pts[:, 1])]]

    def process_custom_map_data(self):
        """ Process the map data """
        # get semantic lanes
        lanes = self.map_data['LANES'][0]
        topos = self.map_data['LANE_TOPOS'][0]
        # get semantic lanes
        semantic_lanes = dict()
        for idx, topo in enumerate(topos):
            lane_points = []
            for lane_idx in topo:
                lane_points.append(lanes[lane_idx][0])
            lane_points = np.concatenate(lane_points, axis=0)
            semantic_lanes[idx] = lane_points
        self.semantic_lanes = semantic_lanes
        self.limits = self.map_data['LIMITS'][0]

    def get_map_limits(self):
        """ Get the map limits """
        return self.limits


class LocalSemanticMap:
    def __init__(self, ego_id, semantic_map):
        self.ego_id = ego_id
        self.map_data = copy.deepcopy(semantic_map.map_data)
        self.semantic_lanes = copy.deepcopy(semantic_map.semantic_lanes)
        self.semantic_lanes_infos = copy.deepcopy(semantic_map.semantic_lanes_infos)
        self.target_lane = None
        self.target_lane_info = None
        self.target_velocity = None
        self.exo_agents = []
        self.ego_agent = None

    def update_target_lane(self, target_lane):
        """ Update the target lane """
        self.target_lane = copy.deepcopy(target_lane)

    def update_target_lane_info(self, target_lane_info):
        self.target_lane_info = target_lane_info

    def update_target_velocity(self, target_velocity):
        """ Update the target velocity """
        self.target_velocity = target_velocity

    def update_observation(self, agents):
        """ Update the surrounding vehicles """
        exo_agents = []
        for agent in agents:
            if agent.id != self.ego_id:
                exo_agents.append(agent)
            else:
                self.ego_agent = agent
        self.exo_agents = exo_agents

    def get_closest_semantic_lane(self, pos, ang, ang_threshold=np.deg2rad(30.0)):
        """ Get the closest semantic lane """
        min_dist = 1e6
        closest_lane = None
        for lane_id, lane in self.semantic_lanes.items():
            dists = np.linalg.norm(lane - pos, axis=1)
            # closest idx on the lane [0, len(lane) - 2]
            closest_idx = min(np.argmin(dists), len(lane) - 2)
            # get the direction of the lane
            lane_dir = lane[closest_idx + 1] - lane[closest_idx]
            lane_dir = lane_dir / np.linalg.norm(lane_dir)
            # filter out lanes that are not in the same direction
            if np.dot(lane_dir, np.array([np.cos(ang), np.sin(ang)])) > np.cos(ang_threshold):
                dist = np.min(dists)
                if dist < min_dist:
                    min_dist = dist
                    closest_lane = lane_id
        return closest_lane


    def get_semantic_lane(self, id):
        """ Get the semantic lane """
        return self.semantic_lanes[id]

    # 【修正】路口锁存值 (Latching)
    _last_valid_lane_width = 3.5

    def get_lane_width_at_position(self, lane_id, position):
        """
        计算指定位置的车道宽度
        
        【修正】增加路口锁存逻辑：
        在十字路口中心，车道宽度可能计算出 10+ 米的异常值。
        此时应沿用进入路口前的锁存宽度。
        
        Args:
            lane_id: 原始地图中的 lane_id
            position: [x, y] 位置坐标
        
        Returns:
            float: 车道宽度 (m)
        """
        DEFAULT_LANE_WIDTH = 3.5
        
        try:
            if self.map_data is None:
                return self._last_valid_lane_width
            
            lane = self.map_data.vector_lane_segments.get(lane_id)
            if lane is None:
                return self._last_valid_lane_width
            
            # 【修正】路口检测和锁存
            # is_intersection 属性表示这是一个交叉路口的车道段
            is_intersection = getattr(lane, 'is_intersection', False)
            
            if is_intersection:
                # 在路口中，使用锁存的上一个有效宽度
                return self._last_valid_lane_width
            
            # 获取左右边界
            left_boundary = lane.left_lane_boundary.xyz[:, :2]
            right_boundary = lane.right_lane_boundary.xyz[:, :2]
            
            # 找到距离 position 最近的边界点
            left_dists = np.linalg.norm(left_boundary - position, axis=1)
            right_dists = np.linalg.norm(right_boundary - position, axis=1)
            
            left_closest = left_boundary[np.argmin(left_dists)]
            right_closest = right_boundary[np.argmin(right_dists)]
            
            # 宽度 = 左右边界点之间的距离
            width = np.linalg.norm(left_closest - right_closest)
            
            # 合理性检查
            if width < 2.0 or width > 6.0:  # 收紧上限到 6.0m
                return self._last_valid_lane_width
            
            # 更新锁存值
            LocalSemanticMap._last_valid_lane_width = float(width)
            
            return float(width)
            
        except Exception as e:
            print(f"[WARN] get_lane_width_at_position failed: {e}")
            return self._last_valid_lane_width

    def get_road_total_width(self, lane_id, position):
        """
        估算道路总宽度（包括当前车道和相邻车道）
        
        Args:
            lane_id: 原始地图中的 lane_id
            position: [x, y] 位置坐标
        
        Returns:
            float: 道路总宽度 (m)
        """
        try:
            if self.map_data is None:
                return self.get_lane_width_at_position(lane_id, position)
            
            lane = self.map_data.vector_lane_segments.get(lane_id)
            if lane is None:
                return self.get_lane_width_at_position(lane_id, position)
            
            # 当前车道宽度
            current_width = self.get_lane_width_at_position(lane_id, position)
            total_width = current_width
            
            # 左侧邻居车道
            if lane.left_neighbor_id is not None:
                left_width = self.get_lane_width_at_position(lane.left_neighbor_id, position)
                total_width += left_width
            
            # 右侧邻居车道
            if lane.right_neighbor_id is not None:
                right_width = self.get_lane_width_at_position(lane.right_neighbor_id, position)
                total_width += right_width
            
            return total_width
            
        except Exception as e:
            print(f"[WARN] get_road_total_width failed: {e}")
            return self.get_lane_width_at_position(lane_id, position)

