import os
import torch
import copy
import numpy as np

# [E1 ablation] PA-LOI risk-layer knobs. Defaults are the deployed values
# (paper Table II); environment overrides exist only for the parameter
# sensitivity sweep and leave normal runs bit-identical when unset.
_PALOI_ETA = float(os.environ.get("PALOI_ETA", "0.6"))
_PALOI_VFLOOR = float(os.environ.get("PALOI_VFLOOR", "2.0"))
_PALOI_WMAX = float(os.environ.get("PALOI_WMAX", "25.0"))
from typing import List, Any, Dict
from shapely.geometry import LineString
from av2.map.lane_segment import LaneType, LaneMarkType
from av2.datasets.motion_forecasting.data_schema import ObjectType

def gpu(data, device):
    """
    Transfer tensor in `data` to gpu recursively
    `data` can be dict, list or tuple
    """
    if isinstance(data, list) or isinstance(data, tuple):
        data = [gpu(x, device=device) for x in data]
    elif isinstance(data, dict):
        data = {key: gpu(_data, device=device) for key, _data in data.items()}
    elif isinstance(data, torch.Tensor):
        data = data.contiguous().to(device, non_blocking=True)
    return data



def from_numpy(data):
    """Recursively transform numpy.ndarray to torch.Tensor.
    """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = from_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [from_numpy(x) for x in data]
    if isinstance(data, np.ndarray):
        """Pytorch now has bool type."""
        data = torch.from_numpy(data)
    return data


def padding_traj_nn(traj):
    n = len(traj)
    # forward
    buff = None
    for i in range(n):
        if np.all(buff == None) and np.all(traj[i] != None):
            buff = traj[i]
        if np.all(buff != None) and np.all(traj[i] == None):
            traj[i] = buff
        if np.all(buff != None) and np.all(traj[i] != None):
            buff = traj[i]
    # backward
    buff = None
    for i in reversed(range(n)):
        if np.all(buff == None) and np.all(traj[i] != None):
            buff = traj[i]
        if np.all(buff != None) and np.all(traj[i] == None):
            traj[i] = buff
        if np.all(buff != None) and np.all(traj[i] != None):
            buff = traj[i]
    return traj


def tgt_gather(batch_size, tgt_nodes_list, tgt_rpe_list):
    tgt_nodes_feat = []
    tgt_rpe_feat = []
    # ~ calc tgt feat
    for tgt_nodes, tgt_rpe in zip(tgt_nodes_list, tgt_rpe_list):
        tgt_nodes_feat.append(tgt_nodes)
        tgt_rpe_feat.append(tgt_rpe)

    tgt_nodes_feat = torch.stack(tgt_nodes_feat, dim=0)
    tgt_rpe_feat = torch.stack(tgt_rpe_feat, dim=0).reshape(batch_size, -1)
    return tgt_nodes_feat, tgt_rpe_feat


def graph_gather(batch_size, graphs):
    '''
        graphs[i]
            node_ctrs           torch.Size([116, N_{pt}, 2])
            node_vecs           torch.Size([116, N_{pt}, 2])
            intersect           torch.Size([116, N_{pt}])
            lane_type           torch.Size([116, N_{pt}, 3])
            cross_left          torch.Size([116, N_{pt}, 3])
            cross_right         torch.Size([116, N_{pt}, 3])
            left                torch.Size([116, N_{pt}])
            right               torch.Size([116, N_{pt}])
            lane_ctrs           torch.Size([116, 2])
            lane_vecs           torch.Size([116, 2])
            num_nodes           1160
            num_lanes           116
    '''
    lane_idcs = list()
    lane_count = 0
    for i in range(batch_size):
        l_idcs = torch.arange(lane_count, lane_count + graphs[i]["num_lanes"])
        lane_idcs.append(l_idcs)
        lane_count = lane_count + graphs[i]["num_lanes"]

    graph = dict()
    for key in ["node_ctrs", "node_vecs", "intersect", "lane_type", "cross_left", "cross_right", "left", "right"]:
        graph[key] = torch.cat([x[key] for x in graphs], 0)
    for key in ["lane_ctrs", "lane_vecs"]:
        graph[key] = [x[key] for x in graphs]

    lanes = torch.cat([graph['node_ctrs'],
                       graph['node_vecs'],
                       graph['intersect'].unsqueeze(2),
                       graph['lane_type'],
                       graph['cross_left'],
                       graph['cross_right'],
                       graph['left'].unsqueeze(2),
                       graph['right'].unsqueeze(2)], dim=-1)  # [N_{lane}, 9, F]
    return lanes, lane_idcs


def actor_gather(batch_size, trajs):
    num_actors = [len(x['TRAJS_CTRS']) for x in trajs]

    act_feats = []
    for i in range(batch_size):
        traj_pos = trajs[i]['TRAJS_POS_OBS']
        traj_disp = torch.zeros_like(traj_pos)
        traj_disp[:, 1:, :] = traj_pos[:, 1:, :] - traj_pos[:, :-1, :]

        act_feat = torch.cat([traj_disp,
                              trajs[i]['TRAJS_ANG_OBS'],
                              trajs[i]['TRAJS_VEL_OBS'],
                              trajs[i]['TRAJS_TYPE'],
                              trajs[i]['PAD_OBS'].unsqueeze(-1)], dim=-1)
        act_feats.append(act_feat)

    act_feats = [x.transpose(1, 2) for x in act_feats]
    actors = torch.cat(act_feats, 0)  # [N_a, feat_len, 50], N_a is agent number in a batch
    actors = actors[..., 2:]  # ! tmp solution
    actor_idcs = []  # e.g. [tensor([0, 1, 2, 3]), tensor([ 4,  5,  6,  7,  8,  9, 10])]
    count = 0
    for i in range(batch_size):
        idcs = torch.arange(count, count + num_actors[i])
        actor_idcs.append(idcs)
        count += num_actors[i]
    return actors, actor_idcs


def collate_fn(batch: List[Any]) -> Dict[str, Any]:
    if len(batch) == 0:
        return None
    batch = from_numpy(batch)
    data = dict()
    data['BATCH_SIZE'] = len(batch)
    # Batching by use a list for non-fixed size
    for key in batch[0].keys():
        data[key] = [x[key] for x in batch]
    '''
        Keys:
        'BATCH_SIZE',
        'ORIG', 'ROT',
        'TRAJS', 'LANE_GRAPH', 'RPE'
    '''

    actors, actor_idcs = actor_gather(data['BATCH_SIZE'], data['TRAJS'])
    lanes, lane_idcs = graph_gather(data['BATCH_SIZE'], data["LANE_GRAPH"])
    tgt_nodes, tgt_rpe = tgt_gather(data['BATCH_SIZE'], data['TGT_NODES'], data['TGT_RPE'])

    data['ACTORS'] = actors
    data['ACTOR_IDCS'] = actor_idcs
    data['LANES'] = lanes
    data['LANE_IDCS'] = lane_idcs
    data['TGT_NODES'] = tgt_nodes
    data['TGT_RPE'] = tgt_rpe
    return data


def get_new_lane_graph(lane_graph, orig, rot, device):
    ret_lane_graph = gpu(copy.deepcopy(lane_graph), device=device)
    # transform the lane_ctrs and lane_vecs
    ret_lane_graph['lane_ctrs'] = torch.matmul(ret_lane_graph['lane_ctrs'] - orig, rot)
    ret_lane_graph['lane_vecs'] = torch.matmul(ret_lane_graph['lane_vecs'], rot)

    return ret_lane_graph


def get_origin_rotation(traj_pos, traj_ang, device):
    obs_len = 50
    orig = traj_pos[obs_len - 1]
    theta = traj_ang[obs_len - 1]
    if isinstance(orig, torch.Tensor):
        rot = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                            [torch.sin(theta), torch.cos(theta)]]).to(device)
    elif isinstance(orig, np.ndarray):
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
    return orig, rot, theta


def get_rpe(ctrs, vecs, radius=100.0):
    # distance encoding
    d_pos = (ctrs.unsqueeze(0) - ctrs.unsqueeze(1)).norm(dim=-1)
    mask = None
    d_pos = d_pos * 2 / radius  # scale [0, radius] to [0, 2]
    pos_rpe = d_pos.unsqueeze(0)

    # angle diff
    cos_a1 = get_cos(vecs.unsqueeze(0), vecs.unsqueeze(1))
    sin_a1 = get_sin(vecs.unsqueeze(0), vecs.unsqueeze(1))
    # print('cos_a1: ', cos_a1.shape, 'sin_a1: ', sin_a1.shape)

    v_pos = ctrs.unsqueeze(0) - ctrs.unsqueeze(1)
    cos_a2 = get_cos(vecs.unsqueeze(0), v_pos)
    sin_a2 = get_sin(vecs.unsqueeze(0), v_pos)
    # print('cos_a2: ', cos_a2.shape, 'sin_a2: ', sin_a2.shape)

    ang_rpe = torch.stack([cos_a1, sin_a1, cos_a2, sin_a2])
    rpe = torch.cat([ang_rpe, pos_rpe], dim=0)
    return rpe, mask


def get_angle(vel):
    return torch.atan2(vel[..., 1], vel[..., 0])


def get_cos(v1, v2):
    ''' input: [M, N, 2], [M, N, 2]
        output: [M, N]
        cos(<a,b>) = (a dot b) / |a||b|
    '''
    v1_norm = v1.norm(dim=-1)
    v2_norm = v2.norm(dim=-1)
    v1_x, v1_y = v1[..., 0], v1[..., 1]
    v2_x, v2_y = v2[..., 0], v2[..., 1]
    cos_dang = (v1_x * v2_x + v1_y * v2_y) / (v1_norm * v2_norm + 1e-10)
    return cos_dang


def get_sin(v1, v2):
    ''' input: [M, N, 2], [M, N, 2]
        output: [M, N]
        sin(<a,b>) = (a x b) / |a||b|
    '''
    v1_norm = v1.norm(dim=-1)
    v2_norm = v2.norm(dim=-1)
    v1_x, v1_y = v1[..., 0], v1[..., 1]
    v2_x, v2_y = v2[..., 0], v2[..., 1]
    sin_dang = (v1_x * v2_y - v1_y * v2_x) / (v1_norm * v2_norm + 1e-10)
    return sin_dang


def get_agent_trajectories(agent_obs, device):
    obs_len = 50

    # * find idcs
    av_idx = None
    exo_idcs = list()  # exclude AV
    key_list = []
    for idx, key in enumerate(agent_obs.keys()):
        if key == 'AV':
            av_idx = idx
        else:
            exo_idcs.append(idx)
        key_list.append(key)

    sorted_idcs = [av_idx] + exo_idcs
    sorted_cat = ["av"] + ["exo"] * len(exo_idcs)
    sorted_tid = [key_list[idx] for idx in sorted_idcs]

    # * get timesteps and timesteps
    ts = np.arange(0, obs_len)  # [0, 1,..., 49]
    ts_obs = ts[obs_len - 1]  # always 49

    # * must follows the pre-defined order
    trajs_pos, trajs_ang, trajs_vel, trajs_type, has_flags = list(), list(), list(), list(), list()
    trajs_tid, trajs_cat = list(), list()  # track id and category
    for k, ind in enumerate(sorted_idcs):
        key = key_list[ind]
        track = agent_obs[key]

        # * pass if no observation at the last timestep
        if track.object_states[-1].observed is False:
            continue

        # * get traj
        observed_flag = np.array([1 if s.observed else 0 for s in track.object_states])

        traj_ts = np.arange(obs_len - len(track.object_states), obs_len)
        traj_ts = traj_ts[observed_flag == 1]

        traj_pos = np.array(
            [list(x.position) if x.observed else [0.0, 0.0] for x in track.object_states])  # [N_{frames}, 2]
        traj_pos = traj_pos[observed_flag == 1]
        traj_ang = np.array([x.heading if x.observed else 0.0 for x in track.object_states])  # [N_{frames}]
        traj_ang = traj_ang[observed_flag == 1]
        traj_vel = np.array(
            [list(x.velocity) if x.observed else [0.0, 0.0] for x in track.object_states])  # [N_{frames}, 2]
        traj_vel = traj_vel[observed_flag == 1]

        # print(has_flag.shape, traj_ts.shape, traj_ts)
        has_flag = np.zeros_like(ts)
        has_flag[traj_ts] = 1
        # object type
        obj_type = np.zeros(7)  # 7 types
        if track.object_type == ObjectType.VEHICLE:
            obj_type[0] = 1
        elif track.object_type == ObjectType.PEDESTRIAN:
            obj_type[1] = 1
        elif track.object_type == ObjectType.MOTORCYCLIST:
            obj_type[2] = 1
        elif track.object_type == ObjectType.CYCLIST:
            obj_type[3] = 1
        elif track.object_type == ObjectType.BUS:
            obj_type[4] = 1
        elif track.object_type == ObjectType.UNKNOWN:
            obj_type[5] = 1
        else:
            obj_type[6] = 1  # for all static objects
        traj_type = np.zeros((len(ts), 7))
        traj_type[traj_ts] = obj_type

        # pad pos, nearest neighbor
        traj_pos_pad = np.full((len(ts), 2), None)
        traj_pos_pad[traj_ts] = traj_pos
        traj_pos_pad = padding_traj_nn(traj_pos_pad)
        # pad ang, nearest neighbor
        traj_ang_pad = np.full(len(ts), None)
        traj_ang_pad[traj_ts] = traj_ang
        traj_ang_pad = padding_traj_nn(traj_ang_pad)
        # pad vel, fill zeros
        traj_vel_pad = np.full((len(ts), 2), 0.0)
        traj_vel_pad[traj_ts] = traj_vel

        trajs_pos.append(traj_pos_pad)
        trajs_ang.append(traj_ang_pad)
        trajs_vel.append(traj_vel_pad)
        trajs_type.append(traj_type)
        has_flags.append(has_flag)
        trajs_tid.append(sorted_tid[k])
        trajs_cat.append(sorted_cat[k])

    
    trajs_pos = np.array(trajs_pos).astype(np.float32)  # [N, 110(50), 2]
    trajs_ang = np.array(trajs_ang).astype(np.float32)  # [N, 110(50)]
    trajs_vel = np.array(trajs_vel).astype(np.float32)  # [N, 110(50), 2]
    trajs_type = np.array(trajs_type).astype(np.int16)  # [N, 110(50), 7]
    has_flags = np.array(has_flags).astype(np.int16)  # [N, 110(50)]
    
    # Convert to Tensor first
    trajs_pos = torch.from_numpy(trajs_pos).to(device)
    trajs_ang = torch.from_numpy(trajs_ang).to(device)
    trajs_vel = torch.from_numpy(trajs_vel).to(device)
    trajs_type = torch.from_numpy(trajs_type).to(device)
    has_flags = torch.from_numpy(has_flags).to(device)

    return (trajs_pos, trajs_ang, trajs_vel, trajs_type, has_flags, trajs_tid, trajs_cat)


def update_lane_graph_from_argo(static_map, orig, rot):
    node_ctrs, node_vecs, lane_type, intersect, cross_left, cross_right, left, right = [], [], [], [], [], [], [], []
    lane_ctrs, lane_vecs = [], []
    NUM_SEG_POINTS = 10
    SEG_LENGTH = 15.0

    for lane_id, lane in static_map.vector_lane_segments.items():
        # get lane centerline
        cl_raw = static_map.get_lane_segment_centerline(lane_id)[:, 0:2]  # use xy
        assert cl_raw.shape[0] == NUM_SEG_POINTS, "[Error] Wrong num of points in lane - {}:{}".format(
            lane_id, cl_raw.shape[0])

        cl_ls = LineString(cl_raw)
        num_segs = np.max([int(np.floor(cl_ls.length / SEG_LENGTH)), 1])
        ds = cl_ls.length / num_segs

        for i in range(num_segs):
            s_lb = i * ds
            s_ub = (i + 1) * ds
            num_sub_segs = NUM_SEG_POINTS

            cl_pts = []
            for s in np.linspace(s_lb, s_ub, num_sub_segs + 1):
                cl_pts.append(cl_ls.interpolate(s))
            ctrln = np.array(LineString(cl_pts).coords)  # [num_sub_segs + 1, 2]
            ctrln = (ctrln - orig).dot(rot)  # to local frame

            anch_pos = np.mean(ctrln, axis=0)
            anch_vec = (ctrln[-1] - ctrln[0]) / np.linalg.norm(ctrln[-1] - ctrln[0])
            anch_rot = np.array([[anch_vec[0], -anch_vec[1]],
                                 [anch_vec[1], anch_vec[0]]])

            lane_ctrs.append(anch_pos)
            lane_vecs.append(anch_vec)

            ctrln = (ctrln - anch_pos).dot(anch_rot)  # to instance frame

            ctrs = np.asarray((ctrln[:-1] + ctrln[1:]) / 2.0, np.float32)
            vecs = np.asarray(ctrln[1:] - ctrln[:-1], np.float32)
            node_ctrs.append(ctrs)  # middle point
            node_vecs.append(vecs)

            # ~ lane type
            lane_type_tmp = np.zeros(3)
            if lane.lane_type == LaneType.VEHICLE:
                lane_type_tmp[0] = 1
            elif lane.lane_type == LaneType.BIKE:
                lane_type_tmp[1] = 1
            elif lane.lane_type == LaneType.BUS:
                lane_type_tmp[2] = 1
            else:
                assert False, "[Error] Wrong lane type"
            lane_type.append(np.expand_dims(lane_type_tmp, axis=0).repeat(num_sub_segs, axis=0))

            # ~ intersection
            if lane.is_intersection:
                intersect.append(np.ones(num_sub_segs, np.float32))
            else:
                intersect.append(np.zeros(num_sub_segs, np.float32))

            # ~ lane marker type
            cross_left_tmp = np.zeros(3)
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

            cross_right_tmp = np.zeros(3)
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

            cross_left.append(np.expand_dims(cross_left_tmp, axis=0).repeat(num_sub_segs, axis=0))
            cross_right.append(np.expand_dims(cross_right_tmp, axis=0).repeat(num_sub_segs, axis=0))

            # ~ has left/right neighbor
            if lane.left_neighbor_id is None:
                left.append(np.zeros(num_sub_segs, np.float32))  # w/o left neighbor
            else:
                left.append(np.ones(num_sub_segs, np.float32))
            if lane.right_neighbor_id is None:
                right.append(np.zeros(num_sub_segs, np.float32))  # w/o right neighbor
            else:
                right.append(np.ones(num_sub_segs, np.float32))

    node_idcs = []  # List of range
    count = 0
    for i, ctr in enumerate(node_ctrs):
        node_idcs.append(range(count, count + len(ctr)))
        count += len(ctr)

    lane_idcs = []  # node belongs to which lane, e.g. [0   0   0 ... 122 122 122]
    for i, idcs in enumerate(node_idcs):
        lane_idcs.append(i * np.ones(len(idcs), np.int16))
    # print("lane_idcs: ", lane_idcs.shape, lane_idcs)

    graph = dict()
    # geometry
    graph['node_ctrs'] = np.stack(node_ctrs, axis=0).astype(np.float32)
    graph['node_vecs'] = np.stack(node_vecs, axis=0).astype(np.float32)
    graph['lane_ctrs'] = np.array(lane_ctrs).astype(np.float32)
    graph['lane_vecs'] = np.array(lane_vecs).astype(np.float32)
    # node features
    graph['lane_type'] = np.stack(lane_type, axis=0).astype(np.int16)
    graph['intersect'] = np.stack(intersect, axis=0).astype(np.int16)
    graph['cross_left'] = np.stack(cross_left, axis=0).astype(np.int16)
    graph['cross_right'] = np.stack(cross_right, axis=0).astype(np.int16)
    graph['left'] = np.stack(left, axis=0).astype(np.int16)
    graph['right'] = np.stack(right, axis=0).astype(np.int16)
    graph['num_nodes'] = graph['node_ctrs'].shape[0] * graph['node_ctrs'].shape[1]
    graph['num_lanes'] = graph['lane_ctrs'].shape[0]
    return graph


def get_closest_point_on_segment(segment, point):
    p1, p2 = segment
    # Vector from p1 to p2
    segment_vector = p2 - p1

    # Projected vector from p1 to p
    projected_vector = torch.dot(point - p1, segment_vector) / torch.dot(segment_vector, segment_vector)

    # Clamp the projection to the segment
    t = torch.clamp(projected_vector, 0, 1)

    # Find the closest point on the segment
    closest = p1 + t * segment_vector
    return closest


def get_distance_to_polyline(polyline, point):
    min_distance = None

    for i in range(len(polyline) - 1):
        segment = (polyline[i], polyline[i + 1])
        closest = get_closest_point_on_segment(segment, point)
        distance = torch.norm(closest - point)

        if min_distance is None or distance < min_distance:
            min_distance = distance

    return min_distance


def get_covariance_matrix(data):
    # check is torch or numpy
    if isinstance(data, torch.Tensor):
        ret_shape = data.shape[:-1] + (2, 2)
        sigma_x = data[..., 0]
        sigma_y = data[..., 1]
        rho = data[..., 2]
        sigma_xy = rho * sigma_x * sigma_y
        return torch.stack([sigma_x ** 2, sigma_xy, sigma_xy, sigma_y ** 2], dim=-1).view(ret_shape)
    elif isinstance(data, np.ndarray):
        ret_shape = data.shape[:-1] + (2, 2)
        sigma_x = data[..., 0]
        sigma_y = data[..., 1]
        rho = data[..., 2]
        sigma_xy = rho * sigma_x * sigma_y
        return np.stack([sigma_x ** 2, sigma_xy, sigma_xy, sigma_y ** 2], axis=-1).reshape(ret_shape)
    else:
        raise ValueError("data should be torch.Tensor or numpy.ndarray")


def get_max_covariance(data):
    # check is torch or numpy
    if isinstance(data, torch.Tensor):
        ret_shape = data.shape[:-1] + (1,)
        sigma_x = data[..., 0]
        sigma_y = data[..., 1]
        # only return the maximum sigma
        return torch.maximum(sigma_x, sigma_y).view(ret_shape)
    elif isinstance(data, np.ndarray):
        ret_shape = data.shape[:-1] + (1,)
        sigma_x = data[..., 0]
        sigma_y = data[..., 1]
        # only return the maximum sigma
        return np.maximum(sigma_x, sigma_y).reshape(ret_shape)
    else:
        raise ValueError("data should be torch.Tensor or numpy.ndarray")


def calculate_adaptive_corridor(lane_width, road_width, ego_vel):
    """
    基于路宽和车速动态计算双层走廊边界
    [修正版] 添加几何约束钳位 (Geometric Clamping)
    
    Args:
        lane_width: 当前车道宽度 (m)
        road_width: 道路总宽度 (m)，包括相邻车道
        ego_vel: 自车速度 (m/s)
    
    Returns:
        d_critical: 内层边界（绝对禁区）
        d_outer: 外层边界（感知范围）
    """
    EGO_WIDTH = 2.0
    SAFETY_MARGIN = 0.2  # 安全余量
    
    # ====== 内层 (d_critical) - 几何约束钳位 ======
    # 1. 动力学需求：基础 0.5m + 速度缓冲
    dynamic_need = 0.5 + 0.03 * abs(ego_vel)
    
    # 2. 几何约束：内层宽度绝不能超过 (车道宽/2 - 0.2m)
    geometric_limit = (lane_width / 2.0) - SAFETY_MARGIN
    
    # 3. 取两者较小值 (关键钳位)
    d_critical = min(dynamic_need, geometric_limit)
    d_critical = max(d_critical, 0.2)  # 兜底
    
    # ====== 外层 (d_outer) - 物理边界约束 ======
    # 外层宽度绝不能超过道路物理边界
    physical_boundary = road_width / 2.0
    d_outer = min(7.0, physical_boundary)  # [Fix] 把 5.0 提升到 7.0
    
    # 确保外层 > 内层
    d_outer = max(d_outer, d_critical + 0.5)
    
    return d_critical, d_outer


def is_obstacle_on_target_lane(obs_pos, target_lane, lane_width=3.5):
    """
    检查障碍物是否在目标车道上或附近
    
    Args:
        obs_pos: 障碍物位置 [x, y] (numpy array)
        target_lane: 目标车道中心线 [N, 2] (numpy array)
        lane_width: 车道宽度 (m)
    
    Returns:
        bool: True 如果障碍物可能阻挡 Ego
    """
    if target_lane is None or len(target_lane) == 0:
        return True  # 无目标车道信息时默认不过滤
    
    # 计算障碍物到车道中心线的最短距离
    dists = np.linalg.norm(target_lane - obs_pos, axis=1)
    min_dist = np.min(dists)
    
    # 如果距离 > 车道宽度的一半 + 余量，说明不在目标车道上
    # 如果距离 > 车道宽度的一半 + 余量，说明不在目标车道上
    # 修正：对于鬼探头检测，我们需要关注路边的遮挡物
    # 原来是 (lane_width / 2.0) + 0.5 (约 2.25m)
    # 修正：对于鬼探头检测，我们需要关注路边的遮挡物
    threshold = (lane_width / 2.0) + 4.5  # [Fix] 扩大检测范围，捕获路边大巴
    
    return min_dist < threshold


def project_to_lateral_distance(ego_pos, ghost_point, lane_heading):
    """
    计算横向距离（用于 KA-RF Sigmoid 计算）
    
    Args:
        ego_pos: 自车位置 [x, y]
        ghost_point: 风险点位置 [x, y]
        lane_heading: 车道方向角 (rad)
    
    Returns:
        float: 横向距离 (m)
    """
    dx = ego_pos[0] - ghost_point[0]
    dy = ego_pos[1] - ghost_point[1]
    
    # 投影到横向平面
    sin_h = np.sin(lane_heading)
    cos_h = np.cos(lane_heading)
    
    lateral_dist = abs(-dx * sin_h + dy * cos_h)
    
    return lateral_dist


def is_separated_by_solid_line(obs_pos, ego_pos, ego_heading, lane_mark_type):
    """
    检查障碍物和 Ego 之间是否有不可跨越的分隔线
    
    Args:
        obs_pos: 障碍物位置 [x, y]
        ego_pos: Ego 位置 [x, y]
        ego_heading: Ego 航向角 (rad)
        lane_mark_type: 车道线类型向量 [crossable, not_crossable, unknown]
    
    Returns:
        bool: True 如果被实线/双黄线分隔（应该过滤）
    """
    # 判断障碍物在 Ego 的左边还是右边
    vec_to_obs = obs_pos - ego_pos
    ego_forward = np.array([np.cos(ego_heading), np.sin(ego_heading)])
    
    # 叉积判断左右
    cross = ego_forward[0] * vec_to_obs[1] - ego_forward[1] * vec_to_obs[0]
    
    # 检查车道线是否不可跨越 (lane_mark_type[1] == 1 表示实线)
    if lane_mark_type is not None and len(lane_mark_type) >= 2:
        is_solid = lane_mark_type[1] > 0.5  # 不可跨越
        if is_solid:
            return True  # 被实线分隔，应该过滤
    
    return False


def calculate_phantom_behavior(longitudinal_dist, lateral_dist, ego_vel):
    """
    【修正版】基于 TTA 和物理可达性的幻影状态机
    
    修正要点：
    1. 人类速度改回 5.0 m/s (合理冲刺速度)
    2. 增加物理可达性检查：鬼需要跑多快才能撞上？
    3. 如果所需速度 > 人类极限，则无需幻影
    
    Args:
        longitudinal_dist: 纵向距离 (m)
        lateral_dist: 横向距离 (m)
        ego_vel: 自车速度 (m/s)
    
    Returns:
        dict: 幻影状态和相关信息
    """
    # 【修正】人类冲刺速度 5.0 m/s (18 km/h，合理上限)
    HUMAN_MAX_SPEED = 5.0
    
    # [PA-LOI Fix] 缩短前瞻时间，防止过早触发 BRAKE 状态
    # 原值 3.0 -> 改为 1.5 (配合 Experiment A/v23 的极限测试)
    LOOKAHEAD_TIME = 1.5  # 秒 (Critical Reaction Time)
    
    result = {
        'state': 'OBSERVE',
        'inject_phantom': False,
        'risk_field_only': True,
        'safe_to_pass': False,
        'tta_ego': float('inf'),
        'tta_human': float('inf'),
        'v_required': 0.0  # 鬼需要的速度
    }
    
    # 计算 TTA
    if ego_vel > 0.1:
        result['tta_ego'] = longitudinal_dist / ego_vel
    if lateral_dist > 0.1:
        result['tta_human'] = lateral_dist / HUMAN_MAX_SPEED
    
    tta_ego = result['tta_ego']
    tta_human = result['tta_human']
    
    # 【关键修正】物理可达性检查
    # 鬼需要跑多快才能在 Ego 到达前拦住 Ego？
    if tta_ego > 0.01:
        v_required = lateral_dist / tta_ego
        result['v_required'] = v_required
    else:
        v_required = float('inf')
    
    # 安全通过条件
    result['safe_to_pass'] = tta_ego < tta_human
    
    # ====== 修正后的状态机 ======
    
    # 物理可达性检查：如果鬼跑断腿也撞不上，无需幻影
    if v_required > HUMAN_MAX_SPEED:
        result['state'] = 'OBSERVE'
        result['inject_phantom'] = False
        result['risk_field_only'] = True
    
    # 距离检查：太远也无需幻影
    elif tta_ego > LOOKAHEAD_TIME:
        result['state'] = 'OBSERVE'
        result['inject_phantom'] = False
        result['risk_field_only'] = True
    
    # 既近，又能撞上 -> 必须处理
    else:
        result['state'] = 'BRAKE'
        result['inject_phantom'] = True
        result['risk_field_only'] = False
    
    return result


def get_semantic_risk_sources(trajs_pos, trajs_vel, trajs_type, trajs_ang, ego_pos, ego_heading, 
                                device='cpu', ego_vel=None, lane_width=3.5, road_width=None,
                                target_lane=None):
    """
    [PA-LOI 升级版] 识别语义级风险源（鬼探头区域）
    
    增强功能：
    1. 动态双层走廊（基于路宽和车速）
    2. TTA 状态机（基于时间而非固定距离）
    3. 目标车道筛选
    
    Args:
        trajs_pos: [N, T, 2] 所有智能体位置轨迹
        trajs_vel: [N, T, 2] 所有智能体速度轨迹
        trajs_type: [N, T, type_dim] 类型 one-hot
        trajs_ang: [N, T] 航向角
        ego_pos: [2] Ego 当前位置
        ego_heading: scalar Ego 当前航向
        device: torch device
        ego_vel: scalar Ego 当前速度 (m/s)，用于 TTA 和动态走廊计算
        lane_width: float 当前车道宽度 (m)
        road_width: float 道路总宽度 (m)，默认使用 lane_width
        target_lane: [M, 2] 目标车道中心线，用于筛选
    
    Returns:
        List of risk dictionaries with 'pos', 'cov', 'weight', 'phantom_state'
    """
    risk_sources = []
    filter_log = []
    
    # 默认速度
    if ego_vel is None:
        ego_vel = 5.0  # 默认 5 m/s
    if road_width is None:
        road_width = lane_width
    
    # ====== PA-LOI 核心：动态走廊计算 ======
    d_critical, d_outer = calculate_adaptive_corridor(lane_width, road_width, ego_vel)
    print(f"[PA-LOI] Dynamic Corridor: d_critical={d_critical:.2f}m, d_outer={d_outer:.2f}m (lane={lane_width:.1f}m, v={ego_vel:.1f}m/s)")
    
    # 尺寸估算 (半长, 半宽)
    DIMENSIONS = {
        'BUS': (6.0, 1.5),
        'VEHICLE': (2.5, 1.0),
    }
    
    STATIC_SPEED_THRES = 0.5  # m/s
    MAX_LONGITUDINAL = 50.0   # 扩展检测范围到 50m
    
    curr_step = -1
    num_agents = len(trajs_pos)
    
    if ego_pos is None:
        ego_pos = trajs_pos[0, curr_step]
    if ego_heading is None:
        ego_heading = trajs_ang[0, curr_step]
    
    ego_forward = torch.stack([torch.cos(ego_heading), torch.sin(ego_heading)])
    
    for i in range(num_agents):
        if i == 0:
            continue
        
        agent_log = {'agent_idx': i, 'passed': False, 'reject_reason': None}
        
        # --- 类型筛选 ---
        agent_type_vec = trajs_type[i, curr_step]
        
        is_occluder = False
        half_len, half_width = 2.5, 1.0
        agent_type_str = 'UNKNOWN'
        
        if agent_type_vec[4] == 1:  # BUS
            is_occluder = True
            half_len, half_width = DIMENSIONS['BUS']
            agent_type_str = 'BUS'
        elif agent_type_vec[0] == 1:  # Vehicle
            is_occluder = True
            half_len, half_width = DIMENSIONS['VEHICLE']
            agent_type_str = 'VEHICLE'
        
        agent_log['type'] = agent_type_str
        
        if not is_occluder:
            agent_log['reject_reason'] = 'NOT_OCCLUDER_TYPE'
            continue
        
        # --- 速度筛选 ---
        vel = trajs_vel[i, curr_step]
        speed = torch.norm(vel).item()
        agent_log['speed'] = speed
        
        if speed > STATIC_SPEED_THRES:
            agent_log['reject_reason'] = f'MOVING (speed={speed:.2f}m/s)'
            continue
        
        # --- 位置计算 ---
        obs_pos = trajs_pos[i, curr_step]
        vec_to_obs = obs_pos - ego_pos
        
        longitudinal = torch.dot(vec_to_obs, ego_forward).item()
        lateral = torch.abs(ego_forward[0] * vec_to_obs[1] - ego_forward[1] * vec_to_obs[0]).item()
        
        agent_log['pos'] = obs_pos.cpu().numpy().tolist()
        agent_log['longitudinal'] = longitudinal
        agent_log['lateral'] = lateral
        
        # --- 纵向筛选 ---
        # 修正：为了防止漏掉刚经过车头的长车(公交)，允许一定的负值
        if longitudinal < -5.0:
            agent_log['reject_reason'] = f'BEHIND_EGO (long={longitudinal:.2f}m)'
            filter_log.append(agent_log)
            continue
            
        if longitudinal > MAX_LONGITUDINAL:
            agent_log['reject_reason'] = f'TOO_FAR (long={longitudinal:.2f}m > {MAX_LONGITUDINAL}m)'
            filter_log.append(agent_log)
            continue
        
        # --- PA-LOI: 动态走廊筛选（使用 d_outer 而非固定值）---
        if lateral > d_outer:
            agent_log['reject_reason'] = f'OUT_OF_CORRIDOR (lat={lateral:.2f}m > d_outer={d_outer:.2f}m)'
            filter_log.append(agent_log)
            continue
        
        # --- PA-LOI: 目标车道筛选 ---
        # 注意：threshold 已放宽到 (lane_width/2) + 2.5
        if target_lane is not None:
            if not is_obstacle_on_target_lane(obs_pos.cpu().numpy(), target_lane, lane_width):
                agent_log['reject_reason'] = f'NOT_ON_TARGET_LANE'
                filter_log.append(agent_log)  # 记录被拒绝的原因
                continue
        
        # === PASSED ALL FILTERS ===
        agent_log['passed'] = True
        agent_log['reject_reason'] = None
        filter_log.append(agent_log)
        
        # --- 计算角点和危险点 ---
        obs_ang = trajs_ang[i, curr_step]
        
        cos_a = torch.cos(obs_ang)
        sin_a = torch.sin(obs_ang)
        
        corners_local = torch.tensor([
            [ half_len,  -half_width],
            [ half_len,   half_width],
            [-half_len,   half_width],
            [-half_len,  -half_width],
        ], device=device, dtype=torch.float32)
        
        rot_matrix = torch.tensor([
            [cos_a, -sin_a],
            [sin_a,  cos_a]
        ], device=device, dtype=torch.float32)
        
        corners_global = torch.mm(corners_local, rot_matrix.T) + obs_pos
        
        # --- 视线切点算法 ---
        vecs_to_corners = corners_global - ego_pos
        angles_to_corners = torch.atan2(vecs_to_corners[:, 1], vecs_to_corners[:, 0])
        angle_ego = torch.atan2(ego_forward[1], ego_forward[0])
        
        relative_angles = angles_to_corners - angle_ego
        relative_angles = torch.atan2(torch.sin(relative_angles), torch.cos(relative_angles))
        
        left_tangent_idx = torch.argmax(relative_angles)
        right_tangent_idx = torch.argmin(relative_angles)
        
        vec_to_obs_center = obs_pos - ego_pos
        cross = ego_forward[0] * vec_to_obs_center[1] - ego_forward[1] * vec_to_obs_center[0]
        
        if cross > 0:
            ghost_point = corners_global[right_tangent_idx]
        else:
            ghost_point = corners_global[left_tangent_idx]
        
        # --- 检查危险点 ---
        vec_to_ghost = ghost_point - ego_pos
        proj_forward = torch.dot(vec_to_ghost, ego_forward)
        
        if proj_forward < 0:
            for log in filter_log:
                if log['agent_idx'] == i and log['passed']:
                    log['passed'] = False
                    log['reject_reason'] = 'GHOST_POINT_BEHIND'
            continue
        
        # --- PA-LOI: Ghost Point 使用动态走廊筛选 ---
        ghost_lateral = torch.abs(ego_forward[0] * vec_to_ghost[1] - ego_forward[1] * vec_to_ghost[0]).item()
        ghost_longitudinal = proj_forward.item()
        
        # 使用 d_outer 作为阈值（动态走廊外边界）
        # 修正：原来使用 d_critical + 0.5 (约1.1m)，对于路边停车场景太小
        ghost_threshold = d_outer
        if ghost_lateral > ghost_threshold:
            for log in filter_log:
                if log['agent_idx'] == i and log['passed']:
                    log['passed'] = False
                    log['reject_reason'] = f'GHOST_LATERAL_TOO_FAR (lat={ghost_lateral:.2f}m > {ghost_threshold:.2f}m)'
            continue
        
        # ====== PA-LOI 核心：TTA 状态机 ======
        phantom_result = calculate_phantom_behavior(ghost_longitudinal, ghost_lateral, ego_vel)
        
        # ============================================================
        # [PA-LOI v55] 自适应安全速度 v_safe
        # 
        # 物理推导：假设行人从遮挡物后方走出到车道中心需要横穿距离 d_ped，
        # 速度为 v_ped。AEB 能提供的最大减速度为 a_aeb。
        # 则 AV 在行人完全进入车道前有 t_react = d_ped / v_ped 的反应 + 制动时间。
        # 在此时间内 AV 需要从 v_safe 刹停：v_safe = a_aeb * t_react
        # 
        # 同时限制最大降速幅度为当前速度的 40%，确保舒适性。
        # 即 v_safe >= 0.6 * ego_vel（最多减 40%）
        # ============================================================
        a_aeb = 4.0     # AEB 制动减速度 (m/s²)
        d_ped = ghost_lateral  # 行人需要横穿的距离 (m)
        v_ped = 2.0     # 假设行人速度 (m/s)
        
        # 物理下界：AV 能以此速度在行人横穿时间内刹停
        t_react = d_ped / v_ped if v_ped > 0 else 1.0
        v_safe_physics = a_aeb * t_react  # 物理上允许的最大通过速度
        
        # 舒适性下界：最多只降 (1 - eta) 的速度（部署值 eta=0.6）
        v_safe_comfort = _PALOI_ETA * ego_vel if ego_vel > 0 else 2.5
        
        # 取两者中的较大值（更宽松的约束），并设置绝对下限 v_floor（部署值 2.0 m/s）
        v_safe = max(min(v_safe_physics, ego_vel), v_safe_comfort, _PALOI_VFLOOR)
        
        # [Fix] 补回被误删的变量定义
        tta_ego = phantom_result['tta_ego']
        
        if tta_ego > 6.5:       
            weight = 0.0        # 6.5秒外：自由驾驶，无视盲区
        elif tta_ego > 2.0:     
            # 6.5s -> 2s：权重线性增加 (0 -> w_max)，产生平滑减速梯度
            # [PA-LOI v58] 权重从15→25，TTA从5s→6.5s（温和调整）
            # 原因：v53的15太弱+5s太晚，导致减速窗口不足
            weight = _PALOI_WMAX * (6.5 - tta_ego) / 4.5
        else:                   
            # < 2s：贴近盲区，权重封顶。
            # 此时若车速降至 2.5m/s，势场 Hinge-Loss 梯度归零，车辆匀速溜过路口！
            weight = _PALOI_WMAX
        
        # --- 标准协方差 (仅影响 evaluate_traj_tree) ---
        sigma = 0.8
        risk_cov = get_risk_covariance(sigma, device=device)
        
        risk_sources.append({
            'type': 'GHOST_PROBE',
            'pos': ghost_point,
            'cov': risk_cov,
            'weight': weight,
            'v_safe': v_safe,  # [v52] 传递 v_safe 给 Planner
            'ghost_lateral': ghost_lateral,
            'ghost_longitudinal': ghost_longitudinal,
            # PA-LOI 新增字段
            'phantom_state': phantom_result['state'],
            'tta_ego': phantom_result['tta_ego'],
            'tta_human': phantom_result['tta_human'],
            'inject_phantom': phantom_result['inject_phantom'],
            'safe_to_pass': phantom_result['safe_to_pass'],
            # 遮挡物几何（基线监督器 B1/B2 使用；PA-LOI 路径不消费这些键）
            'agent_idx': i,
            'occ_pos': obs_pos.detach().cpu().numpy(),
            'occ_heading': float(obs_ang),
            'occ_half_len': float(half_len),
            'occ_half_width': float(half_width),
        })
    
    # === PRINT FILTER LOG ===
    # passed_count = sum(1 for log in filter_log if log['passed'])
    # if len(filter_log) > 0 or len(risk_sources) > 0:
    #     print(f"[PA-LOI RISK] Candidates: {len(filter_log)} | Passed: {passed_count} | Final: {len(risk_sources)}")
    #     for rs in risk_sources:
    #         state_emoji = {'OBSERVE': '👀', 'BRAKE': '🚨', 'PASS': '✅'}.get(rs['phantom_state'], '❓')
    #         print(f"  {state_emoji} Agent {rs['agent_idx']}: state={rs['phantom_state']} | "
    #               f"TTA_ego={rs['tta_ego']:.2f}s TTA_human={rs['tta_human']:.2f}s | "
    #               f"weight={rs['weight']:.1f} | phantom={rs['inject_phantom']}")
    
    # # [DEBUG] 如果有候选者但全部被拒绝，打印拒绝原因
    # if len(risk_sources) == 0 and len(filter_log) > 0:
    #     rejected = [log for log in filter_log if not log.get('passed', False)]
    #     if len(rejected) > 0:
    #         print(f"[PA-LOI DEBUG] All candidates rejected! Top 5 reasons:")
    #         for log in rejected[:5]:
    #             print(f"  - Agent {log.get('agent_idx', '?')} ({log.get('type', '?')}): {log.get('reject_reason', 'UNKNOWN')}")
    
    return risk_sources



def get_risk_covariance(sigma, device='cpu'):
    """
    生成风险区域的协方差矩阵（圆形区域）。
    
    Args:
        sigma: 风险区域半径
        device: torch device
    
    Returns:
        [2, 2] 协方差矩阵
    """
    var = sigma ** 2
    cov = torch.tensor([[var, 0.0], [0.0, var]], device=device, dtype=torch.float32)
    return cov

