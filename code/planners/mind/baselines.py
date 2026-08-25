"""Published occlusion-aware baselines, re-implemented as velocity-bound
supervisors wrapped around the unmodified MIND planner.

B1  ``reachset``  Set-based reachability speed constraint
    (Orzechowski et al., ITSC 2018; Koschi & Althoff, IEEE T-IV 2021).
    A hypothetical worst-case pedestrian stands at the occlusion boundary.
    At every control step the ego must either (a) be able to come to a full
    stop before the conflict point, or (b) provably clear the conflict point
    before the earliest possible pedestrian arrival. Otherwise the commanded
    speed is clamped to the strict stopping-feasibility bound.

B2  ``shadow``    Dynamic shadow tracking with information persistence
    (Nager, Censi, Frazzoli, ICRA 2019).
    Same constraint mapping as B1, but the hypothetical pedestrian is pinned
    to a per-occluder *frontier* that is propagated over time: corridor depth
    that has been observed empty cannot spawn a pedestrian, while a hidden
    pedestrian may advance toward the lane at the worst-case crossing speed.
    The frontier therefore starts deeper than the geometric boundary and is
    re-clamped by line-of-sight clearing as the ego approaches.

Both baselines deliberately reuse the PA-LOI occlusion screening front-end
(``get_semantic_risk_sources``) ONLY as a detector of static occluders and
their sight-line tangent (ghost) points. They consume none of the PA-LOI
shaping machinery: no TTA weight ramp, no hinge potential, no comfort cap,
no crawl floor. They act purely as a post-planner velocity-bound supervisor,
mirroring how the original papers wrap a nominal planner with a formal
verification / constraint layer.

Shared physical parameters are identical to the deployed PA-LOI + AEB stack
(paper Table II) so that the comparison isolates the constraint-mapping
design choice rather than the physics assumptions.
"""

from __future__ import annotations

import math

import numpy as np

# --- Physics shared with the deployed stack (paper Table II) ---
A_BRAKE = 4.0          # braking authority a_b (m/s^2)
TAU_R = 0.2            # system response latency tau_r (s)
DELTA_STOP = 0.5       # stopping safety margin delta (m), as in Eq. (6)
EGO_HALF_WIDTH = 1.0   # w_e / 2 (m)
EGO_HALF_LENGTH = 2.0  # ego half length for the clearing condition (m)
DT_CTRL = 0.2          # discrete control step of the trajectory tree (s)

# --- Baseline-specific assumptions ---
V_PED_MAX = 2.0        # worst-case hidden-pedestrian crossing speed (m/s)
SHADOW_DEPTH = 6.0     # how deep behind the boundary B2 probes visibility (m)
SHADOW_STEP = 0.25     # corridor sampling resolution for B2 (m)


def v_stop(d_s: float) -> float:
    """Strict stopping-feasibility bound, identical to paper Eq. (6)."""
    if d_s <= DELTA_STOP:
        return 0.0
    at = A_BRAKE * TAU_R
    return max(0.0, -at + math.sqrt(at * at + 2.0 * A_BRAKE * (d_s - DELTA_STOP)))


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def _source_geometry(src: dict):
    """Extract (d_s, d_lat) of a screened occlusion source.

    d_s   longitudinal distance from the ego to the lane-center conflict
          point, approximated by the tangent (ghost) point projection that
          the screening front-end already computes.
    d_lat lateral distance from the ego path centerline to the closest
          hidden point (the tangent corner of the occluder).
    """
    return float(src.get("ghost_longitudinal", 0.0)), float(src.get("ghost_lateral", 0.0))


def _bound_from_hypothesis(d_s: float, d_lat: float, ego_v: float) -> float:
    """Map one worst-case pedestrian hypothesis to a velocity bound.

    Pass verification: the ego clears the conflict point before the earliest
    pedestrian arrival -> the source imposes no bound this step. Otherwise
    the ego must stay stopping-feasible: v <= v_stop(d_s).
    """
    if d_s <= -EGO_HALF_LENGTH:
        return float("inf")  # conflict already behind the rear axle
    travel = max(d_lat - EGO_HALF_WIDTH, 0.0)
    t_ped = travel / V_PED_MAX if V_PED_MAX > 1e-6 else float("inf")
    t_pass = (max(d_s, 0.0) + EGO_HALF_LENGTH + DELTA_STOP) / max(ego_v, 0.1) + TAU_R
    if t_pass <= t_ped:
        return float("inf")
    return v_stop(d_s)


class ReachSetVelocityBound:
    """B1: stateless set-based reachability speed constraint."""

    name = "reachset"

    def compute_bound(self, risk_sources, ego_pos, ego_heading, ego_v, dt=DT_CTRL):
        del ego_pos, ego_heading, dt  # geometry comes pre-projected
        bound = float("inf")
        for src in risk_sources:
            d_s, d_lat = _source_geometry(src)
            bound = min(bound, _bound_from_hypothesis(d_s, d_lat, ego_v))
        return bound


def _segment_intersects_rect(p0, p1, center, heading, half_len, half_width) -> bool:
    """Liang-Barsky segment clipping against an oriented rectangle."""
    c, s = math.cos(-heading), math.sin(-heading)
    rot = np.array([[c, -s], [s, c]])
    a = rot @ (np.asarray(p0, dtype=float) - center)
    b = rot @ (np.asarray(p1, dtype=float) - center)
    d = b - a
    t0, t1 = 0.0, 1.0
    for axis, half in ((0, half_len), (1, half_width)):
        if abs(d[axis]) < 1e-12:
            if abs(a[axis]) > half:
                return False
            continue
        lo = (-half - a[axis]) / d[axis]
        hi = (half - a[axis]) / d[axis]
        if lo > hi:
            lo, hi = hi, lo
        t0, t1 = max(t0, lo), min(t1, hi)
        if t0 > t1:
            return False
    return True


class DynamicShadowBound:
    """B2: dynamic shadow tracking with information persistence."""

    name = "shadow"

    def __init__(self):
        # occluder key -> closest possible hidden-pedestrian lateral offset
        self._frontier: dict = {}

    @staticmethod
    def _source_key(src: dict):
        idx = src.get("agent_idx")
        if idx is not None:
            return int(idx)
        pos = _to_numpy(src.get("occ_pos", src.get("pos", (0.0, 0.0))))
        return (round(float(pos[0]), 1), round(float(pos[1]), 1))

    def _first_hidden_lat(self, src, ego_pos, ego_heading):
        """Closest corridor depth whose line of sight is still occluded.

        The crossing corridor is sampled outward from the ghost point along
        the lateral direction away from the ego path. A corridor point is
        cleared when the ego->point segment no longer intersects the
        occluder rectangle. Returns the lateral offset (from the ego path
        centerline) of the first still-hidden sample, or None if the whole
        probed corridor is visible.
        """
        occ_pos = src.get("occ_pos")
        occ_heading = src.get("occ_heading")
        half_len = src.get("occ_half_len")
        half_width = src.get("occ_half_width")
        d_s, d_lat = _source_geometry(src)
        if occ_pos is None or occ_heading is None or d_s <= 0.0:
            return d_lat  # insufficient geometry: keep worst case

        occ_pos = _to_numpy(occ_pos)
        ego_pos = _to_numpy(ego_pos)
        heading = float(ego_heading)
        forward = np.array([math.cos(heading), math.sin(heading)])
        ghost = _to_numpy(src["pos"])

        # Lateral unit vector pointing from the ego path toward the occluder side.
        side = np.array([-forward[1], forward[0]])
        if np.dot(side, ghost - ego_pos) < 0.0:
            side = -side

        # Corridor anchor: the point of the ego path abeam the ghost point.
        anchor = ego_pos + forward * d_s
        steps = int(SHADOW_DEPTH / SHADOW_STEP)
        for k in range(steps + 1):
            lat = d_lat + k * SHADOW_STEP
            point = anchor + side * lat
            if _segment_intersects_rect(
                ego_pos, point, occ_pos, float(occ_heading), float(half_len), float(half_width)
            ):
                return lat
        return None  # fully cleared within the probed depth

    def compute_bound(self, risk_sources, ego_pos, ego_heading, ego_v, dt=DT_CTRL):
        bound = float("inf")
        seen = set()
        for src in risk_sources:
            key = self._source_key(src)
            seen.add(key)
            d_s, d_lat = _source_geometry(src)
            hidden_lat = self._first_hidden_lat(src, ego_pos, ego_heading)

            if hidden_lat is None:
                # Entire probed corridor verified empty: the closest possible
                # pedestrian is beyond the probe depth.
                cleared_floor = d_lat + SHADOW_DEPTH
            else:
                cleared_floor = hidden_lat

            prev = self._frontier.get(key)
            if prev is None:
                # First observation: a pedestrian may already stand at the
                # closest hidden point, but no deeper knowledge exists yet.
                frontier = cleared_floor
            else:
                # Hidden pedestrian may advance toward the lane at V_PED_MAX,
                # but can never be closer than the depth just verified empty.
                frontier = max(prev - V_PED_MAX * dt, cleared_floor)
            self._frontier[key] = frontier

            bound = min(bound, _bound_from_hypothesis(d_s, frontier, ego_v))

        # Forget occluders that left the screening corridor.
        for key in list(self._frontier.keys()):
            if key not in seen:
                del self._frontier[key]
        return bound


SUPERVISORS = {
    ReachSetVelocityBound.name: ReachSetVelocityBound,
    DynamicShadowBound.name: DynamicShadowBound,
}


def make_supervisor(mode: str):
    try:
        return SUPERVISORS[mode]()
    except KeyError:
        raise ValueError(
            f"Unknown BASELINE_MODE {mode!r}; expected one of {sorted(SUPERVISORS)}"
        ) from None


def apply_velocity_bound(ret_ctrl, ego_v: float, bound: float, dt: float = DT_CTRL):
    """Clamp the commanded acceleration so the next-step speed obeys ``bound``.

    Deceleration authority is capped at A_BRAKE, matching the physical
    braking limit shared by every system under test.
    """
    if not math.isfinite(bound):
        return ret_ctrl, False
    a_required = (bound - ego_v) / dt
    a_cmd = min(float(ret_ctrl[0]), max(a_required, -A_BRAKE))
    if a_cmd >= float(ret_ctrl[0]):
        return ret_ctrl, False
    out = np.array(ret_ctrl, dtype=float)
    out[0] = a_cmd
    return out, True
