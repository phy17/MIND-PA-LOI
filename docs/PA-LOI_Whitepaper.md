# PA-LOI Code Implementation & Technical Whitepaper

**Version**: v1.1 (v50 Final Release)
**Scope**: Autonomous Driving Control Algorithm Development, IEEE IV/ITSC Paper Writing
**Core Codebase**: `MIND/planners`

---
## 1. System Architecture

PA-LOI enhances safety at occluded intersections by dynamically modifying the iLQR optimizer's Cost Function based on perception of road topology risks (e.g., blind spots), inducing "spontaneous" defensive driving behavior without explicit rules.

**Data Flow**:
1.  **Map Server**: Provides lane geometry and static obstacle information.
2.  **Utils (`utils.py`)**: 
    *   Identifies occlusion sources (Inject Phantom).
    *   Calculates TTA (Time-to-Arrival).
    *   Calculates Risk Weight ($w_{base}$).
    *   **New in v51**: Sets Safety Threshold $v_{safe}$ (configured to 0.0 for stability).
3.  **Planner (`planner.py`)**: 
    *   Filters invalid risk sources (Critical Fix).
    *   Assembles the Cost Function.
4.  **iLQR Core (`potential.py`)**: 
    *   Calculates Hinge-Loss Cost, Gradient, Hessian.
    *   Iteratively solves for the optimal trajectory.

---

## 2. Core Algorithm Implementation

### 2.1 Hinge-Loss Velocity Potential

**File**: `planners/ilqr/potential.py`
**Class**: `VelocityAwareRiskPotential`

Traditional potential fields penalize position ($\frac{1}{d^2}$), causing vehicles to swerve (lateral avoidance) rather than brake. To **decouple longitudinal and lateral control** and **force deceleration**, we designed a Hinge-Loss Kinetic Potential.

#### Mathematical Model
$$ J_{risk}(s) = w_{base} \cdot \sigma(d_{lat}) \cdot \max(0, v - v_{safe})^2 $$

*   $d_{lat}$: Lateral distance to the risk zone edge (Clearance).
*   $\sigma(\cdot)$: Sigmoid activation function, smoothly enabling risk when laterally close.
*   $\max(0, v - v_{safe})^2$: Hinge-Loss kinetic energy term.
*   $v_{safe}$: Configured to **0.0 m/s** in v50 for maximum stability (Defensive Mode).

#### Code Implementation (Simplified)
```python
class VelocityAwareRiskPotential:
    def get_potential(self, state):
        pos = state[:2]
        v = state[2]
        
        # 1. Calculate Lateral Clearance
        lateral = self._compute_lateral_distance(pos)
        clearance = max(lateral - self.ego_half_width, 0.0)
        
        # 2. Sigmoid Activation
        sig = self._compute_sigmoid(clearance)
        
        # 3. Hinge-Loss Kinetic Cost
        # v50 config: v_safe = 0.0 -> effectively v^2
        excess_vel = max(0.0, v - self.v_safe)
        return self.w_base * sig * (excess_vel ** 2)

    def get_gradient(self, state):
        # Critical Innovation: Explicit Velocity Gradient ∂C/∂v
        # ∂C/∂v = W * Sigmoid * 2 * (v - v_safe)
        gradient = np.zeros(len(state))
        
        v = state[2]
        excess_vel = max(0.0, v - self.v_safe)
        if excess_vel <= 0: return gradient
        
        lateral = self._compute_lateral_distance(state[:2])
        clearance = max(lateral - self.ego_half_width, 0.0)
        sig = self._compute_sigmoid(clearance)
        
        # Only populate velocity channel, force position gradients to 0
        # This prevents "swerving" to minimize cost
        gradient[2] = self.w_base * sig * 2.0 * excess_vel
        return gradient
```

**Design Intent**:
*   **Gradient[2] (Velocity)**: Provides a direct deceleration signal. Larger weight = stronger braking gradient.
*   **Gradient[0,1] (Position)**: Forced to 0. Prioritizes longitudinal control over lateral evasion.

---

### 2.2 Dynamic TTA Weighting

**File**: `planners/mind/utils.py`
**Location**: `get_semantic_risk_sources`

To mimic human "anticipation", weight $w_{base}$ is not fixed but modulated by **Time-to-Arrival (TTA)**.

#### Logic (v50 Configuration)
```python
# [PA-LOI v50] TTA-Weight Mapping
tta = phantom_result['tta_ego']

if tta > 8.0:
    weight = 0.0               # Safe Zone: No interference
elif tta > 3.0:
    # Warning Zone (8s -> 3s): Weight linearly increases 0 -> 15
    # Creates mild drag (acc ~ -0.5 m/s²)
    weight = 15.0 * (8.0 - tta) / 5.0
elif tta > 1.0:
    # Braking Zone (3s -> 1s): Weight linearly increases 15 -> 30
    # Creates significant braking (acc ~ -2.0 m/s²)
    weight = 15.0 + 15.0 * (3.0 - tta) / 2.0
else:
    weight = 30.0              # Emergency Zone: Max braking
```

---

### 2.3 Risk Source Filtering (The Critical Fix)

**File**: `planners/mind/planner.py`
**Location**: `plan` method

This was the root cause of previous failures (v1-v49 tests).

#### Buggy Code (Pre-v50)
```python
# Error: inject_phantom only eligible when TTA < 1.5s
# Result: All long-range (8s-1.5s) anticipation signals were discarded!
active_risk_sources = [r for r in risk_sources if r.get('inject_phantom', False)]
```

#### Fixed Code (v50)
```python
# Correct: Allow risk source as long as weight > 0 (i.e., TTA < 8.0s)
active_risk_sources = [r for r in risk_sources if r.get('weight', 0) > 0]
```

---

## 3. Parameter Dictionary

| Parameter | Variable | Value | Physical Meaning & Turing Guide |
| :--- | :--- | :--- | :--- |
| **Base Weight** | `w_base` | 30.0 | **Max Braking Force**. 30 ≈ -2m/s². <br>If Hinge Loss v_safe > 0, this needs to be 100+. |
| **Lookahead TTA** | `TTA Thresholds` | 8.0s | **Anticipation Horizon**. Larger value = earlier, smoother braking. |
| **Hard Brake TTA** | `TTA Thresholds` | 3.0s | **Braking Onset**. |
| **Lateral Threshold** | `ghost_lateral` | ~3.0m | **Lateral Activation Range**. Auto-calculated. |
| **Sigmoid Steepness**| `k_steep` | 2.0 | **Boundary Hardness**. 2.0 is smooth; 10.0 is wall-like. |
| **Safe Speed** | `v_safe` | 0.0 | **Hinge Threshold**. Set to 0.0 for max stability. |

---

## 4. Reproduction Components

1.  **Environment**: `numpy`, `torch`.
2.  **Config**: `configs/ghost_experiment.json`, set `"enable_ghost_probe": true`.
3.  **Command**: `python experiments/ghost_probe/run_ghost_experiment.py`.
4.  **Expected Output**:
    *   Console: `[PA-LOI] ... cost=xx ... acc=-xx` showing anticipation.
    *   Log: `Collisions: 0`.
    *   Behavior: Smooth linear deceleration from 30m out.

---
**Version**: 1.1
**Author**: Antigravity AI
