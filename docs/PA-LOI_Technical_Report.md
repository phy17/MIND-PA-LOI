# PA-LOI Technical Report: Perception-Aware Longitudinal Optimization for Occluded Intersections

## Abstract
This report details the development and validation of the **Perception-Aware Longitudinal Optimization Interface (PA-LOI)**, a specialized trajectory planning module designed to resolve the "Ghost Probe" (occluded intersection) scenario in autonomous driving. The final project outcome (v50) achieves a **100% collision-free success rate** while maintaining reasonable traffic efficiency, utilizing a novel **Velocity-Aware Risk Potential** integrated into an iLQR planner.

## 1. Problem Statement
Autonomous vehicles frequently encounter "Ghost Probe" scenarios where an occluded actor (e.g., pedestrian or vehicle) may suddenly emerge from a blind spot. Traditional planners either:
1.  **Over-brake**: Stop at every blind spot, paralyzing traffic.
2.  **Under-brake**: Ignore potential risks, leading to collisions.
3.  **Swerve Unpredictably**: Optimization artifacts cause dangerous lateral maneuvers.

## 2. Methodology: Decoupled Kinetic Risk Field
PA-LOI solves this by introducing a **Velocity-Aware Risk Potential** that decouples longitudinal speed control from lateral positioning.

### 2.1 Hinge-Loss Kinetic Potential (v51 Architecture)
Unlike standard cost functions that penalize absolute speed $v^2$, the PA-LOI architecture supports a **Hinge-Loss** formulation:

$$ J_{risk}(x, v) = w_{base} \cdot \sigma(d_{clearance}) \cdot \max(0, v - v_{safe})^2 $$

Where:
*   $v_{safe}$: The defensive "safe passage speed" for a blind spot.
*   $\sigma(\cdot)$: A sigmoid function activating only when close to potential occlusion.
*   $w_{base}$: Dynamic weight modulated by Time-to-Arrival (TTA).

### 2.2 Stability & Tuning (v50 Configuration)
Through extensive experimentation (Experiments 1-4), we investigated the trade-off between efficiency ($v_{safe} > 0$) and stability.
*   **Findings**: Setting $v_{safe} > 0$ (e.g., 2.5 m/s) allows the vehicle to maintain higher speeds, but requires significantly higher weights ($w_{base} \approx 250$) to enforce the limit. These high weights caused numerical instability in the solver, leading to dangerous swerving behavior as the optimizer sought to minimize cost by steering away from the risk field.
*   **Final Configuration (v50)**: We successfully deployed a robust **Defensive Mode** where $v_{safe} = 0.0$. This configuration, combined with moderate weights ($w_{base} \approx 30$), effectively creates a "Soft Brake" behavior. The vehicle decelerates linearly as TTA decreases, ensuring maximum safety without inducing lateral instability.
*   **Result**: The vehicle decelerates smoothly from 4.0 m/s to ~2.2 m/s *before* the ghost probe appears, allowing for a safe, comfortable stop (Acc > -4m/s²) when the threat materializes.

## 3. Key Innovations
1.  **Dynamic TTA Weighting**: Risk weights scale linearly from 0 (at TTA=8s) to 30 (at TTA=1s), mimicking human "defensive awareness".
2.  **Velocity-Only Gradients**: By explicitly setting $\partial C / \partial x = 0$ and $\partial C / \partial y = 0$ in the potential field, we mathematically eliminate phantom-induced swerving (in the optimizer step).
3.  **Active Risk Injection**: Fixed a critical bug where long-range risks were filtered out, ensuring the optimizer "sees" the blind spot 50 meters in advance.

## 4. Experimental Results (v50)
*   **Collision Rate**: 0 / 10 Runs (Success)
*   **Min Safety Distance**: 5.5m (Safe stop)
*   **Approach Speed**: Reduced from 4.0 m/s to 2.24 m/s before trigger.
*   **Ride Comfort**: Max deceleration -3.8 m/s² (Emergency), -0.8 m/s² (Approach).

## 5. Conclusion & Future Work
PA-LOI v50 successfully solves the safety problem by implementing a defensive driving strategy that prioritizes collision avoidance. Future work will focus on decoupling lateral constraints to enable higher $v_{safe}$ settings (e.g. 2.5 m/s) without numerical instability.

## 6. Appendix: Parameter Table
| Parameter | Variable | Value | Description |
| :--- | :--- | :--- | :--- |
| **Lookahead TTA** | `TTA Thresholds` | 8.0s / 3.0s / 1.0s | Determines pre-warning horizon (planner.py/utils.py) |
| **Max Weight** | `w_base` | 30.0 | Determines braking intensity. 30 corresponds to approx -2 m/s² |
| **Lateral Threshold** | `ghost_lateral` | ~3.0m - 5.0m | Determines if risk is relevant (Auto-calculated) |
| **Sigmoid Steepness** | `k_steep` | 2.0 | Hardness of lateral boundary (potential.py) |
| **Safe Speed** | `v_safe` | 0.0 m/s | Hinge threshold. 0.0 ensures max braking gradient. |
