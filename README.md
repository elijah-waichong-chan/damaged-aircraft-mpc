# Damaged Aircraft MPC Guidance Controller

This project explores how **Model Predictive Control (MPC)** can provide **high-level guidance** to a **damaged aircraft** using only a **3D kinematic model**.

Developed as the final project for **UC Berkeley MECENG 231A – Model Predictive Control**.

---

## Overview

The controller outputs *guidance commands* (desired longitudinal acceleration, turn-rate, and climb-angle rate), which are assumed to be tracked by an **ideal lower-level autopilot**.

The core idea is a **two-layer guidance stack**:

1) **Long-horizon geometric planner (convex QP)**  
   Generates a smooth, runway-aligned 3D waypoint reference to the runway threshold.

2) **Short-horizon guidance MPC (convex QP, LTV)**  
   Tracks a local segment of the reference while enforcing constraints and handling degraded authority.

Unlike “replan every step” architectures, the planner **only replans when the MPC declares the current plan infeasible**.

<p align="center">
  <img src="media/guidance_stack.png" alt="Two-layer guidance architecture" width="800">
</p>

---

## Model

### 3D Kinematic Point-Mass Model (Guidance Level)

State:

$$
x = [x,\ y,\ h,\ V,\ \chi,\ \gamma]^T
$$

| State     | Meaning                                | Units   |
|-----------|----------------------------------------|---------|
| $x$       | North position in world frame          | meters  |
| $y$       | East position in world frame           | meters  |
| $h$       | Altitude                               | meters  |
| $V$       | Airspeed magnitude                     | m/s     |
| $\chi$    | Heading angle (0° = North, 90° = East) | radians |
| $\gamma$  | Climb angle                            | radians |

Kinematics:

$$
\dot{x} = V\cos\gamma\cos\chi,\qquad
\dot{y} = V\cos\gamma\sin\chi,\qquad
\dot{h} = V\sin\gamma
$$

$$
\dot{V} = u_{accel},\qquad
\dot{\chi} = u_{\dot{\chi}},\qquad
\dot{\gamma} = u_{\dot{\gamma}}
$$

Inputs (tracked by ideal autopilot):
- $u_{accel}$: longitudinal acceleration command (m/s$^2$)
- $u_{\dot{\chi}}$: heading-rate command (rad/s)
- $u_{\dot{\gamma}}$: climb-angle-rate command (rad/s)

### Linearization for MPC (LTV)

At each control iteration, the nonlinear kinematics are linearized about the current operating point and discretized (e.g., forward Euler) to form an LTV prediction model:

$$
\x_{k+1} = A_d x_k + B_d u_k,\qquad
$$

---

## Guidance Architecture

### Long-Horizon Geometric Planner (Convex QP)

Produces a waypoint sequence $(x_i, y_i, h_i)$ from current aircraft position to runway threshold.

Typical objective terms:
- Smoothness (second differences)
- Glide-slope shaping
- Runway centerline (cross-track) shaping
- Terminal runway heading alignment

**Replanning policy:** planner is called only when the MPC flags infeasibility.

### Short-Horizon Guidance MPC (Convex QP, LTV)

Tracks a short window of the planned reference while enforcing:
- **state envelopes** (e.g., speed bounds, \(\gamma\) bounds)
- **input bounds** (reduced under damage)
- **input rate bounds** (to prevent aggressive/oscillatory commands)

**Reference construction:** the planner directly provides position/altitude references, and the implementation primarily penalizes tracking error in \((x,y,h,V)\). Reference angles for \((\chi,\gamma)\) are computed from path gradients.

---

## Damage + Mode Switching

### Degraded Control Authority (Damage Mode)

Damage is modeled by **tightening input bounds** (and potentially tightening the allowable \(\gamma\) envelope mid-flight) while keeping the same kinematic model. This can render the original runway plan infeasible.

### Tracking Feasibility → Replan Triggers

During flight, the MPC monitors feasibility using:
- a **cross-track error accumulation counter**, and
- a **progress-stall detector** based on insufficient along-path progress.

If triggers persist for several consecutive steps, the current plan is declared infeasible and the planner replans.

### Runway Unreachable → Crash Landing Mode

If (after damage) runway landing is declared infeasible, the system switches to a **secondary reference + MPC structure**:

1) **Online reachability check**  
   Compare remaining horizontal distance along the planned approach polyline vs. maximum achievable horizontal range computed from altitude and shallowest allowable descent.

2) **Crash touchdown selection with no-land zones (ellipses)**  
   The long-horizon planner selects a feasible touchdown point within remaining range while avoiding no-land ellipses by scanning bearings about current heading and rejecting candidates that intersect restricted regions. If the aircraft starts inside a no-land ellipse, an intermediate “escape waypoint” is generated just outside the ellipse boundary.

3) **Crash landing MPC**  
   Tracks the crash polyline using the same convex QP structure, with an added terminal/near-ground penalty encouraging reduced vertical impact severity.

<p align="center">
  <img src="media/damage_reroute.gif" alt="Damaged-aircraft guidance demo" width="800">
</p>

---

## Repository Entry Points

- `flight_control_mpc/path_planner.py` — long-horizon planner + crash planner
- `flight_control_mpc/guidance_mpc.py` — short-horizon MPC + feasibility logic
- `flight_control_mpc/aircraft_model.py` — nonlinear kinematic model + linearization
- `flight_control_mpc/test.py` — runnable scenarios / demos
- `flight_control_mpc/plot.py`, `flight_control_mpc/animation.py` — visualization