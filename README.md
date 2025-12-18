# Damaged Aircraft MPC Guidance Controller
This project explores how **Model Predictive Control (MPC)** can provide **high-level guidance** to a **damaged aircraft** using only a **3D kinematic model**.  

Developed as the final project for **UC Berkeley MECENG 231A – Model Predictive Control**

## 🛫 Introduction
The controller outputs *guidance commands* (desired acceleration, turn rate, and climb-angle rate), which are assumed to be tracked by an **ideal lower-level autopilot**.

The central idea is to combine:

- a **long-horizon glide-path planner**, and  
- a **short-horizon receding-horizon MPC**

to guide an impaired aircraft toward a safe landing runway, even when control authority is degraded.

---

## ✈️ Model
### Kinematic Model

The aircraft is modeled as a **3D point-mass kinematic system** with six states:

$$
x = [x,\ y,\ h,\ V,\ \chi,\ \gamma\]
$$

Where:
| State     | Meaning                                | Units   |
|-----------|----------------------------------------|---------|
| $x$       | North position in world frame          | meters  |
| $y$       | East position in world frame           | meters  |
| $h$       | Altitude                               | meters  |
| $V$       | Airspeed magnitude                     | m/s     |
| $\chi$    | Heading angle (0° = North, 90° = East) | radians |
| $\gamma$  | Climb angle                            | radians |

The kinematic equations of motion are:

$$
\dot{x} = V\cos\gamma\cos\chi,\qquad
\dot{y} = V\cos\gamma\sin\chi,\qquad
\dot{h} = V\sin\gamma
$$

The speed, heading, and climb angle evolve according to **high-level guidance inputs**:

$$
\dot{V} = u_{accel},\qquad
\dot{\chi} = u_{\dot{\chi}},\qquad
\dot{\gamma} = u_{\dot{\gamma}}
$$

Where:
- $u_{accel}$ — commanded longitudinal acceleration  
- $u_{\dot{\chi}}$ — commanded heading rate  
- $u_{\dot{\gamma}}$ — commanded climb-angle rate  

These inputs are assumed to be tracked by an **ideal inner-loop autopilot**, allowing the MPC to operate purely at the guidance level.

### Linearization for MPC

Although the aircraft obeys nonlinear kinematic equations,  
the MPC uses a **linearized model** that is recomputed at each control iteration.

At each step, the Jacobians of the nonlinear dynamics are evaluated around the current
state, producing an LTV (Linear Time-Varying) system:

$$
A_k = \frac{\partial f}{\partial x}\Big\rvert_{x_k}, \qquad 
B_k = \frac{\partial f}{\partial u}\Big\rvert_{x_k},
$$

which is discretized as:

$$
A_d = I + A_k\Delta t, \qquad B_d = B_k\Delta t.
$$

This yields the linear prediction model used inside the MPC:

$$
x_{k+1} = A_d x_k + B_d u_k
$$

This simplification allows us to study **navigation** and **trajectory planning** without modeling aerodynamics, control surfaces, or aircraft attitude.

---

## 🧭 Guidance Architecture
### Long-Horizon Planner
- Fully **convex**
- Optimization-based geometry only trajectory planner
- Generates a smooth runway-aligned 3D trajectory
- Horizon extends all the way to touchdown
- Produces a full-descent reference path
- Only replan when Short-Horizon MPC deemed the planned path is infeasible to track

### Short-Horizon MPC
- Fully **convex**
- Linear time-varying (LTV) MPC
- Tracks the planned path over a short window
- Handles degraded control authority:
  - Limited acceleration
  - Limited turn rate
  - Limited climb/descent rate
- Determines if the path is infeasible to follow
- Outputs high-level rate commands:

$$
\dot{V},\quad
\dot{\chi},\quad
\dot{\gamma}
$$
