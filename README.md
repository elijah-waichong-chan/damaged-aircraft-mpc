# Damaged Aircraft MPC Guidance Controller
This project explores how **Model Predictive Control (MPC)** can provide **high-level guidance** to a **damaged aircraft** using only a **3D kinematic model**.  

The controller outputs *guidance commands* (desired acceleration, turn rate, and climb-angle rate), which are assumed to be tracked by an **ideal lower-level autopilot**.

The central idea is to combine:

- a **long-horizon glide-path planner**, and  
- a **short-horizon receding-horizon MPC**

to guide an impaired aircraft toward a safe landing runway, even when control authority is degraded.

Developed as the final project for  
**UC Berkeley MECENG 231A – Model Predictive Control**

---

## ✈️ Overview

This project focuses on **guidance**, not full flight dynamics.  
The aircraft is modeled as a **point-mass kinematic system**:

### Kinematic Model
$$
x = [x,\; y,\; h,\; V,\; \chi,\; \gamma]
$$

$$
\dot{x} = V\cos\gamma\cos\chi,\quad
\dot{y} = V\cos\gamma\sin\chi,\quad
\dot{h} = V\sin\gamma
$$

$$
\dot{V} = u_T - g\sin\gamma,\quad
\dot{\chi} = u_{\dot{\chi}},\quad
\dot{\gamma} = u_{\dot{\gamma}}
$$

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

allowing convex optimization while still following the nonlinear glide path.

The output of the MPC is interpreted as **high-level commands** to an onboard autopilot, which is assumed to follow these rates within specified limits.

This abstraction allows us to study **navigation** and **trajectory planning** without modeling aerodynamics, control surfaces, or aircraft attitude.

---

## 🧭 Guidance Architecture
### Long-Horizon Planner
- Fully **convex**
- Optimization-based geometry only trajectory planner
- Generates a smooth runway-aligned 3D trajectory
- Horizon extends all the way to touchdown
- Produces a full-descent reference path

### Short-Horizon MPC
- Fully **convex**
- Linear time-varying (LTV) MPC
- Tracks the planned path over a short window
- Handles degraded control authority:
  - Limited acceleration
  - Limited turn rate
  - Limited climb/descent rate
- Outputs high-level rate commands:

$$
\dot{V},\quad
\dot{\chi},\quad
\dot{\gamma}
$$