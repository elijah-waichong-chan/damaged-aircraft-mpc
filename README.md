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

\[
x = [x,\; y,\; h,\; V,\; \chi,\; \gamma]
\]

\[
\dot{x} = V\cos\gamma\cos\chi,\quad
\dot{y} = V\cos\gamma\sin\chi,\quad
\dot{h} = V\sin\gamma
\]

\[
\dot{V} = u_T - g\sin\gamma,\quad
\dot{\chi} = u_{\dot{\chi}},\quad
\dot{\gamma} = u_{\dot{\gamma}}
\]

The output of the MPC is interpreted as **high-level commands** to an onboard autopilot, which is assumed to follow these rates within specified limits.

This abstraction allows us to study **trajectory feasibility**, **path tracking**, and **fault-tolerant guidance** without modeling aerodynamics, control surfaces, or aircraft attitude.

---

## 🧭 Guidance Architecture
### Long-Horizon Planner
- Optimization-based geometry only trajectory planner
- Generates a smooth runway-aligned 3D trajectory
- Horizon extends all the way to touchdown
- Produces a full-descent reference path

### Short-Horizon MPC
- Linear time-varying (LTV) MPC
- Tracks the planned path over a short window
- Handles degraded control authority:
  - Limited acceleration
  - Limited turn rate
  - Limited climb/descent rate
- Outputs:  
  - \( \dot{V} \) — acceleration command  
  - \( \dot{\chi} \) — heading-rate command  
  - \( \dot{\gamma} \) — climb-angle-rate command  

### Kinematic Aircraft Model
- Pure geometric motion  
- No aerodynamics (no lift/drag)  
- Suitable for **guidance-level** reasoning  
- Fast for MPC iteration and visualization
