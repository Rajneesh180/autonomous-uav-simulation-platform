# 3D Kinematics and LoS Communication Theory

This document outlines the explicitly integrated 3D models required for the high-fidelity IEEE simulation environment.

## 1. True 3D Spherical Flight Kinematics
Instead of locking the UAV to a fixed altitude $z_0$, the trajectory optimizer employs spherical coordinate transformations to traverse physical 3D space defined by Yaw ($\psi$) and Pitch ($\theta$).

At any time step $t$, the UAV evaluates motion primitives across discrete steering and pitch offsets. The change in position is governed by:
$$ x(t+1) = x(t) + v \cdot \Delta t \cdot \cos(\theta) \cdot \cos(\psi) $$
$$ y(t+1) = y(t) + v \cdot \Delta t \cdot \cos(\theta) \cdot \sin(\psi) $$
$$ z(t+1) = z(t) + v \cdot \Delta t \cdot \sin(\theta) $$

Where:
*   **$v$** is the bounded UAV velocity (defined by `UAV_STEP_SIZE`).
*   **$\psi$** (Yaw) is the angle in the XY-plane.
*   **$\theta$** (Pitch) is the elevation angle from the XY-plane, limited by mechanical constraints (e.g., $-15^{\circ}, 0^{\circ}, +15^{\circ}$).

## 2. Rician Fading & Elevation-Based Probabilistic LoS
Traditional geometric models assume a flat Earth communication distance. Given a UAV in 3D space $(x_u, y_u, z_u)$ tracking a node at $(x_n, y_n, z_n)$, the straight-line 3D Euclidean distance is:
$$ d_{3D} = \sqrt{(x_u - x_n)^2 + (y_u - y_n)^2 + (z_u - z_n)^2} $$

The probability of maintaining clearly unobstructed Line-of-Sight (LoS) is heavily dependent on the relative elevation angle $E_{\theta}$ between the ground node and the aerial vehicle.
$$ E_{\theta} = \arctan\left( \frac{\max(0, z_u - z_n)}{\sqrt{(x_u - x_n)^2 + (y_u - y_n)^2}} \right) $$

Employing the Rician channel fading approximation, the Probability of LoS ($P_{LoS}$) uses a modified Sigmoid function parameterized by environmental constants $a$ (urban density) and $b$ (environment shape factor):
$$ P_{LoS} = \frac{1}{1 + a \cdot \exp(-b [E_{\theta, deg} - a])} $$

## 3. Path Loss Extrapolation
The expected Path Loss ($PL_{Exp}$) is a probabilistic fusion of both Free-Space LoS ($loss_{los}$) and heavily obstructed Non-LoS ($loss_{nlos}$) states, scaling non-linearly over distance:
$$ PL_{Exp} = P_{LoS} ( FSPL_0 \cdot d_{3D}^{\alpha_{los}} ) + (1 - P_{LoS}) ( FSPL_0 \cdot d_{3D}^{\alpha_{nlos}} ) $$
Where $FSPL_0$ is the reference path loss at 1 meter.
