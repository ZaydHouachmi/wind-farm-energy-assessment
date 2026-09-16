# Phase 2: Ainslie-Based Wake Model — Source-Based Derivation

## Summary

This document is rewritten to reflect the exact implementation in `resgroup/mini-wake`.
It is based on:

- `miniwake/velocity_deficit.py` — a simplified Ainslie/Anderson eddy viscosity far-wake solver
- `miniwake/near_wake_length.py` — Vermeulen (1980) near-wake length formulas

The derivation below follows the source code directly and avoids unsourced constants.

---

## Notation

- $U_0$ — free-stream (ambient) wind speed
- $U_c$ — centerline wake velocity
- $\Delta U_c = U_0 - U_c$ — centerline velocity deficit
- $C_T$ — upwind turbine thrust coefficient
- $I_0$ — ambient turbulence intensity (fraction)
- $x$ — downstream distance from the rotor plane
- $D$ — rotor diameter
- $x^* = x / D$ — normalized downstream distance
- $b$ — characteristic wake width parameter in the Ainslie/Anderson solution
- $\nu_t$ — eddy viscosity

---

## 1. Source Basis in `miniwake/velocity_deficit.py`

The wake solver in `velocity_deficit.py` is implemented through the class `AndersonSimplifiedSolution`.
Its main structure is:

1. Set an initial normalized downstream point: `x^*_0 = 2.0`
2. Compute an initial centerline deficit at that point using a fitted expression
3. Define an eddy-viscosity closure with two terms: ambient and shear-generated
4. Solve an ODE for centerline velocity `U_c(x^*)` using RK45
5. Return the wake deficit for $x^* \ge 2.0$

### 1.1 Constants from the source code

The implementation uses:

- `K1 = 0.015`
- `KAPPA = 0.40`
- `initial_normalised_distance_downstream = 2.0`
- `F(x^*) = 1.0` for $x^* \ge 5.5$, else $0.65 + \sqrt[3]{(x^* - 4.5) / 23.32}$

These constants are hard-coded in the reference implementation.

### 1.2 Initial centerline deficit

The code initializes the deficit at $x^*_0 = 2.0$ as:

$$
\Delta U_c(x^*_0) = C_T - 0.05 - \frac{(16 C_T - 0.5) \, I_0}{10.0}
$$

This is implemented in `calculate_initial_centre_line_deficit()`.
It is a fitted form based on Ainslie/Anderson wake initialization.

### 1.3 Wake width parameter

The local wake width parameter $b$ is computed as:

$$
 b(C_T, \Delta U_c) = \sqrt{\frac{3.56 \, C_T}{8 \, \Delta U_c \, (1 - 0.5 \, \Delta U_c)}}
$$

This comes from `calculate_width()` in `velocity_deficit.py`.

### 1.4 Eddy-viscosity closure

The eddy viscosity is defined in `epsilon()` as:

$$
\nu_t(x^*, U_c) = F(x^*) \left[ K_1 \, b \, \frac{U_0 - U_c}{U_0} + K_{\kappa}^2 \, I_0 \right]
$$

with

- $K_1 = 0.015$
- $K_{\kappa} = 0.40$
- $F(x^*) = \begin{cases}1.0 & x^* \ge 5.5 \\ 0.65 + \sqrt[3]{\frac{x^* - 4.5}{23.32}} & x^* < 5.5\end{cases}$

The ambient turbulence contribution is computed as:

$$
\nu_{t,\text{ambient}} = K_{\kappa}^2 \, I_0
$$

and the shear-generated contribution as:

$$
\nu_{t,\text{shear}} = K_1 \, b \, \frac{U_0 - U_c}{U_0}
$$

### 1.5 ODE for centerline velocity

The centerline velocity derivative is implemented in `derivative_of_center_line_velocity_wrt_distance()`:

Let $u_c = U_c / U_0$. Then:

$$
\frac{d U_c}{d x^*} = 16 \, \nu_t \, \frac{u_c^3 - u_c^2 - u_c + 1}{u_c \, C_T} \, U_0
$$

This is the central marching equation in the reference solver.

### 1.6 Wake deficit evaluation

The public function `calculate_velocity_deficit(C_T, x^*, I_0)` does the following:

- Clamp $x^*$ to $\ge 2.0$.
- Clamp $C_T$ to $\le 1.0$.
- Solve the ODE from $x^*_0 = 2.0$ to the requested $x^*$ with `RK45`.
- Return the computed deficit as the solution at that downstream location.

Thus, in this implementation, the far-wake ODE is only used for normalized distances $x^* \ge 2.0$.

---

## 2. Near-Wake Length in `miniwake/near_wake_length.py`

The near-wake length formula is implemented using Vermeulen (1980) style expressions.

### 2.1 Ambient turbulence erosion rate

The ambient turbulence wake erosion rate is defined piecewise as:

$$
K_a = \begin{cases}
5.0 \, I_0 & I_0 < 0.02 \\
2.5 \, I_0 + 0.05 & I_0 \ge 0.02
\end{cases}
$$

This appears in `calculate_ambient_turbulence_wake_erosion_rate()`.

### 2.2 Flow-field ratio

Vermeulen's flow-field ratio is computed as:

$$
m = \begin{cases}
3.0 & C_T > 0.8888 \\
\frac{1}{\sqrt{1 - C_T}} & C_T \le 0.8888
\end{cases}
$$

This is implemented in `calculate_flow_field_ratio()`.

### 2.3 Tip-speed ratio and mechanical erosion

The tip-speed ratio is:

$$
\lambda = \frac{\omega R}{U_0}
$$

with $\omega = 2 \pi \text{rpm} / 60$ and $R = D/2$.

The mechanical turbulence erosion is:

$$
K_m = 0.012 \, N_b \, \lambda
$$

from `calculate_mechanical_turbulence_wake_erosion_rate()`.

### 2.4 Shear-generated erosion rate

The shear-generated erosion rate is:

$$
K_s = \frac{(1 - m) \sqrt{1.49 + m}}{9.76 (1 + m)}
$$

from `calculate_shear_generated_turbulence_wake_erosion_rate()`.

### 2.5 Inviscid expanded rotor radius

The equivalent inviscid expanded rotor radius is:

$$
R_e = \frac{D}{2} \sqrt{\frac{m + 1}{2}}
$$

from `calculate_radius_of_inviscid_expanded_rotor_disk()`.

### 2.6 Wake-development exponent

The coefficient $n$ is defined as:

$$
n = \frac{\sqrt{A + B m} \left(1 - \sqrt{C + D m}\right)}{\left(1 - \sqrt{A + B m}\right) \sqrt{C + D m}}
$$

with constants:

- $A = 0.214$
- $B = 0.144$
- $C = 0.134$
- $D = 0.124$

This is implemented in `calculate_n()`.

### 2.7 Near-wake length

Finally, the near-wake length is:

$$
L_n = \frac{n \, R_e}{\sqrt{K_a^2 + K_s^2 + K_m^2}}
$$

This is the output of `calculate_near_wake_length()`.

### 2.8 Source attribution

The functions in `near_wake_length.py` are documented with comments referring to Vermeulen, P.E.J. (1980), "An Experimental Analysis of Wind Turbine Wakes," Wind Energy Systems.

---

## 3. Near-Wake / Far-Wake Transition in the Reference Code

### Far-wake solver transition

The code enforces the far-wake solver only for normalized downstream distances $x^* \ge 2.0$.
The first usable far-wake state is initialized at $x^*_0 = 2.0$.

### Near-wake length usage

The `calculate_near_wake_length()` result is used elsewhere in the wake model (for example, added turbulence computations and meander) but it is not directly used to start the far-wake ODE in `velocity_deficit.py`.

### Practical interpretation

- For velocity deficit calculations, this implementation treats $x^*=2.0$ as the transition into the far-wake model.
- The near-wake length formula from Vermeulen provides a physically consistent estimate of where the potential core is eroded, but the velocity-deficit solver itself begins at $x^*=2.0$.

Thus, the near-/far-wake boundary is effectively:

- near wake: $0 < x^* < 2.0$ (handled by separate geometry and turbulence corrections)
- far wake: $x^* \ge 2.0$ (handled by the Ainslie/Anderson ODE solver)

---

## 4. Exact Implementation Notes

The current `miniwake` reference implementation is not a direct analytic reproduction of the original Ainslie paper alone. It is a simplified and calibrated form that combines:

- Ainslie/Anderson-style eddy viscosity closure for wake recovery
- A fitted initial centerline deficit at $x^* = 2.0$
- Vermeulen (1980) formulas for near-wake length and turbulence erosion rates

That means:

- The constants `K1 = 0.015` and `KAPPA = 0.40` are taken directly from the code.
- The starting normalized distance $x^*_0 = 2.0$ is taken directly from the code.
- The near-wake length formula is taken directly from the code and from Vermeulen (1980).

---

## 5. Recommended Source-Based Model Flow for This Repo

1. Use `calculate_near_wake_length()` to estimate the near-wake length from wake and turbulence parameters.
2. Use the Ainslie/Anderson solver in `velocity_deficit.py` for $x^* \ge 2.0$.
3. Preserve the reference implementation constants rather than inventing new empirically tuned values.
4. Document the implementation as a calibrated Ainslie/Anderson solution plus Vermeulen near-wake length.

---

## 6. References

- Ainslie, J. F. (1988). "Calculating the Flowfield in the Wake of Wind Turbines." *Journal of Wind Engineering and Industrial Aerodynamics*.
- Anderson, C. (2009). Simplified eddy viscosity wake model solution.
- Vermeulen, P. E. J. (1980). "An Experimental Analysis of Wind Turbine Wakes." Wind Energy Systems.
