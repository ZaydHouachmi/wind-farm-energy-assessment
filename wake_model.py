"""
wake_model.py
=============
Simple wake deficit models for interactive wind farm analysis.
Includes a viscosity-inspired model and a Jensen wake model.
"""

import numpy as np

try:
    from scipy.integrate import RK45
except ImportError:
    RK45 = None


def calculate_shape_at_position_sq(normalized_position_sq):
    normalized_position_sq_arr = np.asarray(normalized_position_sq)
    shape = np.exp(-3.56 * normalized_position_sq_arr)
    return np.where(normalized_position_sq_arr > 4.0, 0.0, shape)


def calculate_width(thrust_coefficient, velocity_deficit):
    velocity_deficit = np.maximum(velocity_deficit, 1e-6)
    return np.sqrt(3.56 * thrust_coefficient / (8.0 * velocity_deficit * (1.0 - 0.5 * velocity_deficit)))


def real_cubed_root(value):
    value_arr = np.asarray(value)
    return np.sign(value_arr) * np.abs(value_arr) ** (1.0 / 3.0)


class SimpleWakeModel:
    """
    Eddy viscosity-style wake model for demonstration.
    """

    def __init__(self, alpha: float = 0.05, beta: float = 0.5):
        self.alpha = alpha
        self.beta = beta

    def wake_deficit(self, xi: float, eta: float, D: float,
                     thrust_coeff: float = 0.8) -> float:
        # Support scalar or numpy-array inputs for xi and eta
        xi_arr = np.asarray(xi)
        eta_arr = np.asarray(eta)

        # Compute sigma (avoid division by zero)
        sigma = self.alpha * xi_arr + self.beta * D
        sigma = np.maximum(sigma, 1e-6)

        gauss = np.exp(-0.5 * (eta_arr / sigma) ** 2)
        wake_strength = thrust_coeff / (1.0 + (xi_arr / D) ** 2)

        # Zero out upstream (xi <= 0)
        wake = wake_strength * gauss
        wake = np.where(xi_arr <= 0, 0.0, wake)

        # Return scalar when inputs were scalar
        if np.isscalar(xi):
            return float(wake)
        return wake

    def effective_speed(self, u_inflow: float, deficits: np.ndarray) -> float:
        total_deficit = np.sum(deficits)
        total_deficit = np.clip(total_deficit, 0.0, 0.99)
        return u_inflow * (1.0 - total_deficit)


class JensenWakeModel:
    """
    Jensen wake model with linear wake expansion.
    """

    def __init__(self, wake_decay: float = 0.05, thrust_coeff: float = 0.8):
        self.wake_decay = wake_decay
        self.thrust_coeff = thrust_coeff

    def wake_deficit(self, xi: float, eta: float, D: float,
                     thrust_coeff: float | None = None) -> float:
        # Support scalar or numpy-array inputs for xi and eta
        xi_arr = np.asarray(xi)
        eta_arr = np.asarray(eta)

        if thrust_coeff is None:
            thrust_coeff = self.thrust_coeff

        r0 = D / 2.0
        r = r0 + self.wake_decay * xi_arr

        # valid points are where downstream and within wake radius
        inside = (xi_arr > 0) & (np.abs(eta_arr) <= r)

        # Avoid division by zero
        r_safe = np.where(r <= 0, 1e-6, r)
        deficit = (1.0 - np.sqrt(1.0 - thrust_coeff)) * (r0 / r_safe) ** 2
        deficit = np.where(inside, deficit, 0.0)

        if np.isscalar(xi):
            return float(deficit)
        return deficit

    def effective_speed(self, u_inflow: float, deficits: np.ndarray) -> float:
        total_deficit = np.sum(deficits)
        total_deficit = np.clip(total_deficit, 0.0, 0.99)
        return u_inflow * (1.0 - total_deficit)


class EddyViscosityWakeModel:
    """
    Ainslie/mini-wake-style eddy viscosity wake model.
    """

    K1 = 0.015
    KAPPA = 0.40
    INITIAL_NORMALISED_DISTANCE_DOWNSTREAM = 2.0

    def __init__(
            self,
            ambient_turbulence: float = 0.10,
            thrust_coeff: float = 0.8,
            free_stream_velocity: float = 10.0):
        self.ambient_turbulence = ambient_turbulence
        self.thrust_coeff = thrust_coeff
        self.free_stream_velocity = free_stream_velocity
        self._deficit_cache: dict[tuple[float, float], float] = {}
        # Precomputed centerline deficit tables keyed by rounded thrust coeff
        self._deficit_table: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    def _F(self, normalized_downwind_distance):
        normalized_downwind_distance = np.asarray(normalized_downwind_distance)
        return np.where(
            normalized_downwind_distance >= 5.5,
            1.0,
            0.65 + real_cubed_root((normalized_downwind_distance - 4.5) / 23.32)
        )

    def _calculate_initial_centre_line_deficit(self, thrust_coefficient: float) -> float:
        return thrust_coefficient - 0.05 - (16.0 * thrust_coefficient - 0.5) * self.ambient_turbulence / 10.0

    def _center_line_velocity(self, center_line_deficit: float) -> float:
        return self.free_stream_velocity * (1.0 - center_line_deficit)

    def _center_line_deficit(self, center_line_velocity: float) -> float:
        return 1.0 - center_line_velocity / self.free_stream_velocity

    def _ambient_eddy_visocity(self) -> float:
        return EddyViscosityWakeModel.KAPPA * EddyViscosityWakeModel.KAPPA * self.ambient_turbulence

    def _epsilon(self, normalized_downwind_distance: float, center_line_velocity: float, thrust_coefficient: float) -> float:
        velocity_deficit = self._center_line_deficit(center_line_velocity)
        b = calculate_width(thrust_coefficient, velocity_deficit)
        shear_eddy_viscosity = EddyViscosityWakeModel.K1 * b * (self.free_stream_velocity - center_line_velocity) / self.free_stream_velocity
        filter_term = self._F(normalized_downwind_distance)
        return filter_term * (shear_eddy_viscosity + self._ambient_eddy_visocity())

    def _derivative_of_center_line_velocity_wrt_distance(
            self,
            normalized_downwind_distance: float,
            center_line_velocity: float,
            thrust_coefficient: float) -> float:
        epsilon = self._epsilon(normalized_downwind_distance, center_line_velocity, thrust_coefficient)
        uc = center_line_velocity / self.free_stream_velocity
        derivative = 16.0 * epsilon * (uc**3.0 - uc**2.0 - uc + 1.0) / (uc * thrust_coefficient)
        derivative *= self.free_stream_velocity
        return derivative

    def _initial_condition(self, thrust_coefficient: float) -> tuple[float, float]:
        deficit = self._calculate_initial_centre_line_deficit(thrust_coefficient)
        velocity = self._center_line_velocity(deficit)
        return self.INITIAL_NORMALISED_DISTANCE_DOWNSTREAM, velocity

    def _solve_velocity_deficit(self, thrust_coefficient: float, normalized_distance_downwind: float) -> float:
        normalized_distance_downwind = max(2.0, normalized_distance_downwind)
        normalized_distance_downwind = float(normalized_distance_downwind)
        thrust_coefficient = min(1.0, max(0.0, thrust_coefficient))

        cache_key = (round(normalized_distance_downwind, 4), round(thrust_coefficient, 4))
        if cache_key in self._deficit_cache:
            return self._deficit_cache[cache_key]

        x0, u0 = self._initial_condition(thrust_coefficient)
        if normalized_distance_downwind <= x0:
            deficit = self._center_line_deficit(u0)
            self._deficit_cache[cache_key] = deficit
            return deficit

        if RK45 is not None:
            integrator = RK45(
                fun=lambda x, y: [self._derivative_of_center_line_velocity_wrt_distance(x, y[0], thrust_coefficient)],
                t0=x0,
                y0=[u0],
                t_bound=normalized_distance_downwind,
                max_step=0.1,
            )
            while integrator.status == 'running' and integrator.t < normalized_distance_downwind:
                integrator.step()
            u_c = integrator.y[0]
        else:
            x = x0
            u_c = u0
            while x < normalized_distance_downwind:
                h = min(0.1, normalized_distance_downwind - x)
                k1 = self._derivative_of_center_line_velocity_wrt_distance(x, u_c, thrust_coefficient)
                k2 = self._derivative_of_center_line_velocity_wrt_distance(x + 0.5 * h, u_c + 0.5 * h * k1, thrust_coefficient)
                k3 = self._derivative_of_center_line_velocity_wrt_distance(x + 0.5 * h, u_c + 0.5 * h * k2, thrust_coefficient)
                k4 = self._derivative_of_center_line_velocity_wrt_distance(x + h, u_c + h * k3, thrust_coefficient)
                u_c += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                x += h

        deficit = self._center_line_deficit(u_c)
        deficit = max(0.0, deficit)
        self._deficit_cache[cache_key] = deficit
        return deficit

    def _build_deficit_table(self, thrust_coefficient: float, x_star_max: float = 50.0) -> tuple[np.ndarray, np.ndarray]:
        """
        Build an adaptive-resolution table of centerline deficits vs normalized downstream distance
        (x*). Resolution is dense in the near wake (2-6 xD) and coarser downstream. Returns
        (x_grid, deficits) arrays for interpolation.
        """
        key = round(float(thrust_coefficient), 4)
        if key in self._deficit_table:
            return self._deficit_table[key]

        # Dense sampling near-wake and coarser further downstream
        near = np.arange(self.INITIAL_NORMALISED_DISTANCE_DOWNSTREAM, 6.0, 0.02)
        far = np.arange(6.0, max(6.1, float(x_star_max) + 0.1), 0.1)
        x_grid = np.concatenate((near, far))

        # Compute deficits using the accurate solver; results are cached by _solve_velocity_deficit
        deficits = np.empty_like(x_grid, dtype=float)
        for i, xval in enumerate(x_grid):
            deficits[i] = self._solve_velocity_deficit(thrust_coefficient, float(xval))

        self._deficit_table[key] = (x_grid, deficits)
        return x_grid, deficits

    def _interpolated_centerline_deficits(self, thrust_coefficient: float, x_star_values: np.ndarray) -> np.ndarray:
        """
        Return centerline deficits for the input normalized downstream positions using a
        precomputed table and fast interpolation. Builds the table on first use.
        """
        # ensure numpy array
        x_arr = np.asarray(x_star_values, dtype=float)
        out = np.zeros_like(x_arr, dtype=float)

        if x_arr.size == 0:
            return out

        # For positions upstream of the initial normalized distance, use initial deficit
        x0, u0 = self._initial_condition(thrust_coefficient)
        initial_deficit = self._center_line_deficit(u0)

        # Build table up to the max requested x* (but at least a sensible max)
        xmax = max(float(np.max(x_arr)), x0 + 0.1)
        x_grid, deficits = self._build_deficit_table(thrust_coefficient, x_star_max=xmax)

        # For values below x0, assign initial_deficit; else interpolate
        mask_far = x_arr >= x0
        if np.any(mask_far):
            # numpy.interp requires ascending x_grid
            interp_vals = np.interp(x_arr[mask_far], x_grid, deficits, left=deficits[0], right=deficits[-1])
            out[mask_far] = interp_vals

        mask_near = ~mask_far
        if np.any(mask_near):
            out[mask_near] = initial_deficit

        # ensure non-negative
        out = np.maximum(out, 0.0)
        return out

    def wake_deficit(self, xi: float, eta: float, D: float,
                     thrust_coeff: float | None = None) -> float:
        if thrust_coeff is None:
            thrust_coeff = self.thrust_coeff

        xi_arr = np.asarray(xi)
        eta_arr = np.asarray(eta)
        x_star = xi_arr / D
        flat_x_star = x_star.ravel()

        rounded_x_star = np.round(flat_x_star, 4)
        unique_x_star, inverse_indices = np.unique(rounded_x_star, return_inverse=True)
        # Use fast interpolated table for centerline deficits (precomputed for accuracy)
        unique_deficits = self._interpolated_centerline_deficits(thrust_coeff, unique_x_star)

        centerline_deficit = unique_deficits[inverse_indices].reshape(x_star.shape)
        wake_width = calculate_width(thrust_coeff, np.maximum(centerline_deficit, 1e-6)) * D
        normalized_position_sq = (eta_arr / wake_width) ** 2
        shape = calculate_shape_at_position_sq(normalized_position_sq)
        wake = centerline_deficit * shape
        wake = np.where(xi_arr <= 0.0, 0.0, wake)

        if np.isscalar(xi):
            return float(wake)
        return wake

    def effective_speed(self, u_inflow: float, deficits: np.ndarray) -> float:
        total_deficit = np.sum(deficits)
        total_deficit = np.clip(total_deficit, 0.0, 0.99)
        return u_inflow * (1.0 - total_deficit)
