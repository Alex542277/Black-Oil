# ================================================================
# CALIBRATION OF A FRACTIONAL TWO-PHASE FLOW MODEL
# Comparison with Zhong, Li and Kou (2013)
#
# Calibrated parameters:
#   a_v : velocity correction factor
#   a_f : correction factor in fractures
#   a_m : correction factor in matrix
#
# Results:
#   - optimized parameters
#   - MAE, RMSE and MaxAE
#   - average saturation / PVI curves
#   - QQ-plots of residuals
#
# WARNING:
# This model is simplified and does not solve the full two-phase
# pressure equation of the reference article.
# ================================================================

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from scipy import stats


# ================================================================
# 1. APPROXIMATE REFERENCE DATA
# ================================================================

pvi_ref = np.round(np.arange(0.0, 2.5001, 0.05), 2)

Sw_ref = {
    (1.00, 1.00): np.array([
        0.000, 0.050, 0.098, 0.148, 0.195, 0.244, 0.291,
        0.342, 0.389, 0.439, 0.487, 0.538, 0.587, 0.635,
        0.675, 0.697, 0.711, 0.722, 0.730, 0.737, 0.744,
        0.750, 0.755, 0.759, 0.763, 0.767, 0.772, 0.774,
        0.779, 0.781, 0.784, 0.787, 0.790, 0.792, 0.794,
        0.796, 0.798, 0.801, 0.802, 0.805, 0.806, 0.808,
        0.810, 0.812, 0.813, 0.814, 0.816, 0.817, 0.819,
        0.820, 0.820
    ]),

    (0.90, 0.95): np.array([
        0.000, 0.030, 0.058, 0.088, 0.115, 0.144, 0.171,
        0.200, 0.226, 0.255, 0.283, 0.312, 0.340, 0.369,
        0.398, 0.425, 0.456, 0.483, 0.513, 0.542, 0.573,
        0.600, 0.629, 0.652, 0.674, 0.690, 0.701, 0.710,
        0.716, 0.723, 0.729, 0.733, 0.737, 0.741, 0.744,
        0.748, 0.751, 0.754, 0.756, 0.759, 0.762, 0.765,
        0.766, 0.769, 0.772, 0.773, 0.776, 0.777, 0.779,
        0.781, 0.781
    ]),

    (0.85, 0.90): np.array([
        0.000, 0.018, 0.036, 0.054, 0.070, 0.088, 0.105,
        0.121, 0.137, 0.155, 0.171, 0.188, 0.204, 0.222,
        0.239, 0.255, 0.272, 0.288, 0.306, 0.322, 0.340,
        0.356, 0.373, 0.389, 0.407, 0.425, 0.442, 0.460,
        0.476, 0.494, 0.511, 0.530, 0.547, 0.565, 0.583,
        0.600, 0.617, 0.632, 0.647, 0.660, 0.672, 0.682,
        0.690, 0.697, 0.703, 0.708, 0.714, 0.718, 0.721,
        0.725, 0.725
    ])
}

# Check data dimensions
for pair, values in Sw_ref.items():
    if len(values) != len(pvi_ref):
        raise ValueError(
            f"Incorrect number of values for {pair}: "
            f"{len(values)} instead of {len(pvi_ref)}."
        )


# ================================================================
# 2. PHYSICAL AND NUMERICAL PARAMETERS
# ================================================================

Lx = 20.0
Ly = 15.0

phi_matrix = 0.15
phi_fracture = 1.0

K_matrix = 50.0
K_fracture = 1.0e5

mu_w = 1.0
mu_o = 1.0

injection_rate_PVI = 0.2

S_initial = 0.0
S_injected = 1.0

# Mesh used during calibration.
# The time step is deliberately larger to limit
# the cost of the optimization computation.
Nx = 20
Ny = 15

dx = Lx / Nx
dy = Ly / Ny

dt_cal = 0.05
Nt_cal = int(round(2.5 / (injection_rate_PVI * dt_cal)))

assert np.isclose(
    injection_rate_PVI * dt_cal * Nt_cal, 2.5
)

# Maximum number of memory steps kept for alpha < 1.
# This approximation must be tested by varying the window.
memory_window_cal = 80


# ================================================================
# 3. CONSTRUCTION OF THE FRACTURE NETWORK
# ================================================================

x_centers = (np.arange(Nx) + 0.5) * dx
y_centers = (np.arange(Ny) + 0.5) * dy

X, Y = np.meshgrid(x_centers, y_centers)

fracture_mask = np.zeros((Ny, Nx), dtype=bool)


def mark_horizontal(y0, x1, x2, width=None):
    """Marks a horizontal fracture in the mesh."""
    if width is None:
        width = max(dx, dy) * 0.55

    condition = (
        (np.abs(Y - y0) <= width / 2.0)
        & (X >= x1)
        & (X <= x2)
    )
    fracture_mask[condition] = True


def mark_vertical(x0, y1, y2, width=None):
    """Marks a vertical fracture in the mesh."""
    if width is None:
        width = max(dx, dy) * 0.55

    condition = (
        (np.abs(X - x0) <= width / 2.0)
        & (Y >= y1)
        & (Y <= y2)
    )
    fracture_mask[condition] = True


# Illustrative network of the initial model
mark_horizontal(8.0, 0.0, 15.0)
mark_horizontal(3.5, 2.0, 12.0)
mark_horizontal(12.0, 6.0, 16.0)

mark_vertical(3.0, 1.0, 10.0)
mark_vertical(8.0, 2.0, 13.0)
mark_vertical(11.0, 2.0, 12.0)
mark_vertical(14.0, 8.0, 13.0)

print("Number of cells:", Nx * Ny)
print("Fracture cells:", np.count_nonzero(fracture_mask))


# ================================================================
# 4. RELATIVE MOBILITY FUNCTIONS
# ================================================================

def fractional_flow(Sw):
    """
    Water fraction with krw = Sw^3 and kro = (1-Sw)^3.
    Viscosities are taken into account explicitly.
    """
    Sw = np.clip(Sw, 0.0, 1.0)

    krw = Sw**3
    kro = (1.0 - Sw)**3

    lambda_w = krw / mu_w
    lambda_o = kro / mu_o

    denominator = lambda_w + lambda_o

    return np.divide(
        lambda_w,
        denominator,
        out=np.zeros_like(Sw),
        where=denominator > 1e-14
    )


# ================================================================
# 5. SIMPLIFIED SPATIAL OPERATOR
# ================================================================

def spatial_operator(Sw, a_v, a_f, a_m):
    """
    Simplified upwind advection operator.
    The horizontal velocity is imposed, not computed from
    a pressure equation.
    """
    fw = fractional_flow(Sw)

    # Local mobility factor
    mobility_factor = np.where(
        fracture_mask,
        (K_fracture / K_fracture) * a_f,
        (K_matrix / K_fracture) * a_m
    )

    fw_eff = fw * mobility_factor

    # Upwind horizontal flux
    flux_x = np.zeros_like(Sw)

    flux_x[:, 1:] = (
        a_v * (fw_eff[:, 1:] - fw_eff[:, :-1]) / dx
    )

    # No vertical flux in this simplified version
    flux_y = np.zeros_like(Sw)

    return -(flux_x + flux_y)


# ================================================================
# 6. SIMULATION OF THE FRACTIONAL MODEL
# ================================================================

def simulate_model(
    gamma_f,
    gamma_m,
    a_v,
    a_f,
    a_m,
    dt,
    Nt,
    memory_window
):
    """
    Simulation with a truncated fractional convolution.

    The fractional memory is applied separately to the fracture
    and matrix zones. For gamma = 1, the memory coefficients
    are equal to 1.

    IMPORTANT:
    The scheme remains a simplified numerical approximation.
    """
    Sw = np.full((Ny, Nx), S_initial, dtype=float)

    # Memory coefficients
    m = np.arange(memory_window + 1, dtype=float)

    weights_f = (m + 1.0)**gamma_f - m**gamma_f
    weights_m = (m + 1.0)**gamma_m - m**gamma_m

    # History of the spatial operator
    history_f = []
    history_m = []

    # Result: average saturation over time
    Sw_mean = np.zeros(Nt + 1)
    Sw_mean[0] = np.mean(Sw)

    for n in range(Nt):

        F = spatial_operator(Sw, a_v, a_f, a_m)

        F_f = np.where(fracture_mask, F, 0.0)
        F_m = np.where(fracture_mask, 0.0, F)

        history_f.append(F_f.copy())
        history_m.append(F_m.copy())

        # Fractional sum over the memory window
        n_hist = min(n + 1, memory_window + 1)

        memory_f = np.zeros_like(Sw)
        memory_m = np.zeros_like(Sw)

        for j in range(n_hist):
            idx = n - j
            memory_f += weights_f[j] * history_f[idx]
            memory_m += weights_m[j] * history_m[idx]

        # Simplified explicit update
        Sw_new = Sw + dt * (memory_f + memory_m)

        # Injection at the left boundary
        Sw_new[:, 0] = S_injected

        # Physical saturation bounds
        Sw = np.clip(Sw_new, 0.0, 1.0)

        Sw_mean[n + 1] = np.mean(Sw)

    pvi = injection_rate_PVI * dt * np.arange(Nt + 1)

    return pvi, Sw_mean


# ================================================================
# 7. SIMULATION AND INTERPOLATION AT REFERENCE PVI
# ================================================================

def simulate_for_calibration(parameters, pair):
    a_v, a_f, a_m = parameters
    gamma_f, gamma_m = pair

    pvi_sim, Sw_sim = simulate_model(
        gamma_f=gamma_f,
        gamma_m=gamma_m,
        a_v=a_v,
        a_f=a_f,
        a_m=a_m,
        dt=dt_cal,
        Nt=Nt_cal,
        memory_window=memory_window_cal
    )

    return np.interp(pvi_ref, pvi_sim, Sw_sim)


# ================================================================
# 8. OBJECTIVE FUNCTION: JOINT CALIBRATION OF THE THREE CASES
# ================================================================

def compute_residuals(parameters):
    """
    Concatenates the residuals of the three curves to calibrate
    the same triplet of parameters over the three cases.
    """
    all_residuals = []

    for pair in Sw_ref:
        prediction = simulate_for_calibration(parameters, pair)
        residual = prediction - Sw_ref[pair]
        all_residuals.extend(residual)

    return np.asarray(all_residuals, dtype=float)


# ================================================================
# 9. OPTIMIZATION WITH MULTIPLE STARTING POINTS
# ================================================================

initial_guesses = [
    [1.0, 1.0, 1.0],
    [0.5, 1.0, 2.0],
    [2.0, 0.5, 1.0],
    [1.0, 2.0, 0.5],
    [3.0, 1.5, 0.5],
    [0.5, 1.5, 3.0]
]

lower_bounds = [0.1, 0.1, 0.1]
upper_bounds = [10.0, 10.0, 10.0]

best_fit = None
best_cost = np.inf

print("\nStarting calibration...")
print("This operation may take some time.")

for i, guess in enumerate(initial_guesses, start=1):

    print(f"\nTrial {i}/{len(initial_guesses)}")
    print("Starting point:", guess)

    try:
        fit = least_squares(
            compute_residuals,
            x0=np.asarray(guess, dtype=float),
            bounds=(lower_bounds, upper_bounds),
            method="trf",
            loss="linear",
            x_scale="jac",
            max_nfev=12,
            diff_step=0.05,
            ftol=1e-4,
            xtol=1e-4,
            gtol=1e-4,
            verbose=1
        )

        rmse_objective = np.sqrt(np.mean(fit.fun**2))

        print("Combined RMSE:", rmse_objective)
        print("Parameters:", fit.x)

        if np.isfinite(rmse_objective) and rmse_objective < best_cost:
            best_cost = rmse_objective
            best_fit = fit

    except Exception as error:
        print("This trial failed:", error)


if best_fit is None:
    raise RuntimeError(
        "Calibration failed. Check the parameters and data."
    )

a_v_opt, a_f_opt, a_m_opt = best_fit.x

print("\n" + "=" * 65)
print("OPTIMIZED PARAMETERS")
print("=" * 65)
print(f"a_v = {a_v_opt:.6f}")
print(f"a_f = {a_f_opt:.6f}")
print(f"a_m = {a_m_opt:.6f}")
print(f"Combined calibration RMSE = {best_cost:.6f}")


# ================================================================
# 10. FINAL SIMULATIONS WITH FINER TIME STEP
# ================================================================

# Final PVI = 0.2 * 0.01 * 1250 = 2.5
dt_final = 0.01
Nt_final = int(round(2.5 / (injection_rate_PVI * dt_final)))

assert np.isclose(
    injection_rate_PVI * dt_final * Nt_final, 2.5
)

# Larger memory window for the final simulations
memory_window_final = 250

comparison_curves = {}
metrics = {}

for pair, reference in Sw_ref.items():

    gamma_f, gamma_m = pair

    pvi_sim, Sw_sim = simulate_model(
        gamma_f=gamma_f,
        gamma_m=gamma_m,
        a_v=a_v_opt,
        a_f=a_f_opt,
        a_m=a_m_opt,
        dt=dt_final,
        Nt=Nt_final,
        memory_window=memory_window_final
    )

    prediction_at_ref = np.interp(pvi_ref, pvi_sim, Sw_sim)

    residuals = prediction_at_ref - reference

    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals**2))
    maxae = np.max(np.abs(residuals))

    comparison_curves[pair] = (
        pvi_ref.copy(),
        reference.copy(),
        prediction_at_ref.copy()
    )

    metrics[pair] = {
        "MAE": mae,
        "RMSE": rmse,
        "MaxAE": maxae
    }


# ================================================================
# 11. DISPLAY OF ERRORS
# ================================================================

print("\n" + "=" * 65)
print("METRICS ON THE REFERENCE CURVES")
print("=" * 65)

for pair, result in metrics.items():
    print(f"\n(gamma_f, gamma_m) = {pair}")
    print(f"MAE   = {result['MAE']:.6f}")
    print(f"RMSE  = {result['RMSE']:.6f}")
    print(f"MaxAE = {result['MaxAE']:.6f}")


# ================================================================
# 12. PLOT: AVERAGE SATURATION CURVES
# ================================================================

fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)

for ax, pair in zip(axes, Sw_ref.keys()):

    pvi, reference, simulation = comparison_curves[pair]

    ax.plot(
        pvi,
        reference,
        "ko",
        markersize=3,
        label="Reference"
    )

    ax.plot(
        pvi,
        simulation,
        color="crimson",
        linewidth=2,
        label="Our scheme"
    )

    ax.set_title(
        rf"$\gamma_f={pair[0]:.2f},\ "
        rf"\gamma_m={pair[1]:.2f}$"
    )

    ax.set_xlabel("PVI")
    ax.grid(True, alpha=0.3)
    ax.legend()

axes[0].set_ylabel("Average water saturation")

fig.suptitle(
    "Comparison of average saturation curves",
    fontsize=14
)

plt.tight_layout()
plt.show()


# ================================================================
# 13. QQ-PLOTS OF RESIDUALS
# ================================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, pair in zip(axes, Sw_ref.keys()):

    pvi, reference, simulation = comparison_curves[pair]

    residuals = simulation - reference
    residuals = residuals[np.isfinite(residuals)]

    # Centering to examine the shape of the distribution
    residuals_centered = residuals - np.mean(residuals)

    stats.probplot(
        residuals_centered,
        dist="norm",
        plot=ax
    )

    ax.set_title(
        rf"QQ-plot: $\gamma_f={pair[0]:.2f}, "
        rf"\gamma_m={pair[1]:.2f}$"
    )

    ax.set_xlabel("Theoretical normal quantiles")
    ax.set_ylabel("Residual quantiles")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ================================================================
# 14. SHAPIRO-WILK NORMALITY TEST
# ================================================================

print("\n" + "=" * 65)
print("SHAPIRO-WILK NORMALITY TEST")
print("=" * 65)

for pair in Sw_ref:

    _, reference, simulation = comparison_curves[pair]
    residuals = simulation - reference
    residuals = residuals[np.isfinite(residuals)]

    if len(residuals) >= 3:
        W, p_value = stats.shapiro(residuals)

        print(f"\nCase {pair}")
        print(f"W = {W:.6f}")
        print(f"p-value = {p_value:.6f}")

        if p_value > 0.05:
            print(
                "Conclusion: normality is not rejected "
                "at the 5% level."
            )
        else:
            print(
                "Conclusion: normality is rejected "
                "at the 5% level."
            )


# ================================================================
# 15. END
# ================================================================

print("\nCalibration and comparison completed.")
print(
    "Warning: results depend on the mesh, time step, "
    "memory window and model approximations."
)