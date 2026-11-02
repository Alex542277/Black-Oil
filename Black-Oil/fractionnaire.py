"""Etude comparative d'un modele simplifie de saturation fractionnaire.

Le script compare plusieurs ordres alpha, la limite alpha=1 au schema classique,
les raffinements temporel et spatial, et produit des diagnostics et figures.

Limites : ce modele simplifie n'est pas un simulateur Black-Oil complet. Les tests
ci-dessous sont des controles numeriques, pas une validation experimentale.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gamma


# --------------------------- Parametres ------------------------------------
PHI = 0.20
MU_W, MU_O = 3.0, 5.0
VX, VY = 0.01, 0.006
T_CHAR = 1.0       # temps caracteristique t_c, dans l'unite de temps choisie
Q0 = 0.0           # source illustrative; calibrer et preciser ses unites avant usage
SOURCE_DECAY = 1.0
S_INITIAL, S_INJECTED = 0.20, 1.0
INJECTION_WIDTH_FRACTION = 0.15
OUTDIR = Path("fractional_blackoil_results")
OUTDIR.mkdir(exist_ok=True)

# --------------------------- Lois constitutives ----------------------------
def krw(sw):
    return np.clip(sw, 0.0, 1.0) ** 2


def fractional_flow(sw):
    kr = krw(sw)
    return kr / np.maximum(kr + (1.0 - kr) * MU_W / MU_O, 1e-14)


def impose_injection(sw):
    """Impose une bande saturee sur le bord gauche (condition prescrite)."""
    n_inj = max(1, int(round(INJECTION_WIDTH_FRACTION * sw.shape[1])))
    sw[:, :n_inj] = S_INJECTED
    return sw

# --------------------------- Operateur spatial -----------------------------
def spatial_rhs(sw, dx, dy, time):
    """F(S,t)=(-div(fw(S)u)+q)/phi, flux de faces conservatifs amont.

    Les flux exterieurs sont nuls. La bande d'injection est imposee apres chaque
    pas de temps; son effet doit etre pris en compte dans toute etude de bilan.
    """
    ny, nx = sw.shape
    fw = fractional_flow(sw)
    flux_x = np.zeros((ny, nx + 1))
    flux_y = np.zeros((ny + 1, nx))

    # Flux internes en x, avec choix amont selon le signe de VX.
    if VX >= 0:
        flux_x[:, 1:nx] = VX * fw[:, :-1]
    else:
        flux_x[:, 1:nx] = VX * fw[:, 1:]
    # Flux internes en y.
    if VY >= 0:
        flux_y[1:ny, :] = VY * fw[:-1, :]
    else:
        flux_y[1:ny, :] = VY * fw[1:, :]

    div_flux = ((flux_x[:, 1:] - flux_x[:, :-1]) / dx
                + (flux_y[1:, :] - flux_y[:-1, :]) / dy)
    q = np.zeros_like(sw)
    if Q0 != 0.0:
        q[ny // 2, nx // 2] = Q0 * np.exp(-time / SOURCE_DECAY)
    return (-div_flux + q) / PHI

# --------------------------- Solveurs temporels ----------------------------
def initial_field(nx):
    sw = np.full((nx, nx), S_INITIAL, dtype=float)
    return impose_injection(sw)


def simulate_fractional(alpha, nx=20, dt=0.01, final_time=0.20, save_fields=False):
    """Integration produit de convolution de l'equation de Caputo.

    Sw(t_n)=Sw(0)+t_c^(1-alpha)*dt^alpha/Gamma(alpha+1)
            * sum_{j=0}^{n-1} b_(n-1-j) F(Sw_j,t_j),
    b_k=(k+1)^alpha-k^alpha.

    Cette formule explicite est sensible au pas de temps; l'ecretage et la
    condition d'injection peuvent modifier la conservation.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must satisfy 0 < alpha <= 1")
    if nx < 5 or dt <= 0 or final_time <= 0:
        raise ValueError("nx >= 5, dt > 0 et final_time > 0 sont requis")
    nt = int(round(final_time / dt))
    final_time = nt * dt
    dx = dy = 1.0 / nx
    sw0 = initial_field(nx)
    states = [sw0.copy()]
    rhs_history = []
    center = [sw0[nx // 2, nx // 2]]
    mean = [sw0.mean()]
    clipped_count = 0
    weights = np.array([(k + 1)**alpha - k**alpha for k in range(nt)])
    scale = T_CHAR ** (1.0 - alpha)

    for n in range(nt):
        rhs_history.append(spatial_rhs(states[-1], dx, dy, n * dt))
        conv = np.zeros_like(sw0)
        for j in range(n + 1):
            conv += weights[n - j] * rhs_history[j]
        sw_new = sw0 + scale * dt**alpha / gamma(alpha + 1.0) * conv
        clipped_count += int(np.count_nonzero((sw_new < 0.0) | (sw_new > 1.0)))
        sw_new = np.clip(sw_new, 0.0, 1.0)
        impose_injection(sw_new)
        states.append(sw_new.copy())
        center.append(sw_new[nx // 2, nx // 2])
        mean.append(sw_new.mean())

    arr = np.asarray(states)
    return {"alpha": alpha, "time": np.linspace(0, final_time, nt + 1),
            "center": np.asarray(center), "mean": np.asarray(mean),
            "states": arr if save_fields else None, "nx": nx, "dt": dt,
            "final_time": final_time, "clipped_cells_total": clipped_count,
            "min_sw": float(arr.min()), "max_sw": float(arr.max())}


def simulate_classical(nx=20, dt=0.01, final_time=0.20):
    """Schema classique explicite d'Euler pour la saturation (alpha=1)."""
    nt = int(round(final_time / dt))
    final_time = nt * dt
    dx = dy = 1.0 / nx
    sw = initial_field(nx)
    states = [sw.copy()]
    center, mean = [sw[nx // 2, nx // 2]], [sw.mean()]
    clipped_count = 0
    for n in range(nt):
        sw_new = sw + dt * spatial_rhs(sw, dx, dy, n * dt)
        clipped_count += int(np.count_nonzero((sw_new < 0.0) | (sw_new > 1.0)))
        sw = impose_injection(np.clip(sw_new, 0.0, 1.0))
        states.append(sw.copy())
        center.append(sw[nx // 2, nx // 2])
        mean.append(sw.mean())
    arr = np.asarray(states)
    return {"time": np.linspace(0, final_time, nt + 1), "center": np.asarray(center),
            "mean": np.asarray(mean), "states": arr, "nx": nx, "dt": dt,
            "final_time": final_time, "clipped_cells_total": clipped_count,
            "min_sw": float(arr.min()), "max_sw": float(arr.max())}

# --------------------------- Experience principale -------------------------
def run_main_experiment():
    nx, dt, tf = 20, 0.01, 0.20
    alphas = [0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
    results = {a: simulate_fractional(a, nx, dt, tf) for a in alphas}
    classical = simulate_classical(nx, dt, tf)

    # Figure 1 : saturation centrale selon alpha.
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for ax, alpha in zip(axes.flat, alphas):
        r = results[alpha]
        ax.plot(r["time"], r["center"], linewidth=2, label=f"fractional, α={alpha}")
        if alpha == 1.0:
            ax.plot(classical["time"], classical["center"], "--k", linewidth=1.5,
                    label="classic (Euler)")
        ax.set_title(f"α = {alpha}")
        ax.set_xlabel("Times")
        ax.set_ylabel(r"$S_w$ in the center")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Effect of Fractional Order on Central Saturation")
    fig.tight_layout()
    fig.savefig(OUTDIR / "01_alpha_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Figure 2 : alpha=1 contre le solveur classique.
    frac1 = results[1.0]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(frac1["time"], frac1["center"], linewidth=2, label="Integration product, α=1")
    ax.plot(classical["time"], classical["center"], "--", linewidth=2,
            label="Classical Euler")
    ax.set_xlabel("Times")
    ax.set_ylabel(r"$S_w$ in the center")
    ax.set_title("Checking the Limit α=1")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "02_alpha1_vs_classical.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Figure 3 : champs finaux.
    selected = [0.3, 0.5, 0.7, 1.0]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), constrained_layout=True)
    for ax, alpha in zip(axes, selected):
        field = simulate_fractional(alpha, nx, dt, tf, save_fields=True)["states"][-1]
        im = ax.imshow(field, origin="lower", extent=[0, 1, 0, 1], vmin=0, vmax=1,
                       aspect="equal", cmap="viridis")
        ax.set_title(f"α={alpha}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.colorbar(im, ax=axes, label=r"$S_w$")
    fig.suptitle("Saturation fields at the final time step")
    fig.savefig(OUTDIR / "03_final_saturation_maps.png", dpi=300, bbox_inches="tight")
    plt.show()

    diff = np.abs(frac1["center"] - classical["center"])
    rows = []
    for alpha, r in results.items():
        rows.append({"alpha": alpha, "nx": nx, "dt": dt, "final_time": tf,
                     "Sw_center_final": r["center"][-1], "Sw_mean_final": r["mean"][-1],
                     "Sw_min": r["min_sw"], "Sw_max": r["max_sw"],
                     "cells_clipped_total": r["clipped_cells_total"]})
    pd.DataFrame(rows).to_csv(OUTDIR / "alpha_diagnostics.csv", index=False)
    print("\n=== Comparaison alpha=1 / Euler classique ===")
    print(f"Erreur absolue maximale au centre : {diff.max():.6e}")
    print(f"Erreur absolue moyenne au centre : {diff.mean():.6e}")
    print("\n=== Diagnostics par alpha ===")
    print(pd.DataFrame(rows).to_string(index=False))
    return results, classical

# --------------------------- Raffinement -----------------------------------
def refinement_study():
    """Etude de convergence interne : ce n'est pas une validation exacte."""
    alpha, tf = 0.7, 0.20
    temporal_rows, temporal_results = [], []
    nx = 20
    for dt in [0.02, 0.01, 0.005]:
        r = simulate_fractional(alpha, nx, dt, tf)
        temporal_results.append(r)
        temporal_rows.append({"study": "temporal", "alpha": alpha, "nx": nx, "dt": dt,
                              "Sw_center_final": r["center"][-1], "Sw_mean_final": r["mean"][-1],
                              "Sw_min": r["min_sw"], "Sw_max": r["max_sw"],
                              "cells_clipped_total": r["clipped_cells_total"]})
    tref = temporal_results[-1]["center"][-1]
    for row in temporal_rows:
        row["abs_difference_from_finest"] = abs(row["Sw_center_final"] - tref)

    spatial_rows, spatial_results = [], []
    dt = 0.005
    for nx in [10, 20, 40]:
        r = simulate_fractional(alpha, nx, dt, tf)
        spatial_results.append(r)
        spatial_rows.append({"study": "spatial", "alpha": alpha, "nx": nx, "dt": dt,
                            "Sw_center_final": r["center"][-1], "Sw_mean_final": r["mean"][-1],
                            "Sw_min": r["min_sw"], "Sw_max": r["max_sw"],
                            "cells_clipped_total": r["clipped_cells_total"]})
    xref = spatial_results[-1]["center"][-1]
    for row in spatial_rows:
        row["abs_difference_from_finest"] = abs(row["Sw_center_final"] - xref)

    df = pd.DataFrame(temporal_rows + spatial_rows)
    df.to_csv(OUTDIR / "refinement_study.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot([r["dt"] for r in temporal_rows], [r["Sw_center_final"] for r in temporal_rows], "o-", lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("Pas de temps Δt (échelle logarithmique)")
    ax.set_ylabel(r"$S_w$ au centre à $t_f$")
    ax.set_title(f"Raffinement temporel (α={alpha}, Nx={nx})")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTDIR / "04_temporal_refinement.png", dpi=300, bbox_inches="tight")
    plt.show()

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot([r["nx"] for r in spatial_rows], [r["Sw_center_final"] for r in spatial_rows], "o-", lw=2)
    ax.set_xlabel("Nombre de cellules par direction Nx")
    ax.set_ylabel(r"$S_w$ au centre à $t_f$")
    ax.set_title(f"Raffinement spatial (α={alpha}, Δt={dt})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTDIR / "05_spatial_refinement.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("\n=== Etude de raffinement ===")
    print(df.to_string(index=False))
    print("Note: Refinement measures numerical sensitivity, not physical accuracy.")
    return df

# --------------------------- Tests de base ---------------------------------
def run_basic_checks():
    nx, dt, tf = 12, 0.005, 0.05
    frac1 = simulate_fractional(1.0, nx, dt, tf)
    classical = simulate_classical(nx, dt, tf)
    diff = float(np.max(np.abs(frac1["center"] - classical["center"])))
    assert frac1["min_sw"] >= -1e-12 and frac1["max_sw"] <= 1.0 + 1e-12
    assert len(frac1["time"]) == int(round(tf / dt)) + 1
    print("\n=== Controles numeriques ===")
    print(f"Ecart max alpha=1 / Euler : {diff:.6e}")
    print(f"Bornes Sw : [{frac1['min_sw']:.6f}, {frac1['max_sw']:.6f}]")
    if diff > 1e-6:
        print("AVERTISSEMENT : l'ecart alpha=1 / Euler depasse la tolerance; examiner l'ecretage et les conditions imposees.")
    else:
        print("Coherence alpha=1 / Euler verifiee a la tolerance choisie.")

if __name__ == "__main__":
    run_basic_checks()
    run_main_experiment()
    refinement_study()
