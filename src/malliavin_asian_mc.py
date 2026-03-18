import math
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Project A - Malliavin calculus and Monte Carlo methods
# Numerical part for Asian options in the Black-Scholes model
# ------------------------------------------------------------
# Model parameters imposed by the project statement
# x = 100, r = 3%, sigma = 20%, T = 1
# M in {50, 150, 250}
# N = 1000, 3000, ..., 51000
# ------------------------------------------------------------
# Notes
# 1) We follow the discretization required in the statement:
#       A_M^x = (T/M) * sum_{k=0}^M X_{t_k}^x
# 2) For the finite-difference estimator we use common random numbers.
# 3) For method A, the weight from Part A becomes, in the constant-volatility
#    and uniform-grid case, Pi_A = B_{t_1} / (x sigma t_1) = Delta B_0/(x sigma dt).
# ------------------------------------------------------------

@dataclass
class Params:
    x: float = 100.0
    r: float = 0.03
    sigma: float = 0.20
    T: float = 1.0
    K1: float = 100.0
    K2: float = 110.0
    M_list: tuple = (50, 150, 250)
    N_grid: tuple = tuple(range(1000, 51001, 2000))
    eps_list: tuple = (0.25, 0.5, 1.0, 2.0, 5.0)
    seed: int = 123456


PARAMS = Params()
DISC = math.exp(-PARAMS.r * PARAMS.T)
OUTDIR = os.path.join(os.path.dirname(__file__))


def simulate_normalized_paths(M: int, N: int, seed: int):
    """Simulate Z_t such that X_t^x = x * Z_t."""
    rng = np.random.default_rng(seed + M)
    dt = PARAMS.T / M
    dB = math.sqrt(dt) * rng.standard_normal(size=(N, M))
    log_increments = (PARAMS.r - 0.5 * PARAMS.sigma**2) * dt + PARAMS.sigma * dB

    Z = np.empty((N, M + 1))
    Z[:, 0] = 1.0
    Z[:, 1:] = np.exp(np.cumsum(log_increments, axis=1))
    return Z, dB, dt


def summarize(arr: np.ndarray):
    n = arr.size
    mean = arr.mean()
    var = arr.var(ddof=1)
    se = math.sqrt(var / n)
    ci_low = mean - 1.96 * se
    ci_high = mean + 1.96 * se
    return mean, var, se, ci_low, ci_high


# -----------------------------
# Monte Carlo estimators
# -----------------------------

def build_estimators_for_M(M: int, Nmax: int):
    Z, dB, dt = simulate_normalized_paths(M=M, N=Nmax, seed=PARAMS.seed)

    # Approximation imposed by the statement
    A1 = dt * np.sum(Z, axis=1)          # A_M when x = 1
    A_x = PARAMS.x * A1                  # A_M^x

    # Payoffs for prices
    payoff_call = np.maximum(A_x - PARAMS.K1, 0.0)
    payoff_digital = ((A_x > PARAMS.K1) & (A_x < PARAMS.K2)).astype(float)

    price_samples = {
        "call": DISC * payoff_call,
        "digital": DISC * payoff_digital,
    }

    # Malliavin method A (Part A + discrete payoff)
    pi_A = dB[:, 0] / (PARAMS.x * PARAMS.sigma * dt)

    # Malliavin method B (Part B + discrete stochastic integral)
    stochastic_integral = np.sum(Z[:, :-1] * dB, axis=1)
    pi_B = 1.0 / PARAMS.x + 2.0 / (PARAMS.x * PARAMS.sigma * A1) * stochastic_integral

    delta_samples = {
        ("malliavin_A", "call"): DISC * payoff_call * pi_A,
        ("malliavin_A", "digital"): DISC * payoff_digital * pi_A,
        ("malliavin_B", "call"): DISC * payoff_call * pi_B,
        ("malliavin_B", "digital"): DISC * payoff_digital * pi_B,
    }

    # Finite differences with common random numbers
    for eps in PARAMS.eps_list:
        A_plus = (PARAMS.x + eps) * A1
        A_minus = (PARAMS.x - eps) * A1

        fd_call = DISC * (
            np.maximum(A_plus - PARAMS.K1, 0.0) - np.maximum(A_minus - PARAMS.K1, 0.0)
        ) / (2.0 * eps)

        fd_digital = DISC * (
            ((A_plus > PARAMS.K1) & (A_plus < PARAMS.K2)).astype(float)
            - ((A_minus > PARAMS.K1) & (A_minus < PARAMS.K2)).astype(float)
        ) / (2.0 * eps)

        delta_samples[(f"FD_eps_{eps}", "call")] = fd_call
        delta_samples[(f"FD_eps_{eps}", "digital")] = fd_digital

    return {
        "Z": Z,
        "dB": dB,
        "dt": dt,
        "A1": A1,
        "price_samples": price_samples,
        "delta_samples": delta_samples,
    }


# -----------------------------
# Tables and graphics
# -----------------------------

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    Nmax = max(PARAMS.N_grid)
    all_results = {M: build_estimators_for_M(M, Nmax) for M in PARAMS.M_list}

    summary_rows = []
    convergence_rows = []
    eps_rows = []

    for M in PARAMS.M_list:
        res = all_results[M]

        # Summary at N = 51000
        for option in ("call", "digital"):
            mean, var, se, ci_low, ci_high = summarize(res["price_samples"][option])
            summary_rows.append(
                {
                    "M": M,
                    "type": "price",
                    "method": "MC",
                    "option": option,
                    "estimate": mean,
                    "variance": var,
                    "std_error": se,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )

        methods = ["malliavin_A", "malliavin_B"] + [f"FD_eps_{eps}" for eps in PARAMS.eps_list]
        for method in methods:
            for option in ("call", "digital"):
                mean, var, se, ci_low, ci_high = summarize(res["delta_samples"][(method, option)])
                summary_rows.append(
                    {
                        "M": M,
                        "type": "delta",
                        "method": method,
                        "option": option,
                        "estimate": mean,
                        "variance": var,
                        "std_error": se,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                    }
                )

        # Convergence in N
        for N in PARAMS.N_grid:
            for option in ("call", "digital"):
                mean, var, se, ci_low, ci_high = summarize(res["price_samples"][option][:N])
                convergence_rows.append(
                    {
                        "kind": "price",
                        "M": M,
                        "N": N,
                        "method": "MC",
                        "option": option,
                        "estimate": mean,
                        "variance": var,
                        "ci_width": ci_high - ci_low,
                    }
                )

            for method in ("malliavin_A", "malliavin_B", "FD_eps_1.0"):
                for option in ("call", "digital"):
                    mean, var, se, ci_low, ci_high = summarize(res["delta_samples"][(method, option)][:N])
                    convergence_rows.append(
                        {
                            "kind": "delta",
                            "M": M,
                            "N": N,
                            "method": method,
                            "option": option,
                            "estimate": mean,
                            "variance": var,
                            "ci_width": ci_high - ci_low,
                        }
                    )

        # Epsilon effect for FD
        for eps in PARAMS.eps_list:
            for option in ("call", "digital"):
                mean, var, se, ci_low, ci_high = summarize(res["delta_samples"][(f"FD_eps_{eps}", option)])
                eps_rows.append(
                    {
                        "M": M,
                        "epsilon": eps,
                        "option": option,
                        "estimate": mean,
                        "variance": var,
                        "ci_width": ci_high - ci_low,
                    }
                )

    summary_df = pd.DataFrame(summary_rows)
    convergence_df = pd.DataFrame(convergence_rows)
    eps_df = pd.DataFrame(eps_rows)

    summary_df.to_csv(os.path.join(OUTDIR, "summary_results_N51000.csv"), index=False)
    convergence_df.to_csv(os.path.join(OUTDIR, "convergence_results.csv"), index=False)
    eps_df.to_csv(os.path.join(OUTDIR, "fd_epsilon_effect.csv"), index=False)

    # ---------------- plots ----------------
    plt.figure(figsize=(7, 4.5))
    for M in PARAMS.M_list:
        sub = convergence_df[(convergence_df["kind"] == "price") & (convergence_df["option"] == "call") & (convergence_df["M"] == M)]
        plt.plot(sub["N"], sub["estimate"], label=f"M={M}")
    plt.xlabel("Number of simulations N")
    plt.ylabel("Asian call price estimate")
    plt.title("Price convergence - Asian call")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "price_convergence_call.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    for M in PARAMS.M_list:
        sub = convergence_df[(convergence_df["kind"] == "price") & (convergence_df["option"] == "digital") & (convergence_df["M"] == M)]
        plt.plot(sub["N"], sub["estimate"], label=f"M={M}")
    plt.xlabel("Number of simulations N")
    plt.ylabel("Digital Asian price estimate")
    plt.title("Price convergence - digital Asian option")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "price_convergence_digital.png"), dpi=200)
    plt.close()

    for option in ("call", "digital"):
        plt.figure(figsize=(7, 4.5))
        for method, label in (
            ("malliavin_A", "Malliavin A"),
            ("malliavin_B", "Malliavin B"),
            ("FD_eps_1.0", "Finite diff. (eps=1)"),
        ):
            sub = convergence_df[
                (convergence_df["kind"] == "delta")
                & (convergence_df["option"] == option)
                & (convergence_df["M"] == 250)
                & (convergence_df["method"] == method)
            ]
            plt.plot(sub["N"], sub["estimate"], label=label)
        plt.xlabel("Number of simulations N")
        plt.ylabel("Delta estimate")
        plt.title(f"Delta convergence - {option} payoff (M=250)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"delta_convergence_{option}_M250.png"), dpi=200)
        plt.close()

    for option in ("call", "digital"):
        plot_df = summary_df[
            (summary_df["type"] == "delta")
            & (summary_df["option"] == option)
            & (summary_df["method"].isin(["FD_eps_1.0", "malliavin_A", "malliavin_B"]))
        ]
        xloc = np.arange(len(PARAMS.M_list))
        width = 0.24
        plt.figure(figsize=(7, 4.5))
        for idx, (method, label) in enumerate(
            (("FD_eps_1.0", "FD eps=1"), ("malliavin_A", "Mall. A"), ("malliavin_B", "Mall. B"))
        ):
            vals = [plot_df[(plot_df["M"] == M) & (plot_df["method"] == method)]["variance"].iloc[0] for M in PARAMS.M_list]
            plt.bar(xloc + idx * width - width, vals, width=width, label=label)
        plt.xticks(xloc, [f"M={M}" for M in PARAMS.M_list])
        plt.ylabel("Empirical variance")
        plt.title(f"Variance comparison for Delta - {option}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"variance_comparison_delta_{option}.png"), dpi=200)
        plt.close()

    for option in ("call", "digital"):
        plt.figure(figsize=(7, 4.5))
        for M in PARAMS.M_list:
            sub = eps_df[(eps_df["option"] == option) & (eps_df["M"] == M)]
            plt.plot(sub["epsilon"], sub["variance"], marker="o", label=f"M={M}")
        plt.xlabel("epsilon")
        plt.ylabel("Empirical variance of FD Delta")
        plt.title(f"Effect of epsilon on FD variance - {option}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"fd_epsilon_variance_{option}.png"), dpi=200)
        plt.close()

    print("Files written to:", OUTDIR)
    print("Main table:", os.path.join(OUTDIR, "summary_results_N51000.csv"))


if __name__ == "__main__":
    main()
