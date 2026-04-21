import os
import sys
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import k1
from joblib import Parallel, delayed

import triadic_library as triadic

# =============================================================================
# SECTION 1: PARAMETER READING & INITIALIZATION
# =============================================================================
# This script performs a "Refinement Limit" calibration.
# It maintains N_total and Ly constant, while increasing the number of rings (R).
# As R increases, N_per_ring decreases and delta decreases, keeping 2D density fixed.
# Example command: python calibrate_refinement.py 10000 200.0 0.07 0.2 50 42 True 12

try:
    N_total = int(sys.argv[1])  # Total budget of nodes for the whole system
    density_1D_base = float(sys.argv[2])  # Linear density used to fix Lx
    c = float(sys.argv[3])  # Base probability for structural connections
    d0_base = float(sys.argv[4])  # Target 1D d0 (baseline for ring 1)
    max_rings = int(sys.argv[5])  # Maximum resolution (rings) to test
    seed = int(sys.argv[6])  # Random seed

    # Optional parameters handling
    use_parallel = sys.argv[7].lower() == 'true' if len(sys.argv) > 7 else False
    n_jobs = int(sys.argv[8]) if len(sys.argv) > 8 else 1
except IndexError:
    print(
        "Error: Missing arguments. Usage: python calibrate_refinement.py N_total density_1D_base c d0_base max_rings "
        "seed [use_parallel] [n_jobs]"
    )
    sys.exit(1)

# Prompt for precision
user_choice = input("Enter the tolerance for d0 optimization (default 0.01): ")
try:
    tol = float(user_choice)
except ValueError:
    tol = 0.01

init_time = time.time()

# --- THE REFINEMENT GEOMETRY ---
# We fix Lx using the initial total nodes and base density.
# We fix Ly to match Lx for a perfect 2D square comparison.
Lx_fixed = N_total / density_1D_base
Ly_fixed = Lx_fixed

# =============================================================================
# SECTION 2: DIRECTORY SETUP & LOGGER
# =============================================================================
dir_name = (f'results/calibrate/ref_Ntot{N_total}_maxR{max_rings}_dens{density_1D_base}_c{c}'
            f'_seed{seed}/')
if not os.path.exists(dir_name):
    os.makedirs(dir_name)


class Logger(object):
    """
    Custom logger to redirect stdout to both terminal and a text file.
    """

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = Logger(os.path.join(dir_name, "refinement_log.txt"))

print("\n" + "=" * 60)
print("--- STARTING REFINEMENT LIMIT CALIBRATION ---")
print(f"Fixed Budget    : N_total = {N_total} nodes")
print(f"Fixed Universe  : Lx = {Lx_fixed:.2f}, Ly = {Ly_fixed:.2f} (Square)")
print(f"Refinement      : Increasing R from 1 to {max_rings}")
print(f"Tolerance       : {tol:.4f}")
print("=" * 60)


# =============================================================================
# SECTION 3: CALIBRATION FUNCTIONS
# =============================================================================

def theoretical_degree_refined(R: int, N_tot: int, Lx: float, Ly: float, c: float, d0: float) -> float:
    """
    Calculates the theoretical <k> for the refinement case where rho_1D changes with R.

    :param R: int, number of rings
    :param N_tot: int, total nodes in the system
    :param Lx: float, width
    :param Ly: float, height
    :param c: float, connectivity constant
    :param d0: float, decay length
    :return: float, theoretical mean degree
    """
    delta = Ly / R
    # In refinement, linear density inside each ring decreases as we add more rings
    rho_1D_current = (N_tot / R) / Lx

    k_tot = 0.0
    for n in range(R):
        dy = min(n, R - n) * delta
        if dy == 0:
            k_tot += 2 * c * rho_1D_current * d0
        else:
            k_tot += 2 * c * rho_1D_current * dy * k1(dy / d0)
    return k_tot


def theoretical_optimal_d0_refined(target_k: float, R: int, N_tot: int, Lx: float, Ly: float, c: float) -> float:
    """
    Finds the theoretical d0 for the refinement case.

    :param target_k: float, target mean degree
    :param R: int, number of rings
    :param N_tot: int, total nodes
    :param Lx: float, width
    :param Ly: float, height
    :param c: float, connectivity constant
    :return: float, theoretical optimal d0
    """

    def objective(d: float) -> float:
        return theoretical_degree_refined(R, N_tot, Lx, Ly, c, d) - target_k

    return float(brentq(objective, 1e-6, 15.0))


def find_optimal_d0(target_k: float, N_ring: int, Lx: float, num_rings: int, delta: float, c: float,
                    initial_d0: float, seed: int, tol: float = 0.01) -> tuple:
    """
    Bisection algorithm to find d0 in the refined network.

    :param target_k: float, target <k>
    :param N_ring: int, nodes per ring (refined)
    :param Lx: float, width
    :param num_rings: int, number of rings
    :param delta: float, separation
    :param c: float, connectivity base
    :param initial_d0: float, guess
    :param seed: int, seed
    :param tol: float, error margin
    :return: tuple (d0_opt, k_final, G_opt)
    """
    low, high = 1e-6, 10.0
    diff, iteration, mid_d0 = 1e9, 0, initial_d0

    while diff > tol and iteration < 150:
        mid_d0 = (low + high) / 2.0
        np.random.seed(seed)
        _, G, _ = triadic.coupled_rings_structural_network(N_ring, num_rings, Lx, delta, c, mid_d0)
        k_emp = np.mean([d for n, d in G.degree()])
        diff = abs(k_emp - target_k)
        if k_emp < target_k:
            low = mid_d0
        else:
            high = mid_d0
        iteration += 1
        if (high - low) < 1e-10: break
    return mid_d0, k_emp, G


def process_refinement_step(r: int, target_k: float, N_total: int, Lx: float, Ly: float, c: float, d0_base: float,
                            seed: int) -> dict:
    """
    Worker for a single resolution step.

    :param r: int, current number of rings
    :param target_k: float, target <k>
    :param N_total: int, fixed total nodes
    :param Lx: float, fixed width
    :param Ly: float, fixed height
    :param c: float, constant
    :param d0_base: float, starting d0
    :param seed: int, seed
    :return: dict with results
    """
    # Refinement Logic
    current_n_ring = N_total // r
    current_delta = Ly / r

    logs = [f"\n--- Resolution R = {r} ---",
            f"  -> Params  : N_per_ring = {current_n_ring}, Delta = {current_delta:.6f}"]

    # Scaled Calibration
    d0_opt, k_scaled, G_scaled = find_optimal_d0(target_k, current_n_ring, Lx, r, current_delta, c, d0_base, seed, tol)
    logs.append(f"  -> Found d0: {d0_opt:.6f} (k = {k_scaled:.4f})")

    # Distance
    Gcc = sorted(nx.connected_components(G_scaled), key=len, reverse=True)
    dist = triadic.get_topological_distances(G_scaled.subgraph(Gcc[0]), sample_size=500)[1] if Gcc else np.nan
    logs.append(f"  -> Topo. Distance: {dist:.2f}")

    return {
        'num_rings': r, 'delta': current_delta, 'n_per_ring': current_n_ring,
        'd0_opt': d0_opt, 'k_scaled': k_scaled, 'dist_scaled': dist, 'log_messages': "\n".join(logs)
    }


# =============================================================================
# SECTION 4: MAIN CALIBRATION LOOP
# =============================================================================
print("\n[Phase 1] Baseline (R=1, N_per_ring=N_total)...")
np.random.seed(seed)
_, G_base, _ = triadic.coupled_rings_structural_network(N_total, 1, Lx_fixed, Ly_fixed, c, d0_base)
target_k = np.mean([d for n, d in G_base.degree()])
print(f"-> Target <k> established: {target_k:.4f}")

results = []
results.append({'num_rings': 1, 'delta': Ly_fixed, 'n_per_ring': N_total, 'd0_opt': d0_base, 'k_scaled': target_k,
                'dist_scaled': np.nan})

print("\n[Phase 2] Refining Mesh...")
rings_seq = list(range(2, max_rings + 1))

if use_parallel:
    with Parallel(n_jobs=n_jobs, return_as="generator") as parallel:
        gen = parallel(
            delayed(process_refinement_step)(r, target_k, N_total, Lx_fixed, Ly_fixed, c, d0_base, seed) for r in
            rings_seq)
        for res in gen:
            print(res['log_messages'])
            del res['log_messages']
            results.append(res)
    results.sort(key=lambda x: x['num_rings'])
else:
    for r in rings_seq:
        res = process_refinement_step(r, target_k, N_total, Lx_fixed, Ly_fixed, c, d0_base, seed)
        print(res['log_messages'])
        del res['log_messages']
        results.append(res)

# =============================================================================
# SECTION 5: EXPORT & PLOTTING
# =============================================================================
print("\n[Phase 3] Generating Refinement Plots...")
df = pd.DataFrame(results)
df.to_csv(os.path.join(dir_name, "refinement_data.csv"), index=False)
fig_dir = os.path.join(dir_name, "figures");
os.makedirs(fig_dir, exist_ok=True)

rings_int = np.arange(1, df['num_rings'].max() + 1)
d0_theo = [theoretical_optimal_d0_refined(target_k, r, N_total, Lx_fixed, Ly_fixed, c) for r in rings_int]

# Plot 1: d0 Scaling (Refinement)
plt.figure(figsize=(8, 5))
plt.plot(df['num_rings'], df['d0_opt'], 'D', color='teal', label='Exp. Refinement')
plt.plot(rings_int, d0_theo, '-', color='orange', label='Theory (Variable rho_1D)')
plt.title("Refinement Limit: d0 Scaling with Fixed N_total")
plt.xlabel("Resolution (Number of Rings)")
plt.ylabel("Optimal d0")
plt.grid(True, alpha=0.3);
plt.legend();
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "plot_1_d0_refinement.png"), dpi=400);
plt.close()

# Plot 2: Conservation
plt.figure(figsize=(8, 5))
plt.plot(df['num_rings'], df['k_scaled'], 's-', color='forestgreen')
plt.axhline(target_k, color='black', linestyle=':')
plt.title("Degree Conservation during Mesh Refinement")
plt.xlabel("Resolution (R)")
plt.ylabel("<k>")
plt.grid(True, alpha=0.3);
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "plot_2_k_conservation.png"), dpi=400);
plt.close()

print(f"\nRefinement completed. Results in {dir_name}")