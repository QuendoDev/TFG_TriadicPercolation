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
# The script expects 7 mandatory arguments and 3 optional from the console.
# Example command: python calibrate_fixed_delta.py 1000 200.0 0.07 0.2 15 0.2 42 125 True 12
# The order is: N_per_ring density_1D c d0_base loop_max_rings delta_factor seed [target_rings] [use_parallel] [n_jobs]

try:
    N_per_ring = int(sys.argv[1])  # Number of nodes PER RING
    density = float(sys.argv[2])  # Linear Node density (Nodes / Length)
    c = float(sys.argv[3]) if len(sys.argv) > 3 else float(sys.argv[3])  # Base probability for structural connections
    d0_base = float(sys.argv[4])  # Target 1D d0 (used as baseline for ring 1)
    loop_max_rings = int(sys.argv[5])  # Maximum number of rings for the continuous loop
    delta_factor = float(sys.argv[6])  # Separation factor (delta = delta_factor * d0_base)
    seed = int(sys.argv[7])  # Random seed

    # Optional parameters handling
    target_rings = int(sys.argv[8]) if len(sys.argv) > 8 and sys.argv[8] != "0" else None
    use_parallel = sys.argv[9].lower() == 'true' if len(sys.argv) > 9 else False
    n_jobs = int(sys.argv[10]) if len(sys.argv) > 10 else 1
except IndexError:
    print(
        "Error: Missing arguments. Usage: python calibrate_fixed_delta.py N_per_ring density c d0_base loop_max_rings "
        "delta_factor seed [target_rings] [use_parallel] [n_jobs]\n"
        "(Pass '0' for target_rings if you want to skip it but use parallel flags)"
    )
    sys.exit(1)

# Interactive prompt to decide some parameters
user_choice = input("Do you want to include the massive target (e.g. 125) in the plots? (y/n): ").lower()
plot_massive = True if user_choice == 'y' else False
user_choice = input("Enter the tolerance for d0 optimization (default 0.02): ")
try:    tol = float(user_choice)
except ValueError:    tol = 0.02

# Track the total time
init_time = time.time()

# Fixed physical distance between rings based on the base 1D connectivity scale
delta = delta_factor * d0_base

# =============================================================================
# SECTION 2: DIRECTORY SETUP & LOGGER
# =============================================================================
dir_name = (f'results/calibrate/fdelta_loop{loop_max_rings}_target{target_rings}_N{N_per_ring}_dens{density}_c{c}'
            f'_d0base{d0_base}_dfact{delta_factor}'
            f'_seed{seed}/')
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

class Logger(object):
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


sys.stdout = Logger(os.path.join(dir_name, "calibration_log.txt"))

print("\n")
print("=" * 60)
print("--- LOG FILE INITIALIZED ---")
print("=" * 60)
print("--- STARTING DIMENSIONAL CALIBRATION ---")
print(f"Base Parameters : N_per_ring={N_per_ring}, Linear Density 1D={density}, c={c}, seed={seed}")
print(f"Scaling Target  : Continuous loop up to {loop_max_rings} rings")
if target_rings and target_rings > loop_max_rings:
    print(f"Isolated Target : Massive final calculation at {target_rings} rings")
print(f"Ring Separation : delta = {delta:.4f} (Factor: {delta_factor} * d0_base)")
print(f"Tolerance for d0 optimization: {tol:.4f}")
print(f"Execution Mode  : {'PARALLEL' if use_parallel else 'SEQUENTIAL'} (Cores: {n_jobs})")
print("=" * 60)


# =============================================================================
# SECTION 3: CALIBRATION FUNCTIONS
# =============================================================================
def theoretical_degree(R: int, c: float, rho_1D: float, d0: float, delta: float) -> float:
    """
    Calculates the exact theoretical structural degree <k> in a discrete Torus
    using modified Bessel functions.

    :param R: int, number of rings
    :param c: float, structural connectivity constant
    :param rho_1D: float, 1D linear density
    :param d0: float, characteristic decay length
    :param delta: float, separation between rings
    :return: float, exact mathematical mean degree
    """
    k_tot = 0.0
    for n in range(R):
        dy = min(n, R - n) * delta
        if dy == 0:
            k_tot += 2 * c * rho_1D * d0
        else:
            k_tot += 2 * c * rho_1D * dy * k1(dy / d0)
    return k_tot


def theoretical_optimal_d0(target_k: float, R: int, c: float, rho_1D: float, delta: float) -> float:
    """
    Finds the exact mathematical d0 required to maintain a target degree <k>
    by finding the root of the theoretical degree equation.

    :param target_k: float, target mean degree
    :param R: int, number of rings
    :param c: float, structural connectivity constant
    :param rho_1D: float, 1D linear density
    :param delta: float, separation between rings
    :return: float, theoretical optimal d0
    """
    def objective(d: float) -> float:
        return theoretical_degree(R, c, rho_1D, d, delta) - target_k

    # Use scipy's brentq to find the root quickly between a very small and large d0
    return float(brentq(objective, 1e-6, 2.0))


def find_optimal_d0(target_k: float, N_per_ring: int, Lx: float, num_rings: int, delta: float, c: float,
                    initial_d0: float, seed: int, tol: float = 0.02) -> tuple:
    """
    Find the optimal d0 using an adaptive bisection method that converges based on tolerance.

    :param target_k: float, target mean degree
    :param N_per_ring: int, nodes per ring
    :param Lx: float, X-axis length
    :param num_rings: int, number of rings
    :param delta: float, Y-axis separation
    :param c: float, structural connectivity base
    :param initial_d0: float, first guess
    :param seed: int, random seed
    :param tol: float, precision threshold for <k>
    :return: tuple (optimal_d0, final_k, G_optimized)
    """
    low = 1e-6
    high = initial_d0 * 2.0

    # Dynamic expansion of the upper bound to ensure the root is bracketed
    np.random.seed(seed)
    _, G_test, _ = triadic.coupled_rings_structural_network(N_per_ring, num_rings, Lx, delta, c, high)
    while np.mean([d for n, d in G_test.degree()]) < target_k:
        high *= 2.0
        np.random.seed(seed)
        _, G_test, _ = triadic.coupled_rings_structural_network(N_per_ring, num_rings, Lx, delta, c, high)

    # Adaptive loop: it runs until the error is within tolerance or d0 delta is negligible
    diff = 1e9
    iteration = 0
    mid_d0 = initial_d0

    while diff > tol and iteration < 200:  # Safety cap at 200 to avoid infinite loops
        mid_d0 = (low + high) / 2.0
        np.random.seed(seed)
        _, G, _ = triadic.coupled_rings_structural_network(N_per_ring, num_rings, Lx, delta, c, mid_d0)

        k_emp = np.mean([d for n, d in G.degree()])
        diff = abs(k_emp - target_k)

        if k_emp < target_k:
            low = mid_d0
        else:
            high = mid_d0

        iteration += 1
        # If the search space is too small to continue, break
        if (high - low) < 1e-9:
            break

    return mid_d0, k_emp, G


def process_ring(r: int, target_k: float, N_per_ring: int, density: float, delta: float, c: float, d0_base: float,
                 seed: int) -> dict:
    """
    Worker function to process a single ring configuration. Isolated for parallel execution.

    :param r: int, the ring configuration
    :param target_k: float, the target empirical mean degree to achieve
    :param N_per_ring: int, number of nodes per ring
    :param density: float, density parameter
    :param delta: float, separation distance between rings
    :param c: float, base structural connectivity parameter
    :param d0_base: float, base d0 for scaling
    :param seed: int, random seed for reproducibility
    :return: dict with results for this ring configuration
    """
    Lx_r = N_per_ring / density
    Ly_r = r * delta

    logs = [f"\n--- Analyzing system with {r} rings ---",
            f"  -> Geometry : Lx = {Lx_r:.4f}, Ly = {Ly_r:.4f}"]

    # 1. Unscaled Test
    np.random.seed(seed)
    _, G_unscaled, _ = triadic.coupled_rings_structural_network(N_per_ring, r, Lx_r, delta, c, d0_base)
    k_unscaled = np.mean([d for n, d in G_unscaled.degree()])
    logs.append(f"  -> Unscaled <k> (Explosion) : {k_unscaled:.4f} (Using d0 = {d0_base:.6f})")

    # 2. Scaled Calibration
    logs.append(f"  -> Searching for optimal d0 to maintain <k> = {target_k:.4f}...")
    d0_opt, k_scaled, G_scaled = find_optimal_d0(target_k, N_per_ring, Lx_r, r, delta, c, d0_base, seed, tol)
    logs.append(f"  -> Found optimal d0         : {d0_opt:.6f} (Resulting <k> = {k_scaled:.4f})")

    # 3. Topological Distance
    Gcc_scaled = sorted(nx.connected_components(G_scaled), key=len, reverse=True)
    if len(Gcc_scaled) > 0:
        G0_scaled = G_scaled.subgraph(Gcc_scaled[0])
        _, dist_scaled = triadic.get_topological_distances(G0_scaled, sample_size=500)
    else:
        dist_scaled = np.nan

    logs.append(f"  -> Scaled Topo. Distance    : {dist_scaled:.2f} hops")

    # Only return the scalars to prevent memory overload in inter-process communication
    return {
        'num_rings': r,
        'd0_opt': d0_opt,
        'k_scaled': k_scaled,
        'k_unscaled': k_unscaled,
        'dist_scaled': dist_scaled,
        'log_messages': "\n".join(logs)
    }


# =============================================================================
# SECTION 4: MAIN CALIBRATION LOOP
# =============================================================================
print("\n[Phase 1] Establishing 1D Baseline (1 Ring)...")

# Fix Lx purely based on 1D linear density to maintain universe length constant
Lx_1 = N_per_ring / density
Ly_1 = 1 * delta

np.random.seed(seed)
_, G_base, _ = triadic.coupled_rings_structural_network(N_per_ring, 1, Lx_1, 0.0, c, d0_base)
target_k = np.mean([d for n, d in G_base.degree()])

# Get baseline topological distance
Gcc_base = sorted(nx.connected_components(G_base), key=len, reverse=True)
G0_base = G_base.subgraph(Gcc_base[0])
_, dist_base = triadic.get_topological_distances(G0_base, sample_size=500)

print(f"-> Base Geometry : Lx = {Lx_1:.2f}, Ly = {Ly_1:.4f}")
print(f"-> Target <k>    : {target_k:.4f} connections/node")
print(f"-> Base Distance : {dist_base:.2f} hops")

# Data collection lists
results = []
results.append({
    'num_rings': 1,
    'd0_opt': d0_base,
    'k_scaled': target_k,
    'k_unscaled': target_k,
    'dist_scaled': dist_base
})

print("\n[Phase 2] Executing Scaling and Calibration Sequence...")

# Generate the sequence of rings to calculate: The continuous loop + the final isolated target
rings_sequence = list(range(2, loop_max_rings + 1))
if target_rings is not None and target_rings > loop_max_rings:
    rings_sequence.append(target_rings)

if use_parallel:
    print(f"-> Dispatching {len(rings_sequence)} tasks to Joblib (n_jobs={n_jobs})...")
    print("   (Logs will appear in real-time as workers finish their rings)\n")

    # Use a generator to yield results as soon as they are completed
    # 'return_as="generator"' is the key for real-time output
    with Parallel(n_jobs=n_jobs, return_as="generator") as parallel:
        results_generator = parallel(
            delayed(process_ring)(r, target_k, N_per_ring, density, delta, c, d0_base, seed)
            for r in rings_sequence
        )

        # We iterate over the generator: it blocks until the NEXT task is finished
        for res in results_generator:
            print(res['log_messages'])  # Prints instantly as the worker finishes
            del res['log_messages']  # Clean memory
            results.append(res)

    # Critical: Since workers finish at different times, we must sort the final list
    # so the plots (CSV) follow the order 1, 2, 3, 4...
    results.sort(key=lambda x: x['num_rings'])
else:
    # Sequential execution
    for r in rings_sequence:
        res = process_ring(r, target_k, N_per_ring, density, delta, c, d0_base, seed)
        print(res['log_messages'])
        del res['log_messages']
        results.append(res)

# =============================================================================
# SECTION 5: DATA EXPORT & PLOTTING
# =============================================================================
print("\n[Phase 3] Saving results and generating plots...")

df = pd.DataFrame(results)
df.to_csv(os.path.join(dir_name, "calibration_data.csv"), index=False)

fig_dir = os.path.join(dir_name, "figures")
os.makedirs(fig_dir, exist_ok=True)

# Filter data for cleaner visualization if the user chose 'n'
df_plot = df if plot_massive else df[df['num_rings'] <= loop_max_rings]

print("-> Calculating theoretical overlay curves...")
rings_int = np.arange(1, df_plot['num_rings'].max() + 1)
k_theo_explosion = [theoretical_degree(r, c, density, d0_base, delta) for r in rings_int]
d0_theo_curve = [theoretical_optimal_d0(target_k, r, c, density, delta) for r in rings_int]

# 1. Connectivity Evolution Plot
plt.figure(figsize=(8, 5))
plt.plot(df_plot['num_rings'], df_plot['k_unscaled'], marker='o', linestyle='', color='crimson', alpha=0.6,
         label='Unscaled (Explosion) - Exp.')
plt.plot(df_plot['num_rings'], df_plot['k_scaled'], marker='s', linestyle='', color='forestgreen',
         label='Calibrated - Exp.')
plt.plot(rings_int, k_theo_explosion, color='black', linestyle='--', linewidth=1.5, zorder=1,
         label='Theoretical Math (Bessel)')
plt.axhline(target_k, color='black', linestyle=':', alpha=0.5, label=f'Target <k> ({target_k:.2f})')
plt.title("Degree Conservation across Dimensions: Theory vs Experiment")
plt.xlabel("Number of Rings (System Size)")
plt.ylabel("Average Structural Degree <k>")
plt.grid(True, linestyle=':', alpha=0.6)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "plot_1_connectivity.png"), dpi=400)
plt.close()

# 2. Scaling Law Plot (d0 vs Rings)
plt.figure(figsize=(8, 5))
plt.plot(df_plot['num_rings'], df_plot['d0_opt'], marker='D', linestyle='', color='indigo', markersize=6,
         label='Optimal d0 (Bisection) - Exp.')
plt.plot(rings_int, d0_theo_curve, color='orange', linestyle='-', linewidth=2.5, zorder=1,
         label='Theoretical Optimal (Root Finding)')
plt.title("Structural Decay Scaling Law: Theory vs Experiment")
plt.xlabel("Number of Rings (System Size)")
plt.ylabel("Optimal Decay Length (d0)")
plt.grid(True, linestyle=':', alpha=0.6)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "plot_2_d0_scaling.png"), dpi=400)
plt.close()

# 3. Topological Distance Plot
plt.figure(figsize=(8, 5))
plt.plot(df_plot['num_rings'], df_plot['dist_scaled'], marker='D', color='darkcyan', linewidth=2)
plt.title("Topological Path Length Expansion")
plt.xlabel("Number of Rings (System Size)")
plt.ylabel("Mean Shortest Path (Hops)")
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "plot_3_distances.png"), dpi=400)
plt.close()

total_time = time.time() - init_time
print(f"-> All plots successfully saved in: {fig_dir}")
print("\n" + "=" * 60)
print(f"   CALIBRATION COMPLETED SUCCESSFULLY")
print(f"   Total Time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
print("=" * 60 + "\n")