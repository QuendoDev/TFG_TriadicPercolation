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
# The script expects 6 mandatory arguments and 2 optional from the console.
# Example command: python calibrate_fixed_size.py 1000 200.0 0.07 0.2 50 42 True 12
# The order is: N_per_ring density_1D c d0_base max_rings seed [use_parallel] [n_jobs]

try:
    N_per_ring = int(sys.argv[1])  # Number of nodes PER RING
    density = float(sys.argv[2])  # Linear Node density (Nodes / Length)
    c = float(sys.argv[3])  # Base probability for structural connections
    d0_base = float(sys.argv[4])  # Target 1D d0 (used as baseline for ring 1)
    max_rings = int(sys.argv[5])  # Maximum number of rings to pack inside the fixed space
    seed = int(sys.argv[6])  # Random seed

    # Optional parameters handling
    use_parallel = sys.argv[7].lower() == 'true' if len(sys.argv) > 7 else False
    n_jobs = int(sys.argv[8]) if len(sys.argv) > 8 else 1
except IndexError:
    print(
        "Error: Missing arguments. Usage: python calibrate_fixed_size.py N_per_ring density c d0_base max_rings "
        "seed [use_parallel] [n_jobs]"
    )
    sys.exit(1)

# Interactive prompt for tolerance to maintain consistency with the other calibration script
user_choice = input("Enter the tolerance for d0 optimization (default 0.02): ")
try:
    tol = float(user_choice)
except ValueError:
    tol = 0.005

# Track the total time
init_time = time.time()

# --- THE CONTINUUM GEOMETRY ---
# Fix the universe dimensions. Lx is fixed by density. Ly is set to equal Lx for a Perfect Square Torus.
Lx_fixed = N_per_ring / density
Ly_fixed = Lx_fixed

# =============================================================================
# SECTION 2: DIRECTORY SETUP & LOGGER
# =============================================================================
# Save everything inside the general 'results/' folder to keep the root directory clean
dir_name = (f'results/continuum_RINGS_maxR{max_rings}_N{N_per_ring}_dens{density}_c{c}_d0base'
            f'{d0_base}_seed{seed}/')
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


sys.stdout = Logger(os.path.join(dir_name, "continuum_log.txt"))

print("\n")
print("=" * 60)
print("--- LOG FILE INITIALIZED ---")
print("=" * 60)
print("--- STARTING FIXED SIZE CALIBRATION (CONTINUUM LIMIT) ---")
print(f"Base Parameters : N_per_ring={N_per_ring}, Linear Density 1D={density}, c={c}, seed={seed}")
print(f"Fixed Universe  : Lx = {Lx_fixed:.2f}, Ly = {Ly_fixed:.2f} (Perfect Square Torus)")
print(f"Densification   : Packing up to {max_rings} rings inside fixed Ly")
print(f"Tolerance       : {tol:.4f}")
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

    diff = 1e9
    iteration = 0
    mid_d0 = initial_d0

    # Adaptive loop: it runs until the error is within tolerance or d0 delta is negligible
    while diff > tol and iteration < 200:
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
        if (high - low) < 1e-9:
            break

    return mid_d0, k_emp, G


def process_continuum_ring(r: int, target_k: float, N_per_ring: int, Lx: float, Ly: float, c: float, d0_base: float,
                           seed: int) -> dict:
    """
    Worker function to process a single ring configuration in the continuum limit.

    :param r: int, the ring configuration
    :param target_k: float, the target empirical mean degree to achieve
    :param N_per_ring: int, number of nodes per ring
    :param Lx: float, fixed X length
    :param Ly: float, fixed Y length
    :param c: float, base structural connectivity parameter
    :param d0_base: float, base d0 for scaling
    :param seed: int, random seed
    :return: dict with results for this configuration
    """
    # The core of the continuum limit: Delta shrinks as R increases while Ly remains fixed
    delta_r = Ly / r

    logs = [f"\n--- Analyzing Continuum System with {r} rings ---",
            f"  -> Geometry : Lx = {Lx:.4f}, Ly = {Ly:.4f} (FIXED), Delta = {delta_r:.6f}"]

    # 1. Unscaled Test
    np.random.seed(seed)
    _, G_unscaled, _ = triadic.coupled_rings_structural_network(N_per_ring, r, Lx, delta_r, c, d0_base)
    k_unscaled = np.mean([d for n, d in G_unscaled.degree()])
    logs.append(f"  -> Unscaled <k> (Explosion) : {k_unscaled:.4f} (Using constant d0 = {d0_base:.6f})")

    # 2. Scaled Calibration
    logs.append(f"  -> Searching for optimal shrinking d0 to maintain <k> = {target_k:.4f}...")
    d0_opt, k_scaled, G_scaled = find_optimal_d0(target_k, N_per_ring, Lx, r, delta_r, c, d0_base, seed, tol)
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
        'delta': delta_r,
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

# For 1 ring in a box of Ly_fixed, the separation to itself is exactly Ly_fixed
np.random.seed(seed)
_, G_base, _ = triadic.coupled_rings_structural_network(N_per_ring, 1, Lx_fixed, Ly_fixed, c, d0_base)
target_k = np.mean([d for n, d in G_base.degree()])

# Get baseline topological distance
Gcc_base = sorted(nx.connected_components(G_base), key=len, reverse=True)
G0_base = G_base.subgraph(Gcc_base[0])
_, dist_base = triadic.get_topological_distances(G0_base, sample_size=500)

print(f"-> Target <k>    : {target_k:.4f} connections/node")
print(f"-> Base Distance : {dist_base:.2f} hops")

# Data collection lists
results = []
results.append({
    'num_rings': 1,
    'delta': Ly_fixed,
    'd0_opt': d0_base,
    'k_scaled': target_k,
    'k_unscaled': target_k,
    'dist_scaled': dist_base
})

print("\n[Phase 2] Executing Continuum Packing Sequence...")
rings_sequence = list(range(2, max_rings + 1))

if use_parallel:
    print(f"-> Dispatching {len(rings_sequence)} tasks to Joblib (n_jobs={n_jobs})...")
    print("   (Logs will appear in real-time as workers finish their rings)\n")

    with Parallel(n_jobs=n_jobs, return_as="generator") as parallel:
        results_generator = parallel(
            delayed(process_continuum_ring)(r, target_k, N_per_ring, Lx_fixed, Ly_fixed, c, d0_base, seed)
            for r in rings_sequence
        )
        for res in results_generator:
            print(res['log_messages'])
            del res['log_messages']
            results.append(res)

    # Sort the final list to ensure ordered plotting
    results.sort(key=lambda x: x['num_rings'])
else:
    # Sequential execution
    for r in rings_sequence:
        res = process_continuum_ring(r, target_k, N_per_ring, Lx_fixed, Ly_fixed, c, d0_base, seed)
        print(res['log_messages'])
        del res['log_messages']
        results.append(res)

# =============================================================================
# SECTION 5: DATA EXPORT & PLOTTING
# =============================================================================
print("\n[Phase 3] Saving results and generating plots...")
df = pd.DataFrame(results)
df.to_csv(os.path.join(dir_name, "continuum_data.csv"), index=False)

fig_dir = os.path.join(dir_name, "figures")
os.makedirs(fig_dir, exist_ok=True)

print("-> Calculating theoretical overlay curves (Continuum Limit)...")
rings_int = np.arange(1, df['num_rings'].max() + 1)

# In the continuum limit, delta is continuously shrinking as (Ly_fixed / R)
k_theo_explosion = [theoretical_degree(r, c, density, d0_base, Ly_fixed / r) for r in rings_int]
d0_theo_curve = [theoretical_optimal_d0(target_k, r, c, density, Ly_fixed / r) for r in rings_int]

# 1. Connectivity Evolution Plot (Conservation vs Explosion)
plt.figure(figsize=(8, 5))
plt.plot(df['num_rings'], df['k_unscaled'], marker='o', linestyle='', color='crimson', alpha=0.6,
         label='Unscaled (Explosion) - Exp.')
plt.plot(df['num_rings'], df['k_scaled'], marker='s', linestyle='', color='forestgreen',
         label='Calibrated - Exp.')
plt.plot(rings_int, k_theo_explosion, color='black', linestyle='--', linewidth=1.5, zorder=1,
         label='Theoretical Math (Bessel)')
plt.axhline(target_k, color='black', linestyle=':', alpha=0.5, label=f'Target <k> ({target_k:.2f})')
plt.title("Degree Conservation in Continuum Limit: Theory vs Experiment")
plt.xlabel("Number of Rings Packed (Inverse of Delta)")
plt.ylabel("Average Structural Degree <k>")
plt.grid(True, linestyle=':', alpha=0.6)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "plot_1_connectivity.png"), dpi=400)
plt.close()

# 2. Scaling Law Plot (d0 Shrinking vs Rings)
plt.figure(figsize=(8, 5))
plt.plot(df['num_rings'], df['d0_opt'], marker='D', linestyle='', color='teal', markersize=6,
         label='Optimal d0 (Bisection) - Exp.')
plt.plot(rings_int, d0_theo_curve, color='orange', linestyle='-', linewidth=2.5, zorder=1,
         label='Theoretical Optimal (Root Finding)')
plt.title("Continuum Limit: Optimal Decay Length vs Density")
plt.xlabel("Number of Rings Packed (Inverse of Delta)")
plt.ylabel("Optimal Decay Length (d0)")
plt.grid(True, linestyle=':', alpha=0.6)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "plot_2_continuum_d0.png"), dpi=400)
plt.close()

# 3. Topological Distance Plot
plt.figure(figsize=(8, 5))
plt.plot(df['num_rings'], df['dist_scaled'], marker='D', color='darkcyan', linewidth=2)
plt.title("Topological Path Length in Continuum Limit")
plt.xlabel("Number of Rings Packed (Inverse of Delta)")
plt.ylabel("Mean Shortest Path (Hops)")
plt.grid(True, linestyle=':', alpha=0.6)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "plot_3_distances.png"), dpi=400)
plt.close()

total_time = time.time() - init_time
print(f"-> All plots successfully saved in: {fig_dir}")
print("\n" + "=" * 60)
print(f"   CONTINUUM CALIBRATION COMPLETED SUCCESSFULLY")
print(f"   Total Time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
print("=" * 60 + "\n")