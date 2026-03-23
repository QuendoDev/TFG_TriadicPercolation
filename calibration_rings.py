import os
import sys
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

import triadic_library as triadic

# =============================================================================
# SECTION 1: PARAMETER READING & INITIALIZATION
# =============================================================================
# The script expects exactly 8 arguments from the console.
# Example command: python calibration_rings.py 10000 1000.0 0.07 0.2 5 0.1 42
# The order is: N density_1D c d0_base max_rings delta_factor seed

try:
    N = int(sys.argv[1])  # Number of nodes
    density = float(sys.argv[2])  # Linear Node density (Nodes / Length)
    c = float(sys.argv[3]) if len(sys.argv) > 3 else float(sys.argv[3])  # Base probability for structural connections
    d0_base = float(sys.argv[4])  # Target 1D d0 (used as baseline for ring 1)
    max_rings = int(sys.argv[5])  # Maximum number of rings to scale up to
    delta_factor = float(sys.argv[6])  # Separation factor (delta = delta_factor * d0_base)
    seed = int(sys.argv[7])  # Random seed
except IndexError:
    print(
        "Error: Missing arguments. Usage: python calibration_rings.py N density c d0_base max_rings delta_factor seed")
    sys.exit(1)

# Track the total time
init_time = time.time()

# Fixed physical distance between rings based on the base 1D connectivity scale
delta = delta_factor * d0_base

# =============================================================================
# SECTION 2: DIRECTORY SETUP & LOGGER
# =============================================================================
dir_name = f'calibration_RINGS_max{max_rings}_N{N}_dens{density}_c{c}_d0base{d0_base}_dfact{delta_factor}_seed{seed}/'

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

print("=" * 60)
print("--- LOG FILE INITIALIZED ---")
print("=" * 60)
print("--- STARTING DIMENSIONAL CALIBRATION ---")
print(f"Base Parameters : N={N}, Linear Density 1D={density}, c={c}, seed={seed}")
print(f"Scaling Target  : Max Rings = {max_rings}")
print(f"Ring Separation : delta = {delta:.4f} (Factor: {delta_factor} * d0_base)")
print("=" * 60)


# =============================================================================
# SECTION 3: CALIBRATION FUNCTIONS
# =============================================================================

def find_optimal_d0(target_k: float, N: int, Lx: float, num_rings: int, delta: float, c: float,
                    initial_d0: float, seed: int, tol: float = 0.05, max_iter: int = 40) -> tuple:
    """
    Binary search algorithm to find the optimal d0 that yields a specific empirical <k>.

    :param target_k: float, the target empirical mean degree to achieve
    :param N: int, number of nodes
    :param Lx: float, continuous length of the rings
    :param num_rings: int, number of discrete rings
    :param delta: float, separation distance between rings
    :param c: float, base structural connectivity parameter
    :param initial_d0: float, starting guess for d0
    :param seed: int, random seed (strictly enforced for monotonic root finding)
    :param tol: float, acceptable error margin for <k>
    :param max_iter: int, maximum number of bisection iterations

    :return: tuple (optimal_d0, final_empirical_k, optimized_Graph)
    """
    low = 0.0001
    high = initial_d0 * 2.0

    # Quick expansion if high is not high enough
    np.random.seed(seed)
    _, G_test, _ = triadic.random_coupled_rings_netw_PBC(N, Lx, num_rings, delta, c, high)
    while np.mean([d for n, d in G_test.degree()]) < target_k:
        high *= 1.5
        np.random.seed(seed)
        _, G_test, _ = triadic.random_coupled_rings_netw_PBC(N, Lx, num_rings, delta, c, high)

    mid_d0 = initial_d0
    best_G = None
    best_k = 0.0

    # Binary search loop
    for _ in range(max_iter):
        mid_d0 = (low + high) / 2.0

        # Reset seed to guarantee deterministic noise pattern across evaluations
        np.random.seed(seed)
        _, G, _ = triadic.random_coupled_rings_netw_PBC(N, Lx, num_rings, delta, c, mid_d0)

        k_emp = np.mean([d for n, d in G.degree()])
        best_G = G
        best_k = k_emp

        if abs(k_emp - target_k) <= tol:
            break
        elif k_emp < target_k:
            low = mid_d0
        else:
            high = mid_d0

    return mid_d0, best_k, best_G


# =============================================================================
# SECTION 4: MAIN CALIBRATION LOOP
# =============================================================================
print("\n[Phase 1] Establishing 1D Baseline (1 Ring)...")

# Fix Lx purely based on 1D linear density to maintain universe length constant
Lx_1 = N / density
Ly_1 = 1 * delta

np.random.seed(seed)
_, G_base, _ = triadic.random_coupled_rings_netw_PBC(N, Lx_1, 1, delta, c, d0_base)
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

print("\n[Phase 2] Executing Scaling and Calibration Loop...")

for r in range(2, max_rings + 1):
    print(f"\n--- Analyzing system with {r} rings ---")
    Lx_r = N / density
    Ly_r = r * delta

    # 1. Unscaled Test (What happens if we don't scale d0?)
    np.random.seed(seed)
    _, G_unscaled, _ = triadic.random_coupled_rings_netw_PBC(N, Lx_r, r, delta, c, d0_base)
    k_unscaled = np.mean([d for n, d in G_unscaled.degree()])
    print(f"  -> Unscaled <k> (Explosion) : {k_unscaled:.2f} (Using d0 = {d0_base:.4f})")

    # 2. Scaled Calibration (Finding optimal d0)
    print(f"  -> Searching for optimal d0 to maintain <k> = {target_k:.2f}...")
    d0_opt, k_scaled, G_scaled = find_optimal_d0(target_k, N, Lx_r, r, delta, c, d0_base, seed)
    print(f"  -> Found optimal d0         : {d0_opt:.4f} (Resulting <k> = {k_scaled:.2f})")

    # 3. Topological Distance of the Calibrated Network
    Gcc_scaled = sorted(nx.connected_components(G_scaled), key=len, reverse=True)
    if len(Gcc_scaled) > 0:
        G0_scaled = G_scaled.subgraph(Gcc_scaled[0])
        _, dist_scaled = triadic.get_topological_distances(G0_scaled, sample_size=500)
    else:
        dist_scaled = np.nan

    print(f"  -> Scaled Topo. Distance    : {dist_scaled:.2f} hops")

    # Save results
    results.append({
        'num_rings': r,
        'd0_opt': d0_opt,
        'k_scaled': k_scaled,
        'k_unscaled': k_unscaled,
        'dist_scaled': dist_scaled
    })

# =============================================================================
# SECTION 5: DATA EXPORT & PLOTTING
# =============================================================================
print("\n[Phase 3] Saving results and generating plots...")

df = pd.DataFrame(results)
df.to_csv(os.path.join(dir_name, "calibration_data.csv"), index=False)

fig_dir = os.path.join(dir_name, "figures")
os.makedirs(fig_dir, exist_ok=True)

# Plot 1: The Connectivity Explosion (k vs Rings)
plt.figure(figsize=(8, 5))
plt.plot(df['num_rings'], df['k_unscaled'], marker='o', color='red', linestyle='-', linewidth=2,
         label=r'Unscaled (Fixed $d_0$)')
plt.plot(df['num_rings'], df['k_scaled'], marker='s', color='green', linestyle='--', linewidth=2,
         label=r'Scaled (Calibrated $d_0$)')
plt.axhline(target_k, color='gray', linestyle=':', label=r'Target $\langle k \rangle$')
plt.title("The Connectivity Explosion: Unscaled vs Scaled Density")
plt.xlabel("Number of Rings (Dimensional Expansion)")
plt.ylabel(r"Average Structural Degree $\langle k \rangle$")
plt.xticks(df['num_rings'])
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "plot_1_connectivity_explosion.png"), dpi=400)
plt.close()

# Plot 2: The Scaling Law (Optimal d0 vs Rings)
plt.figure(figsize=(8, 5))
plt.plot(df['num_rings'], df['d0_opt'], marker='o', color='purple', linestyle='-', linewidth=2.5)
plt.title(r"Scaling Law: Optimal $d_0$ required to conserve $\langle k \rangle$")
plt.xlabel("Number of Rings (Dimensional Expansion)")
plt.ylabel(r"Optimal Decay Length ($d_0$)")
plt.xticks(df['num_rings'])
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "plot_2_d0_scaling_law.png"), dpi=400)
plt.close()

# Plot 3: The Dimensional Shortcut (Topological Distance vs Rings)
plt.figure(figsize=(8, 5))
plt.plot(df['num_rings'], df['dist_scaled'], marker='D', color='teal', linestyle='-', linewidth=2.5,
         label='Giant Component')
plt.title("The Dimensional Shortcut: Topological Distance vs Space")
plt.xlabel("Number of Rings (Dimensional Expansion)")
plt.ylabel("Average Shortest Path (Hops)")
plt.xticks(df['num_rings'])
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "plot_3_topological_shortcut.png"), dpi=400)
plt.close()

total_time = time.time() - init_time
print(f"-> All plots successfully saved in: {fig_dir}")
print("\n" + "=" * 60)
print(f"   CALIBRATION COMPLETED SUCCESSFULLY")
print(f"   Total Time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
print("=" * 60 + "\n")