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
# Example command: python calibrate_refinement.py 10000 200.0 0.07 0.2 60 42 True 12

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
user_choice = input("Enter the tolerance for d0 optimization (default 0.0025): ")
try:
    tol = float(user_choice)
except ValueError:
    tol = 0.0025

init_time = time.time()

# --- THE REFINEMENT GEOMETRY ---
# We fix Lx using the initial total nodes and base density.
# We fix Ly to match Lx for a perfect 2D square comparison.
Lx_fixed = N_total / density_1D_base
Ly_fixed = Lx_fixed

# =============================================================================
# SECTION 2: DIRECTORY SETUP & LOGGER
# =============================================================================
dir_name = (f'results/calibrate/ref/Ntot{N_total}_maxR{max_rings}_dens{density_1D_base}_c{c}'
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


def find_optimal_d0(target_k: float, N_total: int, Lx: float, Ly: float, num_rings: int, c: float,
                    initial_d0: float, seed: int, tol: float = 0.01) -> tuple:
    """
    Bisection algorithm to find d0 in the refined network using fixed N_total.

    :param target_k: float, target <k>
    :param N_total: int, total nodes (fixed)
    :param Lx: float, width
    :param Ly: float, height
    :param num_rings: int, number of rings
    :param c: float, connectivity base
    :param initial_d0: float, guess
    :param seed: int, seed
    :param tol: float, error margin
    :return: tuple (d0_opt, k_final, G_opt, nodes)
    """
    low, high = 1e-6, 10.0
    diff, iteration, mid_d0 = 1e9, 0, initial_d0

    while diff > tol and iteration < 150:
        mid_d0 = (low + high) / 2.0
        np.random.seed(seed)
        nodes, G, _ = triadic.coupled_rings_structural_network_fixed_N(N_total, num_rings, Lx, Ly, c, mid_d0)
        k_emp = np.mean([d for n, d in G.degree()])
        diff = abs(k_emp - target_k)
        if k_emp < target_k:
            low = mid_d0
        else:
            high = mid_d0
        iteration += 1
        if (high - low) < 1e-10: break
    return mid_d0, k_emp, G, nodes


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
    current_delta = Ly / r

    logs = [f"\n--- Resolution R = {r} ---",
            f"  -> Params  : N_total = {N_total}, Delta = {current_delta:.6f}"]

    # Scaled Calibration
    d0_opt, k_scaled, G_scaled, nodes = find_optimal_d0(target_k, N_total, Lx, Ly, r, c, d0_base, seed, tol)
    logs.append(f"  -> Nodes generated: {G_scaled.number_of_nodes()} (Target: {N_total})")
    logs.append(f"  -> Found d0: {d0_opt:.6f} (k = {k_scaled:.4f})")

    # Distance
    Gcc = sorted(nx.connected_components(G_scaled), key=len, reverse=True)
    dist = triadic.get_topological_distances(G_scaled.subgraph(Gcc[0]), sample_size=1000)[1] if Gcc else np.nan
    logs.append(f"  -> Topo. Distance: {dist:.2f}")

    # Geometry & Isotropy
    frac_horiz, angles = compute_geometric_metrics(G_scaled, nodes, Lx, Ly)
    logs.append(f"  -> 2D Isotropy : {frac_horiz:.2f}% Horizontal Links")

    return {
        'num_rings': r, 'delta': current_delta, 'n_total': N_total,
        'd0_opt': d0_opt, 'k_scaled': k_scaled, 'dist_scaled': dist,
        'frac_horiz': frac_horiz, 'angles': angles,
        'log_messages': "\n".join(logs)
    }


def compute_geometric_metrics(G: nx.Graph, nodes: np.ndarray, Lx: float, Ly: float) -> tuple:
    """
    Computes the fraction of horizontal links and the angle distribution of edges.
    Takes Periodic Boundary Conditions into account to measure true topological angles.

    :param G: networkx graph
    :param nodes: np.ndarray, nodes coordinates
    :param Lx: float, width
    :param Ly: float, height
    :return: tuple (frac_horiz, angles_array)
    """
    edges = np.array(G.edges())
    if len(edges) == 0:
        return 100.0, np.array([])

    I, J = edges[:, 0], edges[:, 1]
    X_i, Y_i = nodes[I, 0], nodes[I, 1]
    X_j, Y_j = nodes[J, 0], nodes[J, 1]

    # Shortest distance under Periodic Boundary Conditions
    dx = np.abs(X_i - X_j)
    dx = np.minimum(dx, Lx - dx)

    dy = np.abs(Y_i - Y_j)
    dy = np.minimum(dy, Ly - dy)

    # Is the link more horizontal than vertical?
    horiz_mask = dx > dy
    frac_horiz = np.mean(horiz_mask) * 100.0

    # Angle in degrees (0 to 90). dx and dy are strictly positive.
    # 0 = Purely Horizontal, 90 = Purely Vertical
    angles = np.degrees(np.arctan2(dy, dx))

    return frac_horiz, angles


# =============================================================================
# SECTION 4: MAIN CALIBRATION LOOP
# =============================================================================
print("\n[Phase 1] Baseline (R=1, All nodes in 1 ring)...")
np.random.seed(seed)
nodes_base, G_base, _ = triadic.coupled_rings_structural_network_fixed_N(N_total, 1, Lx_fixed, Ly_fixed,
                                                                         c, d0_base)
print(f"-> Nodes generated in graph: {G_base.number_of_nodes()} (Target: {N_total})")
target_k = np.mean([d for n, d in G_base.degree()])
print(f"-> Target <k> established: {target_k:.4f}")

# Compute geometry for baseline
frac_horiz_base, angles_base = compute_geometric_metrics(G_base, nodes_base, Lx_fixed, Ly_fixed)
print(f"-> Baseline 2D Isotropy: {frac_horiz_base:.2f}% Horizontal Links")

results = []
results.append({'num_rings': 1, 'delta': Ly_fixed, 'n_total': N_total, 'd0_opt': d0_base, 'k_scaled': target_k,
                'dist_scaled': np.nan, 'frac_horiz': frac_horiz_base, 'angles': angles_base})

print("\n[Phase 2] Refining Mesh...")
rings_seq = list(range(2, max_rings + 1))

if use_parallel:
    print(f"-> Dispatching {len(rings_seq)} tasks to Joblib (n_jobs={n_jobs})...")
    print("   (Logs will appear in real-time as workers finish their rings)\n")
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
# Drop the massive 'angles' arrays before saving to CSV to prevent formatting errors and heavy files
df_export = df.drop(columns=['angles'], errors='ignore')
df_export.to_csv(os.path.join(dir_name, "refinement_data.csv"), index=False)
fig_dir = os.path.join(dir_name, "figures");
os.makedirs(fig_dir, exist_ok=True)

rings_int = np.arange(1, df['num_rings'].max() + 1)
# Calculate theoretical curves for plotting
d0_theo_curve = [theoretical_optimal_d0_refined(target_k, r, N_total, Lx_fixed, Ly_fixed, c) for r in rings_int]
# Theoretical <k> if d0 was kept at the baseline (showing the drift caused by density changes)
k_theo_unscaled = [theoretical_degree_refined(r, N_total, Lx_fixed, Ly_fixed, c, d0_base) for r in rings_int]

# 1. Connectivity Evolution Plot
plt.figure(figsize=(8, 5))
plt.plot(rings_int, k_theo_unscaled, color='crimson', linestyle='--', linewidth=2, zorder=1,
         label='Unscaled Theory (Fixed d0)')
plt.plot(df['num_rings'], df['k_scaled'], marker='s', linestyle='', color='forestgreen',
         label='Calibrated - Exp.')
plt.axhline(target_k, color='black', linestyle=':', alpha=0.5, label=f'Target <k> ({target_k:.2f})')
plt.title("Degree Conservation across Resolution: Theory vs Experiment")
plt.xlabel("Number of Rings (Resolution)")
plt.ylabel("Average Structural Degree <k>")
plt.grid(True, linestyle=':', alpha=0.6)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "plot_1_connectivity.png"), dpi=400)
plt.close()

# 2. Scaling Law Plot (d0 vs Rings)
plt.figure(figsize=(8, 5))
plt.plot(df['num_rings'], df['d0_opt'], marker='D', linestyle='', color='indigo', markersize=6,
         label='Optimal d0 (Bisection) - Exp.')
plt.plot(rings_int, d0_theo_curve, color='orange', linestyle='-', linewidth=2.5, zorder=1,
         label='Theoretical Optimal (Root Finding)')
plt.title("Structural Decay Scaling Law: Theory vs Experiment")
plt.xlabel("Number of Rings (Resolution)")
plt.ylabel("Optimal Decay Length (d0)")
plt.grid(True, linestyle=':', alpha=0.6)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "plot_2_d0_scaling.png"), dpi=400)
plt.close()

# 3. Topological Distance Plot
plt.figure(figsize=(8, 5))
plt.plot(df['num_rings'], df['dist_scaled'], marker='D', linestyle='-', color='darkcyan', linewidth=2)
plt.title("Topological Path Length Evolution")
plt.xlabel("Number of Rings (Resolution)")
plt.ylabel("Mean Shortest Path (Hops)")
plt.grid(True, linestyle=':', alpha=0.6)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "plot_3_distances.png"), dpi=400)
plt.close()

# 4. Directional Isotropy Plot
plt.figure(figsize=(8, 5))
plt.plot(df['num_rings'], df['frac_horiz'], marker='o', linestyle='-', color='purple', linewidth=2)
plt.axhline(50.0, color='black', linestyle='--', alpha=0.7, label='Perfect 2D Isotropy (50%)')
plt.title("Directional Isotropy Transition (1D to 2D)")
plt.xlabel("Number of Rings (Resolution)")
plt.ylabel("Horizontal Links (%)")
plt.ylim(40, 105)
plt.grid(True, linestyle=':', alpha=0.6)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "plot_4_isotropy.png"), dpi=400)
plt.close()

# 5. Mean Angle Evolution
mean_angles = []
std_angles = []

# Calculate the mean angle and standard deviation for each resolution
for index, row in df.iterrows():
    angs = row['angles']
    if isinstance(angs, np.ndarray) and len(angs) > 0:
        mean_angles.append(np.mean(angs))
        std_angles.append(np.std(angs))
    else:
        # For R=1 (1D pure), all links are strictly 0 degrees
        mean_angles.append(0.0)
        std_angles.append(0.0)

mean_angles = np.array(mean_angles)
std_angles = np.array(std_angles)

plt.figure(figsize=(8, 5))
# Plot the standard deviation as a shaded confidence interval
plt.fill_between(df['num_rings'], mean_angles - std_angles, mean_angles + std_angles, color='teal', alpha=0.15)
# Plot the main average line
plt.plot(df['num_rings'], mean_angles, marker='o', linestyle='-', color='teal', linewidth=2, label='Mean Edge Angle')
# The mathematical proof of 2D isotropy: exactly 45 degrees
plt.axhline(45.0, color='black', linestyle='--', alpha=0.7, label='Perfect 2D Isotropy (45°)')

plt.title("Average Link Angle Evolution (1D -> 2D)")
plt.xlabel("Number of Rings (Resolution)")
plt.ylabel("Mean Angle (Degrees)")
plt.ylim(-5, 60)
plt.grid(True, linestyle=':', alpha=0.6)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "plot_5_mean_angle.png"), dpi=400)
plt.close()

total_time = time.time() - init_time
print(f"-> All plots successfully saved in: {fig_dir}")
print("\n" + "=" * 60)
print("   CALIBRATION COMPLETED SUCCESSFULLY")
print(f"   Total Time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
print("=" * 60 + "\n")