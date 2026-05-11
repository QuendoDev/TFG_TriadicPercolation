import os
import random
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
# This script performs the "Regulatory Refinement Limit" calibration.
# It reads the structural d0_opt from the previous structural calibration CSV,
# generates the exact same structural network, and finds the optimal dr.
# Example command: python calibrate_refinement_reg.py 10000 200.0 0.07 0.2 0.03 0.03 0.2 60 num=5 True 12

try:
    N_total = int(sys.argv[1])  # Total budget of nodes
    density_1D_base = float(sys.argv[2])  # Linear density
    c = float(sys.argv[3])  # Structural base probability
    d0_base = float(sys.argv[4])  # Target 1D d0 (used to find the structural folder)
    cpos = float(sys.argv[5])  # Positive regulation base probability
    cneg = float(sys.argv[6])  # Negative regulation base probability
    dr_base = float(sys.argv[7])  # Target 1D dr (baseline for ring 1)
    max_rings = int(sys.argv[8])  # Maximum resolution

    # Seed logic (Only used to determine the directory name)
    seed_arg = sys.argv[9]
    if seed_arg.startswith('num='):
        expected_seed_count = int(seed_arg.split('=')[1])
    elif ',' in seed_arg:
        expected_seed_count = len(seed_arg.split(','))
    else:
        expected_seed_count = 1

    # Optional parameters handling
    use_parallel = sys.argv[10].lower() == 'true' if len(sys.argv) > 10 else False
    n_jobs = int(sys.argv[11]) if len(sys.argv) > 11 else 1
except IndexError:
    print(
        "Error: Missing arguments. Usage: python calibrate_refinement_reg.py N_total density c d0 cpos "
        "cneg dr max_rings [seed|seed1,seed2|num=X] [use_parallel] [n_jobs]"
    )
    sys.exit(1)

user_choice = input("Enter the tolerance for dr optimization (default 0.0025): ")
try:
    tol = float(user_choice)
except ValueError:
    tol = 0.0025

init_time = time.time()

Lx_fixed = N_total / density_1D_base
Ly_fixed = Lx_fixed

# =============================================================================
# SECTION 2: DIRECTORY SETUP & CSV READING
# =============================================================================
# 1. Locate the structural data
struct_dir = (f'results/calibrate/ref/Ntot{N_total}_maxR{max_rings}_dens{density_1D_base}_c{c}'
              f'_seeds{expected_seed_count}/')
csv_path = os.path.join(struct_dir, "refinement_raw_data.csv")

if not os.path.exists(csv_path):
    print(f"CRITICAL ERROR: Structural RAW data not found at {csv_path}")
    print("Please run calibrate_refinement.py first with the SAME ensemble parameters.")
    sys.exit(1)

df_struct = pd.read_csv(csv_path)

# Get the exact list of seeds from the structural dataframe to ensure we only process the seeds that were actually
# generated in the structural calibration. This is crucial for reproducibility and to avoid any mismatches.
seeds = df_struct['seed'].unique().tolist()

# 2. Set up the regulatory directory
dir_name = (f'results/calibrate/ref_reg/Ntot{N_total}_maxR{max_rings}_dens{density_1D_base}'
            f'_c{c}_cpos{cpos}_cneg{cneg}_seeds{len(seeds)}/')
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


sys.stdout = Logger(os.path.join(dir_name, "refinement_reg_log.txt"))

print("\n" + "=" * 60)
print("--- STARTING REGULATORY REFINEMENT CALIBRATION ---")
print(f"Reading structural data from: {csv_path}")
print(f"Structural Params: N_total={N_total}, Lx={Lx_fixed:.2f}, Ly={Ly_fixed:.2f}, c={c}, d0_base={d0_base}")
print(f"Regulatory Params: c+={cpos}, c-={cneg}, target dr_base={dr_base}")
print(f"Ensemble Size    : {len(seeds)} seeds retrieved from structural CSV")
print(f"Tolerance        : {tol:.4f}")
print("=" * 60)


# =============================================================================
# SECTION 3: CALIBRATION FUNCTIONS
# =============================================================================
def theoretical_degree_refined(R: int, N_tot: int, Lx: float, Ly: float, c: float, d0: float) -> float:
    """
    Calculates the theoretical structural <k> for the coupled rings geometry.

    :param R: int, number of rings
    :param N_tot: int, total nodes
    :param Lx: float, width
    :param Ly: float, height
    :param c: float, structural connectivity base
    :param d0: float, structural decay length
    :return: float, theoretical mean structural degree
    """
    delta = Ly / R
    rho_1D_current = (N_tot / R) / Lx

    k_tot = 0.0
    for n in range(R):
        dy = min(n, R - n) * delta
        if dy == 0:
            k_tot += 2 * c * rho_1D_current * d0
        else:
            k_tot += 2 * c * rho_1D_current * dy * k1(dy / d0)
    return k_tot


def theoretical_degree_reg_refined(R: int, N_tot: int, Lx: float, Ly: float, c: float, d0_opt: float, c_reg: float,
                                   dr: float) -> float:
    """
    Calculates the theoretical <kappa> (regulatory degree) for the refinement case.

    :param R: int, number of rings
    :param N_tot: int, total nodes
    :param Lx: float, width
    :param Ly: float, height
    :param c: float, structural connectivity base
    :param d0_opt: float, optimal structural decay (at this R)
    :param c_reg: float, regulatory base probability (cpos or cneg)
    :param dr: float, regulatory decay length
    :return: float, theoretical regulatory mean degree
    """
    delta = Ly / R
    # Structural <k> is constant, so total links NL is roughly constant
    # We use the theoretical formula for structural <k> to find the link density
    k_struct = theoretical_degree_refined(R, N_tot, Lx, Ly, c, d0_opt)
    NL_theo = (N_tot * k_struct) / 2
    rho_L_current = (NL_theo / R) / Lx

    kappa_tot = 0.0
    for n in range(R):
        dy = min(n, R - n) * delta
        if dy == 0:
            kappa_tot += 2 * c_reg * rho_L_current * dr
        else:
            kappa_tot += 2 * c_reg * rho_L_current * dy * k1(dy / dr)
    return kappa_tot


def theoretical_optimal_dr_refined(target_kreg: float, R: int, N_tot: int, Lx: float, Ly: float, c: float,
                                   d0_opt: float, c_reg: float) -> float:
    """
    Finds the theoretical dr for the refinement case using root finding.

    :param target_kreg: float, target mean positive regulatory degree
    :param R: int, number of rings
    :param N_tot: int, total nodes
    :param Lx: float, width
    :param Ly: float, height
    :param c: float, structural connectivity base
    :param d0_opt: float, optimal structural decay (at this R)
    :param c_reg: float, regulatory base probability (cpos or cneg)
    :return: float, theoretical regulatory mean degree
    """

    def objective(d: float) -> float:
        return theoretical_degree_reg_refined(R, N_tot, Lx, Ly, c, d0_opt, c_reg, d) - target_kreg

    return float(brentq(objective, 1e-6, 15.0))


def find_optimal_dr(target_kpos: float, N_total: int, Lx: float, Ly: float, num_rings: int,
                    c: float, cpos: float, cneg: float, d0_opt: float, initial_dr: float,
                    seed: int, tol: float = 0.0025) -> tuple:
    """
    Bisection algorithm to find the optimal regulatory decay length (dr).

    :param target_kpos: float, target mean positive regulatory degree
    :param N_total: int, total nodes
    :param Lx: float, universe width
    :param Ly: float, universe height
    :param num_rings: int, number of rings
    :param c: float, structural base probability
    :param cpos: float, positive regulation base probability
    :param cneg: float, negative regulation base probability
    :param d0_opt: float, calibrated structural decay length
    :param initial_dr: float, initial guess for dr
    :param seed: int, random seed
    :param tol: float, error tolerance
    :return: tuple, (optimal_dr, kpos_emp, kneg_emp)
    """
    # 1. Regenerate the EXACT SAME structural network using the saved seed and optimal d0
    np.random.seed(seed)
    nodes, G, _ = triadic.coupled_rings_structural_network_fixed_N(N_total, num_rings, Lx, Ly, c, d0_opt)
    edges = np.array(G.edges())

    if len(edges) == 0:
        return 0.0, 0.0, 0.0

    # 2. Extract midpoints from the structural links
    links_mid, _ = triadic.midpoints_rings_PBC(nodes, Lx, Ly, edges[:, 0], edges[:, 1])

    low, high = 1e-6, 15.0
    diff, iteration, mid_dr = 1e9, 0, initial_dr

    # 3. Bisection Loop
    while diff > tol and iteration < 150:
        mid_dr = (low + high) / 2.0

        # Lock the seed inside the bisection so the regulatory random mask is deterministic
        # for a given dr, allowing the bisection to converge smoothly.
        np.random.seed(seed + 99)

        adjpos, adjneg = triadic.coupled_rings_regulatory_network(nodes, links_mid, Lx, Ly, mid_dr, cpos, cneg)

        # Calculate positive mean degree
        kpos_emp = adjpos.sum() / N_total
        diff = abs(kpos_emp - target_kpos)

        if kpos_emp < target_kpos:
            low = mid_dr
        else:
            high = mid_dr

        iteration += 1
        if (high - low) < 1e-10:
            break

    # Calculate the negative degree achieved with this optimal dr
    kneg_emp = adjneg.sum() / N_total

    return mid_dr, kpos_emp, kneg_emp


def process_reg_step(row: pd.Series, target_kpos: float, N_total: int, Lx: float, Ly: float,
                     c: float, cpos: float, cneg: float, dr_base: float, tol: float) -> dict:
    """
    Worker for a single resolution step in regulatory calibration.

    :param row: pandas Series, row from the structural calibration dataframe
    :param target_kpos: float, target mean positive regulatory degree
    :param N_total: int, total nodes
    :param Lx: float, universe width
    :param Ly: float, universe height
    :param c: float, structural base probability
    :param cpos: float, positive regulation base probability
    :param cneg: float, negative regulation base probability
    :param dr_base: float, initial guess for dr
    :param tol: float, error tolerance
    :return: dict, results for the current ring resolution
    """
    r = int(row['num_rings'])
    current_seed = int(row['seed'])
    d0_opt = float(row['d0_opt'])
    delta = float(row['delta'])

    logs = [f"\n--- Res R = {r} | Seed = {current_seed} ---",
            f"  -> Structural: d0_opt = {d0_opt:.6f}"]

    dr_opt, kpos_scaled, kneg_scaled = find_optimal_dr(
        target_kpos, N_total, Lx, Ly, r, c, cpos, cneg, d0_opt, dr_base, current_seed, tol
    )

    logs.append(f"  -> Found dr: {dr_opt:.6f}")
    logs.append(f"  -> k_pos   : {kpos_scaled:.4f} (Target: {target_kpos:.4f})")
    logs.append(f"  -> k_neg   : {kneg_scaled:.4f}")

    return {
        'num_rings': r, 'seed': current_seed, 'delta': delta, 'n_total': N_total, 'd0_opt': d0_opt,
        'dr_opt': dr_opt, 'kpos_scaled': kpos_scaled, 'kneg_scaled': kneg_scaled,
        'log_messages': "\n".join(logs)
    }


# =============================================================================
# SECTION 4: MAIN CALIBRATION LOOP
# =============================================================================
print("\n[Phase 1] Baseline Regulatory Target (R=1)...")
results = []
target_kpos_dict = {}
target_kneg_dict = {}

for s in seeds:
    print(f"\n-> Baseline establishing with seed {s}...")
    # Get the specific d0_opt for R=1 for THIS seed from the structural dataframe
    row_base = df_struct[(df_struct['num_rings'] == 1) &
                         (df_struct['seed'] == s)].iloc[0]
    d0_opt_base = float(row_base['d0_opt'])
    print(f"-> Using structural d0_opt: {d0_opt_base:.6f}")

    # Calculate the exact empirical k_pos and k_neg for the 1D baseline using dr_base
    np.random.seed(s)
    nodes_1, G_1, _ = triadic.coupled_rings_structural_network_fixed_N(N_total, 1, Lx_fixed, Ly_fixed, c,
                                                                       d0_opt_base)
    print(f"-> Nodes generated in structural graph: {G_1.number_of_nodes()} (Target: {N_total})")
    edges_1 = np.array(G_1.edges())
    links_mid_1, _ = triadic.midpoints_rings_PBC(nodes_1, Lx_fixed, Ly_fixed, edges_1[:, 0], edges_1[:, 1])

    np.random.seed(s + 99)
    adjpos_1, adjneg_1 = triadic.coupled_rings_regulatory_network(nodes_1, links_mid_1, Lx_fixed, Ly_fixed, dr_base,
                                                                  cpos, cneg)

    target_kpos = adjpos_1.sum() / N_total
    target_kneg = adjneg_1.sum() / N_total

    target_kpos_dict[s] = target_kpos
    target_kneg_dict[s] = target_kneg

    print(f"-> Target <k_pos> established: {target_kpos:.4f}")
    print(f"-> Target <k_neg> established: {target_kneg:.4f}")

    results.append({
        'num_rings': 1, 'seed': s, 'delta': Ly_fixed, 'n_total': N_total, 'd0_opt': d0_opt_base,
        'dr_opt': dr_base, 'kpos_scaled': target_kpos, 'kneg_scaled': target_kneg
    })

print(f"-> Baselines established across {len(seeds)} seeds.")

print("\n[Phase 2] Refining Regulatory Mesh...")
rows_to_process = [row for _, row in df_struct.iterrows() if row['num_rings'] > 1]

if use_parallel:
    print(f"-> Dispatching {len(rows_to_process)} tasks to Joblib (n_jobs={n_jobs})...")
    print("   (Logs will appear in real-time as workers finish their rings)\n")
    executor = Parallel(n_jobs=n_jobs, return_as="generator")
    gen = executor(
        delayed(process_reg_step)(
            row, target_kpos_dict[int(row['seed'])], N_total, Lx_fixed, Ly_fixed, c, cpos, cneg, dr_base, tol
        ) for row in rows_to_process
    )
    for res in gen:
        print(res['log_messages'])
        del res['log_messages']
        results.append(res)
    results.sort(key=lambda x: (x['num_rings'], x['seed']))
else:
    for row in rows_to_process:
        res = process_reg_step(row, target_kpos_dict[int(row['seed'])], N_total, Lx_fixed, Ly_fixed, c, cpos, cneg,
                               dr_base, tol)
        print(res['log_messages'])
        del res['log_messages']
        results.append(res)

# =============================================================================
# SECTION 5: EXPORT & PLOTTING
# =============================================================================
print("\n[Phase 3] Generating Regulatory Plots...")
df = pd.DataFrame(results)
df.to_csv(os.path.join(dir_name, "refinement_reg_data.csv"), index=False)

# Group by ring resolution to calculate ensemble statistics
df_agg = df.groupby('num_rings').agg(['mean', 'std']).reset_index()
df_agg.to_csv(os.path.join(dir_name, "refinement_reg_agg_data.csv"), index=False)

fig_dir = os.path.join(dir_name, "figures")
os.makedirs(fig_dir, exist_ok=True)

rings_int = df_agg['num_rings'].values

target_kpos_mean = float(np.mean(list(target_kpos_dict.values())))
target_kneg_mean = float(np.mean(list(target_kneg_dict.values())))

# To calculate theoretical curves, we use the ensemble mean of d0_opt for each ring
mean_d0_opts = df_agg['d0_opt']['mean'].values

# 1. Calculate Theoretical dr curve
# We use the d0_opt from the structural dataframe for each R
dr_theo_curve = [theoretical_optimal_dr_refined(target_kpos_mean, r, N_total, Lx_fixed, Ly_fixed, c,
                 mean_d0_opts[i], cpos) for i, r in enumerate(rings_int)]

# 2. Calculate Unscaled Regulatory Degrees (both pos and neg)
kpos_unscaled = [theoretical_degree_reg_refined(r, N_total, Lx_fixed, Ly_fixed, c,
                 mean_d0_opts[i], cpos, dr_base) for i, r in enumerate(rings_int)]

kneg_unscaled = [theoretical_degree_reg_refined(r, N_total, Lx_fixed, Ly_fixed, c,
                 mean_d0_opts[i], cneg, dr_base) for i, r in enumerate(rings_int)]

# Plot 1: dr Scaling Law
plt.figure(figsize=(8, 5))
plt.fill_between(rings_int, df_agg['dr_opt']['mean'] - df_agg['dr_opt']['std'],
                 df_agg['dr_opt']['mean'] + df_agg['dr_opt']['std'], color='crimson', alpha=0.2)
plt.plot(rings_int, df_agg['dr_opt']['mean'], marker='D', linestyle='', color='crimson', markersize=5,
         label='Exp. Refinement (dr)')
plt.plot(rings_int, dr_theo_curve, color='orange', linestyle='-', linewidth=2.5, zorder=1,
         label='Theoretical Optimal')
plt.title("Regulatory Decay Scaling Law: Theory vs Experiment")
plt.xlabel("Number of Rings (Resolution)")
plt.ylabel("Optimal Decay Length (dr)")
plt.grid(True, linestyle=':', alpha=0.6)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "plot_1_dr_scaling.png"), dpi=400)
plt.close()

# Plot 2: Dual Degree Conservation (pos and neg in the same plot)
plt.figure(figsize=(8, 5))
# Unscaled (Reference)
plt.plot(rings_int, kpos_unscaled, color='gray', linestyle='--', linewidth=1.5, label='Unscaled Theory (pos)')
# Only plot the second unscaled line if cneg is actually different from cpos
if cpos != cneg:
    plt.plot(rings_int, kneg_unscaled, color='silver', linestyle='-.', linewidth=1.5,
             label='Unscaled Theory (neg)')

# Positive Connections (Empirical Scaled)
plt.errorbar(rings_int, df_agg['kpos_scaled']['mean'], yerr=df_agg['kpos_scaled']['std'], fmt='s',
             color='royalblue', ecolor='darkblue', capsize=3, alpha=0.8, label='Empirical <k_pos>')
plt.axhline(target_kpos_mean, color='navy', linestyle=':', alpha=0.8, linewidth=2,
            label=f'Target <k_pos> ({target_kpos_mean:.2f})')

# Negative Connections (Empirical Scaled)
plt.errorbar(rings_int, df_agg['kneg_scaled']['mean'], yerr=df_agg['kneg_scaled']['std'], fmt='o',
             color='tomato', ecolor='darkred', capsize=3, alpha=0.8, label='Empirical <k_neg>')
plt.axhline(target_kneg_mean, color='darkred', linestyle=':', alpha=0.8, linewidth=2,
            label=f'Target <k_neg> ({target_kneg_mean:.2f})')


plt.title("Regulatory Degree Conservation: Scaled vs Unscaled")
plt.xlabel("Number of Rings (Resolution)")
plt.ylabel("Average Regulatory Degree <k>")
plt.grid(True, linestyle=':', alpha=0.6)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))

# Put the legend outside if it overlaps, or keep it inside if it fits well
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "plot_2_k_conservation_dual.png"), dpi=400)
plt.close()

total_time = time.time() - init_time
print(f"-> All plots successfully saved in: {fig_dir}")
print("\n" + "=" * 60)
print("   REGULATORY CALIBRATION COMPLETED SUCCESSFULLY")
print(f"   Total Time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
print("=" * 60 + "\n")