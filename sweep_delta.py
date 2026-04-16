import sys
import os
import time
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

import triadic_library as triadic

# =============================================================================
# SECTION 1: PARAMETER READING & INITIALIZATION
# =============================================================================
# Example command: python sweep_delta.py 2000 200.0 0.07 0.2 42
# The usage is python sweep_delta.py N_per_ring density c d0_base seed

try:
    N_per_ring = int(sys.argv[1])
    density = float(sys.argv[2])
    c = float(sys.argv[3])
    d0_base = float(sys.argv[4])
    seed = int(sys.argv[5])
except IndexError:
    print("Usage: python sweep_delta.py N_per_ring density c d0_base seed")
    sys.exit(1)

init_time = time.time()
Lx = N_per_ring / density
num_rings = 2  # We fix it to 2 rings to study the pure 1D -> 2D transition

# Create directory
dir_name = f'results/sweep_DELTA_N{N_per_ring}_dens{density}_c{c}_d0base{d0_base}_seed{seed}/'
os.makedirs(dir_name, exist_ok=True)

print("=" * 60)
print("--- STARTING DELTA SWEEP (DIMENSIONAL CROSSOVER) ---")
print(f"Parameters: N_per_ring={N_per_ring}, Lx={Lx:.2f}, Rings={num_rings}, c={c}, d0={d0_base}")
print("=" * 60)

# 1. Get 1D Baseline (1 Ring)
np.random.seed(seed)
_, G_base, _ = triadic.coupled_rings_structural_network(N_per_ring, 1, Lx, 0.0, c, d0_base)
k_baseline = np.mean([d for n, d in G_base.degree()])
print(f"-> 1D Baseline <k> (1 Ring) : {k_baseline:.2f}")

# 2. Sweep delta_factor from 0.01 (microscopic) to 2.0 (macroscopic)
delta_factors = np.linspace(0.01, 6.0, 50)
results = []

print("\nSweeping delta values...")
for df in delta_factors:
    delta = df * d0_base
    np.random.seed(seed)

    # Generate the 2-ring network
    _, G, _ = triadic.coupled_rings_structural_network(N_per_ring, num_rings, Lx, delta, c, d0_base)
    k_emp = np.mean([d for n, d in G.degree()])

    results.append({'delta_factor': df, 'delta': delta, 'k': k_emp})
    # Print a quick progress bar
    sys.stdout.write('.')
    sys.stdout.flush()

print("\n\n[Phase 3] Saving results and generating plots...")
df_res = pd.DataFrame(results)
df_res.to_csv(os.path.join(dir_name, "delta_sweep_data.csv"), index=False)

# 3. Plotting the Crossover
plt.figure(figsize=(8, 5))
plt.axhline(k_baseline, color='gray', linestyle='--', linewidth=2, label=f'1D Baseline (<k>={k_baseline:.2f})')
plt.plot(df_res['delta_factor'], df_res['k'], marker='o', color='crimson', markersize=5, linestyle='-',
         linewidth=2, label='2 Rings <k>')

# Frontier A: When the system STARTS to uncouple (drops 5% from its empirical maximum)
k_max = df_res['k'].max()
threshold_start = k_max * 0.95
cross_start_df = df_res[df_res['k'] < threshold_start]

if not cross_start_df.empty:
    cross_start = cross_start_df.iloc[0]['delta_factor']
    plt.axvline(cross_start, color='orange', linestyle='--', linewidth=2,
                label=f'Starts uncoupling ~ {cross_start:.2f} * d0')

# Frontier B: When the system is FULLY uncoupled (reaches 105% of the 1D baseline)
threshold_end = k_baseline * 1.05
cross_end_df = df_res[df_res['k'] < threshold_end]

if not cross_end_df.empty:
    cross_end = cross_end_df.iloc[0]['delta_factor']
    plt.axvline(cross_end, color='blue', linestyle=':', linewidth=2,
                label=f'Fully uncoupled ~ {cross_end:.2f} * d0')

# Highlight the transition zone
if not cross_start_df.empty and not cross_end_df.empty:
    plt.axvspan(cross_start, cross_end, color='yellow', alpha=0.15, label='Dimensional Transition Zone')

plt.title("Dimensional Crossover: When does the Y-axis matter?")
plt.xlabel(r"Separation Factor ($\Delta / d_0$)")
plt.ylabel(r"Average Structural Degree $\langle k \rangle$")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(dir_name, "dimensional_crossover.png"), dpi=400)
plt.close()

print(f"-> Sweep completed in {time.time() - init_time:.2f} seconds.")
print(f"-> Check '{dir_name}' for the plot!")