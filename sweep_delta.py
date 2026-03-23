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
# Example command: python sweep_delta.py 10000 1000.0 0.07 0.2 42

try:
    N = int(sys.argv[1])
    density = float(sys.argv[2])
    c = float(sys.argv[3])
    d0_base = float(sys.argv[4])
    seed = int(sys.argv[5])
except IndexError:
    print("Usage: python sweep_delta.py N density c d0_base seed")
    sys.exit(1)

init_time = time.time()
Lx = N / density
num_rings = 2  # We fix it to 2 rings to study the pure 1D -> 2D transition

# Create directory
dir_name = f'sweep_DELTA_N{N}_dens{density}_c{c}_d0base{d0_base}_seed{seed}/'
os.makedirs(dir_name, exist_ok=True)

print("=" * 60)
print("--- STARTING DELTA SWEEP (DIMENSIONAL CROSSOVER) ---")
print(f"Parameters: N={N}, Lx={Lx:.2f}, Rings={num_rings}, c={c}, d0={d0_base}")
print("=" * 60)

# 1. Get 1D Baseline (1 Ring)
np.random.seed(seed)
_, G_base, _ = triadic.random_coupled_rings_netw_PBC(N, Lx, 1, 0.0, c, d0_base)
k_baseline = np.mean([d for n, d in G_base.degree()])
print(f"-> 1D Baseline <k> (1 Ring) : {k_baseline:.2f}")

# 2. Sweep delta_factor from 0.01 (microscopic) to 2.0 (macroscopic)
delta_factors = np.linspace(0.01, 2.0, 30)
results = []

print("\nSweeping delta values...")
for df in delta_factors:
    delta = df * d0_base
    np.random.seed(seed)

    # Generate the 2-ring network
    _, G, _ = triadic.random_coupled_rings_netw_PBC(N, Lx, num_rings, delta, c, d0_base)
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
plt.plot(df_res['delta_factor'], df_res['k'], marker='o', color='crimson', markersize=5, linestyle='-', linewidth=2,
         label='2 Rings <k>')

# Find the point where <k> drops by 5% from baseline (a good proxy for the crossover threshold)
threshold_k = k_baseline * 0.95
crossover_df = df_res[df_res['k'] < threshold_k]
if not crossover_df.empty:
    cross_point = crossover_df.iloc[0]['delta_factor']
    plt.axvline(cross_point, color='blue', linestyle=':', linewidth=2,
                label=f'Crossover starts ~ {cross_point:.2f} * d0')

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