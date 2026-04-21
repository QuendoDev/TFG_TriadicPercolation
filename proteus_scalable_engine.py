import sys
import os
import time
import json
import random
import numpy as np
import networkx as nx

from scipy.stats import circmean, circstd

import triadic_library as triadic

# =============================================================================
# SECTION 1: PARAMETER READING & INITIALIZATION
# =============================================================================
# The script expects exactly 12 arguments from the console.
# Example command: python proteus_scalable_engine.py 1000 400 0.00 200.0 0.2 0.2 0.03 0.03 0.07 42 40 0.2
# The order is: N_per_ring Tmax p density d0 dr cpos cneg c seed num_rings delta_factor

try:
    N_per_ring = int(sys.argv[1])    # Number of nodes PER RING
    Tmax = int(sys.argv[2]) # Maximum simulation time
    p = float(sys.argv[3])  # Probability of a link not breaking randomly
    density = float(sys.argv[4])    # Node density (it tries to emulate the 2D density)
    d0 = float(sys.argv[5]) # Typical scale for structural connections
    dr = float(sys.argv[6]) # Typical scale for regulatory connections
    cpos = float(sys.argv[7])   # Base probability for positive regulatory
    cneg = float(sys.argv[8])   # Base probability for negative regulatory
    c = float(sys.argv[9])  # Base probability for structural connections
    seed = int(sys.argv[10])    # Random seed for reproducibility
    num_rings = int(sys.argv[11])   # Number of coupled discrete rings
    delta_factor = float(sys.argv[12])  # Separation factor between rings (Delta = d0 * delta_factor)
except IndexError:
    print(
        "Error: Missing arguments. Usage: python proteus_scalable_engine.py N_per_ring Tmax p density d0 dr cpos cneg c"
        " seed num_rings delta_factor")
    sys.exit(1)

# Track the total time
init_time = time.time()

# Set the random seeds
np.random.seed(seed)
random.seed(seed)

# Geometry calculation for the Discrete Torus
# This method will work because of some equations:
# 1) rho_2D = N_tot / area_tot
# 2) N_tot = N_ring * R (number of rings)
# 3) area_tot = L_x * L_y
# 4) L_y = Delta * R
# Substituting 1,2,3,4 we get: rho_2D = N_ring * R / L_x * Delta * R
# If L_x is defined as L_x = N_ring / rho_1D then rho_2D = rho_1D / Delta, which means that the 2D density is
# effectively controlled by the choice of Delta (or delta_factor) and doesn't depend on R.
# Total number of nodes in the system
N = N_per_ring * num_rings

# Physical distance between adjacent rings
delta = delta_factor * d0

# Total dimensions of the periodic box
Lx = N_per_ring / density
Ly = num_rings * delta

L = np.asarray([Lx, Ly])

# =============================================================================
# SECTION 2: DIRECTORY SETUP
# =============================================================================
dir_name = (f'results/scalable/R{num_rings}_p{p:.2f}_N{N}_T{Tmax}_rho{density}_dr{dr:.2f}_d0{d0:.2f}'
            f'_cpos{cpos:.2f}_cneg{cneg:.2f}_c{c:.2f}_dfact{delta_factor:.2f}_seed{seed}/')

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

sys.stdout = Logger(os.path.join(dir_name, "simulation_log.txt"))

print("="*60)
print("--- LOG FILE INITIALIZED ---")
print("=" * 60)
print("--- STARTING SIMULATION (COUPLED RINGS) ---")
print(f"Parameters: N_per_ring={N_per_ring}, Total_N={N}, Tmax={Tmax}, p={p}")
print(f"Geometry  : num_rings={num_rings}, Linear Density={density}, delta={delta:.4f} (factor={delta_factor})")
print(f"Dimensions: Lx={Lx:.2f}, Ly={Ly:.4f} (Periodic in both axes)")
print(f"Distances : d0={d0}, dr={dr}")
print(f"Connectiv.: cpos={cpos}, cneg={cneg}, c={c}")
print(f"Seed      : {seed}")
print("=" * 60)

# =============================================================================
# SECTION 3: NETWORK GENERATION & STATIC METRICS
# =============================================================================
print("\n[Phase 1] Generating spatial networks...")
net_start_time = time.time()

# 1. Structural Network
print('Creating structural network (Discrete 2D)...')
struct_start = time.time()

nodes, G, adj = triadic.coupled_rings_structural_network(
    N_per_ring=N_per_ring,
    num_rings=num_rings,
    Lx=Lx,
    delta=delta,
    c=c,
    d0=d0
)

lij = np.array(G.edges())
I = lij[:, 0]
J = lij[:, 1]
del lij

struct_end = time.time()
print(f'-> Structural network created in: {struct_end - struct_start:.3f} seconds.')

# 2. Regulatory Network
print('Creating regulatory network...')
reg_start = time.time()
# The mathematical continuous representation of this discrete space allows to safely use the 2D midpoints
links, NL = triadic.midpoints_square_PBC(nodes, L, I, J)
adjpos, adjneg = triadic.regulatory_network_square(nodes, links, L, dr, cpos, cneg)

NL_pos = np.count_nonzero(adjpos)
NL_neg = np.count_nonzero(adjneg)

reg_end = time.time()
print(f'-> Regulatory network created in: {reg_end - reg_start:.3f} seconds.')
net_end_time = time.time()
print(f"-> Networks created in {net_end_time - net_start_time:.2f} seconds.")

print("\n" + "-" * 45)
print("               NETWORK SUMMARY")
print("-" * 45)
print(f" Total Nodes (N)                : {N}")
print(f" Total Structural Links (NL)    : {NL}")
print(f" Total Positive Regulations (+) : {NL_pos}")
print(f" Total Negative Regulations (-) : {NL_neg}")
print("-" * 45 + "\n")

# 3. Giant Component Isolation
Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
G0 = G.subgraph(Gcc[0])
ag = list(Gcc[0])
print(f"-> Giant component isolated with {len(Gcc[0])} nodes.")

# 4. Compute Static Topology Metrics
print("\n[Phase 2] Calculating static topological metrics...")
# Since the space is a hybrid, theoretical bounds are an approximation. Pass dim=2 to use 2D integration
degrees_data = triadic.calculate_degrees(G, adj, adjpos, adjneg, density, d0, dr, c, cpos, cneg, dim=2)
sanity_check = np.isclose(degrees_data['k_real_mean'], 2 * degrees_data['k_mat_mean'], atol=0.1)

print("\n" + "=" * 60)
print("--- DEGREES: EMPIRICAL (HYBRID GEOMETRY) ---")
print(" Note: Theoretical values are approx. due to discrete Y-axis.")
print(f"\n1. STRUCTURAL DEGREES <k>     : {degrees_data['k_real_mean']:.2f} +- {degrees_data['k_real_std']:.2f}")
print(f"2. REG. IN-DEGREES (+/-)      : {degrees_data['kappa_in_pos_mean']:.2f} / "
      f"{degrees_data['kappa_in_neg_mean']:.2f}")
print(f"3. REG. OUT-DEGREES (+/-)     : {degrees_data['kappa_out_pos_mean']:.2f} / "
      f"{degrees_data['kappa_out_neg_mean']:.2f}")
print(f"   Sanity Check (Real ~ 2*Mat): {sanity_check}")
print("=" * 60 + "\n")

path_lengths_t0, avg_distance_t0 = triadic.get_topological_distances(G0, sample_size=500)
print(f"-> Structural baseline average distance: {avg_distance_t0:.2f} hops.")

r_vals, N_r_vals = triadic.get_fractal_mass_radius(G0, max_hops=int(N / 2), sample_size=100)

print("-> Calculating Topological vs Geometric distance sample...")
sample_nodes = list(G0.nodes())
num_pairs = 3000
topo_distances, geom_distances = [], []

for _ in range(num_pairs):
    u, v = random.sample(sample_nodes, 2)
    topo_d = nx.shortest_path_length(G0, u, v)
    if topo_d > 0:
        dx = triadic.distance_PBC_1D_pair(nodes[u, 0], np.array([nodes[v, 0]]), Lx)[0]
        dy = triadic.distance_PBC_1D_pair(nodes[u, 1], np.array([nodes[v, 1]]), Ly)[0]
        geom_distances.append(np.sqrt(dx**2 + dy**2))
        topo_distances.append(topo_d)

print("-> Calculating structural link lengths...")
DX = triadic.distance_PBC_1D_pair(nodes[I, 0], nodes[J, 0], Lx)
DY = triadic.distance_PBC_1D_pair(nodes[I, 1], nodes[J, 1], Ly)
link_lengths = np.sqrt(DX**2 + DY**2)

avg_link_length = float(np.mean(link_lengths))
max_link_per_node = np.zeros(N)
for node_idx in range(N):
    connected_links_mask = (I == node_idx) | (J == node_idx)
    if np.any(connected_links_mask):
        max_link_per_node[node_idx] = np.max(link_lengths[connected_links_mask])

avg_max_link_length = float(np.mean(max_link_per_node))
print(f"-> Average link length: {avg_link_length:.4f}")

# =============================================================================
# SECTION 4: TIME DYNAMICS
# =============================================================================
print("\n[Phase 3] Starting time evolution...")
dyn_start_time = time.time()

statenodes = np.zeros((Tmax, N), dtype=bool)
statenodes[0, ag] = True

RT = np.zeros((Tmax, 3))
RT[0, 0] = len(ag) / float(N)
if len(Gcc) > 1:
    RT[0, 1] = len(Gcc[1]) / float(N)
    if len(Gcc) > 2:
        RT[0, 2] = len(Gcc[2]) / float(N)

spatial_dyn = np.full((Tmax, 6), np.nan)
theta_init = nodes[:, 0] * (2. * np.pi / Lx)

if len(Gcc) > 0:
    spatial_dyn[0, 0], spatial_dyn[0, 1] = (circmean(theta_init[ag], high=2 * np.pi, low=0),
                                            circstd(theta_init[ag], high=2 * np.pi, low=0))
    if len(Gcc) > 1:
        ag2_init = list(Gcc[1])
        spatial_dyn[0, 2], spatial_dyn[0, 3] = (circmean(theta_init[ag2_init], high=2 * np.pi, low=0),
                                                circstd(theta_init[ag2_init], high=2 * np.pi, low=0))
        if len(Gcc) > 2:
            ag3_init = list(Gcc[2])
            spatial_dyn[0, 4], spatial_dyn[0, 5] = (circmean(theta_init[ag3_init], high=2 * np.pi, low=0),
                                                    circstd(theta_init[ag3_init], high=2 * np.pi, low=0))

for it in range(1, Tmax):
    RT[it, :], agn, spatial_dyn[it, :] = triadic.itera_rings(
        Lx, statenodes[it - 1, :], nodes, links, I, J, adjpos, adjneg, p
    )
    statenodes[it, agn] = True
    if it % 10 == 0:
        print(f"   Iter {it}/{Tmax} completed ({(it / Tmax) * 100:.1f}%)")

dyn_end_time = time.time()
print(f"-> Dynamics finished in {dyn_end_time - dyn_start_time:.2f} seconds.")

print("\n[Phase 4] Calculating steady state topology...")
active_nodes_indices = np.where(statenodes[-1, :])[0]
G_active = G.subgraph(active_nodes_indices)

if len(G_active.nodes()) > 0:
    largest_cc_nodes = max(nx.connected_components(G_active), key=len)
    G0_active = G_active.subgraph(largest_cc_nodes)
    path_lengths_tfinal, avg_distance_tfinal = triadic.get_topological_distances(G0_active, sample_size=500)
else:
    path_lengths_tfinal, avg_distance_tfinal = [], 0.0

print(f"-> Active distance at t=Tmax: {avg_distance_tfinal:.2f} hops.")

# =============================================================================
# SECTION 5: DATA EXPORT
# =============================================================================
print("\n[Phase 5] Saving results...")

np.savetxt(os.path.join(dir_name, 'RT.txt'), RT)
np.savetxt(os.path.join(dir_name, 'spatial_dyn.txt'), spatial_dyn)
np.savez_compressed(os.path.join(dir_name, 'statenodes.npz'), statenodes=statenodes)
np.save(os.path.join(dir_name, 'nodes_coords.npy'), nodes)

np.savez_compressed(
    os.path.join(dir_name, 'topology_data.npz'),
    r_vals=r_vals, N_r_vals=N_r_vals, path_lengths_t0=path_lengths_t0,
    path_lengths_tfinal=path_lengths_tfinal, topo_distances=topo_distances,
    geom_distances=geom_distances, link_lengths=link_lengths
)

summary_data = {
    "parameters": {
        "N": int(N), "Tmax": int(Tmax), "p": float(p), "density": float(density),
        "L": L.tolist(), "geometry": "RINGS", "num_rings": int(num_rings), "delta_factor": float(delta_factor),
        "d0": float(d0), "dr": float(dr), "cpos": float(cpos), "cneg": float(cneg), "c": float(c), "seed": int(seed)
    },
    "network":{
        "NL": int(NL), "NL_pos": int(NL_pos), "NL_neg": int(NL_neg)
    },
    "link_lengths": {
        "avg_link_length": float(avg_link_length), "avg_max_link_length": float(avg_max_link_length)
    },
    "distances": {
        "mean_hops_t0": float(avg_distance_t0), "mean_hops_tfinal": float(avg_distance_tfinal)
    },
    "degrees_summary": {
        "k_real_mean": float(degrees_data["k_real_mean"]),
        "k_real_std": float(degrees_data["k_real_std"]),
        "kappa_in_pos_mean": float(degrees_data["kappa_in_pos_mean"]),
        "kappa_in_neg_mean": float(degrees_data["kappa_in_neg_mean"]),
        "kappa_out_pos_mean": float(degrees_data["kappa_out_pos_mean"]),
        "kappa_out_neg_mean": float(degrees_data["kappa_out_neg_mean"])
    },
    "degrees_arrays": {
        "k_real_array": degrees_data["k_real_array"].tolist(),
        "kappa_out_pos_array": degrees_data["kappa_out_pos_array"].tolist(),
        "kappa_out_neg_array": degrees_data["kappa_out_neg_array"].tolist(),
        "kappa_in_pos_array": degrees_data["kappa_in_pos_array"].tolist(),
        "kappa_in_neg_array": degrees_data["kappa_in_neg_array"].tolist()
    }
}

with open(os.path.join(dir_name, 'summary_metrics.json'), 'w') as json_file:
    json.dump(summary_data, json_file, indent=4)

total_time = time.time() - init_time
print(f"-> Results successfully saved in: {dir_name}")
print("\n" + "=" * 60)
print(f"   SCRIPT COMPLETED SUCCESSFULLY")
print(f"   Total Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print("=" * 60 + "\n")