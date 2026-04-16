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
# The script expects exactly 10 arguments from the console.
# Example command: python proteus_1D_engine.py 10000 400 0.00 1000.0 0.2 0.2 0.03 0.03 0.07 42
# The order is python proteus_1D_engine.py N Tmax p density d0 dr cpos cneg c seed

try:
    N = int(sys.argv[1])    # Number of nodes
    Tmax = int(sys.argv[2]) # Maximum simulation time
    p = float(sys.argv[3])  # Probability of a link not breaking randomly
    density = float(sys.argv[4])    # Node density in the 1D space
    d0 = float(sys.argv[5]) # Typical scale for structural connections
    dr = float(sys.argv[6]) # Typical scale for regulatory connections
    cpos = float(sys.argv[7])   # Base probability for positive regulatory connections
    cneg = float(sys.argv[8])   # Base probability for negative regulatory connections
    c = float(sys.argv[9])  # Base probability for structural connections
    seed = int(sys.argv[10])    # Random seed for reproducibility
except IndexError:
    print("Error: Missing arguments. Usage: python proteus_1D_engine.py N Tmax p density d0 dr cpos cneg c seed")
    sys.exit(1)

# Track the total time
init_time = time.time()

# Set the random seeds to ensure 100% reproducible networks!
np.random.seed(seed)
random.seed(seed)

# System size (length of the 1D space) calculated dynamically
L = float(N) / density

# =============================================================================
# SECTION 2: DIRECTORY SETUP
# =============================================================================
# Create a unique directory name based on the core parameters and the seed
dir_name = (f'results/1D_p{p:.2f}_N{N}_T{Tmax}_rho{density}_dr{dr:.2f}_d0{d0:.2f}'
            f'_cpos{cpos:.2f}_cneg{cneg:.2f}_c{c:.2f}_seed{seed}/')

if not os.path.exists(dir_name):
    os.makedirs(dir_name)


# Logger class made for saving all the console prints into a txt file
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        # This flush method is needed for python 3 compatibility.
        # This handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        self.terminal.flush()
        self.log.flush()

# Redirect all prints to the Logger, which will save them in a file called "simulation_log.txt"
sys.stdout = Logger(os.path.join(dir_name, "simulation_log.txt"))

print("="*60)
print("--- LOG FILE INITIALIZED ---")

print("=" * 60)
print("--- STARTING SIMULATION (1D RING) ---")
print(f"Parameters: N={N}, Tmax={Tmax}, p={p}, density={density}, L={L:.2f}")
print(f"Distances: d0={d0}, dr={dr}")
print(f"Connectivities: cpos={cpos}, cneg={cneg}, c={c}")
print(f"Random seed: {seed}")
print("=" * 60)

# =============================================================================
# SECTION 3: NETWORK GENERATION & STATIC METRICS
# =============================================================================
print("\n[Phase 1] Generating spatial networks...")
net_start_time = time.time()

# 1. Structural Network
print('Creating structural network...')
struct_start = time.time()
nodes, G, adj = triadic.random_uniform_line_netw_PBC(N, L, c, d0)

# Extract indices of the links for the regulatory network
lij = np.array(G.edges())
I = lij[:, 0]
J = lij[:, 1]
del lij

struct_end = time.time()
print(f'-> Structural network created in: {struct_end - struct_start:.3f} seconds.')

# 2. Regulatory Network
print('Creating regulatory network...')
reg_start = time.time()
links, NL = triadic.midpoints_line_PBC(nodes, L, I, J)
adjpos, adjneg = triadic.regulatory_network_line(nodes, links, L, dr, cpos, cneg)

# Count total positive and negative regulations
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
ag = list(Gcc[0])   # Vector with the indices of the nodes in the Giant component
print(f"-> Giant component isolated with {len(Gcc[0])} nodes.")

# 4. Compute Static Topology Metrics
print("\n[Phase 2] Calculating static topological metrics...")
# Degrees calculation using the library
degrees_data = triadic.calculate_degrees(G, adj, adjpos, adjneg, density, d0, dr, c, cpos, cneg, dim=1)

# Sanity check: The real mean degree should be approximately twice the matrix mean degree, so this bool should be
# True if the empirical data matches the expected relationship.
sanity_check = np.isclose(degrees_data['k_real_mean'], 2 * degrees_data['k_mat_mean'], atol=0.1)

print("\n" + "=" * 60)
print("--- DEGREES: THEORETICAL VS EMPIRICAL (1-DIM) ---")

print("\n1. STRUCTURAL DEGREES <k> (Connections per node):")
print(f"   Theoretical           : {degrees_data['k_theo']:.2f}")
print(f"   Empirical (Real G)    : {degrees_data['k_real_mean']:.2f} +- {degrees_data['k_real_std']:.2f}  "
      f"<-- True connections")
print(f"   Empirical (Matrix adj): {degrees_data['k_mat_mean']:.2f} +- {degrees_data['k_mat_std']:.2f}  "
      f"<-- Half (upper triangular)")
print(f"   Sanity Check (Real ~ 2*Matrix): {sanity_check}")

print("\n2. REGULATORY IN-DEGREES <kappa_in> (Regulators per link):")
print(f"   Theoretical (+)       : {degrees_data['kappa_in_pos_theo']:.2f}")
print(f"   Empirical (+)         : {degrees_data['kappa_in_pos_mean']:.2f} +- {degrees_data['kappa_in_pos_std']:.2f}")
print(f"   Theoretical (-)       : {degrees_data['kappa_in_neg_theo']:.2f}")
print(f"   Empirical (-)         : {degrees_data['kappa_in_neg_mean']:.2f} +- {degrees_data['kappa_in_neg_std']:.2f}")

print("\n3. REGULATORY OUT-DEGREES <kappa_out> (Links regulated per node):")
print(f"   Theoretical (+)       : {degrees_data['kappa_out_pos_theo']:.2f}")
print(f"   Empirical (+)         : {degrees_data['kappa_out_pos_mean']:.2f} +- {degrees_data['kappa_out_pos_std']:.2f}")
print(f"   Theoretical (-)       : {degrees_data['kappa_out_neg_theo']:.2f}")
print(f"   Empirical (-)         : {degrees_data['kappa_out_neg_mean']:.2f} +- {degrees_data['kappa_out_neg_std']:.2f}")
print("=" * 60 + "\n")

# Shortest paths (Topological distances at t=0)
path_lengths_t0, avg_distance_t0 = triadic.get_topological_distances(G0, sample_size=500)
print(f"-> Structural baseline average distance: {avg_distance_t0:.2f} hops.")

# Fractal dimension (Hausdorff Mass-Radius scaling)
# Let the algorithm find the true diameter (no max_hops limit passed)
r_vals, N_r_vals = triadic.get_fractal_mass_radius(G0, max_hops=int(N / 2), sample_size=100)

# Topological vs Geometric Distances (Raw data for local plotting)
print("-> Calculating Topological vs Geometric distance sample...")
sample_nodes = list(G0.nodes())
num_pairs = 3000
topo_distances = []
geom_distances = []

for _ in range(num_pairs):
    u, v = random.sample(sample_nodes, 2)
    # Topological distance (hops)
    topo_d = nx.shortest_path_length(G0, u, v)

    if topo_d > 0:  # Avoid self-loops
        # Geometric distance (physical length in the 1D ring)
        geom_d = triadic.distance_PBC_1D_pair(nodes[u, 0], np.array([nodes[v, 0]]), L)[0]
        topo_distances.append(topo_d)
        geom_distances.append(geom_d)

print("-> Calculating structural link lengths...")
# Calculate the physical distance of all structural links considering PBC
link_lengths = triadic.distance_PBC_1D_pair(nodes[I, 0], nodes[J, 0], L)
avg_link_length = float(np.mean(link_lengths))

# Find the maximum link length for each node
max_link_per_node = np.zeros(N)
for node_idx in range(N):
    # Find all links connected to this node (either as I or J)
    connected_links_mask = (I == node_idx) | (J == node_idx)
    if np.any(connected_links_mask):
        max_link_per_node[node_idx] = np.max(link_lengths[connected_links_mask])

avg_max_link_length = float(np.mean(max_link_per_node))

print(f"-> Average link length: {avg_link_length:.4f}")
print(f"-> Average of the maximum link per node: {avg_max_link_length:.4f}")

# =============================================================================
# SECTION 4: TIME DYNAMICS
# =============================================================================
print("\n[Phase 3] Starting time evolution...")
dyn_start_time = time.time()

# States array (active / inactive) of the nodes
# Initial state: only nodes in the Giant component are active
statenodes = np.zeros((Tmax, N), dtype=bool)
statenodes[0, ag] = True

# Arrays to store dynamic metrics over time (fraction of active nodes)
RT = np.zeros((Tmax, 3))
# Giant cluster (first cluster)
RT[0, 0] = len(ag) / float(N)
if len(Gcc) > 1:
    # Second cluster
    RT[0, 1] = len(Gcc[1]) / float(N)
    if len(Gcc) > 2:
        # Third cluster
        RT[0, 2] = len(Gcc[2]) / float(N)

# Columns represent: [Mean_Cluster1, Std_Cluster1, Mean_Cluster2, Std_Cluster2, Mean_Cluster3, Std_Cluster3] where mean
# is the position of the center of mass of the cluster and std is the spatial spread or width (std) of the cluster
spatial_dyn = np.full((Tmax, 6), np.nan)

# Evaluate spatial metrics for the initial condition (t=0) for first, second and third cluster
theta_init = nodes[:, 0] * (2. * np.pi / L)
if len(Gcc) > 0:
    spatial_dyn[0, 0] = circmean(theta_init[ag], high=2 * np.pi, low=0)
    spatial_dyn[0, 1] = circstd(theta_init[ag], high=2 * np.pi, low=0)
    if len(Gcc) > 1:
        ag2_init = list(Gcc[1])
        spatial_dyn[0, 2] = circmean(theta_init[ag2_init], high=2 * np.pi, low=0)
        spatial_dyn[0, 3] = circstd(theta_init[ag2_init], high=2 * np.pi, low=0)
        if len(Gcc) > 2:
            ag3_init = list(Gcc[2])
            spatial_dyn[0, 4] = circmean(theta_init[ag3_init], high=2 * np.pi, low=0)
            spatial_dyn[0, 5] = circstd(theta_init[ag3_init], high=2 * np.pi, low=0)


# Dynamic loop
for it in range(1, Tmax):
    # Perform one iteration without any visualization logic
    RT[it, :], agn, spatial_dyn[it, :] = triadic.itera_line(
        L, statenodes[it - 1, :], nodes, links, I, J, adjpos, adjneg, p
    )

    # Activate the nodes in the new giant component
    statenodes[it, agn] = True

    if it % 10 == 0:
        print(f"   Iter {it}/{Tmax} completed ({(it / Tmax) * 100:.1f}%)")

dyn_end_time = time.time()
print(f"-> Dynamics finished in {dyn_end_time - dyn_start_time:.2f} seconds.")

# Calculate final steady state distance
print("\n[Phase 4] Calculating steady state topology...")

# 1. Identify which nodes are active (True) at the LAST time step (Tmax-1)
active_nodes_indices = np.where(statenodes[-1, :])[0]

# 2. Create a "subgraph" using only those active nodes and their base structural connections
G_active = G.subgraph(active_nodes_indices)

# 3. Find the Giant Component of this active network and calculate the average distance within it
if len(G_active.nodes()) > 0:
    largest_cc_nodes = max(nx.connected_components(G_active), key=len)
    G0_active = G_active.subgraph(largest_cc_nodes)
    path_lengths_tfinal, avg_distance_tfinal = triadic.get_topological_distances(G0_active, sample_size=500)
else:
    path_lengths_tfinal = []
    avg_distance_tfinal = 0.0

print(f"-> Active distance at t=Tmax: {avg_distance_tfinal:.2f} hops.")

# =============================================================================
# SECTION 5: DATA EXPORT (LIGHTWEIGHT FILES)
# =============================================================================
print("\n[Phase 5] Saving results...")

# 1. Save heavy arrays using NumPy compressed format (.npz) or raw text (.txt)

# RT: Array (Tmax, 3) with the fraction of active nodes for the top 3 clusters over time
# Saved to plot the Giant Component size fluctuations (R) vs Time
np.savetxt(os.path.join(dir_name, 'RT.txt'), RT)

# spatial_dyn: Array (Tmax, 6) with circular mean (position) and std deviation (width) of the clusters
# Saved to plot the spatial movement and pattern width of the clusters over time
np.savetxt(os.path.join(dir_name, 'spatial_dyn.txt'), spatial_dyn)

# statenodes: Boolean array (Tmax, N) tracking if a node is active (True) or inactive (False) at each step
# Highly compressed. Saved specifically to generate the Spatiotemporal Raster Plot locally
np.savez_compressed(os.path.join(dir_name, 'statenodes.npz'), statenodes=statenodes)

# nodes: Array (N, 1) with the physical 1D coordinates of each node
# Saved to provide the Y-axis (spatial position) for the Spatiotemporal Raster Plot
np.save(os.path.join(dir_name, 'nodes_coords.npy'), nodes)

# We save the fractal distribution and distances arrays compressed to save disk space
np.savez_compressed(
    os.path.join(dir_name, 'topology_data.npz'),
    r_vals=r_vals,
    N_r_vals=N_r_vals,
    path_lengths_t0=path_lengths_t0,
    path_lengths_tfinal=path_lengths_tfinal,
    topo_distances=topo_distances,
    geom_distances=geom_distances,
    link_lengths=link_lengths
)

# 2. Save all scalar metrics and parameters in a clean JSON file
summary_data = {
    "parameters": {
        "N": int(N), "Tmax": int(Tmax), "p": float(p), "density": float(density), "L": float(L),
        "d0": float(d0), "dr": float(dr), "cpos": float(cpos), "cneg": float(cneg), "c": float(c), "seed": int(seed)
    },
    "network":{
        "NL": int(NL),
        "NL_pos": int(NL_pos),
        "NL_neg": int(NL_neg)
    },
    "link_lengths": {
        "avg_link_length": float(avg_link_length),
        "avg_max_link_length": float(avg_max_link_length)
    },
    "distances": {
        "mean_hops_t0": float(avg_distance_t0),
        "mean_hops_tfinal": float(avg_distance_tfinal)
    },
    "degrees_summary": {
        # Theoretical values
        "k_theo": float(degrees_data["k_theo"]),
        "kappa_in_pos_theo": float(degrees_data["kappa_in_pos_theo"]),
        "kappa_in_neg_theo": float(degrees_data["kappa_in_neg_theo"]),
        "kappa_out_pos_theo": float(degrees_data["kappa_out_pos_theo"]),
        "kappa_out_neg_theo": float(degrees_data["kappa_out_neg_theo"]),

        # Empirical Means and Stds
        "k_real_mean": float(degrees_data["k_real_mean"]),
        "k_real_std": float(degrees_data["k_real_std"]),
        "kappa_in_pos_mean": float(degrees_data["kappa_in_pos_mean"]),
        "kappa_in_pos_std": float(degrees_data["kappa_in_pos_std"]),
        "kappa_in_neg_mean": float(degrees_data["kappa_in_neg_mean"]),
        "kappa_in_neg_std": float(degrees_data["kappa_in_neg_std"]),
        "kappa_out_pos_mean": float(degrees_data["kappa_out_pos_mean"]),
        "kappa_out_pos_std": float(degrees_data["kappa_out_pos_std"])
    },
    "degrees_arrays": {
        # Full arrays for histogram plotting locally
        "k_real_array": degrees_data["k_real_array"].tolist(),
        "kappa_out_pos_array": degrees_data["kappa_out_pos_array"].tolist(),
        "kappa_out_neg_array": degrees_data["kappa_out_neg_array"].tolist(),
        "kappa_in_pos_array": degrees_data["kappa_in_pos_array"].tolist(),
        "kappa_in_neg_array": degrees_data["kappa_in_neg_array"].tolist()
    }
}

with open(os.path.join(dir_name, 'summary_metrics.json'), 'w') as json_file:
    json.dump(summary_data, json_file, indent=4)

# Get the final simulation time to calculate the total time
finish_time = time.time()
total_time = finish_time - init_time

print(f"-> Results successfully saved in: {dir_name}")
print("\n" + "=" * 60)
print(f"   SCRIPT COMPLETED SUCCESSFULLY")
print(f"   Total Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print("=" * 60 + "\n")