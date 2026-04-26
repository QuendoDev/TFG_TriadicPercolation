import gc
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
# SECTION 1: PARAMETER READING & SEED PARSING
# =============================================================================
# This script expects 11 parameters and the seed initializer
# Example commands:
# python proteus_2D_engine.py 10000 1000 0.00 100.0 0.25 0.25 0.2 0.2 0.4 1.0 RC 42
# python proteus_2D_engine.py 10000 1000 0.00 100.0 0.25 0.25 0.2 0.2 0.4 1.0 RC 42,43,44
# python proteus_2D_engine.py 10000 1000 0.00 100.0 0.25 0.25 0.2 0.2 0.4 1.0 RC num=5
# Usage: python proteus_2D_engine.py N Tmax p density d0 dr cpos cneg c RC_factor geometry [seed|seed1,seed2|num=X]

# Track the total time
init_time = time.time()

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
    RC_factor = float(sys.argv[10]) # Geometry type (1 for 2D square, >1 for rectangular) (Lx/Ly)
    geometry = sys.argv[11] # Geometry type (SQ for square, RC for rectangular)

    # SEED PARSING LOGIC
    if len(sys.argv) > 12:
        seed_arg = sys.argv[12].strip()
        if seed_arg.startswith("num="):
            n_seeds = int(seed_arg.split("=")[1])
            seeds = [random.randint(1, 999999) for _ in range(n_seeds)]
        elif "," in seed_arg:
            seeds = [int(s) for s in seed_arg.split(",")]  # List of specific seeds
        else:
            seeds = [int(seed_arg)]  # Single seed
    else:
        # Default to 5 seeds if no argument is provided
        seeds = [random.randint(1, 999999) for _ in range(5)]

except Exception as e:
    print(f"Error parsing arguments: {e}")
    print("Usage: python proteus_2D_engine.py N Tmax p density d0 dr cpos cneg c RC_factor geometry "
          "[seed|seed1,seed2|num=X]")
    sys.exit(1)

# System size
if geometry == 'SQ':  # Square
    Lx = np.sqrt(N / density)
    Ly = Lx
    L = np.asarray([Lx, Ly])  # Sides of the network area
elif geometry == 'RC':    # Rectangular
    Ly = np.sqrt(N / (density * RC_factor))
    Lx = RC_factor * Ly
    L = np.asarray([Lx , Ly]) # Sides of the network area

# =============================================================================
# SECTION 2: DIRECTORY SETUP
# =============================================================================
# The base directory now contains only the physical parameters
base_dir_name = (f'results/2D/'
                 f'{geometry}{RC_factor}_N{N}_T{Tmax}_p{p:.2f}_c{c}_cpos{cpos}_cneg{cneg}_d0{d0:.2f}_dr{dr:.2f}')
if not os.path.exists(base_dir_name):
    os.makedirs(base_dir_name, exist_ok=True)


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
sys.stdout = Logger(os.path.join(base_dir_name, "simulation_log.txt"))

print("="*60)
print("--- LOG FILE INITIALIZED ---")

print("=" * 60)
print("--- STARTING SIMULATION (2D) ---")
print(f"Parameters: N={N}, Tmax={Tmax}, p={p}, density={density}")
print(f"Distances: d0={d0}, dr={dr}")
print(f"Connectivities: cpos={cpos}, cneg={cneg}, c={c}")
print(f"Geometry: {geometry}, Lx={Lx}, Ly={Ly}, RC_factor={RC_factor}")
print(f"Seeds to run: {seeds}")
print(f"Base Output Dir : {base_dir_name}")
print("=" * 60)

# =============================================================================
# MAIN SIMULATION LOOP (OVER SEEDS)
# =============================================================================
for current_seed in seeds:
    seed_time = time.time()
    print(f"\n>>> Starting simulation for SEED = {current_seed} <<<")

    # Create specific seed directory
    seed_dir = os.path.join(base_dir_name, f'seed={current_seed}')
    os.makedirs(seed_dir, exist_ok=True)

    np.random.seed(current_seed)

    # =============================================================================
    # SECTION 3: NETWORK GENERATION & STATIC METRICS
    # =============================================================================
    print("\n[Phase 1] Generating spatial networks...")
    net_start_time = time.time()

    # 1. Structural Network
    print('Creating structural network...')
    struct_start = time.time()
    nodes, G, adj = triadic.random_uniform_square_netw_PBC(N, L, c, d0)

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
    links, NL = triadic.midpoints_square_PBC(nodes, L, I, J)
    adjpos, adjneg = triadic.regulatory_network_square(nodes, links, L, dr, cpos, cneg)

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
    degrees_data = triadic.calculate_degrees(G, adj, adjpos, adjneg, density, d0, dr, c, cpos, cneg, dim=2)

    # Sanity check: The real mean degree should be approximately twice the matrix mean degree, so this bool should be
    # True if the empirical data matches the expected relationship.
    sanity_check = np.isclose(degrees_data['k_real_mean'], 2 * degrees_data['k_mat_mean'], atol=0.1)

    print("\n" + "=" * 60)
    print("--- DEGREES: THEORETICAL VS EMPIRICAL (2-DIM) ---")

    print("\n1. STRUCTURAL DEGREES <k> (Connections per node):")
    print(f"   Theoretical           : {degrees_data['k_theo']:.2f}")
    print(f"   Empirical (Real G)    : {degrees_data['k_real_mean']:.2f} +- {degrees_data['k_real_std']:.2f}  "
          f"<-- True connections")
    print(f"   Empirical (Matrix adj): {degrees_data['k_mat_mean']:.2f} +- {degrees_data['k_mat_std']:.2f}  "
          f"<-- Half (upper triangular)")
    print(f"   Sanity Check (Real ~ 2*Matrix): {sanity_check}")

    print("\n2. REGULATORY IN-DEGREES <kappa_in> (Regulators per link):")
    print(f"   Theoretical (+)       : {degrees_data['kappa_in_pos_theo']:.2f}")
    print(f"   Empirical (+)         : {degrees_data['kappa_in_pos_mean']:.2f} +- "
          f"{degrees_data['kappa_in_pos_std']:.2f}")
    print(f"   Theoretical (-)       : {degrees_data['kappa_in_neg_theo']:.2f}")
    print(f"   Empirical (-)         : {degrees_data['kappa_in_neg_mean']:.2f} +- "
          f"{degrees_data['kappa_in_neg_std']:.2f}")

    print("\n3. REGULATORY OUT-DEGREES <kappa_out> (Links regulated per node):")
    print(f"   Theoretical (+)       : {degrees_data['kappa_out_pos_theo']:.2f}")
    print(f"   Empirical (+)         : {degrees_data['kappa_out_pos_mean']:.2f} +- "
          f"{degrees_data['kappa_out_pos_std']:.2f}")
    print(f"   Theoretical (-)       : {degrees_data['kappa_out_neg_theo']:.2f}")
    print(f"   Empirical (-)         : {degrees_data['kappa_out_neg_mean']:.2f} +- "
          f"{degrees_data['kappa_out_neg_std']:.2f}")
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
            # 2D Geometric distance with PBC (Pythagoras)
            dx = triadic.distance_PBC_1D_pair(nodes[u, 0], np.array([nodes[v, 0]]), Lx)[0]
            dy = triadic.distance_PBC_1D_pair(nodes[u, 1], np.array([nodes[v, 1]]), Ly)[0]
            geom_d = np.sqrt(dx ** 2 + dy ** 2)

            topo_distances.append(topo_d)
            geom_distances.append(geom_d)

    print("-> Calculating structural link lengths...")
    # Calculate the physical distance of all structural links considering 2D PBC
    DX = triadic.distance_PBC_1D_pair(nodes[I, 0], nodes[J, 0], Lx)
    DY = triadic.distance_PBC_1D_pair(nodes[I, 1], nodes[J, 1], Ly)
    link_lengths = np.sqrt(DX**2 + DY**2)

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


    # Dynamic loop
    for it in range(1, Tmax):
        # Perform one iteration without any visualization logic
        RT[it, :], agn = triadic.itera_square(
            statenodes[it - 1, :], nodes, links, I, J, adjpos, adjneg, p
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
    np.savetxt(os.path.join(seed_dir, 'RT.txt'), RT)

    # statenodes: Boolean array (Tmax, N) tracking if a node is active (True) or inactive (False) at each step
    # Highly compressed. Saved specifically to generate the Spatiotemporal Raster Plot locally
    np.savez_compressed(os.path.join(seed_dir, 'statenodes.npz'), statenodes=statenodes)

    # nodes: Array (N, 2) with the physical 2D coordinates of each node
    # Saved to provide the Y-axis (spatial position) for the Spatiotemporal Raster Plot
    np.save(os.path.join(seed_dir, 'nodes_coords.npy'), nodes)

    # We save the fractal distribution and distances arrays compressed to save disk space
    np.savez_compressed(
        os.path.join(seed_dir, 'topology_data.npz'),
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
            "N": int(N), "Tmax": int(Tmax), "p": float(p), "density": float(density), "L": L.tolist(),
            "d0": float(d0), "dr": float(dr), "cpos": float(cpos), "cneg": float(cneg), "c": float(c),
            "seed": int(current_seed),
            "RC_factor": float(RC_factor), "geometry": geometry
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

    with open(os.path.join(seed_dir, 'summary_metrics.json'), 'w') as json_file:
        json.dump(summary_data, json_file, indent=4)

    print(f"-> Results successfully saved in: {base_dir_name}")

    seed_total_time = time.time() - seed_time
    print(f"-> Seed {current_seed} completed in {seed_total_time:.2f} seconds ({seed_total_time/60:.2f} minutes)")

    # -------------------------------------------------------------------------
    # MEMORY MANAGEMENT: Clear large arrays to free RAM for the next seed
    # -------------------------------------------------------------------------
    # Delete heavy topology and regulatory structures
    del nodes, G, adj, I, J, links, adjpos, adjneg
    del Gcc, G0, ag

    # Delete degrees and distance calculation data
    del degrees_data, path_lengths_t0, r_vals, N_r_vals
    del sample_nodes, topo_distances, geom_distances
    del DX, DY, link_lengths, max_link_per_node

    # Delete dynamic simulation arrays
    del statenodes, RT

    # Delete final steady state calculation variables
    del active_nodes_indices, G_active, largest_cc_nodes, G0_active, path_lengths_tfinal

    # Delete the summary dictionary
    del summary_data

    # Force Python's Garbage Collector to reclaim memory immediately
    gc.collect()

# Get the final simulation time to calculate the total time
finish_time = time.time()
total_time = finish_time - init_time

print(f"\n-> Results successfully saved in: {base_dir_name}")
print("\n" + "=" * 60)
print(f"   SCRIPT COMPLETED SUCCESSFULLY")
print(f"   Total Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print("=" * 60 + "\n")