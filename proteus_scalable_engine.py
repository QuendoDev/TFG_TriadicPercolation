import sys
import os
import time
import json
import random
import numpy as np
import networkx as nx
import gc

from scipy.stats import circmean, circstd
from scipy.special import k1

import triadic_library as triadic

# =============================================================================
# SECTION 1: PARAMETER READING & SEED PARSING
# =============================================================================
# This script runs the scalable "Refinement" model (Coupled Rings).
# It maintains N_total and Ly constant, scaling the number of rings (R).
# Usage: python proteus_scalable_engine.py N_total density_1D R Tmax p c cpos cneg d0_opt dr_opt [seed]
# Example: python proteus_scalable_engine.py 10000 200.0 15 400 0.85 0.07 0.03 0.03 0.45 0.32 num=5

init_time = time.time()

try:
    N_total = int(sys.argv[1])  # Total budget of nodes
    density_1D_base = float(sys.argv[2])  # Linear density (used to fix Lx)
    num_rings = int(sys.argv[3])  # R: Number of rings (Resolution)
    Tmax = int(sys.argv[4])  # Maximum simulation time
    p = float(sys.argv[5])  # Probability of a link not breaking randomly
    c = float(sys.argv[6])  # Structural base probability
    cpos = float(sys.argv[7])  # Positive regulation base probability
    cneg = float(sys.argv[8])  # Negative regulation base probability
    d0_opt = float(sys.argv[9])  # Calibrated structural decay length for this R
    dr_opt = float(sys.argv[10])  # Calibrated regulatory decay length for this R

    # SEED PARSING LOGIC
    if len(sys.argv) > 11:
        seed_arg = sys.argv[11].strip()
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
    print("Usage: python proteus_scalable_engine.py N_total density_1D R Tmax p c cpos cneg d0_opt dr_opt [seed]")
    sys.exit(1)

# Refinement Geometry Definitions
Lx = N_total / density_1D_base
Ly = Lx
delta = Ly / num_rings

# =============================================================================
# SECTION 2: DIRECTORY SETUP & LOGGER
# =============================================================================
base_dir_name = (f'results/scalable/N{N_total}_R{num_rings}_T{Tmax}_p{p:.2f}'
                 f'_c{c}_cpos{cpos}_cneg{cneg}')
os.makedirs(base_dir_name, exist_ok=True)


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


sys.stdout = Logger(os.path.join(base_dir_name, "simulation_log.txt"))

print("="*60)
print("--- LOG FILE INITIALIZED ---")

print("=" * 60)
print("--- STARTING SIMULATION (SCALABLE COUPLED RINGS) ---")
print(f"Geometry   : N_total={N_total}, density={density_1D_base}, Rings(R)={num_rings}, Lx={Lx:.2f}, Ly={Ly:.2f}, "
      f"Delta={delta:.4f}")
print(f"Dynamics   : Tmax={Tmax}, p={p}")
print(f"Calibrated : d0_opt={d0_opt:.4f}, dr_opt={dr_opt:.4f}")
print(f"Connectivities: cpos={cpos}, cneg={cneg}, c={c}")
print(f"Seeds      : {seeds}")
print(f"Base Output Dir : {base_dir_name}")
print("=" * 60)

# =============================================================================
# MAIN SIMULATION LOOP (OVER SEEDS)
# =============================================================================
for current_seed in seeds:
    seed_time = time.time()
    print(f"\n>>> Starting simulation for SEED = {current_seed} <<<")

    seed_dir = os.path.join(base_dir_name, f'seed={current_seed}')
    os.makedirs(seed_dir, exist_ok=True)
    np.random.seed(current_seed)

    # =============================================================================
    # SECTION 3: NETWORK GENERATION
    # =============================================================================
    print("\n[Phase 1] Generating scalable spatial networks...")
    net_start_time = time.time()

    # 1. Structural Network (Fixed N total)
    print('Creating structural network...')
    struct_start = time.time()
    nodes, G, adj = triadic.coupled_rings_structural_network_fixed_N(N_total, num_rings, Lx, Ly, c, d0_opt)
    edges = np.array(G.edges())
    I, J = edges[:, 0], edges[:, 1]
    NL = len(edges)

    struct_end = time.time()
    print(f'-> Structural network created in: {struct_end - struct_start:.3f} seconds.')

    # 2. Regulatory Network
    print('Creating regulatory network...')
    reg_start = time.time()
    links_mid, _ = triadic.midpoints_rings_PBC(nodes, Lx, Ly, I, J)
    adjpos, adjneg = triadic.coupled_rings_regulatory_network(nodes, links_mid, Lx, Ly, dr_opt, cpos, cneg)

    NL_pos = adjpos.nnz
    NL_neg = adjneg.nnz

    reg_end = time.time()
    print(f'-> Regulatory network created in: {reg_end - reg_start:.3f} seconds.')

    net_end_time = time.time()
    print(f"-> Networks created in {net_end_time - net_start_time:.2f} seconds.")

    print("\n" + "-" * 45)
    print("               NETWORK SUMMARY")
    print(f" Total Nodes (N)                : {N_total}")
    print(f" Experimental Nodes (N from the graph): {len(G.nodes())}")
    print(f" Total Structural Links (NL)    : {NL}")
    print(f" Total Positive Regulations (+) : {NL_pos}")
    print(f" Total Negative Regulations (-) : {NL_neg}")
    print("-" * 45 + "\n")

    # 3. Giant Component Isolation
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G0 = G.subgraph(Gcc[0])
    ag = list(Gcc[0])
    print(f"-> Initial Giant component isolated with {len(ag)} nodes.")

    # 4. Static Topo & Geometric Metrics
    print("\n[Phase 2] Calculating static topological metrics...")
    # Degrees calculation using the library
    degrees_data = triadic.calculate_degrees(G, adj, adjpos, adjneg, density_1D_base, d0_opt, dr_opt, c, cpos, cneg,
                                             dim="RINGS", num_rings=num_rings, Lx=Lx, Ly=Ly)

    # Sanity check: The real mean degree should be approximately twice the matrix mean degree, so this bool should be
    # True if the empirical data matches the expected relationship.
    sanity_check = np.isclose(degrees_data['k_real_mean'], 2 * degrees_data['k_mat_mean'], atol=0.1)

    print("\n" + "=" * 60)
    print("--- DEGREES: THEORETICAL VS EMPIRICAL (SCALABLE) ---")

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
    path_lengths_t0, avg_distance_t0 = triadic.get_topological_distances(G0, sample_size=2500)
    print(f"-> Structural baseline average distance: {avg_distance_t0:.2f} hops.")

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
    max_link_per_node = np.zeros(N_total)
    for node_idx in range(N_total):
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
    statenodes = np.zeros((Tmax, N_total), dtype=bool)
    statenodes[0, ag] = True

    # Arrays to store dynamic metrics over time (fraction of active nodes)
    RT = np.zeros((Tmax, 3))
    for i in range(min(3, len(Gcc))):
        RT[0, i] = len(Gcc[i]) / float(N_total)

    for it in range(1, Tmax):
        # Perform one iteration without any visualization logic
        RT[it, :], agn = triadic.itera_rings(
            statenodes[it - 1, :], I, J, adjpos, adjneg, p
        )

        # Activate the nodes in the new giant component
        statenodes[it, agn] = True
        if it % 10 == 0:
            print(f"   Iter {it}/{Tmax} completed")

    dyn_end_time = time.time()
    print(f"-> Dynamics finished in {dyn_end_time - dyn_start_time:.2f} seconds.")

    # Calculate final steady state distance
    print("\n[Phase 4] Calculating steady state topology...")

    # 1. Identify which nodes are active (True) at the LAST time step (Tmax-1)
    active_nodes = np.where(statenodes[-1, :])[0]

    # 2. Create a "subgraph" using only those active nodes and their base structural connections
    G_active = G.subgraph(active_nodes)

    # 3. Find the Giant Component of this active network and calculate the average distance within it
    if len(G_active.nodes()) > 0:
        largest_cc = max(nx.connected_components(G_active), key=len)
        path_lengths_tfinal, avg_distance_tfinal = triadic.get_topological_distances(G_active.subgraph(largest_cc),
                                                                                     sample_size=2500)
    else:
        largest_cc = None
        path_lengths_tfinal = []
        avg_distance_tfinal = 0.0

    print(f"-> Active distance at t=Tmax: {avg_distance_tfinal:.2f} hops.")

    # =============================================================================
    # SECTION 5: DATA EXPORT
    # =============================================================================
    print(f"\n[Phase 5] Saving results for seed = {current_seed}...")

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

    # We save the distances arrays compressed to save disk space
    np.savez_compressed(
        os.path.join(seed_dir, 'topology_data.npz'),
        path_lengths_t0=path_lengths_t0,
        path_lengths_tfinal=path_lengths_tfinal,
        topo_distances=topo_distances,
        geom_distances=geom_distances,
        link_lengths=link_lengths
    )

    # 2. Save all scalar metrics and parameters in a clean JSON file
    summary_data = {
        "parameters": {
            "N": int(N_total), "Tmax": int(Tmax), "p": float(p), "density": float(density_1D_base),
            "num_rings": num_rings, "Lx": float(Lx), "Ly": float(Ly), "delta": float(delta),
            "d0": float(d0_opt), "dr": float(dr_opt), "cpos": float(cpos), "cneg": float(cneg), "c": float(c),
            "seed": int(current_seed)
        },
        "network": {
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
    del nodes, G, adj, I, J, links_mid, adjpos, adjneg
    del Gcc, G0, ag

    # Delete degrees and distance calculation data
    del degrees_data, path_lengths_t0
    del sample_nodes, topo_distances, geom_distances
    del DX, DY, link_lengths, max_link_per_node

    # Delete dynamic simulation arrays
    del statenodes, RT

    # Delete final steady state calculation variables
    del active_nodes, G_active, largest_cc, path_lengths_tfinal

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
print(f"   Total Time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
print("=" * 60 + "\n")