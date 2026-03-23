import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

# =============================================================================
# GLOBAL ANALYSIS SETUP
# =============================================================================
print("=" * 60)
print(" GLOBAL ANALYSIS SETUP")
print("=" * 60)
print("Define the baseline parameters for the Global Plots.")
print("(Press Enter on any parameter to use the default value shown in brackets)")

try:
    print("\n" + "-" * 45)
    print(" 1D SIMULATION TARGETS")
    print("-" * 45)
    t_Tmax_1d = int(input("Target Tmax [400]: ") or 400)
    t_N_1d    = int(input("Target N for plots 1, 2, 3 and 4 [10000]: ") or 10000)
    t_c_1d    = float(input("Target c [0.07]: ") or 0.07)
    t_cpos_1d = float(input("Target cpos [0.03]: ") or 0.03)
    t_cneg_1d = float(input("Target cneg [0.03]: ") or 0.03)
    t_d0_1d   = float(input("Target d0 [0.2]: ") or 0.2)
    t_dr_1d   = float(input("Target dr [0.2]: ") or 0.2)
    t_pfss_str_1d = input("Target 'p' for Fractal FSS plot 5 (or press Enter to skip): ")
    t_pfss_1d = float(t_pfss_str_1d) if t_pfss_str_1d.strip() else None

    print("\n" + "-" * 45)
    print(" 2D SIMULATION TARGETS")
    print("-" * 45)
    t_Tmax_2d = int(input("Target Tmax [1000]: ") or 1000)
    t_N_2d    = int(input("Target N for plots 1, 2, 3 and 4 [10000]: ") or 10000)
    t_c_2d    = float(input("Target c [0.4]: ") or 0.4)
    t_cpos_2d = float(input("Target cpos [0.2]: ") or 0.2)
    t_cneg_2d = float(input("Target cneg [0.2]: ") or 0.2)
    t_d0_2d   = float(input("Target d0 [0.25]: ") or 0.25)
    t_dr_2d   = float(input("Target dr [0.25]: ") or 0.25)
    t_RC_2d = float(input("Target RC factor (Lx/Ly) [1.0]: ") or 1.0)
    t_pfss_str_2d = input("Target 'p' for Fractal FSS plot 5 (or press Enter to skip): ")
    t_pfss_2d = float(t_pfss_str_2d) if t_pfss_str_2d.strip() else None

    print("\n" + "-" * 45)
    print(" RINGS SIMULATION TARGETS")
    print("-" * 45)
    t_Tmax_rings = int(input("Target Tmax [1000]: ") or 1000)
    t_N_rings = int(input("Target N for plots 1, 2, 3 and 4 [10000]: ") or 10000)
    t_c_rings = float(input("Target c [0.4]: ") or 0.4)
    t_cpos_rings = float(input("Target cpos [0.2]: ") or 0.2)
    t_cneg_rings = float(input("Target cneg [0.2]: ") or 0.2)
    t_d0_rings = float(input("Target d0 [0.25]: ") or 0.25)
    t_dr_rings = float(input("Target dr [0.25]: ") or 0.25)
    t_num_rings = int(input("Target num_rings [2]: ") or 2)
    t_dfact_rings = float(input("Target delta_factor [0.1]: ") or 0.1)
    t_pfss_str_rings = input("Target 'p' for Fractal FSS plot 5 (or press Enter to skip): ")
    t_pfss_rings = float(t_pfss_str_rings) if t_pfss_str_rings.strip() else None

except ValueError:
    print("Invalid input! Exiting setup. Please use numeric values.")
    exit()

# =============================================================================
# CONFIGURATION
# =============================================================================
# Base directory where all 'results_1D_...' folders are located
base_dir = "./"
# Transient time (iterations to ignore for steady-state calculations)
TT = 1

# Create a multi-line plot comparing different structural/regulatory parameters?
do_param_comparison = True

# Lists for the 1D global plots
global_R_vs_p_1D = []
global_R_time_series_1D = []
all_R_fluctuations_1D = []
steady_state_distances_1D = []

# List for the all-N 1D fractal graph
global_fractal_data_1D = []

# Lists for the 2D global plots
global_R_vs_p_2D = []
global_R_time_series_2D = []
all_R_fluctuations_2D = []
steady_state_distances_2D = []

# List for the all-N 2D fractal graph
global_fractal_data_2D = []

# Lists for the RINGS global plots
global_R_vs_p_RINGS = []
global_R_time_series_RINGS = []
all_R_fluctuations_RINGS = []
steady_state_distances_RINGS = []

# List for the all-N RINGS fractal graph
global_fractal_data_RINGS = []

# Find all simulation result folders (both 1D and 2D geometries)
result_folders_all = glob.glob(os.path.join(base_dir, "results_*"))

# Filter out anything that isn't a directory (like files) or the global_figures folder
result_folders = [f for f in result_folders_all if os.path.isdir(f) and "global" not in os.path.basename(f)]

if not result_folders:
    print("No result folders found. Please check the base_dir.")
    exit()

# Count how many are 1D, RINGS, and 2D
count_1d = sum(1 for folder in result_folders if "results_1D_" in os.path.basename(folder))
count_rings = sum(1 for folder in result_folders if "results_RINGS_" in os.path.basename(folder))
count_2d = len(result_folders) - count_1d - count_rings

print("=" * 60)
print(f"Found {len(result_folders)} simulation folders in total:")
print(f"  -> 1D Simulations (Ring)     : {count_1d}")
print(f"  -> Coupled Rings (Discrete)  : {count_rings}")
print(f"  -> 2D Simulations (Toroidal) : {count_2d}")

# =============================================================================
# INDIVIDUAL FOLDER ANALYSIS LOOP
# =============================================================================
for folder in sorted(result_folders):
    print("=" * 60)
    print(f"Processing: {os.path.basename(folder)}")

    # ---------------------------------------------------------
    # 1. LOAD DATA
    # ---------------------------------------------------------
    json_path = os.path.join(folder, "summary_metrics.json")
    if not os.path.exists(json_path):
        print(f"-> Missing JSON file in {folder}. Skipping...")
        continue

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract parameters
    p = data["parameters"]["p"]
    N = data["parameters"]["N"]
    Tmax = data["parameters"]["Tmax"]
    d0 = data["parameters"]["d0"]
    dr = data["parameters"]["dr"]
    c = data["parameters"]["c"]
    cpos = data["parameters"]["cpos"]
    cneg = data["parameters"]["cneg"]

    # --- DIMENSION DETECTION & TARGET ROUTING ---
    if "geometry" in data["parameters"] and data["parameters"]["geometry"] == "RINGS":
        dim = "RINGS"
        geometry = "RINGS"
        num_rings_val = data["parameters"]["num_rings"]
        delta_factor_val = data["parameters"]["delta_factor"]
        L_list = data["parameters"]["L"]
        Lx, Ly = L_list[0], L_list[1]

        target_Tmax, target_N = t_Tmax_rings, t_N_rings
        target_c, target_cpos, target_cneg = t_c_rings, t_cpos_rings, t_cneg_rings
        target_d0, target_dr = t_d0_rings, t_dr_rings
        target_p_fss = t_pfss_rings

        match_topology = (round(c, 3) == round(target_c, 3) and
                          round(cpos, 3) == round(target_cpos, 3) and
                          round(cneg, 3) == round(target_cneg, 3) and
                          round(d0, 1) == round(target_d0, 1) and
                          round(dr, 3) == round(target_dr, 3) and
                          Tmax == target_Tmax and
                          num_rings_val == t_num_rings and
                          round(delta_factor_val, 2) == round(t_dfact_rings, 2))

    elif "geometry" in data["parameters"]:
        dim = 2
        geometry = data["parameters"]["geometry"]
        RC_factor = data["parameters"]["RC_factor"]
        # In 2D, L is saved as a list [Lx, Ly]
        L_list = data["parameters"]["L"]
        Lx, Ly = L_list[0], L_list[1]

        # Load the 2D targets the user inputted
        target_Tmax, target_N = t_Tmax_2d, t_N_2d
        target_c, target_cpos, target_cneg = t_c_2d, t_cpos_2d, t_cneg_2d
        target_d0, target_dr = t_d0_2d, t_dr_2d
        target_RC = t_RC_2d
        target_p_fss = t_pfss_2d

        match_topology = (round(c, 3) == round(target_c, 3) and
                          round(cpos, 3) == round(target_cpos, 3) and
                          round(cneg, 3) == round(target_cneg, 3) and
                          round(d0, 1) == round(target_d0, 1) and
                          round(dr, 3) == round(target_dr, 3) and
                          Tmax == target_Tmax and
                          round(RC_factor, 2) == round(target_RC, 2))
    else:
        dim = 1
        geometry = "1D"
        RC_factor = 1.0
        # In 1D, L is just a float
        L = data["parameters"]["L"]

        # Load the 1D targets the user inputted
        target_Tmax, target_N = t_Tmax_1d, t_N_1d
        target_c, target_cpos, target_cneg = t_c_1d, t_cpos_1d, t_cneg_1d
        target_d0, target_dr = t_d0_1d, t_dr_1d
        target_RC = 1.0
        target_p_fss = t_pfss_1d

        match_topology = (round(c, 3) == round(target_c, 3) and
                          round(cpos, 3) == round(target_cpos, 3) and
                          round(cneg, 3) == round(target_cneg, 3) and
                          round(d0, 1) == round(target_d0, 1) and
                          round(dr, 3) == round(target_dr, 3) and
                          Tmax == target_Tmax)

    # Extract metrics
    deg_sum = data["degrees_summary"]
    deg_arr = data["degrees_arrays"]

    # Load dynamic arrays
    RT = np.loadtxt(os.path.join(folder, "RT.txt"))

    try:
        spatial_dyn = np.loadtxt(os.path.join(folder, "spatial_dyn.txt"))
    except FileNotFoundError:
        # In 2D it is not necessary
        spatial_dyn = None

    # Load topology compressed data
    try:
        topo_data = np.load(os.path.join(folder, "topology_data.npz"))
    except FileNotFoundError:
        print("-> Missing topology_data.npz. Skipping topology plots.")
        topo_data = None

    # Load spatial compressed data (for raster plots)
    try:
        statenodes = np.load(os.path.join(folder, "statenodes.npz"))['statenodes']
        nodes_coords = np.load(os.path.join(folder, "nodes_coords.npy"))
    except FileNotFoundError:
        print("-> Missing spatial data. Skipping Raster Plot.")
        statenodes = None

    # Extract the fluctuations of R after the transient time TT for the violin plot and append to the global list
    # with the corresponding value of p for each R value. That 0 is from the giant component
    steady_R = RT[TT:, 0]

    if match_topology and N == target_N:
        if dim == 1:
            for r_val in steady_R:
                all_R_fluctuations_1D.append({'p': p, 'R': r_val})

            steady_state_distances_1D.append({
                'p': p,
                'dist_t0': data["distances"]["mean_hops_t0"],
                'dist_tfinal': data["distances"]["mean_hops_tfinal"]
            })

            global_R_time_series_1D.append({
                'p': p,
                'time': np.arange(len(RT[:, 0])),
                'R': RT[:, 0]
            })
        elif dim == "RINGS":
            for r_val in steady_R:
                all_R_fluctuations_RINGS.append({'p': p, 'R': r_val})

            steady_state_distances_RINGS.append({
                'p': p,
                'dist_t0': data["distances"]["mean_hops_t0"],
                'dist_tfinal': data["distances"]["mean_hops_tfinal"]
            })

            global_R_time_series_RINGS.append({
                'p': p,
                'time': np.arange(len(RT[:, 0])),
                'R': RT[:, 0]
            })
        elif dim == 2:
            for r_val in steady_R:
                all_R_fluctuations_2D.append({'p': p, 'R': r_val})

            steady_state_distances_2D.append({
                'p': p,
                'dist_t0': data["distances"]["mean_hops_t0"],
                'dist_tfinal': data["distances"]["mean_hops_tfinal"]
            })

            global_R_time_series_2D.append({
                'p': p,
                'time': np.arange(len(RT[:, 0])),
                'R': RT[:, 0]
            })

    # Append global data for the R vs p parameter comparison plot
    if Tmax == target_Tmax and N == target_N:
        if len(steady_R) > 0:
            if dim == 1:
                global_R_vs_p_1D.append({
                    'p': p,
                    'R_mean': np.mean(steady_R),
                    'c': c, 'cpos': cpos, 'cneg': cneg, 'd0': d0, 'dr': dr
                })
            elif dim == "RINGS":
                global_R_vs_p_RINGS.append({
                    'p': p,
                    'R_mean': np.mean(steady_R),
                    'c': c, 'cpos': cpos, 'cneg': cneg, 'd0': d0, 'dr': dr
                })
            elif dim == 2:
                global_R_vs_p_2D.append({
                    'p': p,
                    'R_mean': np.mean(steady_R),
                    'c': c, 'cpos': cpos, 'cneg': cneg, 'd0': d0, 'dr': dr
                })

    # Create an output folder for figures inside the result folder to keep things organized
    fig_dir = os.path.join(folder, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 2. DEGREE DISTRIBUTION HISTOGRAMS
    # ---------------------------------------------------------
    print("-> Generating Degree Distributions...")

    # Structural Degree (physically connected nodes)
    plt.figure(figsize=(6, 4))
    sns.histplot(deg_arr['k_real_array'], bins=30, color='skyblue', kde=True, stat='density')
    plt.axvline(deg_sum['k_theo'], color='red', linestyle='--', linewidth=2,
                label=f"Theo Mean: {deg_sum['k_theo']:.1f}")
    plt.title("Structural Degree Distribution P(k)")
    plt.xlabel("Degree k (Connections per node)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"hist_structural_degree_p{p:.2f}.png"), dpi=400)
    plt.close()

    # Regulatory Out-Degree (nodes regulating links)
    plt.figure(figsize=(6, 4))
    min_out = min(min(deg_arr['kappa_out_pos_array']), min(deg_arr['kappa_out_neg_array']))
    max_out = max(max(deg_arr['kappa_out_pos_array']), max(deg_arr['kappa_out_neg_array']))
    mis_bins_out = np.linspace(min_out, max_out, 30)
    sns.histplot(deg_arr['kappa_out_pos_array'], bins=mis_bins_out, color='green', alpha=0.5, kde=True, stat='density',
                 label='Positive (+)')
    sns.histplot(deg_arr['kappa_out_neg_array'], bins=mis_bins_out, color='red', alpha=0.5, kde=True, stat='density',
                 label='Negative (-)')
    plt.axvline(deg_sum['kappa_out_pos_theo'], color='darkgreen', linestyle='--', linewidth=2, label='Theo. Mean (+)')
    plt.axvline(deg_sum['kappa_out_neg_theo'], color='darkred', linestyle='--', linewidth=2, label='Theo. Mean (-)')
    plt.title("Regulatory Out-Degree Distribution P(k_out)")
    plt.xlabel("Out-Degree (Nodes regulating links)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"hist_regulatory_out_degree_p{p:.2f}.png"), dpi=400)
    plt.close()

    # Regulatory In-Degree (links regulated by nodes)
    plt.figure(figsize=(6, 4))
    min_out = min(min(deg_arr['kappa_in_pos_array']), min(deg_arr['kappa_in_neg_array']))
    max_out = max(max(deg_arr['kappa_in_pos_array']), max(deg_arr['kappa_in_neg_array']))
    mis_bins_out = np.linspace(min_out, max_out, 30)
    sns.histplot(deg_arr['kappa_in_pos_array'], bins=mis_bins_out, color='green', alpha=0.5, kde=True, stat='density',
                 label='Positive (+)')
    sns.histplot(deg_arr['kappa_in_neg_array'], bins=mis_bins_out, color='red', alpha=0.5, kde=True, stat='density',
                 label='Negative (-)')
    plt.axvline(deg_sum['kappa_in_pos_theo'], color='darkgreen', linestyle='--', linewidth=2, label='Theo. Mean (+)')
    plt.axvline(deg_sum['kappa_in_neg_theo'], color='darkred', linestyle='--', linewidth=2, label='Theo. Mean (-)')
    plt.title("Regulatory In-Degree Distribution P(k_in)")
    plt.xlabel("In-Degree (Links regulated by nodes)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"hist_regulatory_in_degree_p{p:.2f}.png"), dpi=400)
    plt.close()

    # ---------------------------------------------------------
    # 3. TOPOLOGY & DISTANCE PLOTS
    # ---------------------------------------------------------
    if topo_data is not None:
        print("-> Generating Topology & Fractal Plots...")

        path_lengths_t0 = topo_data['path_lengths_t0']
        path_lengths_tfinal = topo_data['path_lengths_tfinal']
        mean_hops_t0 = data["distances"]["mean_hops_t0"]
        mean_hops_tfinal = data["distances"]["mean_hops_tfinal"]

        # A.1. Topological Distance Histogram (t=0)
        if len(path_lengths_t0) > 0:
            plt.figure(figsize=(6, 4))
            max_hop_t0 = max(path_lengths_t0)
            bins_t0 = np.arange(0.5, max_hop_t0 + 1.5, 1)
            plt.hist(path_lengths_t0, bins=bins_t0, color='mediumpurple', edgecolor='black', alpha=0.7, density=True)
            plt.axvline(mean_hops_t0, color='red', linestyle='dashed', linewidth=2,
                        label=f'Mean (t=0): {mean_hops_t0:.2f}')
            plt.title("Topological Distance Distribution at t=0")
            plt.xlabel("Number of Hops (Topological Distance)")
            plt.ylabel("Probability Density")
            plt.xticks(range(1, max_hop_t0 + 1))
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f"hist_topological_distances_t0_p{p:.2f}.png"), dpi=400)
            plt.close()

        # A.2. Topological Distance Histogram (t=Tmax)
        if len(path_lengths_tfinal) > 0:
            plt.figure(figsize=(6, 4))
            max_hop_tfinal = max(path_lengths_tfinal)
            bins_tfinal = np.arange(0.5, max_hop_tfinal + 1.5, 1)
            plt.hist(path_lengths_tfinal, bins=bins_tfinal, color='darkorange', edgecolor='black', alpha=0.7,
                     density=True)
            plt.axvline(mean_hops_tfinal, color='red', linestyle='dashed', linewidth=2,
                        label=f'Mean (t=Tmax): {mean_hops_tfinal:.2f}')
            plt.title("Active Topological Distance Distribution at t=Tmax")
            plt.xlabel("Number of Hops (Topological Distance)")
            plt.ylabel("Probability Density")
            plt.xticks(range(1, max_hop_tfinal + 1))
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f"hist_topological_distances_tfinal_p{p:.2f}.png"), dpi=400)
            plt.close()

        # A.3. Topological Distance Histogram (Combined t=0 vs t=Tmax)
        if len(path_lengths_t0) > 0 and len(path_lengths_tfinal) > 0:
            plt.figure(figsize=(6, 4))

            # Find the global max hop to align bins perfectly for both distributions
            global_max_hop = max(max(path_lengths_t0), max(path_lengths_tfinal))
            global_bins = np.arange(0.5, global_max_hop + 1.5, 1)

            # Plot both histograms with transparency
            plt.hist(path_lengths_t0, bins=global_bins, color='mediumpurple', edgecolor='black', alpha=0.5,
                     density=True, label='t=0 (Structural)')
            plt.hist(path_lengths_tfinal, bins=global_bins, color='darkorange', edgecolor='black', alpha=0.6,
                     density=True, label='t=Tmax (Active)')

            # Add mean lines with labels for the legend
            plt.axvline(mean_hops_t0, color='purple', linestyle='dashed', linewidth=2,
                        label=f'Mean t=0 ({mean_hops_t0:.1f})')
            plt.axvline(mean_hops_tfinal, color='orangered', linestyle='dashed', linewidth=2,
                        label=f'Mean t=Tmax ({mean_hops_tfinal:.1f})')

            plt.title("Topological Distance Shift (Structural vs Active)")
            plt.xlabel("Number of Hops (Topological Distance)")
            plt.ylabel("Probability Density")
            plt.xticks(range(1, global_max_hop + 1))
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f"hist_topological_distances_combined_p{p:.2f}.png"), dpi=400)
            plt.close()

        # B. Link Length Distribution (Exponential Decay verification)
        link_lengths = topo_data['link_lengths']
        plt.figure(figsize=(6, 4))
        sns.histplot(link_lengths, bins=40, color='coral', stat='density', alpha=0.6, label='Empirical Data')
        # Overlay theoretical exponential decay: P(d) ~ exp(-d/d0)
        x_vals = np.linspace(0, max(link_lengths), 1000)
        # Overlay theoretical decay based on dimensionality
        if dim == 1:
            # 1D: Pure exponential decay
            y_vals = (1 / d0) * np.exp(-x_vals / d0)
            plt.plot(x_vals, y_vals, color='red', linewidth=2, linestyle='--', label=r'Theory: $\sim e^{-d/d_0}$')
        elif dim == 2 or dim == "RINGS":
            # 2D: Probability proportional to ring area (d) * exponential decay
            # The normalized PDF is (x / d0^2) * exp(-x/d0)
            y_vals = (x_vals / (d0 ** 2)) * np.exp(-x_vals / d0)
            plt.plot(x_vals, y_vals, color='red', linewidth=2, linestyle='--',
                     label=r'Theory: $\sim d \cdot e^{-d/d_0}$')
        plt.title("Physical Link Length Distribution")
        plt.xlabel("Physical Distance (Geometry)")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"hist_link_lengths_p{p:.2f}.png"), dpi=400)
        plt.close()

        # C. Fractal Dimension (Hausdorff) with slope comparisons
        r_vals = np.asarray(topo_data['r_vals'])
        N_r_vals = np.asarray(topo_data['N_r_vals'])

        # 1. Clean the data (remove the flat line)
        valid_indices = (r_vals > 0) & (N_r_vals > 0)
        r_clean = r_vals[valid_indices]
        N_clean = N_r_vals[valid_indices]

        # Find the exact moment the network saturates (reaches maximum nodes)
        # and cut the arrays there so we don't plot an infinite flat line.
        first_max_idx = np.argmax(N_clean)
        r_clean = r_clean[:first_max_idx + 1]
        N_clean = N_clean[:first_max_idx + 1]

        log_r = np.log10(r_clean)
        log_N = np.log10(N_clean)

        if target_p_fss is not None and round(p, 3) == round(target_p_fss, 3) and match_topology:
            if dim == 1:
                global_fractal_data_1D.append({
                    'N': N,
                    'log_r': log_r,
                    'log_N': log_N
                })
            elif dim == "RINGS":
                global_fractal_data_RINGS.append({
                    'N': N,
                    'log_r': log_r,
                    'log_N': log_N
                })
            elif dim == 2:
                global_fractal_data_2D.append({
                    'N': N,
                    'log_r': log_r,
                    'log_N': log_N
                })

        plt.figure(figsize=(6, 4))
        plt.plot(log_r, log_N, marker='o', linestyle='-', color='teal', linewidth=2, label='Simulation Data')

        if len(log_r) >= 3:
            x_start, y_start = log_r[0], log_N[0]
            x_final, y_final = log_r[-1], log_N[-1]

            # 2. Calculate crossover point mathematically
            # Line early: y = 4 * (x - x_start) + y_start
            # Line late:  y = 1 * (x - x_final) + y_final
            # Intersection (3x = 4*x_start - x_final + y_final - y_start):
            x_cross = (4 * x_start - x_final + y_final - y_start) / 3.0
            y_cross = 4.0 * (x_cross - x_start) + y_start

            # 3. Early theory (slope = 4)
            # Extend from the first point to slightly past the crossover
            x_early = np.array([x_start, x_cross + 0.1])
            y_early = 4.0 * (x_early - x_start) + y_start
            plt.plot(x_early, y_early, color='orange', linestyle='--', linewidth=2.5,
                     label=r'Early Theory ($d_H=4$)')

            # 4. Asymptotic theory (slope = 1)
            # Extend from slightly before the crossover to the final point
            x_late = np.array([max(0, x_cross - 0.1), x_final])
            y_late = 1.0 * (x_late - x_final) + y_final
            plt.plot(x_late, y_late, color='red', linestyle='--', linewidth=2.5,
                     label=r'Asymptotic Theory ($d_H=1$)')

            # Mark the crossover point
            plt.plot(x_cross, y_cross, 'ko', markersize=5, label=r'Crossover Scale ($r_\times$)')

        plt.title("Fractal Dimension Scaling (Hausdorff)")
        plt.xlabel(r"$\log_{10}(r)$ [Topological Radius / Hops]")
        plt.ylabel(r"$\log_{10}(N(r))$ [Cumulative Mass / Nodes]")
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"fractal_dimension_p{p:.2f}.png"), dpi=400)
        plt.close()

        # D. Topological vs Geometric Distance Plot
        topo_dist = topo_data['topo_distances']
        geom_dist = topo_data['geom_distances']
        plt.figure(figsize=(6, 4))
        plt.scatter(geom_dist, topo_dist, color='purple', alpha=0.3, s=10)

        # Calculate moving average
        bins = np.linspace(0, max(geom_dist), 15)
        bin_means = []
        for i in range(len(bins) - 1):
            mask = (geom_dist >= bins[i]) & (geom_dist < bins[i + 1])
            if np.any(mask):
                bin_means.append(np.mean(topo_dist[mask]))
            else:
                bin_means.append(np.nan)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        plt.plot(bin_centers, bin_means, color='red', linewidth=3, label='Average Trend')
        plt.title("Topological vs Geometric Distance")
        plt.xlabel("Geometric Distance (Physical Space)")
        plt.ylabel("Topological Distance (Hops)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"topo_vs_geom_p{p:.2f}.png"), dpi=400)
        plt.close()

    # ---------------------------------------------------------
    # 4. TIME DYNAMICS PLOTS
    # ---------------------------------------------------------
    print("-> Generating Time Dynamics Plots...")

    # A. Giant Component Evolution
    plt.figure(figsize=(6, 4))
    plt.title(f"Fraction of Active Nodes Over Time (p={p:.2f})")
    plt.plot(RT[:, 0], label='Cluster 1 (Giant)', color='magenta')
    if RT.shape[1] > 1: plt.plot(RT[:, 1], label='Cluster 2', color='green')
    if RT.shape[1] > 2: plt.plot(RT[:, 2], label='Cluster 3', color='blue')
    plt.xlabel('Time')
    plt.ylabel('Fraction of active nodes')
    plt.xlim(0, Tmax)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='-', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"Giant_evolution_p{p:.2f}.png"), dpi=400)
    plt.close()

    if dim == 1 or dim == "RINGS":
        # B.1. Phase Space (R vs Average Angle for Cluster 1)
        plt.figure(figsize=(6, 4))
        plt.title("Phase Space: Size vs Angular Position (Cluster 1)")
        sc1 = plt.scatter(spatial_dyn[:, 0], RT[:, 0], c=np.arange(Tmax), cmap='viridis', s=15, alpha=0.8)
        plt.colorbar(sc1, label='Time (iterations)')
        plt.xlabel('Average Angular Position (rad)')
        plt.ylabel('R (Fraction of active nodes)')
        plt.xlim(0, 2 * np.pi)
        plt.ylim(0, 1.0)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"PhaseSpace_Cluster1_p{p:.2f}.png"), dpi=400)
        plt.close()

        # B.2. Phase Space (R vs Average Angle for Cluster 2)
        plt.figure(figsize=(6, 4))
        plt.title("Phase Space: Size vs Angular Position (Cluster 2)")
        sc2 = plt.scatter(spatial_dyn[:, 2], RT[:, 1], c=np.arange(Tmax), cmap='plasma', s=15, alpha=0.8)
        plt.colorbar(sc2, label='Time (iterations)')
        plt.xlabel('Average Angular Position (rad)')
        plt.ylabel('R (Fraction of active nodes)')
        plt.xlim(0, 2 * np.pi)
        max_r2 = np.max(RT[:, 1])
        plt.ylim(0, max_r2 + 0.05 if max_r2 > 0 else 0.1)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"PhaseSpace_Cluster2_p{p:.2f}.png"), dpi=400)
        plt.close()

        # B.3. Phase Space (R vs Average Angle for Cluster 3)
        plt.figure(figsize=(6, 4))
        plt.title("Phase Space: Size vs Angular Position (Cluster 3)")
        sc3 = plt.scatter(spatial_dyn[:, 4], RT[:, 2], c=np.arange(Tmax), cmap='cool', s=15, alpha=0.8)
        plt.colorbar(sc3, label='Time (iterations)')
        plt.xlabel('Average Angular Position (rad)')
        plt.ylabel('R (Fraction of active nodes)')
        plt.xlim(0, 2 * np.pi)
        max_r3 = np.max(RT[:, 2])
        plt.ylim(0, max_r3 + 0.05 if max_r3 > 0 else 0.1)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"PhaseSpace_Cluster3_p{p:.2f}.png"), dpi=400)
        plt.close()

        # B.4. Phase Space (R vs Average Angle for all 3 Clusters)
        plt.figure(figsize=(6, 4))
        plt.title("Phase Space: Size vs Angular Position (All Clusters)")
        # 1. Draw faint trajectory lines to clearly show the path over time
        plt.plot(spatial_dyn[:, 0], RT[:, 0], color='purple', alpha=0.3, linewidth=1, zorder=1)
        if RT.shape[1] > 1:
            plt.plot(spatial_dyn[:, 2], RT[:, 1], color='green', alpha=0.3, linewidth=1, zorder=1)
        if RT.shape[1] > 2:
            plt.plot(spatial_dyn[:, 4], RT[:, 2], color='blue', alpha=0.3, linewidth=1, zorder=1)

        # 2. Scatter points using monochromatic sequential colormaps
        # Time is represented by how dark the color is (light -> dark)
        sc1 = plt.scatter(spatial_dyn[:, 0], RT[:, 0], c=np.arange(Tmax), cmap='Purples',
                          s=25, edgecolor='black', linewidth=0.5, alpha=0.9, zorder=2)
        if RT.shape[1] > 1:
            sc2 = plt.scatter(spatial_dyn[:, 2], RT[:, 1], c=np.arange(Tmax), cmap='Greens',
                              s=25, edgecolor='black', linewidth=0.5, alpha=0.9, zorder=2)
        if RT.shape[1] > 2:
            sc3 = plt.scatter(spatial_dyn[:, 4], RT[:, 2], c=np.arange(Tmax), cmap='Blues',
                              s=25, edgecolor='black', linewidth=0.5, alpha=0.9, zorder=2)

        # 3. Add a unified, neutral colorbar for Time
        # We create a dummy Greys scalar mappable so it doesn't favor any specific cluster color
        sm = plt.cm.ScalarMappable(cmap='Greys', norm=plt.Normalize(vmin=0, vmax=Tmax))
        cbar = plt.colorbar(sm, ax=plt.gca())
        cbar.set_label('Time (Iterations: Light -> Dark)')

        # 4. Create a custom legend for the clusters
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=8,
                   label='Cluster 1 (Giant)'),
        ]
        if RT.shape[1] > 1:
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8,
                       label='Cluster 2'))
        if RT.shape[1] > 2:
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8,
                       label='Cluster 3'))

        plt.legend(handles=legend_elements, loc='upper right')

        plt.xlabel('Average Angular Position (rad)')
        plt.ylabel('R (Fraction of active nodes)')
        plt.xlim(0, 2 * np.pi)
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"PhaseSpace_All_Cluster_p{p:.2f}.png"), dpi=400)
        plt.close()

    # C. Spatiotemporal Raster Plot
    if statenodes is not None:
        if dim == 1:
            plt.figure(figsize=(8, 5))

            # np.where searches automatically where statenodes is True
            # It returns two arrays: one with the time indices and another with the node indices of the active nodes
            active_times, active_node_indices = np.where(statenodes)

            # We extract the exact spatial position 'y' of those active nodes
            y_positions = nodes_coords[active_node_indices, 0]

            # Draw all the points at the same time
            # s=0.5 makes the points smaller, and alpha=0.8 gives them some transparency to better visualize overlaps
            plt.scatter(active_times, y_positions, color='black', s=0.5, alpha=0.8)
            plt.title(f"Spatiotemporal Evolution (p={p:.2f})")
            plt.xlabel('Time (iterations)')
            plt.ylabel('Spatial Position (L)')
            plt.xlim(0, Tmax)
            plt.ylim(0, L)
            plt.grid(True, linestyle='-', alpha=0.3, color='lightgray')
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f"raster_plot_p{p:.2f}.png"), dpi=400)
            plt.close()

        elif dim == 2 or dim == "RINGS":
            plt.figure(figsize=(6, 6))

            # We create a heatmap based on the frequency of activation of each node
            activation_freq = np.sum(statenodes, axis=0) / float(Tmax)

            sc = plt.scatter(nodes_coords[:, 0], nodes_coords[:, 1], c=activation_freq, cmap='magma', s=3, alpha=0.6)
            plt.colorbar(sc, label='Activation Frequency')
            plt.title(f"2D Spatial Activity Heatmap (p={p:.2f})")
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.xlim(0, Lx)
            plt.ylim(0, Ly)
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f"heatmap_2D_p{p:.2f}.png"), dpi=400)
            plt.close()

    if dim == 1 or dim == "RINGS":
        # D.1. Spatial position tracking (angular position vs time) of Cluster 1
        plt.figure(figsize=(6, 4))
        plt.title("Spatial Position Over Time (Cluster 1)")
        plt.plot(spatial_dyn[:, 0], 'mo', markersize=3, label='Cluster 1 (Giant)')
        plt.xlabel('Time (iterations)')
        plt.ylabel('Average Angular Position (rad)')
        plt.ylim(0, 2 * np.pi)
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"Position_Time_Cluster1_p{p:.2f}.png"), dpi=400)
        plt.close()

        # D.2. Spatial position tracking (angular position vs time) of Cluster 2
        if spatial_dyn.shape[1] > 2:
            plt.figure(figsize=(6, 4))
            plt.title("Spatial Position Over Time (Cluster 2)")
            plt.plot(spatial_dyn[:, 2], 'go', markersize=3, label='Cluster 2')
            plt.xlabel('Time (iterations)')
            plt.ylabel('Average Angular Position (rad)')
            plt.ylim(0, 2 * np.pi)
            plt.grid(True, linestyle=':', alpha=0.5)
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f"Position_Time_Cluster2_p{p:.2f}.png"), dpi=400)
            plt.close()

        # D.3. Spatial position tracking (angular position vs time) of Cluster 3
        if spatial_dyn.shape[1] > 4:
            plt.figure(figsize=(6, 4))
            plt.title("Spatial Position Over Time (Cluster 3)")
            plt.plot(spatial_dyn[:, 4], 'bo', markersize=3, label='Cluster 3')
            plt.xlabel('Time (iterations)')
            plt.ylabel('Average Angular Position (rad)')
            plt.ylim(0, 2 * np.pi)
            plt.grid(True, linestyle=':', alpha=0.5)
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f"Position_Time_Cluster3_p{p:.2f}.png"), dpi=400)
            plt.close()

        # D.4. Spatial position tracking (angular position vs time) of all 3 Clusters
        plt.figure(figsize=(6, 4))
        plt.title("Spatial Position of All Clusters")
        plt.plot(spatial_dyn[:, 0], 'mo', markersize=3, label='Cluster 1 (Giant)')
        if spatial_dyn.shape[1] > 2:
            plt.plot(spatial_dyn[:, 2], 'go', markersize=3, label='Cluster 2')
        if spatial_dyn.shape[1] > 4:
            plt.plot(spatial_dyn[:, 4], 'bo', markersize=3, label='Cluster 3')
        plt.xlabel('Time (iterations)')
        plt.ylabel('Average Angular Position (rad)')
        plt.ylim(0, 2 * np.pi)
        plt.grid(True, linestyle='-', alpha=0.3)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"Position_Time_All_p{p:.2f}.png"), dpi=400)
        plt.close()

        # E.1. Spatial width tracking (pattern width vs time) of Cluster 1
        plt.figure(figsize=(6, 4))
        plt.title("Spatial Width Over Time (Cluster 1)")
        plt.plot(spatial_dyn[:, 1], 'm-', linewidth=2, label='Cluster 1 (Giant)')
        plt.xlabel('Time (iterations)')
        plt.ylabel('Pattern Width (Circular Std)')
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"Width_Time_Cluster1_p{p:.2f}.png"), dpi=400)
        plt.close()

        # E.2. Spatial width tracking (pattern width vs time) of Cluster 2
        if spatial_dyn.shape[1] > 3:
            plt.figure(figsize=(6, 4))
            plt.title("Spatial Width Over Time (Cluster 2)")
            plt.plot(spatial_dyn[:, 3], 'g-', linewidth=2, label='Cluster 2')
            plt.xlabel('Time (iterations)')
            plt.ylabel('Pattern Width (Circular Std)')
            plt.grid(True, linestyle=':', alpha=0.5)
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f"Width_Time_Cluster2_p{p:.2f}.png"), dpi=400)
            plt.close()

        # E.3. Spatial width tracking (pattern width vs time) of Cluster 3
        if spatial_dyn.shape[1] > 5:
            plt.figure(figsize=(6, 4))
            plt.title("Spatial Width Over Time (Cluster 3)")
            plt.plot(spatial_dyn[:, 5], 'b-', linewidth=2, label='Cluster 3')
            plt.xlabel('Time (iterations)')
            plt.ylabel('Pattern Width (Circular Std)')
            plt.grid(True, linestyle=':', alpha=0.5)
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f"Width_Time_Cluster3_p{p:.2f}.png"), dpi=400)
            plt.close()

        # E.4. Spatial width tracking (pattern width vs time) of all 3 Clusters
        plt.figure(figsize=(6, 4))
        plt.title("Spatial Width of All Clusters")
        plt.plot(spatial_dyn[:, 1], 'm-', linewidth=2, label='Cluster 1 (Giant)')
        if spatial_dyn.shape[1] > 3:
            plt.plot(spatial_dyn[:, 3], 'g-', linewidth=2, label='Cluster 2')
        if spatial_dyn.shape[1] > 5:
            plt.plot(spatial_dyn[:, 5], 'b-', linewidth=2, label='Cluster 3')
        plt.xlabel('Time (iterations)')
        plt.ylabel('Pattern Width (Circular Std)')
        plt.grid(True, linestyle='-', alpha=0.3)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"Width_Time_All_p{p:.2f}.png"), dpi=400)
        plt.close()

# =============================================================================
# GLOBAL ANALYSIS (Comparisons across different 'p' values)
# =============================================================================
print("\n" + "=" * 60)
print("Generating Global Comparative Plots...")

# Dictionary to easily loop over both dimensions without duplicating plotting code
global_data_collections = {
    "1D": {
        "time_series": global_R_time_series_1D,
        "fluctuations": all_R_fluctuations_1D,
        "distances": steady_state_distances_1D,
        "R_vs_p": global_R_vs_p_1D,
        "fractal": global_fractal_data_1D
    },
    "RINGS": {
        "time_series": global_R_time_series_RINGS,
        "fluctuations": all_R_fluctuations_RINGS,
        "distances": steady_state_distances_RINGS,
        "R_vs_p": global_R_vs_p_RINGS,
        "fractal": global_fractal_data_RINGS
    },
    "2D": {
        "time_series": global_R_time_series_2D,
        "fluctuations": all_R_fluctuations_2D,
        "distances": steady_state_distances_2D,
        "R_vs_p": global_R_vs_p_2D,
        "fractal": global_fractal_data_2D
    }
}

for current_dim, g_data in global_data_collections.items():
    # Skip if no data exists for this dimension
    if not g_data["time_series"] and not g_data["R_vs_p"] and not g_data["fractal"]:
        continue

    print(f"\n-> Generating global comparative plots for {current_dim}...")
    global_fig_dir = os.path.join(base_dir, f"global_figures_{current_dim}")
    os.makedirs(global_fig_dir, exist_ok=True)

    # Get the specific targets used for this dimension (for titles)
    if current_dim == "1D":
        dim_N, dim_p_fss = t_N_1d, t_pfss_1d
    elif current_dim == "RINGS":
        dim_N, dim_p_fss = t_N_rings, t_pfss_rings
    else:
        dim_N, dim_p_fss = t_N_2d, t_pfss_2d

    # 1. COMPARATIVE TIME EVOLUTION OF R FOR ALL THE GENERATED P
    if g_data["time_series"]:
        print("-> Generating global Time Dynamics comparison...")
        plt.figure(figsize=(8, 5))

        # Sort the series by 'p' so the legend appears ordered
        sorted_ts = sorted(g_data["time_series"], key=lambda x: x['p'])

        # Create a color palette to differentiate 'p' values
        colors = sns.color_palette("viridis", n_colors=len(sorted_ts))

        for i, data_series in enumerate(sorted_ts):
            plt.plot(data_series['time'], data_series['R'], color=colors[i],
                     linewidth=2, label=f"p = {data_series['p']:.2f}")

        plt.title(f"Evolution of the Giant Component over Time ({current_dim}, N={dim_N})")
        plt.xlabel("Time (iterations)")
        plt.ylabel("Fraction of active nodes (R)")
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle='--', alpha=0.5)

        # Move the legend outside the plot so it doesn't overlap the curves
        plt.legend(title="Control Parameter", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(global_fig_dir, "global_R_vs_Time_comparison.png"), dpi=400)
        plt.close()

    # 2. VIOLIN PLOT FOR R FLUCTUATIONS IN 1 DIMENSION
    if g_data["fluctuations"]:
        print("-> Generating global Fluctuations comparison...")
        df_fluctuations = pd.DataFrame(g_data["fluctuations"])
        plt.figure(figsize=(8, 5))

        sns.violinplot(data=df_fluctuations, x='p', y='R', inner=None, color='lightgray',
                       alpha=0.5, linewidth=1, density_norm='width')
        sns.stripplot(data=df_fluctuations, x='p', y='R', size=2.5, alpha=0.6, jitter=True, palette='viridis', hue='p',
                      legend=False)

        plt.title(f"Steady State Fluctuations of R ({current_dim}, N={dim_N})")
        plt.xlabel("Control Parameter (p)")
        plt.ylabel("Size of the Giant Component (R)")
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(global_fig_dir, "global_violin_fluctuations_R.png"), dpi=400)
        plt.close()

    # 3. DISTANCE VS P PLOT IN 1 DIMENSION
    if g_data["distances"]:
        print("-> Generating global Distance vs p comparison plot...")
        df_distances = pd.DataFrame(g_data["distances"]).sort_values(by='p')
        plt.figure(figsize=(6, 4))

        # Plotting baseline t=0 (taking the mean of all t=0 distances as reference)
        avg_baseline = df_distances['dist_t0'].mean()
        plt.axhline(avg_baseline, color='gray', linestyle='--', linewidth=2,
                    label=f'Structural baseline (t=0): {avg_baseline:.2f}')

        # Plotting steady state distances
        plt.plot(df_distances['p'], df_distances['dist_tfinal'], marker='o', color='darkorange', linestyle='-',
                 linewidth=2, markersize=6, label='Steady State (t=Tmax)')

        plt.title(f"Steady State Topological Distance vs p ({current_dim}, N={dim_N})")
        plt.xlabel("Control Parameter (p)")
        plt.ylabel("Average Shortest Path (Hops)")
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(global_fig_dir, "global_distance_vs_p.png"), dpi=400)
        plt.close()

    # 4. R VS P COMPARISON PLOT (MULTIPLE LINES FOR DIFFERENT PARAMETERS)
    if do_param_comparison and g_data["R_vs_p"]:
        print("-> Generating R vs p parameter comparison plot...")
        df_params = pd.DataFrame(g_data["R_vs_p"])

        # Automatically detect which parameters actually vary in your folders (excluding 'p' and 'R_mean')
        param_cols = ['c', 'cpos', 'cneg', 'd0', 'dr']
        varying_params = [col for col in param_cols if df_params[col].nunique() > 1]

        plt.figure(figsize=(8, 6))

        if not varying_params:
            # If no parameters varied (you only tested different p's for one network setup)
            grouped = df_params.groupby('p')['R_mean'].mean().reset_index()
            plt.plot(grouped['p'], grouped['R_mean'], marker='o', linewidth=2, color='magenta',
                     label='Base parameters')
        else:
            # Group the data by whatever parameters you varied and plot a line for each group
            for name, group in df_params.groupby(varying_params):
                group = group.sort_values(by='p')

                # Create a dynamic legend label based on what changed
                if isinstance(name, tuple):
                    label = ", ".join([f"{var}={val}" for var, val in zip(varying_params, name)])
                else:
                    label = f"{varying_params[0]}={name}"

                plt.plot(group['p'], group['R_mean'], marker='o', linewidth=2, label=label)

        plt.title(f"Steady State Giant Component (R) vs p ({current_dim}, N={dim_N})")
        plt.xlabel("Control Parameter (p)")
        plt.ylabel("Average Size of the Giant Component (R)")
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle='--', alpha=0.6)

        # Put the legend outside the plot if there are many varying parameters
        plt.legend(title="Varying Parameters", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(global_fig_dir, "global_R_vs_p_comparison.png"), dpi=400)
        plt.close()

    # 5. FRACTAL DIMENSION FINITE-SIZE SCALING PLOT (MULTIPLE LINES FOR DIFFERENT N)
    if g_data["fractal"] and dim_p_fss is not None:
        print("-> Generating Finite-Size Scaling Plot for Fractal Dimension...")

        # Sort by N ascending so colors and legend are ordered perfectly
        g_data["fractal"].sort(key=lambda x: x['N'])

        plt.figure(figsize=(8, 6))
        colors = sns.color_palette("viridis", n_colors=len(g_data["fractal"]))

        # Pre-calculate crossovers to draw the Early Theory line ONLY ONCE
        max_x_cross, x_start_global, y_start_global = 0, 0, 0
        crossovers = []

        for data_fss in g_data["fractal"]:
            lr, lN = data_fss['log_r'], data_fss['log_N']
            if len(lr) >= 3:
                x_cross = (4 * lr[0] - lr[-1] + lN[-1] - lN[0]) / 3.0
                y_cross = 4.0 * (x_cross - lr[0]) + lN[0]
                crossovers.append((x_cross, y_cross, lr[-1], lN[-1]))
                if x_cross > max_x_cross:
                    max_x_cross = x_cross
                    x_start_global, y_start_global = lr[0], lN[0]
            else:
                crossovers.append(None)

        # Plot Early Theory (slope=4) exactly ONCE spanning to the max crossover
        if max_x_cross > 0:
            x_early = np.array([x_start_global, max_x_cross + 0.1])
            y_early = 4.0 * (x_early - x_start_global) + y_start_global
            plt.plot(x_early, y_early, color='orange', linestyle='--', linewidth=2, alpha=0.6,
                     label=r'Early Theory ($d_H=4$)')

        added_late, added_cross = False, False

        # Plot simulations and individual asymptotic theories
        for i, data_fss in enumerate(g_data["fractal"]):
            lr, lN, current_N = data_fss['log_r'], data_fss['log_N'], data_fss['N']

            plt.plot(lr, lN, marker='o', linestyle='-', color=colors[i], linewidth=2,
                     label=f'Simulation (N={current_N})')

            if crossovers[i]:
                x_cross, y_cross, x_final, y_final = crossovers[i]
                l_late = r'Asymptotic Theory ($d_H=1$)' if not added_late else "_nolegend_"
                l_cross = r'Crossover Scale ($r_\times$)' if not added_cross else "_nolegend_"

                x_late = np.array([max(0, x_cross - 0.1), x_final])
                y_late = 1.0 * (x_late - x_final) + y_final
                plt.plot(x_late, y_late, color='red', linestyle='--', linewidth=1.5, alpha=0.4, label=l_late)
                plt.plot(x_cross, y_cross, 'ko', markersize=5, label=l_cross)
                added_late, added_cross = True, True

        plt.title(f"Fractal Dimension Finite-Size Scaling ({current_dim}, p={dim_p_fss})")
        plt.xlabel(r"$\log_{10}(r)$ [Topological Radius / Hops]")
        plt.ylabel(r"$\log_{10}(N(r))$ [Cumulative Mass / Nodes]")
        plt.grid(True, linestyle=':', alpha=0.7)

        # Sorting the legend
        handles, labels = plt.gca().get_legend_handles_labels()

        def legend_sort_key(text):
            if "Simulation" in text:
                # Take the N so it gets sorted like 2500 -> 5000 -> ...
                n_val = int(text.split("N=")[1].split(")")[0])
                return 0, n_val
            elif "Early" in text:
                return 1, 0
            elif "Asymptotic" in text:
                return 2, 0
            elif "Crossover" in text:
                return 3, 0
            else:
                return 4, 0

        sorted_pairs = sorted(zip(handles, labels), key=lambda pair: legend_sort_key(pair[1]))
        sorted_handles, sorted_labels = zip(*sorted_pairs)

        plt.legend(sorted_handles, sorted_labels, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(global_fig_dir, f"global_fractal_scaling_p{dim_p_fss:.2f}.png"), dpi=400)
        plt.close()

print("\n" + "=" * 60)
print(" ALL ANALYSIS COMPLETED SUCCESSFULLY!")
print("=" * 60 + "\n")