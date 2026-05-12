import os
import json
import random
import glob
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

import triadic_library as triadic
import make_video as mv


def get_frames_interactive(Tmax):
    """
    Function to get frames interactively

    :param Tmax: Maximum time step (exclusive) for frame selection

    :return: List of selected frame indices
    """
    print("\n" + "=" * 45)
    print(f" FRAME SELECTION WIZARD (for Tmax = {Tmax}) ")
    print("=" * 45)
    print("How would you like to select the frames?")
    print(f"  1. All frames (0 to {Tmax - 1})")
    print("  2. Specific intervals (e.g., start to end by steps)")
    print("  3. Specific single points (e.g., frame 0, 50, and last)")
    print("  4. Both (Intervals + Single points)")

    choice = input("\nSelect an option [1]: ").strip()
    if not choice: choice = '1'

    frames = set()
    history = []

    if choice == '1':
        return list(range(Tmax))

    if choice in ['2', '4']:
        num_int_str = input("\nHow many intervals do you want to define? [1]: ").strip()
        num_int = int(num_int_str) if num_int_str.isdigit() else 1

        for i in range(num_int):
            print(f"\n--- Interval {i + 1} ---")

            if history:
                print("  Currently selected ranges:")
                for entry in history:
                    print(f"    ✓ {entry}")
                print("  --------------------------")

            start_str = input(f"  Start frame (0 to {Tmax - 1}) [0]: ").strip()
            end_str = input(f"  End frame (0 to {Tmax - 1}, or -1 for last) [-1]: ").strip()
            step_str = input("  Step (e.g., 5 for every 5th frame) [1]: ").strip()

            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else -1
            step = int(step_str) if step_str else 1

            # Convert negative indices to real indices
            if start < 0: start += Tmax
            if end < 0: end += Tmax

            # Safety bounds
            start = max(0, min(start, Tmax - 1))
            end = max(0, min(end, Tmax - 1))
            step = max(1, step)

            # Add range to set (handles backwards ranges just in case)
            if start <= end:
                frames.update(range(start, end + 1, step))
            else:
                frames.update(range(start, end - 1, -step))

            history.append(f"Interval from {start} to {end} (step: {step})")

    if choice in ['3', '4']:
        print("\n--- Single Points ---")

        if history:
            print("  Currently selected ranges:")
            for entry in history:
                print(f"    ✓ {entry}")
            print("  --------------------------")

        pts_str = input("Enter frames separated by commas (e.g., 0, 50, -1): ").strip()
        if pts_str:
            for p in pts_str.split(','):
                if p.strip():
                    try:
                        val = int(p.strip())
                        if val < 0: val += Tmax
                        val = max(0, min(val, Tmax - 1))
                        frames.add(val)
                    except ValueError:
                        pass

    if not frames:
        print("\nNo frames selected. Defaulting to first and last frame.")
        return [0, Tmax - 1]

    return sorted(list(frames))

# =============================================================================
# CONFIGURATION
# =============================================================================
# Define the target folder you want to visualize
target_folder = "./results/"

# =============================================================================
# 1. LOAD DATA & REBUILD EXACT NETWORK USING THE SEED
# =============================================================================
print("=" * 60)
print(" FRAME VIEWER SETUP")
print("=" * 60)
print(f"Loading data from: {target_folder}")

# Load parameters
with open(os.path.join(target_folder, "summary_metrics.json"), 'r') as f:
    data = json.load(f)

prm = data["parameters"]
N, p = prm["N"], prm["p"]
c, d0, dr = prm["c"], prm["d0"], prm["dr"]
cpos, cneg, seed = prm["cpos"], prm["cneg"], prm["seed"]
Tmax_actual = prm["Tmax"]

# --- DIMENSION DETECTION ---
if "num_rings" in prm:
    dim = "RINGS"
    geometry = "RINGS"
    num_rings = prm["num_rings"]
    RC_factor = 1.0
    Lx, Ly = prm["Lx"], prm["Ly"]
    L = np.asarray([Lx, Ly])
    fig_width, fig_height = 8.0, 8.0
elif "geometry" in prm:
    dim = 2
    geometry = prm["geometry"]
    RC_factor = prm["RC_factor"]
    L_list = prm["L"]
    Lx, Ly = L_list[0], L_list[1]
    L = np.asarray([Lx, Ly])

    # Auto-adjust figure size to perfectly match the rectangle proportions!
    fig_width = 8.0
    fig_height = 8.0 / RC_factor
else:
    dim = 1
    geometry = "1D"
    RC_factor = 1.0
    L = prm["L"]
    fig_width, fig_height = 8.0, 8.0

# Load node states over time
try:
    statenodes = np.load(os.path.join(target_folder, "statenodes.npz"))['statenodes']
except FileNotFoundError:
    print("Error: statenodes.npz not found. Cannot plot frames.")
    exit()

# Setting up some values
print("\n" + "-" * 45)
print(f" Detected Geometry: {geometry} ({dim}D), N = {N}")
print("-" * 45)

if dim == 1:
    polar_input = input("Use polar (circular) representation? (y/n) [y]: ").strip().lower()
    polar = False if polar_input == 'n' else True
else:
    polar = False

frames_to_plot = get_frames_interactive(Tmax_actual)
print(f"\n-> Perfectly understood! Selected {len(frames_to_plot)} frames to render.")
print("-" * 45)

# Calculate recommended marker size based on geometry and visual style
if dim == 2 or dim == "RINGS":
    rec_solid = 4.0 # Solid continuous color blobs
    rec_spaced = 1.5    # Individual points (good for drawing background links)
    hint = f"{rec_solid} for solid pattern, {rec_spaced} to see links"
    default_ms = rec_solid
else:
    default_ms = 4.0 if N <= 500 else 1.5
    hint = f"~{default_ms} for 1D"

ms_input = input(f"Point size (markersize)? [Suggestion: {hint}] (Press Enter for {default_ms}): ").strip()
try:
    ms = float(ms_input) if ms_input else default_ms
except ValueError:
    print(f"Invalid input. Using default: {default_ms}")
    ms = default_ms

draw_links_input = input("Draw structural background links? (y/n) [y]: ").strip().lower()
draw_links = False if draw_links_input == 'n' else True

make_video_input = input("Create MP4 videos after rendering? (y/n) [n]: ").strip().lower()
do_video = True if make_video_input == 'y' else False

fps = 5
if do_video:
    fps_input = input("Video FPS (frames per second)? [5]: ").strip()
    fps = int(fps_input) if fps_input.isdigit() else 5

print("-" * 45 + "\n")

print(f"Rebuilding base network (Seed: {seed}, Dim: {dim}D)...")
np.random.seed(seed)
random.seed(seed)

if dim == 1:
    # Rebuild 1D structural network exactly as it was
    nodes, G, adj = triadic.random_uniform_line_netw_PBC(N, L, c, d0)
    lij = np.array(G.edges())
    I, J = lij[:, 0], lij[:, 1]

    # Rebuild 1D regulatory network exactly as it was
    links, NL = triadic.midpoints_line_PBC(nodes, L, I, J)
    adjpos, adjneg = triadic.regulatory_network_line(nodes, links, L, dr, cpos, cneg)

    # Pre-calculate polar coordinates for speed
    theta_nodes = nodes[:, 0] * (2. * np.pi / L)
    x_nodes, y_nodes = np.cos(theta_nodes), np.sin(theta_nodes)

    theta_links = links[:, 0] * (2. * np.pi / L)
    x_links, y_links = np.cos(theta_links), np.sin(theta_links)

    # Format for LineCollection
    start_points = np.column_stack((x_nodes[I], y_nodes[I]))
    end_points = np.column_stack((x_nodes[J], y_nodes[J]))
    segments = np.stack((start_points, end_points), axis=1)

elif dim == 2:
    # Rebuild 2D structural network
    nodes, G, adj = triadic.random_uniform_square_netw_PBC(N, L, c, d0)
    lij = np.array(G.edges())
    I, J = lij[:, 0], lij[:, 1]

    # Rebuild 2D regulatory network
    links, NL = triadic.midpoints_square_PBC(nodes, L, I, J)
    adjpos, adjneg = triadic.regulatory_network_square(nodes, links, L, dr, cpos, cneg)

    # Format for LineCollection in 2D (Direct X, Y coordinates)
    start_points = nodes[I]
    end_points = nodes[J]
    segments = np.stack((start_points, end_points), axis=1)

elif dim == "RINGS":
    # Rebuild scalable structural network
    nodes, G, adj = triadic.coupled_rings_structural_network_fixed_N(N, num_rings, Lx, Ly, c, d0)
    lij = np.array(G.edges())
    I, J = lij[:, 0], lij[:, 1]

    # Rebuild scalable regulatory network
    links, NL = triadic.midpoints_rings_PBC(nodes, Lx, Ly, I, J)
    adjpos, adjneg = triadic.coupled_rings_regulatory_network(nodes, links, Lx, Ly, dr, cpos, cneg)

    # Format for LineCollection (Direct X, Y coordinates)
    start_points = nodes[I]
    end_points = nodes[J]
    segments = np.stack((start_points, end_points), axis=1)

# Create output folders
path_struct = os.path.join(target_folder, "frames_structural")
path_reg = os.path.join(target_folder, "frames_regulatory")
os.makedirs(path_struct, exist_ok=True)
os.makedirs(path_reg, exist_ok=True)

old_struct_files = glob.glob(os.path.join(path_struct, "*.*"))
old_reg_files = glob.glob(os.path.join(path_reg, "*.*"))

if old_struct_files or old_reg_files:
    print("\n" + "!" * 45)
    print(" WARNING: OUTPUT FOLDERS NOT EMPTY")
    print("!" * 45)
    print(f" Found {len(old_struct_files)} old files in structural folder.")
    print(f" Found {len(old_reg_files)} old files in regulatory folder.")
    print(" (Mixing old and new frames could corrupt the final MP4 video).")

    clean_input = input("\nDo you want to delete the old files before rendering? (y/n) [y]: ").strip().lower()

    if clean_input != 'n':
        print(" -> Sweeping old files...")
        for f in old_struct_files + old_reg_files:
            try:
                os.remove(f)
            except OSError:
                pass
        print(" -> Folders are now clean and ready!")
    else:
        overwrite_input = input(" Do you want to OVERWRITE existing frames if they match? (y/n) [y]: ").strip().lower()

        if overwrite_input == 'n':
            frames_to_keep = []
            for it in frames_to_plot:
                actual_it = Tmax_actual - 1 if it == -1 else it
                s_file = os.path.join(path_struct, f"structural_t{actual_it:04d}.png")
                r_file = os.path.join(path_reg, f"regulatory_t{actual_it:04d}.png")

                if os.path.exists(s_file) or os.path.exists(r_file):
                    continue
                else:
                    frames_to_keep.append(it)

            skipped_count = len(frames_to_plot) - len(frames_to_keep)
            frames_to_plot = frames_to_keep
            print(f" -> Smart Resume: Skipped {skipped_count} frames that already exist.")
            print(f" -> Remaining frames to render: {len(frames_to_plot)}")
        else:
            print(" -> Keeping old files. Matching frames WILL be overwritten.")
print("-" * 45 + "\n")

# =============================================================================
# 2. FRAME GENERATION LOOP
# =============================================================================
print(f"Generating {len(frames_to_plot)} requested frames...")

for it in frames_to_plot:
    # Handle negative indices (e.g., -1 for the last frame)
    actual_it = Tmax_actual - 1 if it == -1 else it
    if actual_it >= Tmax_actual:
        print(f"Skipping t={it} (Exceeds maximum simulated time {Tmax_actual - 1})")
        continue

    print(f" -> Rendering Frame t = {actual_it:04d}...")

    # ---------------------------------------------------------
    # A. IDENTIFY ACTIVE NODES AND CLUSTERS AT THIS TIME STEP
    # ---------------------------------------------------------
    active_mask = statenodes[actual_it, :]
    active_indices = np.where(active_mask)[0]

    agn, ag2, ag3 = [], [], []

    if len(active_indices) > 0:
        # Create subgraph of only currently active nodes
        G_active = G.subgraph(active_indices)
        # Find clusters (connected components)
        clusters = sorted(nx.connected_components(G_active), key=len, reverse=True)

        if len(clusters) > 0: agn = list(clusters[0])
        if len(clusters) > 1: ag2 = list(clusters[1])
        if len(clusters) > 2: ag3 = list(clusters[2])

        # Multiply the network matrix by the CURRENTLY active nodes
        # Use .dot() instead of np.matmul to safely handle both dense (1D/2D) and sparse (RINGS) matrices
        # True if the link receives at least 1 positive signal from an active node
        current_pos_signal = adjpos.transpose().dot(active_mask) > 0

        # True if the link receives at least 1 negative signal from an active node
        current_neg_signal = adjneg.transpose().dot(active_mask) > 0

        # Dynamics logic: Inhibition overrides excitation.
        # A link is inhibited if it receives ANY negative signal.
        dyn_inhibited = current_neg_signal

        # A link is excited ONLY if it receives positive signals AND NO negative signals.
        dyn_excited = current_pos_signal & (~current_neg_signal)

        # ---------------------------------------------------------
        # B. PLOT 1: STRUCTURAL NETWORK
        # ---------------------------------------------------------
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        if draw_links:
            # Draw physical cables (background) for both dimensions
            lc = LineCollection(segments, colors='black', alpha=0.1, linewidths=0.3)
            ax.add_collection(lc)

        if dim == 1:
            if polar:
                # Plot all nodes as inactive first (Cyan)
                ax.plot(x_nodes, y_nodes, "o", color='cyan', markersize=ms, alpha=0.3, markeredgewidth=0)

                # Overwrite with active clusters
                if len(agn) > 0: ax.plot(x_nodes[agn], y_nodes[agn], "o", color='magenta', markersize=ms * 1.1,
                                         markeredgewidth=0)
                if len(ag2) > 0: ax.plot(x_nodes[ag2], y_nodes[ag2], "o", color='lime', markersize=ms * 1.1,
                                         markeredgewidth=0)
                if len(ag3) > 0: ax.plot(x_nodes[ag3], y_nodes[ag3], "o", color='blue', markersize=ms * 1.1,
                                         markeredgewidth=0)

                ax.set_xlim(-1.1, 1.1)
                ax.set_ylim(-1.1, 1.1)
                ax.set_aspect('equal')
            else:
                # Linear representation
                ax.plot(nodes[:, 0], np.zeros(N), "o", color='cyan', markersize=ms, alpha=0.3)
                if len(agn) > 0: ax.plot(nodes[agn, 0], np.zeros(len(agn)), "o", color='magenta',
                                         markersize=ms * 1.1)
                if len(ag2) > 0: ax.plot(nodes[ag2, 0], np.zeros(len(ag2)), "o", color='lime',
                                         markersize=ms * 1.1)
                if len(ag3) > 0: ax.plot(nodes[ag3, 0], np.zeros(len(ag3)), "o", color='blue',
                                         markersize=ms * 1.1)
                ax.set_ylim(-0.1, 0.1)

        elif dim == 2 or dim == "RINGS":
            # Plot all nodes as inactive first (Cyan)
            ax.plot(nodes[:, 0], nodes[:, 1], "o", color='cyan', markersize=ms, alpha=0.3, markeredgewidth=0)

            # Overwrite with active clusters
            if len(agn) > 0: ax.plot(nodes[agn, 0], nodes[agn, 1], "o", color='magenta', markersize=ms * 1.1,
                                     markeredgewidth=0)
            if len(ag2) > 0: ax.plot(nodes[ag2, 0], nodes[ag2, 1], "o", color='lime', markersize=ms * 1.1,
                                     markeredgewidth=0)
            if len(ag3) > 0: ax.plot(nodes[ag3, 0], nodes[ag3, 1], "o", color='blue', markersize=ms * 1.1,
                                     markeredgewidth=0)

            ax.set_xlim(0, Lx)
            ax.set_ylim(0, Ly)
            ax.set_aspect('equal')

        # --- CUSTOM LEGEND (OUTSIDE THE PLOT) ---
        # Draw time text on ax with a white stroke for contrast
        ax.text(0.98, 0.02, f't={actual_it:04d}', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=12, fontweight='bold', color='black',
                path_effects=[pe.withStroke(linewidth=3, foreground="white")])

        legend_elements_struct = []

        if draw_links:
            legend_elements_struct.append(
                Line2D([0], [0], color='black', lw=2, alpha=0.3, label='Structural Links')
            )

        # Base elements that always appear
        legend_elements_struct.extend([
            Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', markersize=8,
                   label='Inactive Nodes'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='magenta', markersize=8,
                   label='Cluster 1 (Giant)')
        ])

        # Conditional elements: Only add them to the legend if they actually exist in this frame
        if len(ag2) > 0:
            legend_elements_struct.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor='lime', markersize=8,
                       label='Cluster 2')
            )
        if len(ag3) > 0:
            legend_elements_struct.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8,
                       label='Cluster 3')
            )

        leg = ax.legend(handles=legend_elements_struct, loc='center left', bbox_to_anchor=(1.05, 0.5),
                        title="Structural State", frameon=True, edgecolor='lightgray', facecolor='white',
                        framealpha=0.9, borderaxespad=0.)
        leg.get_title().set_fontweight('bold')

        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(path_struct, f"structural_t{actual_it:04d}.png"), dpi=400, bbox_inches='tight')
        plt.close()

        # ---------------------------------------------------------
        # C. PLOT 2: REGULATORY NETWORK (LINK STATES)
        # ---------------------------------------------------------
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        if dim == 1:
            if polar:
                # Plot all structural midpoints (Cyan)
                ax.plot(x_links, y_links, "o", color='cyan', markersize=ms*1.0, alpha=1.0, markeredgewidth=0)

                # Plot links receiving positive regulation (Magenta)
                if np.any(dyn_excited):
                    ax.plot(x_links[dyn_excited], y_links[dyn_excited], "o", color='magenta',
                            markersize=ms * 1.1, alpha=1.0, markeredgewidth=0)

                # Plot links receiving negative regulation (Black)
                if np.any(dyn_inhibited):
                    ax.plot(x_links[dyn_inhibited], y_links[dyn_inhibited], "o", color='black',
                            markersize=ms * 1.1, alpha=1.0, markeredgewidth=0)

                ax.set_xlim(-1.1, 1.1)
                ax.set_ylim(-1.1, 1.1)
                ax.set_aspect('equal')
            else:
                # Linear representation
                ax.plot(links[:, 0], np.zeros(NL), "o", color='cyan', markersize=ms * 1.0, alpha=1.0)
                if np.any(dyn_excited): ax.plot(links[dyn_excited, 0], np.zeros(np.sum(dyn_excited)), "o",
                                                 color='magenta', markersize=ms * 1.1)
                if np.any(dyn_inhibited): ax.plot(links[dyn_inhibited, 0], np.zeros(np.sum(dyn_inhibited)), "o",
                                                 color='black', markersize=ms * 1.1)
                ax.set_ylim(-0.1, 0.1)

        elif dim == 2 or dim == "RINGS":
            ax.plot(links[:, 0], links[:, 1], "o", color='cyan', markersize=ms * 1.0, alpha=1.0,
                    markeredgewidth=0)
            if np.any(dyn_excited): ax.plot(links[dyn_excited, 0], links[dyn_excited, 1], "o", color='magenta',
                                             markersize=ms * 1.1, alpha=1.0, markeredgewidth=0)
            if np.any(dyn_inhibited): ax.plot(links[dyn_inhibited, 0], links[dyn_inhibited, 1], "o",
                                              color='black', markersize=ms * 1.1, alpha=1.0, markeredgewidth=0)

            ax.set_xlim(0, Lx)
            ax.set_ylim(0, Ly)
            ax.set_aspect('equal')

        # --- CUSTOM LEGEND (OUTSIDE THE PLOT) ---
        # Draw time text on ax with a white stroke for contrast
        ax.text(0.98, 0.02, f't={actual_it:04d}', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=12, fontweight='bold', color='black',
                path_effects=[pe.withStroke(linewidth=3, foreground="white")])

        legend_elements_reg = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', markersize=8,
                   label='Neutral Midpoints'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='magenta', markersize=8,
                   label='Excited Links (+)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8,
                   label='Inhibited Links (-)')
        ]
        leg = ax.legend(handles=legend_elements_reg, loc='center left', bbox_to_anchor=(1.05, 0.5),
                        title="Regulatory State", frameon=True, edgecolor='lightgray', facecolor='white',
                        framealpha=0.9, borderaxespad=0.)
        leg.get_title().set_fontweight('bold')

        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(path_reg, f"regulatory_t{actual_it:04d}.png"), dpi=400, bbox_inches='tight')
        plt.close()

print("\n" + "=" * 60)
print("-> VISUALIZATION COMPLETED. Check the output folders!")

if do_video:
    print("-" * 60)
    print("-> INITIATING VIDEO COMPILATION...")
    mv.make_video(target_folder, fps=fps)

print("=" * 60 + "\n")