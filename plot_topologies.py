import os
import sys
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.collections import LineCollection

import triadic_library as triadic

# =============================================================================
# SECTION 1: PARAMETER READING & INITIALIZATION
# =============================================================================
# This script generates clean, publication-ready visualizations of the structural networks for the TFG memory.
# It creates 3 distinct topological representations: 1D Ring, 2D Plane, and Coupled.

# Visualization parameters (Kept small specifically to make links visible)
N_vis = 2000
Lx = 10.0
Ly = 10.0
c = 0.07
d0 = 0.5
seed = 42

init_time = time.time()

# =============================================================================
# SECTION 2: DIRECTORY SETUP & LOGGER
# =============================================================================
dir_name = 'results/topologies/'
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

print("\n" + "=" * 60)
print("--- STARTING TOPOLOGICAL VISUALIZATION SUITE ---")
print(f"Visualization Nodes : N = {N_vis}")
print(f"Universe            : Lx = {Lx}, Ly = {Ly}")
print(f"Connectivity        : c = {c}, d0 = {d0}")
print("=" * 60)


# =============================================================================
# SECTION 3: PLOTTING FUNCTIONS
# =============================================================================

def plot_1d_ring(N: int, Lx: float, c: float, d0: float, seed: int, save_path: str):
    """
    Generates and plots a 1D network with PBC mapped onto a circular ring.

    :param N: int, number of nodes
    :param Lx: float, length of the 1D domain
    :param c: float, connection probability base
    :param d0: float, structural decay length
    :param seed: int, random seed
    :param save_path: str, path to save the figure
    """
    print("  -> Generating 1D Ring topology...")
    np.random.seed(seed)

    # Generate 1D network using the specific library function
    nodes, G, _ = triadic.random_uniform_line_netw_PBC(N, Lx, c, d0)

    # Map 1D x-coordinates [0, Lx] to polar coordinates [0, 2*pi]
    radius = Lx / (2 * np.pi)
    pos_polar = {}

    # nodes is an array of shape (N, 1)
    for i in G.nodes():
        x_val = nodes[i, 0]
        theta = 2 * np.pi * (x_val / Lx)
        # Convert polar to Cartesian for plotting
        x_plot = radius * np.cos(theta)
        y_plot = radius * np.sin(theta)
        pos_polar[i] = (x_plot, y_plot)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Build LineCollection for edges
    lines = []
    for u, v in G.edges():
        lines.append([pos_polar[u], pos_polar[v]])

    lc = LineCollection(lines, colors='gray', linewidths=0.5, alpha=0.5, zorder=1)
    ax.add_collection(lc)

    # Plot nodes
    x_nodes = [pos_polar[n][0] for n in G.nodes()]
    y_nodes = [pos_polar[n][1] for n in G.nodes()]
    ax.scatter(x_nodes, y_nodes, s=20, c='coral', edgecolors='white', linewidths=0.2, zorder=2)

    ax.set_aspect('equal')
    ax.axis('off')  # Hide axes for clean memory image
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "topology_1D_ring.png"), dpi=400, transparent=False, bbox_inches='tight')
    plt.close()
    print("     [OK] Saved 1D Ring.")


def plot_2d_square(N: int, Lx: float, Ly: float, c: float, d0: float, seed: int, save_path: str):
    """
    Generates and plots a 2D network with PBC on a flat square.
    Filters out wrap-around edges to keep the visualization clean.

    :param N: int, number of nodes
    :param Lx: float, width of the domain
    :param Ly: float, height of the domain
    :param c: float, connection probability base
    :param d0: float, structural decay length
    :param seed: int, random seed
    :param save_path: str, path to save the figure
    """
    print("  -> Generating 2D Square topology...")
    np.random.seed(seed)

    # Generate 2D network using the specific library function
    # Passing [Lx, Ly] as expected by random_uniform_square_netw_PBC
    nodes, G, _ = triadic.random_uniform_square_netw_PBC(N, [Lx, Ly], c, d0)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Build LineCollection for edges (Filtering PBC wrap-around for visual clarity)
    lines = []
    for u, v in G.edges():
        x1, y1 = nodes[u, 0], nodes[u, 1]
        x2, y2 = nodes[v, 0], nodes[v, 1]

        # Only draw the line if it doesn't cross the entire universe (PBC filter)
        if abs(x1 - x2) < Lx / 2.0 and abs(y1 - y2) < Ly / 2.0:
            lines.append([[x1, y1], [x2, y2]])

    lc = LineCollection(lines, colors='gray', linewidths=0.5, alpha=0.5, zorder=1)
    ax.add_collection(lc)

    # Plot nodes
    x_nodes = nodes[:, 0]
    y_nodes = nodes[:, 1]
    ax.scatter(x_nodes, y_nodes, s=20, c='coral', edgecolors='none', zorder=2)

    # Set limits strictly to the box
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_aspect('equal')

    # Keep the box frame but remove the ticks for a cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "topology_2D_square.png"), dpi=400, transparent=False,
                bbox_inches='tight')
    plt.close()
    print("     [OK] Saved 2D Square.")


def plot_coupled_model(save_path: str):
    """
    Generates and plots the coupled rings model.
    TODO: Implement once the specific calibration branch (Refinement vs Density) is chosen.
    """
    print("  -> Generating Coupled Rings topology...")
    print("     [TODO] Implementation pending final calibration strategy choice.")
    pass


# =============================================================================
# SECTION 4: MAIN EXECUTION
# =============================================================================

print("\n[Phase 1] Executing rendering functions...")

plot_1d_ring(N_vis, Lx, c, d0, seed, dir_name)
plot_2d_square(N_vis, Lx, Ly, c, d0, seed, dir_name)
plot_coupled_model(dir_name)

total_time = time.time() - init_time
print("\n" + "=" * 60)
print(f"   VISUALIZATION SUITE COMPLETED SUCCESSFULLY")
print(f"   Total Time: {total_time:.2f} seconds")
print(f"   Outputs saved to: {dir_name}")
print("=" * 60 + "\n")