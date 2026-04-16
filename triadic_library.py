import random
import networkx as nx
import numpy as np

import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.stats import circmean, circstd


# =============================================================================
# SECTION 1: GENERAL NETWORK TOPOLOGY & METRICS (1D & 2D COMPATIBLE)
# =============================================================================

def calculate_degrees(G: nx.Graph, adj: np.ndarray, adjpos: np.ndarray, adjneg: np.ndarray,
                      density: float, d0: float, dr: float, c: float, cpos: float, cneg: float, dim=1) -> dict:
    """
    Calculate theoretical and empirical degrees for structural and regulatory networks.

    :param G: networkx graph, structural network
    :param adj: numpy array (N, N), structural adjacency matrix (upper triangular)
    :param adjpos: numpy array (N, NL), positive regulatory adjacency matrix
    :param adjneg: numpy array (N, NL), negative regulatory adjacency matrix
    :param density: float, node density (N / L for 1D, N / L^2 for 2D)
    :param d0: float, decay length for structural connectivity
    :param dr: float, decay length for regulatory connectivity
    :param c: float, connectivity parameter for structural network
    :param cpos: float, connectivity parameter for positive regulation
    :param cneg: float, connectivity parameter for negative regulation
    :param dim: int, dimensionality of the system (1 or 2)

    :return: dict, containing theoretical and empirical arrays and means
    """
    # ==========================================
    # 1. THEORETICAL VALUES (GEOMETRY DEPENDENT)
    # ==========================================
    if dim == 1:
        # 1D integral gives 2*d
        # Structural degree <k>: Average number of physical connections per node
        k_theo = 2 * density * d0 * c

        # Regulatory IN-degree <kappa_in>: Average number of nodes regulating a SINGLE LINK
        # This is the "pressure" exerted on a link.
        kappa_in_pos_theo = 2 * density * dr * cpos
        kappa_in_neg_theo = 2 * density * dr * cneg

        rho_L = density * (k_theo / 2)
        kappa_out_pos_theo = 2 * rho_L * dr * cpos
        kappa_out_neg_theo = 2 * rho_L * dr * cneg

    elif dim == 2:
        # 2D polar integral gives 2 * pi * d^2
        k_theo = density * 2 * np.pi * (d0 ** 2) * c
        kappa_in_pos_theo = density * 2 * np.pi * (dr ** 2) * cpos
        kappa_in_neg_theo = density * 2 * np.pi * (dr ** 2) * cneg

        rho_L = density * (k_theo / 2)
        kappa_out_pos_theo = rho_L * 2 * np.pi * (dr ** 2) * cpos
        kappa_out_neg_theo = rho_L * 2 * np.pi * (dr ** 2) * cneg
    else:
        raise ValueError("Dimensionality (dim) must be 1 or 2.")

    # ==========================================
    # 2. EMPIRICAL VALUES
    # ==========================================

    # --- A. STRUCTURAL DEGREES ---
    # REAL degree (from networkx): Actual number of connections per node.
    k_real_array = np.array([d for n, d in G.degree()])

    # MATRIX degree (from adj): Since adj is an upper triangular matrix (np.triu),
    # summing its rows only counts "outgoing" connections to the right, ignoring incoming ones.
    # It should be exactly half of the real degree.
    k_mat_array = np.sum(adj, axis=1)

    # --- B. REGULATORY IN-DEGREES (Focus on LINKS) ---
    # Summing over axis=0 (nodes) tells us how many nodes point to each link.
    kappa_in_pos_array = np.sum(adjpos, axis=0)
    kappa_in_neg_array = np.sum(adjneg, axis=0)

    # --- C. REGULATORY OUT-DEGREES (Focus on NODES) ---
    # Summing over axis=1 (links) tells us how many links are targeted by each node.
    kappa_out_pos_array = np.sum(adjpos, axis=1)
    kappa_out_neg_array = np.sum(adjneg, axis=1)

    return {
        "k_theo": k_theo,
        "k_real_array": k_real_array,
        "k_real_mean": np.mean(k_real_array),
        "k_real_std": np.std(k_real_array),
        "k_mat_mean": np.mean(k_mat_array),
        "k_mat_std": np.std(k_mat_array),
        "kappa_in_pos_theo": kappa_in_pos_theo,
        "kappa_in_pos_array": kappa_in_pos_array,
        "kappa_in_pos_mean": np.mean(kappa_in_pos_array),
        "kappa_in_pos_std": np.std(kappa_in_pos_array),
        "kappa_in_neg_theo": kappa_in_neg_theo,
        "kappa_in_neg_array": kappa_in_neg_array,
        "kappa_in_neg_mean": np.mean(kappa_in_neg_array),
        "kappa_in_neg_std": np.std(kappa_in_neg_array),
        "kappa_out_pos_theo": kappa_out_pos_theo,
        "kappa_out_pos_array": kappa_out_pos_array,
        "kappa_out_pos_mean": np.mean(kappa_out_pos_array),
        "kappa_out_pos_std": np.std(kappa_out_pos_array),
        "kappa_out_neg_theo": kappa_out_neg_theo,
        "kappa_out_neg_array": kappa_out_neg_array,
        "kappa_out_neg_mean": np.mean(kappa_out_neg_array),
        "kappa_out_neg_std": np.std(kappa_out_neg_array)
    }


def get_topological_distances(G0: nx.Graph, sample_size: int = 500) -> tuple:
    """
    Compute the distribution of shortest path lengths in the Giant Component. It uses only the Giant Component (G0)
    because paths to isolated islands are infinite. It takes a sample of sample_size nodes (default=500) to avoid heavy
    computation (statistically sufficient).

    :param G0: networkx graph (Giant Component)
    :param sample_size: number of nodes to sample for calculation

    :return: (list of path lengths, mean distance)
    """
    nodes_to_sample = min(sample_size, len(G0.nodes()))
    sampled_nodes = random.sample(list(G0.nodes()), nodes_to_sample)
    path_lengths = []

    # Calculate the distance from our sampled nodes to ALL other nodes in G0
    for source in sampled_nodes:
        # Returns a dictionary with {target_node: number_of_hops}
        lengths = nx.single_source_shortest_path_length(G0, source)
        # Save all distances greater than 0 (to avoid counting the distance from a node to itself)
        path_lengths.extend([l for target, l in lengths.items() if l > 0])

    # The "typical distance" will be the mean of all the obtained lengths
    # Safe check to avoid RuntimeWarning (Mean of empty slice) if no paths were found
    if not path_lengths:
        return path_lengths, 0.0

    return path_lengths, float(np.mean(path_lengths))


def get_fractal_mass_radius(G0: nx.Graph, max_hops: int = 10, sample_size: int = 100) -> tuple:
    """
    Compute the average mass N(r) for different topological radii r. It uses the giant component G0 to avoid infinite
    distances. It only cares about the first few hops to see the growth trend.

    :param G0: networkx graph (Giant Component)
    :param max_hops: maximum number of hops to consider
    :param sample_size: number of nodes to sample for calculation

    :return: (r_values, N_r_values)
    """
    # Average over a sample of nodes to get smooth data
    nodes_to_sample = min(sample_size, len(G0.nodes()))
    sampled_nodes = random.sample(list(G0.nodes()), nodes_to_sample)

    # Dictionary to store the average mass N(r) for each radius r
    mass_vs_radius = {r: [] for r in range(1, max_hops + 1)}
    for source in sampled_nodes:
        # Get distances from source to all other nodes in G0
        lengths = nx.single_source_shortest_path_length(G0, source)

        # For each radius r, count how many nodes are at distance <= r
        for r in range(1, max_hops + 1):
            mass = sum(1 for node, dist in lengths.items() if 0 < dist <= r)
            mass_vs_radius[r].append(mass)

    # Calculate the average mass for each radius across all sampled nodes
    r_vals = np.array(list(mass_vs_radius.keys()))
    N_r_vals = np.array([np.mean(mass_vs_radius[r]) for r in r_vals])
    return r_vals, N_r_vals


# =============================================================================
# SECTION 2: 1D WORLD (LINE/RING GEOMETRY)
# =============================================================================

def distance_PBC_1D(nodes1: np.ndarray, nodes2: np.ndarray, L: float) -> np.ndarray:
    """
    Compute the distance between two sets of points in 1D with Periodic Boundary Conditions.

    :param nodes1: numpy array (N1,), coordinates of the first set of points
    :param nodes2: numpy array (N2,), coordinates of the second set of points
    :param L: float, size of the periodic domain

    :return: numpy array (N1, N2), distance matrix between each pair of points
    """
    # Reshape nodes and nodes2
    nodes1 = nodes1[:, np.newaxis]
    nodes2 = nodes2[np.newaxis, :]

    # Compute the difference
    dif = nodes1 - nodes2
    dif = np.absolute(L / 2 - np.absolute(L / 2 - np.absolute(dif)))
    return dif

def distance_PBC_1D_pair(nodes1: np.ndarray, nodes2: np.ndarray, L: float) -> np.ndarray:
    """
    Compute the distance between pairs of points in 1D with Periodic Boundary Conditions.

    :param nodes1: numpy array (N,), coordinates of the first set of points
    :param nodes2: numpy array (N,), coordinates of the second set of points
    :param L: float, size of the periodic domain

    :return: numpy array (N,), distance between each corresponding pair
    """
    dif = nodes1 - nodes2
    dif = np.absolute(L / 2 - np.absolute(L / 2 - np.absolute(dif)))
    return dif


def random_uniform_line_netw_PBC(N: int, L: float, c: float, d0: float) -> tuple:
    """
    Create a random spatial network with nodes uniformly distributed in a 1D periodic segment.

    :param N: int, number of nodes
    :param L: float, size of the segment
    :param c: float, base connectivity probability
    :param d0: float, decay length of the connectivity

    :return: tuple (nodes, G, adj), where nodes is (N,1) array, G is nx.Graph, and adj is (N,N) bool array
    """
    nodes = np.random.rand(N, 1) * L
    # Measure distances between each pair of nodes with PBC
    D = distance_PBC_1D(nodes[:, 0], nodes[:, 0], L)

    # Probability for nodes to be connected
    P = c * np.exp(-D / d0)
    del D

    # Create adjacency matrix
    adj = np.triu((P - np.identity(N)) > np.random.rand(N, N)).astype(bool)
    del P

    # Create nx graph object from adjacency matrix
    G = nx.from_numpy_array(adj)

    return nodes, G, adj


def midpoints_line_PBC(nodes: np.ndarray, L: float, I: np.ndarray, J: np.ndarray) -> tuple:
    """
    Compute the midpoints of links in a 1D periodic segment.

    :param nodes: numpy array (N, 1), coordinates of the nodes
    :param L: float, size of the segment
    :param I: numpy array (NL,), indices of the first nodes of each link
    :param J: numpy array (NL,), indices of the second nodes of each link

    :return: tuple (xl, NL), where xl is (NL, 1) midpoint coordinates and NL is number of links
    """
    xl2 = np.mod(0.5 * (L + nodes[I] + nodes[J]), L)
    NL = I.shape[0]
    xl1 = 0.5 * (nodes[I] + nodes[J])

    dist1 = distance_PBC_1D_pair(nodes[I, 0], xl1[:, 0], L)
    dist2 = distance_PBC_1D_pair(nodes[I, 0], xl2[:, 0], L)

    cond = (dist1 < dist2)[:, np.newaxis]
    xl = xl1 * cond + xl2 * (np.logical_not(cond))

    return xl, NL


def regulatory_network_line(nodes: np.ndarray, links: np.ndarray, L: float, dr: float,
                            cpos: float, cneg: float) -> tuple:
    """
    Create positive and negative regulatory networks between nodes and links in 1D.

    :param nodes: numpy array (N, 1), coordinates of the nodes
    :param links: numpy array (NL, 1), coordinates of the links
    :param L: float, size of the segment
    :param dr: float, decay length of the regulation
    :param cpos: float, base probability for positive regulation
    :param cneg: float, base probability for negative regulation

    :return: tuple (adjpos, adjneg), both being (N, NL) boolean adjacency matrices
    """
    N, NL = nodes.shape[0], links.shape[0]
    adjpos, adjneg = np.zeros((N, NL), dtype=bool), np.zeros((N, NL), dtype=bool)

    for i, node in enumerate(nodes):
        DL = distance_PBC_1D_pair(np.asarray(node[0]), links[:, 0], L)

        PLpos, PLneg = cpos * np.exp(-DL / dr), cneg * np.exp(-DL / dr) # Probability of pos/neg regulation nodes-links

        ran2 = np.random.rand(NL)
        if cpos + cneg > 1: ran2 = ran2 * (cpos + cneg)

        adjpos[i, :] = (PLpos > ran2)   # Positive regulation adjacency matrix
        adjneg[i, :] = ((ran2 > PLpos) & (ran2 < (PLpos + PLneg)))  # Negative regulation adjacency matrix

    return adjpos, adjneg


def itera_line(L: float, statenode1: np.ndarray, nodes: np.ndarray, links: np.ndarray,
               I: np.ndarray, J: np.ndarray, adjpos: np.ndarray, adjneg: np.ndarray, p: float) -> tuple:
    """
    Perform one iteration of the triadic dynamics in 1D.

    :param L: float, size of the segment
    :param statenode1: numpy array (N,), active state of nodes at t-1
    :param nodes: numpy array (N, 1), coordinates of nodes
    :param links: numpy array (NL, 1), coordinates of links
    :param I: numpy array (NL,), indices of first node of each link
    :param J: numpy array (NL,), indices of second node of each link
    :param adjpos: numpy array (N, NL), positive regulatory matrix
    :param adjneg: numpy array (N, NL), negative regulatory matrix
    :param p: float, probability of a link not breaking randomly

    :return: tuple (RT, agn, spatial_metrics), fraction of active nodes, list of active indices, and circular stats
    """
    N, NL = nodes.shape[0], links.shape[0]

    # Obtain the new state of the links
    # Regulation dynamics
    # Three conditions for a link to be active:
    # 1) It does not break randomly (with probability p): p is the probability of surviving
    f1 = p > np.random.rand(NL)
    # 2) It has at least one active node exerting positive regulation
    f2 = np.matmul(adjpos.transpose(), statenode1) > 0
    # 3) It has no active node exerting negative regulation
    f3 = np.logical_not(np.matmul(adjneg.transpose(), statenode1))

    # State of the links in this iteration (active / inactive)
    statelink = f1 * f2 * f3

    # Indexes of active links. It will be used to create the new giant component
    linkid = np.where(statelink)[0]

    # Defining the new adjacency matrix it is a sparse matrix
    nadj = csr_matrix((np.ones(linkid.shape[0]), (I[linkid], J[linkid])), shape=(N, N))

    # Compute the giant component in this iteration
    Gn = nx.from_scipy_sparse_array(nadj)
    del nadj
    Gccn = sorted(nx.connected_components(Gn), key=len, reverse=True)

    # Initialize spatial metrics as NaN. If a cluster dies, it remains NaN
    mean1, std1, mean2, std2, mean3, std3 = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    theta = nodes[:, 0] * (2. * np.pi / L)

    # Giant component (first cluster)
    if len(Gccn) > 0:
        agn = list(Gccn[0])  # Vector of indices of the nodes in the Giant component
        # Calculate circular mean and width for the giant component
        mean1, std1 = circmean(theta[agn], high=2 * np.pi, low=0), circstd(theta[agn], high=2 * np.pi, low=0)

        ngn = len(Gccn[0])  # Number of nodes in the Giant component
    else:
        agn, ngn = [], 0

    # Second and third cluster
    ag2_len, ag3_len = 0, 0
    if len(Gccn) > 1:
        ag2 = list(Gccn[1])
        # Calculate circular mean and width for the second cluster
        mean2, std2 = circmean(theta[ag2], high=2 * np.pi, low=0), circstd(theta[ag2], high=2 * np.pi, low=0)
        ag2_len = len(Gccn[1])
        if len(Gccn) > 2:
            ag3 = list(Gccn[2])
            # Calculate circular mean and width for the third cluster
            mean3, std3 = circmean(theta[ag3], high=2 * np.pi, low=0), circstd(theta[ag3], high=2 * np.pi, low=0)
            ag3_len = len(Gccn[2])

    # Fraction of nodes that are active in this step:
    RT = np.asarray([ngn, ag2_len, ag3_len]) * 1. / N
    return RT, agn, [mean1, std1, mean2, std2, mean3, std3]


# =============================================================================
# SECTION 3: 2D WORLD (SQUARE GEOMETRY)
# =============================================================================

def PBC_distance(nodes: np.ndarray, L: list | tuple | float | np.ndarray) -> np.ndarray:
    """
    Compute the distance between points in 2D with Periodic Boundary Conditions.

    :param nodes: numpy array (N, 2), coordinates of the points
    :param L: list or float, size of the periodic box (scalar L or vector [Lx, Ly])

    :return: numpy array (N, N), distances between each pair of points
    """
    # Safe check in case L is passed as a pure float instead of a list
    L_safe = np.atleast_1d(L)

    # Distance between points with PBC
    if len(L_safe) == 1:
        Lx = L_safe[0]
        Ly = L_safe[0]
    else:
        Lx = L_safe[0]
        Ly = L_safe[1]

    DX = distance_PBC_1D(nodes[:, 0], nodes[:, 0], Lx)
    DY = distance_PBC_1D(nodes[:, 1], nodes[:, 1], Ly)
    D = np.sqrt(np.multiply(DX, DX) + np.multiply(DY, DY))

    return D


def random_uniform_square_netw_PBC(N: int, L: float | list | np.ndarray, c: float, d0: float) -> tuple:
    """
    Create a random spatial network with nodes uniformly distributed in a 2D periodic square.

    :param N: int, number of nodes
    :param L: float, size of the square (side length)
    :param c: float, connectivity parameter
    :param d0: float, decay length of the connectivity

    :return: tuple (nodes, G, adj), where nodes is (N,2) array, G is nx.Graph, and adj is (N,N) bool array
    """
    nodes = np.random.rand(N, 2) * L
    # Measure distances between each pair of nodes with PBC
    D = PBC_distance(nodes, L)

    # Probability for nodes to be connected
    P = c * np.exp(-D / d0)
    del D

    # Create adjacency matrix
    adj = np.triu((P - np.identity(N)) > np.random.rand(N, N)).astype(bool)
    del P

    # Create nx graph object from adjacency matrix
    G = nx.from_numpy_array(adj)

    return nodes, G, adj


def midpoints_square_PBC(nodes: np.ndarray, L: float | list | np.ndarray, I: np.ndarray, J: np.ndarray) -> tuple:
    """
    Compute the midpoints of the links in a 2D square with periodic boundary conditions.

    :param nodes: numpy array (N, 2), coordinates of the nodes
    :param L: float or array-like, size of the square (can be a scalar L or vector [Lx, Ly])
    :param I: numpy array (NL,), indices of the first nodes of each link
    :param J: numpy array (NL,), indices of the second nodes of each link

    :return: tuple (xl, NL), where xl is (NL, 2) midpoint coordinates and NL is the number of links
    """
    L_safe = np.atleast_1d(L)
    if len(L_safe) == 1:
        Lx = L_safe[0]
        Ly = L_safe[0]
        xl2 = np.mod(0.5 * (L + nodes[I] + nodes[J]), L)
    else:
        Lx = L_safe[0]
        Ly = L_safe[1]
        xl2 = np.stack((np.mod(0.5 * (Lx + nodes[I, 0] + nodes[J, 0]), Lx),
                        np.mod(0.5 * (Ly + nodes[I, 1] + nodes[J, 1]), Ly))).T

    NL = I.shape[0]
    xl1 = 0.5 * (nodes[I] + nodes[J])

    # Distance in X and Y to determine the correct midpoint under PBC
    dist1x = distance_PBC_1D_pair(nodes[I, 0], xl1[:, 0], Lx)
    dist2x = distance_PBC_1D_pair(nodes[I, 0], xl2[:, 0], Lx)

    dist1y = distance_PBC_1D_pair(nodes[I, 1], xl1[:, 1], Ly)
    dist2y = distance_PBC_1D_pair(nodes[I, 1], xl2[:, 1], Ly)

    condx = dist1x < dist2x
    condy = dist1y < dist2y
    cond = np.stack((condx, condy)).T

    # Final coordinates selection based on shortest distance
    xl = xl1 * cond + xl2 * (np.logical_not(cond))
    return xl, NL


def regulatory_network_square(nodes: np.ndarray, links: np.ndarray, L: float | list | np.ndarray,
                              dr: float, cpos: float, cneg: float) -> tuple:
    """
    Create positive and negative regulatory networks between nodes and links in 2D.

    :param nodes: numpy array (N, 2), coordinates of the nodes
    :param links: numpy array (NL, 2), coordinates of the links
    :param L: float or array-like, size of the square (scalar L or vector [Lx, Ly])
    :param dr: float, decay length of the regulation
    :param cpos: float, base probability for positive regulation
    :param cneg: float, base probability for negative regulation

    :return: tuple (adjpos, adjneg), both being (N, NL) boolean adjacency matrices
    """
    N = nodes.shape[0]
    NL = links.shape[0]

    # Handle both scalar and vector L for Periodic Boundary Conditions
    L_safe = np.atleast_1d(L)
    if len(L_safe) == 1:
        Lx = L_safe[0]
        Ly = L_safe[0]
    else:
        Lx = L_safe[0]
        Ly = L_safe[1]

    adjpos = np.zeros((N, NL), dtype=bool)
    adjneg = np.zeros((N, NL), dtype=bool)

    for i, node in enumerate(nodes):
        DX = distance_PBC_1D_pair(np.asarray(node[0]), links[:, 0], Lx)
        DY = distance_PBC_1D_pair(np.asarray(node[1]), links[:, 1], Ly)
        DL = np.sqrt(np.multiply(DX, DX) + np.multiply(DY, DY))

        PLpos = cpos * np.exp(-DL / dr) # Probability of positive regulation nodes-links
        PLneg = cneg * np.exp(-DL / dr) # Probability of negative regulation nodes-links

        ran2 = np.random.rand(NL)
        if cpos + cneg > 1:
            ran2 = ran2 * (cpos + cneg)

        adjpos[i, :] = (PLpos > ran2)   # Positive regulation adjacency matrix
        adjneg[i, :] = ((ran2 > PLpos) & (ran2 < (PLpos + PLneg)))  # Negative regulation adjacency matrix

    return adjpos, adjneg


def itera_square(statenode1: np.ndarray, nodes: np.ndarray, links: np.ndarray,
                 I: np.ndarray, J: np.ndarray, adjpos: np.ndarray, adjneg: np.ndarray, p: float) -> tuple:
    """
    Perform one iteration of the triadic dynamics in 2D (Headless version).

    :param statenode1: numpy array (N,), active state of nodes at t-1
    :param nodes: numpy array (N, 2), coordinates of nodes
    :param links: numpy array (NL, 2), coordinates of links
    :param I: numpy array (NL,), indices of first node of each link
    :param J: numpy array (NL,), indices of second node of each link
    :param adjpos: numpy array (N, NL), positive regulatory matrix
    :param adjneg: numpy array (N, NL), negative regulatory matrix
    :param p: float, probability of a link not breaking randomly

    :return: tuple (RT, agn), where RT contains cluster fractions and agn is the giant component indices
    """
    N = nodes.shape[0]
    NL = links.shape[0]

    # Obtain the new state of the links
    # Regulation dynamics
    # Three conditions for a link to be active:
    # 1) It does not break randomly (with probability p): p is the probability of surviving
    f1 = p > np.random.rand(NL)
    # 2) It has at least one active node exerting positive regulation
    f2 = np.matmul(adjpos.transpose(), statenode1) > 0
    # 3) It has no active node exerting negative regulation
    f3 = np.logical_not(np.matmul(adjneg.transpose(), statenode1))

    # State of the links at in this iteration (active / inactive)
    statelink = f1 * f2 * f3
    # Indexes of active links. It will be used to create the new giant component
    linkid = np.where(statelink)[0]

    # Defining the new adjacency matrix it is a sparse matrix
    nadj = csr_matrix((np.ones(linkid.shape[0]), (I[linkid], J[linkid])), shape=(N, N))

    # Compute the giant component in this iteration
    Gn = nx.from_scipy_sparse_array(nadj)
    del nadj
    Gccn = sorted(nx.connected_components(Gn), key=len, reverse=True)  # Set of connected components

    if len(Gccn) > 0:
        ngn = len(Gccn[0])  # Number of nodes in the Giant component
        agn = list(Gccn[0])  # Vector of indices of the nodes in the Giant component
    else:
        ngn = 0
        agn = []

    # Calculate fractions for the first three clusters
    ag2_len = len(Gccn[1]) if len(Gccn) > 1 else 0
    ag3_len = len(Gccn[2]) if len(Gccn) > 2 else 0

    # This will be the output of the function: the new set of active nodes in the first three clusters for the next step
    RT = np.asarray([ngn, ag2_len, ag3_len]) * 1. / N

    return RT, agn


# =============================================================================
# SECTION 4: SCALABLE WORLD (DISCRETE RINGS)
# =============================================================================

def random_coupled_rings_netw_PBC(N: int, Lx: float, num_rings: int, delta: float, c: float, d0: float) -> tuple:
    """
    Generate a spatial network with nodes distributed across discrete coupled rings with PBC in both axes.

    :param N: int, total number of nodes
    :param Lx: float, length of the rings (X-axis)
    :param num_rings: int, number of discrete rings
    :param delta: float, separation distance between adjacent rings (Y-axis)
    :param c: float, base connectivity probability
    :param d0: float, decay length of the connectivity

    :return: tuple, (nodes (N, 2) array, G nx.Graph, adj (N, N) bool array)
    """
    nodes = np.zeros((N, 2))

    # Randomly distribute all X coordinates along the ring length Lx
    nodes[:, 0] = np.random.rand(N) * Lx

    # Assign Y coordinates based on discrete rings. Each node is randomly assigned to one of the num_rings rings,
    # separated by delta.
    ring_indices = np.random.randint(0, num_rings, size=N)
    nodes[:, 1] = ring_indices * delta

    # Total Y length for periodic boundaries across the rings
    Ly = num_rings * delta

    # Compute Euclidean distances handling Periodic Boundary Conditions in both axes
    DX = distance_PBC_1D(nodes[:, 0], nodes[:, 0], Lx)
    DY = distance_PBC_1D(nodes[:, 1], nodes[:, 1], Ly)
    D = np.sqrt(np.multiply(DX, DX) + np.multiply(DY, DY))

    # Probability for nodes to be connected
    P = c * np.exp(-D / d0)
    del D

    # Create adjacency matrix (upper triangular)
    adj = np.triu((P - np.identity(N)) > np.random.rand(N, N)).astype(bool)
    del P

    # Create nx graph object
    G = nx.from_numpy_array(adj)

    return nodes, G, adj


def coupled_rings_structural_network(N_per_ring: int, num_rings: int, Lx: float, delta: float, c: float,
                                     d0: float, cutoff_factor: float = 5.0) -> tuple:
    """
    Generate a structural network of coupled rings maintaining density per ring,
    with Periodic Boundary Conditions (PBC) in both X and Y axes.

    :param N_per_ring: int, number of nodes in each 1D ring
    :param num_rings: int, total number of coupled rings
    :param Lx: float, length of the periodic boundary box in the X axis
    :param delta: float, vertical separation between adjacent rings
    :param c: float, base connectivity probability
    :param d0: float, decay length of the connectivity
    :param cutoff_factor: float, factor to limit distance calculations for optimization
    :return: tuple, (nodes (N_tot, 2) array, G nx.Graph, adj (N_tot, N_tot) sparse bool matrix)
    """
    # ---------------------------------------------------------
    # 1. GENERATE COORDINATES
    # ---------------------------------------------------------
    N_tot = N_per_ring * num_rings
    Ly = num_rings * delta  # Total Y length needed for PBC in Y

    # X is uniformly random, Y is fixed per ring (0, delta, 2*delta...)
    X = np.random.rand(N_tot) * Lx
    Y = np.repeat(np.arange(num_rings) * delta, N_per_ring)

    nodes = np.column_stack((X, Y))

    # ---------------------------------------------------------
    # 2. FIND POSSIBLE PAIRS AND ESTABLISH LINKS
    # ---------------------------------------------------------
    cutoff_dist = cutoff_factor * d0
    I, J = [], []

    for i in range(N_tot):
        # -- Y DISTANCE --
        if num_rings > 1:
            dy = distance_PBC_1D_pair(Y[i], Y, Ly)
        else:
            dy = np.abs(Y[i] - Y)

        # Fast 1D filter: discard nodes too far in Y before doing heavy math
        valid_y_mask = dy <= cutoff_dist

        # Discard previous nodes (j <= i) to avoid duplicate edges and ensure upper triangular adj
        valid_y_mask[:i + 1] = False

        # Get the actual indices of surviving candidates
        valid_indices = np.where(valid_y_mask)[0]

        if len(valid_indices) == 0:
            continue

        # -- X DISTANCE --
        dx = distance_PBC_1D_pair(X[i], X[valid_indices], Lx)

        # -- TOTAL EUCLIDEAN DISTANCE --
        dist = np.sqrt(np.multiply(dx, dx) + np.multiply(dy[valid_indices], dy[valid_indices]))

        # -- FINAL CIRCULAR CUTOFF FILTER --
        final_mask = dist <= cutoff_dist
        final_indices = valid_indices[final_mask]
        final_dists = dist[final_mask]

        # -- PROBABILITY THROW --
        if len(final_indices) > 0:
            p_ij = c * np.exp(-final_dists / d0)
            connected_mask = np.random.rand(len(p_ij)) < p_ij

            connected_j = final_indices[connected_mask]

            # Store edges
            I.extend([i] * len(connected_j))
            J.extend(connected_j)

    # ---------------------------------------------------------
    # 3. BUILD GRAPH AND SPARSE MATRIX
    # ---------------------------------------------------------
    # Create nx graph object
    G = nx.Graph()
    G.add_nodes_from(range(N_tot))
    G.add_edges_from(zip(I, J))

    # Create adjacency matrix (upper triangular, sparse boolean)
    adj = csr_matrix((np.ones(len(I), dtype=bool), (I, J)), shape=(N_tot, N_tot))

    return nodes, G, adj


def itera_rings(Lx: float, statenode1: np.ndarray, nodes: np.ndarray, links: np.ndarray,
                I: np.ndarray, J: np.ndarray, adjpos: np.ndarray, adjneg: np.ndarray, p: float) -> tuple:
    """
    Perform one iteration of the triadic dynamics for coupled rings, tracking spatial metrics along the X-axis.

    :param Lx: float, length of the rings (X-axis)
    :param statenode1: numpy array (N,), active state of nodes at t-1
    :param nodes: numpy array (N, 2), coordinates of nodes
    :param links: numpy array (NL, 2), coordinates of links
    :param I: numpy array (NL,), indices of first node of each link
    :param J: numpy array (NL,), indices of second node of each link
    :param adjpos: numpy array (N, NL), positive regulatory matrix
    :param adjneg: numpy array (N, NL), negative regulatory matrix
    :param p: float, probability of a link not breaking randomly

    :return: tuple, (RT, agn, spatial_metrics) containing fractions, giant component indices, and circular stats
    """
    N, NL = nodes.shape[0], links.shape[0]

    # 1) Probability of surviving
    f1 = p > np.random.rand(NL)
    # 2) Positive regulation
    f2 = np.matmul(adjpos.transpose(), statenode1) > 0
    # 3) No negative regulation
    f3 = np.logical_not(np.matmul(adjneg.transpose(), statenode1))

    statelink = f1 * f2 * f3
    linkid = np.where(statelink)[0]

    nadj = csr_matrix((np.ones(linkid.shape[0]), (I[linkid], J[linkid])), shape=(N, N))
    Gn = nx.from_scipy_sparse_array(nadj)
    del nadj
    Gccn = sorted(nx.connected_components(Gn), key=len, reverse=True)

    mean1, std1, mean2, std2, mean3, std3 = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # Track phase exclusively along the continuous X-axis
    theta = nodes[:, 0] * (2. * np.pi / Lx)

    if len(Gccn) > 0:
        agn = list(Gccn[0])
        mean1, std1 = circmean(theta[agn], high=2 * np.pi, low=0), circstd(theta[agn], high=2 * np.pi, low=0)
        ngn = len(Gccn[0])
    else:
        agn, ngn = [], 0

    ag2_len, ag3_len = 0, 0
    if len(Gccn) > 1:
        ag2 = list(Gccn[1])
        mean2, std2 = circmean(theta[ag2], high=2 * np.pi, low=0), circstd(theta[ag2], high=2 * np.pi, low=0)
        ag2_len = len(Gccn[1])
        if len(Gccn) > 2:
            ag3 = list(Gccn[2])
            mean3, std3 = circmean(theta[ag3], high=2 * np.pi, low=0), circstd(theta[ag3], high=2 * np.pi, low=0)
            ag3_len = len(Gccn[2])

    RT = np.asarray([ngn, ag2_len, ag3_len]) * 1. / N
    return RT, agn, [mean1, std1, mean2, std2, mean3, std3]


# =============================================================================
# SECTION 5: UTILITIES
# =============================================================================
def calculate_cutoff_distance(c: float, d0: float, p_min: float = 1e-4) -> float:
    """
    Calculate the cutoff distance beyond which the connection probability
    falls below a specified minimum threshold.

    :param c: float, base connection probability (probability at distance 0)
    :param d0: float, structural decay length
    :param p_min: float, minimum probability threshold to consider a link possible (default 0.01%)
    :return: float, the maximum distance (cutoff) to evaluate
    """
    # If the selected threshold is bigger than the maximum probability, cutoff is 0
    if p_min >= c:
        return 0.0

    cutoff_dist = -d0 * np.log(p_min / c)
    return cutoff_dist