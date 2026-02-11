"""
UAV Path Planning with Clustering and Live Cost Visualization

This script demonstrates:
1. Controlled generation of IoT clusters in a 100x100 meter area
2. Recovery of cluster centers using unsupervised clustering (K-Means, Fuzzy C-Means)
3. UAV path minimization by visiting cluster centers using TSP
4. Live visualization of UAV movement with incremental path length (P) and energy cost (C)

All quantities are expressed in physical units (meters, Joules).
"""

import pygame
import numpy as np
import itertools
from sklearn.cluster import KMeans

# =========================================================
# CONFIGURATION PARAMETERS
# =========================================================

# Screen resolution for visualization (pixels)
WIDTH, HEIGHT = 800, 800

# Logical size of the environment (meters)
MAP_SIZE = 100.0

# Conversion factor: meters → pixels
# Used to map physical coordinates to screen coordinates
SCALE = WIDTH / MAP_SIZE

# Number of clusters to generate in the area
NUM_CLUSTERS = 4

# Number of IoT nodes per cluster
NODES_PER_CLUSTER = 20

# Radius of each cluster (meters)
CLUSTER_RADIUS = 10.0

# UAV motion parameters
UAV_SPEED = 0.4  # meters per frame (controls animation speed)
ENERGY_PER_METER = 1.0  # Joules per meter (linear energy model)

# =========================================================
# INITIALIZE PYGAME ENVIRONMENT
# =========================================================

pygame.init()

# Create a window for visualization
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Window title
pygame.display.set_caption("Clustering & Path Minimization")

# Clock object to control frame rate
clock = pygame.time.Clock()

# Font used for all on-screen text
font = pygame.font.SysFont("Arial", 14)

# =========================================================
# COORDINATE TRANSFORMATION FUNCTION
# =========================================================


def to_screen(pt):
    """
    Converts Cartesian coordinates (origin at bottom-left, meters)
    into Pygame screen coordinates (origin at top-left, pixels).

    This ensures the visualization behaves like a standard X–Y graph.
    """
    x = int(pt[0] * SCALE)
    y = int((MAP_SIZE - pt[1]) * SCALE)
    return x, y


# =========================================================
# DATA COLLECTION CENTER (START AND END POINT)
# =========================================================

# The UAV always starts and ends at the data collection center
# Placed at the geometric center of the area
DCC = np.array([50.0, 50.0])

# =========================================================
# TRUE (GROUND-TRUTH) CLUSTER CENTERS
# =========================================================

# These centers define where IoT devices are deployed physically
# They are NOT given to the clustering algorithms
true_centers = np.array([[25, 25], [75, 25], [25, 75], [75, 75]])

# =========================================================
# GENERATE IOT NODES AROUND EACH TRUE CENTER
# =========================================================

nodes = []  # Stores all node coordinates
true_labels = []  # Stores the true cluster index for coloring

for idx, center in enumerate(true_centers):
    for _ in range(NODES_PER_CLUSTER):
        # Randomly sample points uniformly inside a circle
        r = CLUSTER_RADIUS * np.sqrt(np.random.rand())
        theta = 2 * np.pi * np.random.rand()

        # Convert polar coordinates to Cartesian
        node_position = center + np.array([r * np.cos(theta), r * np.sin(theta)])

        nodes.append(node_position)
        true_labels.append(idx)

# Convert list to NumPy array for numerical operations
nodes = np.array(nodes)

# =========================================================
# K-MEANS CLUSTERING (HARD ASSIGNMENT)
# =========================================================

# K-Means attempts to recover cluster centers using only node positions
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)

# Assign each node to a cluster
labels_km = kmeans.fit_predict(nodes)

# Extract the estimated cluster centers
centers_km = kmeans.cluster_centers_

# =========================================================
# FUZZY C-MEANS CLUSTERING (SOFT ASSIGNMENT)
# =========================================================


def fuzzy_c_means(X, c, m=2, iters=100):
    """
    Implements a basic Fuzzy C-Means clustering algorithm.

    X     : data points
    c     : number of clusters
    m     : fuzziness parameter
    iters : number of iterations
    """
    n = X.shape[0]

    # Initialize random membership values
    U = np.random.rand(n, c)
    U /= U.sum(axis=1, keepdims=True)

    for _ in range(iters):
        # Raise membership matrix to power m
        um = U**m

        # Compute cluster centers
        centers = (um.T @ X) / um.sum(axis=0)[:, None]

        # Compute distances between points and centers
        dist = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        dist = np.maximum(dist, 1e-6)  # avoid division by zero

        # Update membership matrix
        U = 1.0 / (dist ** (2 / (m - 1)))
        U /= U.sum(axis=1, keepdims=True)

    return centers


# Run Fuzzy C-Means clustering
centers_fcm = fuzzy_c_means(nodes, NUM_CLUSTERS)

# =========================================================
# CLUSTER CENTER RECOVERY ERROR
# =========================================================


def center_error(recovered, true):
    """
    Computes mean distance between recovered centers
    and the nearest true cluster center.

    This measures how accurately clustering recovers
    the physical deployment centers.
    """
    return np.mean([min(np.linalg.norm(r - t) for t in true) for r in recovered])


# Calculate errors for both clustering methods
err_km = center_error(centers_km, true_centers)
err_fcm = center_error(centers_fcm, true_centers)

# =========================================================
# PATH MINIMIZATION USING TSP
# =========================================================


def path_length(path):
    """
    Computes total length of a path defined
    by an ordered list of waypoints.
    """
    return sum(np.linalg.norm(path[i] - path[i + 1]) for i in range(len(path) - 1))


# Solve TSP by brute force (feasible for small cluster count)
best_path = None
best_cost = float("inf")

for perm in itertools.permutations(centers_km):
    candidate_path = [DCC] + list(perm) + [DCC]
    candidate_cost = path_length(candidate_path)

    if candidate_cost < best_cost:
        best_cost = candidate_cost
        best_path = np.array(candidate_path)

# =========================================================
# UAV STATE AND LIVE METRICS
# =========================================================

# UAV starts at the data collection center
uav_pos = best_path[0].copy()

# Index of the next waypoint to visit
target_idx = 1

# Accumulated path length (meters)
path_P = 0.0

# Accumulated energy cost (Joules)
cost_C = 0.0

# Flag indicating completion of the full tour
completed = False

# =========================================================
# COLOR DEFINITIONS
# =========================================================

NODE_COLORS = [(0, 120, 255), (0, 200, 0), (255, 180, 0), (200, 0, 200)]

UAV_COLOR = (30, 144, 255)  # Blue for UAV
DCC_COLOR = (220, 20, 60)  # Red for DCC

# =========================================================
# MAIN SIMULATION LOOP
# =========================================================

running = True
while running:
    clock.tick(60)  # maintain ~60 FPS

    # Handle window close event
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # ---------------- UAV MOTION ----------------
    if target_idx < len(best_path):
        target = best_path[target_idx]
        direction = target - uav_pos
        distance = np.linalg.norm(direction)

        if distance > 0:
            # Move UAV by a fixed step or remaining distance
            step = min(UAV_SPEED, distance)
            movement = (direction / distance) * step
            uav_pos += movement

            # Update live path and energy metrics
            path_P += step
            cost_C += step * ENERGY_PER_METER

            # Switch to next waypoint if reached
            if step == distance:
                target_idx += 1
    else:
        completed = True

    # ---------------- DRAWING ----------------
    screen.fill((255, 255, 255))

    # Draw grid and axis ticks every 10 meters
    for i in range(0, 101, 10):
        pygame.draw.line(
            screen, (235, 235, 235), to_screen([i, 0]), to_screen([i, 100])
        )
        pygame.draw.line(
            screen, (235, 235, 235), to_screen([0, i]), to_screen([100, i])
        )

        screen.blit(
            font.render(str(i), True, (0, 0, 0)),
            (to_screen([i, 0])[0] - 8, HEIGHT - 18),
        )
        screen.blit(font.render(str(i), True, (0, 0, 0)), (5, to_screen([0, i])[1] - 6))

    # Draw cluster boundaries
    for c in true_centers:
        pygame.draw.circle(
            screen, (200, 200, 200), to_screen(c), int(CLUSTER_RADIUS * SCALE), 1
        )

    # Draw IoT nodes
    for i, p in enumerate(nodes):
        pygame.draw.circle(screen, NODE_COLORS[true_labels[i]], to_screen(p), 3)

    # Draw true cluster centers
    for c in true_centers:
        pygame.draw.circle(screen, (0, 0, 0), to_screen(c), 6)

    # Draw K-Means estimated centers
    for c in centers_km:
        pygame.draw.circle(screen, (0, 0, 0), to_screen(c), 10, 2)

    # Draw DCC
    pygame.draw.circle(screen, DCC_COLOR, to_screen(DCC), 9)

    # Draw optimal path
    for i in range(len(best_path) - 1):
        pygame.draw.line(
            screen,
            (120, 120, 120),
            to_screen(best_path[i]),
            to_screen(best_path[i + 1]),
            3,
        )

    # Draw UAV
    pygame.draw.circle(screen, UAV_COLOR, to_screen(uav_pos), 7)

    # Draw moving P and C labels while UAV is in motion
    if not completed:
        px, py = to_screen(uav_pos)
        screen.blit(
            font.render(f"P = {path_P:.1f} m", True, (0, 0, 0)), (px + 10, py - 20)
        )
        screen.blit(
            font.render(f"C = {cost_C:.1f} J", True, (0, 0, 0)), (px + 10, py - 6)
        )

    # Draw final summary after completion
    if completed:
        summary = [
            f"Total Path Traversed : {path_P:.2f} m",
            f"Total Energy Cost   : {cost_C:.2f} J",
            f"UAV Speed           : {UAV_SPEED * 60:.1f} m/s",
            f"K-Means Error       : {err_km:.2f} m",
            f"Fuzzy Error         : {err_fcm:.2f} m",
        ]
        for i, text in enumerate(summary):
            screen.blit(font.render(text, True, (0, 0, 0)), (520, 10 + i * 18))

    pygame.display.flip()

pygame.quit()
