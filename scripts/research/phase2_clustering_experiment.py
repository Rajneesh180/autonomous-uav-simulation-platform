"""
UAV Clustering Research Framework
----------------------------------
Modes:
- BATCH  : prints comparison table
- VISUAL : shows annotated visualization

Features:
- Uniform vs Non-Uniform dataset toggle
- Metric overlay
- Cluster coloring
- True vs Estimated centers
- Error vectors
- Noise highlighting
- Path length display
"""

# =========================================================
# SWITCHES
# =========================================================
EXPERIMENT_MODE = "VISUAL"  # "BATCH" or "VISUAL"
ACTIVE_METHOD = "kmeans"  # kmeans | dbscan | agglo | gmm
DATA_MODE = "NON_UNIFORM"  # "UNIFORM" or "NON_UNIFORM"

# =========================================================
# IMPORTS
# =========================================================
import numpy as np
import time
import itertools

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

# =========================================================
# DATA CONFIG
# =========================================================
MAP_SIZE = 100.0
NUM_CLUSTERS = 4

if DATA_MODE == "UNIFORM":
    cluster_sizes = [20, 20, 20, 20]
    cluster_radii = [10, 10, 10, 10]
    noise_points = 0
else:
    cluster_sizes = [30, 10, 25, 15]
    cluster_radii = [5, 12, 20, 8]
    noise_points = 12

true_centers = np.array([[20, 20], [80, 25], [30, 75], [70, 70]])

# =========================================================
# DATA GENERATION
# =========================================================
nodes = []
for i, c in enumerate(true_centers):
    for _ in range(cluster_sizes[i]):
        r = cluster_radii[i] * np.sqrt(np.random.rand())
        t = 2 * np.pi * np.random.rand()
        nodes.append(c + np.array([r * np.cos(t), r * np.sin(t)]))

noise = np.random.uniform(0, MAP_SIZE, (noise_points, 2))
nodes.extend(noise)
nodes = np.array(nodes)


# =========================================================
# UTILITIES
# =========================================================
def center_error(recovered, true):
    return np.mean([min(np.linalg.norm(r - t) for t in true) for r in recovered])


def safe_metrics(X, labels):
    if len(set(labels)) < 2 or -1 in labels:
        return 0, 999, 0
    return (
        silhouette_score(X, labels),
        davies_bouldin_score(X, labels),
        calinski_harabasz_score(X, labels),
    )


# =========================================================
# CLUSTER ENGINE
# =========================================================
def run_clustering(method, X):
    start = time.time()

    if method == "kmeans":
        m = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
        labels = m.fit_predict(X)
        centers = m.cluster_centers_

    elif method == "dbscan":
        m = DBSCAN(eps=10, min_samples=5)
        labels = m.fit_predict(X)
        centers = np.array(
            [X[labels == i].mean(axis=0) for i in set(labels) if i != -1]
        )

    elif method == "agglo":
        m = AgglomerativeClustering(n_clusters=NUM_CLUSTERS)
        labels = m.fit_predict(X)
        centers = np.array([X[labels == i].mean(axis=0) for i in range(NUM_CLUSTERS)])

    elif method == "gmm":
        m = GaussianMixture(n_components=NUM_CLUSTERS, random_state=42)
        labels = m.fit_predict(X)
        centers = m.means_

    runtime = (time.time() - start) * 1000
    sil, dbi, ch = safe_metrics(X, labels)
    err = center_error(centers, true_centers)

    return centers, labels, runtime, sil, dbi, ch, err


# =========================================================
# BATCH MODE
# =========================================================
methods = ["kmeans", "dbscan", "agglo", "gmm"]
if EXPERIMENT_MODE == "BATCH":
    print("\nAlgorithm Comparison\n")
    print("Algo | Sil | DBI | CH | Time(ms) | Error")
    print("-----------------------------------------")
    for m in methods:
        _, _, t, s, d, c, e = run_clustering(m, nodes)
        print(f"{m:<6}|{s:>5.2f}|{d:>5.2f}|{c:>5.1f}|{t:>9.1f}|{e:>6.2f}")
    exit()

# =========================================================
# VISUAL MODE
# =========================================================
import pygame

WIDTH = 800
HEIGHT = 800
SCALE = WIDTH / MAP_SIZE
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
font = pygame.font.SysFont("Arial", 14)
clock = pygame.time.Clock()


def to_screen(p):
    return int(p[0] * SCALE), int((MAP_SIZE - p[1]) * SCALE)


centers, labels, runtime, sil, dbi, ch, err = run_clustering(ACTIVE_METHOD, nodes)

# =========================================================
# TSP PATH
# =========================================================
DCC = np.array([50.0, 50.0])


def path_len(p):
    return sum(np.linalg.norm(p[i] - p[i + 1]) for i in range(len(p) - 1))


best = None
cost = float("inf")
for perm in itertools.permutations(centers):
    p = [DCC] + list(perm) + [DCC]
    c = path_len(p)
    if c < cost:
        cost = c
        best = np.array(p)

# =========================================================
# UAV
# =========================================================
uav = best[0].copy()
idx = 1
speed = 0.5

COLORS = [(0, 120, 255), (255, 150, 0), (0, 200, 0), (200, 0, 200)]

running = True
while running:
    clock.tick(60)
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

    if idx < len(best):
        d = best[idx] - uav
        dist = np.linalg.norm(d)
        if dist > 0:
            step = min(speed, dist)
            uav += d / dist * step
            if step == dist:
                idx += 1

    screen.fill((255, 255, 255))

    # Nodes colored
    for i, p in enumerate(nodes):
        col = (255, 0, 0) if labels[i] == -1 else COLORS[labels[i] % 4]
        pygame.draw.circle(screen, col, to_screen(p), 3)

    # True centers
    for tc in true_centers:
        pygame.draw.circle(screen, (0, 200, 0), to_screen(tc), 6)

    # Estimated centers + error vectors
    for ec in centers:
        pygame.draw.circle(screen, (0, 0, 0), to_screen(ec), 8, 2)
        nearest = min(true_centers, key=lambda t: np.linalg.norm(t - ec))
        pygame.draw.line(screen, (255, 0, 0), to_screen(ec), to_screen(nearest), 1)

    # Path
    for i in range(len(best) - 1):
        pygame.draw.line(
            screen, (120, 120, 120), to_screen(best[i]), to_screen(best[i + 1]), 2
        )

    pygame.draw.circle(screen, (255, 0, 0), to_screen(uav), 6)

    # Metric Overlay Panel
    panel = [
        f"Method: {ACTIVE_METHOD}",
        f"Data  : {DATA_MODE}",
        f"Sil   : {sil:.2f}",
        f"DBI   : {dbi:.2f}",
        f"CH    : {ch:.1f}",
        f"Err   : {err:.2f}m",
        f"Run   : {runtime:.1f}ms",
        f"Path  : {cost:.1f}m",
    ]
    for i, t in enumerate(panel):
        screen.blit(font.render(t, True, (0, 0, 0)), (600, 10 + i * 18))

    pygame.display.flip()

pygame.quit()
"""
UAV CLUSTERING RESEARCH FRAMEWORK
---------------------------------
This framework demonstrates clustering + UAV path planning.

MODES
-----
BATCH  -> Prints algorithm comparison table in terminal
VISUAL -> Animated UAV traversal with clustering visualization

FEATURES
--------
• Uniform vs Non-Uniform dataset toggle
• Cluster coloring
• True vs Estimated centers
• Error vectors
• Noise highlighting
• UAV path length display
• Live metric overlay
"""

# =========================================================
# ======================== SWITCHES =======================
# =========================================================
"""
EXPERIMENT MODE OPTIONS
-----------------------
"BATCH"  : Numeric comparison table
"VISUAL" : Interactive UAV visualization
"""
EXPERIMENT_MODE = "VISUAL"

"""
ALGORITHM OPTIONS (VISUAL MODE)
--------------------------------
"K-Means"
"DBSCAN"
"Agglomerative Hierarchical"
"Gaussian Mixture Models"
"""
ACTIVE_METHOD = "K-Means"

"""
DATASET OPTIONS
---------------
"UNIFORM"      : Equal cluster sizes and radii
"NON_UNIFORM"  : Unequal sizes, radii, and noise
"""
DATA_MODE = "NON_UNIFORM"

# =========================================================
# ======================== IMPORTS ========================
# =========================================================
import numpy as np
import time
import itertools

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

# =========================================================
# ======================== DATA CONFIG ====================
# =========================================================
MAP_SIZE = 100.0
NUM_CLUSTERS = 4
np.random.seed(42)

if DATA_MODE == "UNIFORM":
    cluster_sizes = [20, 20, 20, 20]
    cluster_radii = [10, 10, 10, 10]
    noise_points = 0
else:
    cluster_sizes = [30, 10, 25, 15]
    cluster_radii = [5, 12, 20, 8]
    noise_points = 12

true_centers = np.array([[20, 20], [80, 25], [30, 75], [70, 70]])

# =========================================================
# ======================== DATA GENERATION ================
# =========================================================
nodes = []
for i, c in enumerate(true_centers):
    for _ in range(cluster_sizes[i]):
        r = cluster_radii[i] * np.sqrt(np.random.rand())
        t = 2 * np.pi * np.random.rand()
        nodes.append(c + np.array([r * np.cos(t), r * np.sin(t)]))

noise = np.random.uniform(0, MAP_SIZE, (noise_points, 2))
nodes.extend(noise)
nodes = np.array(nodes)


# =========================================================
# ======================== METRIC UTILITIES ===============
# =========================================================
def center_error(recovered, true):
    return np.mean([min(np.linalg.norm(r - t) for t in true) for r in recovered])


def safe_metrics(X, labels):
    if len(set(labels)) < 2 or -1 in labels:
        return 0, 999, 0
    return (
        silhouette_score(X, labels),
        davies_bouldin_score(X, labels),
        calinski_harabasz_score(X, labels),
    )


# =========================================================
# ======================== CLUSTER ENGINE =================
# =========================================================
def run_clustering(method, X):
    start = time.time()

    if method == "K-Means":
        m = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
        labels = m.fit_predict(X)
        centers = m.cluster_centers_

    elif method == "DBSCAN":
        m = DBSCAN(eps=10, min_samples=5)
        labels = m.fit_predict(X)
        centers = np.array(
            [X[labels == i].mean(axis=0) for i in set(labels) if i != -1]
        )

    elif method == "Agglomerative Hierarchical":
        m = AgglomerativeClustering(n_clusters=NUM_CLUSTERS)
        labels = m.fit_predict(X)
        centers = np.array([X[labels == i].mean(axis=0) for i in range(NUM_CLUSTERS)])

    elif method == "Gaussian Mixture Models":
        m = GaussianMixture(n_components=NUM_CLUSTERS, random_state=42)
        labels = m.fit_predict(X)
        centers = m.means_

    runtime = (time.time() - start) * 1000
    sil, dbi, ch = safe_metrics(X, labels)
    err = center_error(centers, true_centers)

    return centers, labels, runtime, sil, dbi, ch, err


# =========================================================
# ======================== BATCH MODE =====================
# =========================================================
methods = ["K-Means", "DBSCAN", "Agglomerative Hierarchical", "Gaussian Mixture Models"]

if EXPERIMENT_MODE == "BATCH":
    print("\nCLUSTERING ALGORITHM COMPARISON\n")
    header = f"{'Algorithm':30s} | {'Silhouette Score':16s} | {'Davies–Bouldin Index':20s} | {'Calinski–Harabasz Score':23s} | {'Runtime (ms)':12s} | {'Center Error':12s}"
    print(header)
    print("-" * len(header))
    for m in methods:
        _, _, rt, s, d, c, e = run_clustering(m, nodes)
        print(f"{m:30s} | {s:16.3f} | {d:20.3f} | {c:23.1f} | {rt:12.1f} | {e:12.2f}")
    exit()

# =========================================================
# ======================== VISUAL MODE ====================
# =========================================================
import pygame

WIDTH = 820
HEIGHT = 820
SCALE = WIDTH / MAP_SIZE
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("UAV Clustering Visualization")
font = pygame.font.SysFont("Arial", 16)
clock = pygame.time.Clock()


def to_screen(p):
    return int(p[0] * SCALE), int((MAP_SIZE - p[1]) * SCALE)


centers, labels, runtime, sil, dbi, ch, err = run_clustering(ACTIVE_METHOD, nodes)

# =========================================================
# ======================== TSP PATH =======================
# =========================================================
DCC = np.array([50.0, 50.0])


def path_len(p):
    return sum(np.linalg.norm(p[i] - p[i + 1]) for i in range(len(p) - 1))


best = None
cost = float("inf")
for perm in itertools.permutations(centers):
    p = [DCC] + list(perm) + [DCC]
    c = path_len(p)
    if c < cost:
        cost = c
        best = np.array(p)

# =========================================================
# ======================== UAV STATE ======================
# =========================================================
uav = best[0].copy()
idx = 1
speed = 0.5
COLORS = [(52, 152, 219), (231, 76, 60), (46, 204, 113), (155, 89, 182)]

running = True
while running:
    clock.tick(60)
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

    if idx < len(best):
        d = best[idx] - uav
        dist = np.linalg.norm(d)
        if dist > 0:
            step = min(speed, dist)
            uav += d / dist * step
            if step == dist:
                idx += 1

    screen.fill((240, 240, 240))

    # Nodes
    for i, p in enumerate(nodes):
        col = (200, 200, 200) if labels[i] == -1 else COLORS[labels[i] % 4]
        pygame.draw.circle(screen, col, to_screen(p), 3)

    # True centers
    for tc in true_centers:
        pygame.draw.circle(screen, (0, 180, 0), to_screen(tc), 6)

    # Estimated centers + error vectors
    for ec in centers:
        pygame.draw.circle(screen, (0, 0, 0), to_screen(ec), 8, 2)
        nearest = min(true_centers, key=lambda t: np.linalg.norm(t - ec))
        pygame.draw.line(screen, (255, 0, 0), to_screen(ec), to_screen(nearest), 1)

    # Path
    for i in range(len(best) - 1):
        pygame.draw.line(
            screen, (120, 120, 120), to_screen(best[i]), to_screen(best[i + 1]), 2
        )

    pygame.draw.circle(screen, (255, 0, 0), to_screen(uav), 6)

    # ===== METRIC OVERLAY PANEL =====
    panel = pygame.Surface((300, 170))
    panel.set_alpha(230)
    panel.fill((255, 255, 255))
    screen.blit(panel, (500, 20))

    lines = [
        f"Algorithm                : {ACTIVE_METHOD}",
        f"Dataset                  : {DATA_MODE}",
        f"Silhouette Score         : {sil:.2f}",
        f"Davies–Bouldin Index     : {dbi:.2f}",
        f"Calinski–Harabasz Score  : {ch:.1f}",
        f"Center Recovery Error    : {err:.2f} m",
        f"Runtime                  : {runtime:.1f} ms",
        f"UAV Path Length          : {cost:.1f} m",
    ]

    for i, t in enumerate(lines):
        screen.blit(font.render(t, True, (20, 20, 20)), (520, 35 + i * 22))

    pygame.display.flip()

pygame.quit()
