import numpy as np
import pygame
import time
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

# =========================================================
# CONFIGURATION
# =========================================================
ACTIVE_METHOD = "gmm"  # kmeans | dbscan | agglo | gmm
DATA_MODE = "UNIFORM"  # UNIFORM | NON_UNIFORM

WIDTH, HEIGHT = 800, 800
MAP_SIZE = 100
SCALE = WIDTH / MAP_SIZE

ANIM_DURATION = 8.0  # seconds (slow scientific reveal)

# =========================================================
# DATASET GENERATION
# =========================================================
true_centers = np.array([[20, 20], [80, 25], [30, 75], [70, 70]])

if DATA_MODE == "UNIFORM":
    sizes = [20, 20, 20, 20]
    radii = [10, 10, 10, 10]
    noise = 0
else:
    sizes = [30, 10, 25, 15]
    radii = [5, 12, 20, 8]
    noise = 12

nodes = []
for i, c in enumerate(true_centers):
    for _ in range(sizes[i]):
        r = radii[i] * np.sqrt(np.random.rand())
        t = 2 * np.pi * np.random.rand()
        nodes.append(c + [r * np.cos(t), r * np.sin(t)])

nodes.extend(np.random.uniform(0, MAP_SIZE, (noise, 2)))
nodes = np.array(nodes)

# =========================================================
# CLUSTERING
# =========================================================
start = time.time()

if ACTIVE_METHOD == "kmeans":
    model = KMeans(n_clusters=4, random_state=42)
elif ACTIVE_METHOD == "dbscan":
    model = DBSCAN(eps=10, min_samples=5)
elif ACTIVE_METHOD == "agglo":
    model = AgglomerativeClustering(n_clusters=4)
else:
    model = GaussianMixture(n_components=4)

labels = model.fit_predict(nodes)

# Estimated centers
if ACTIVE_METHOD == "kmeans":
    centers = model.cluster_centers_
elif ACTIVE_METHOD == "gmm":
    centers = model.means_
else:
    centers = np.array(
        [nodes[labels == i].mean(axis=0) for i in set(labels) if i != -1]
    )

runtime = (time.time() - start) * 1000

valid = len(set(labels)) > 1
sil = silhouette_score(nodes, labels) if valid else 0
dbi = davies_bouldin_score(nodes, labels) if valid else 999
ch = calinski_harabasz_score(nodes, labels) if valid else 0

# Center recovery error
errors = []
for c in centers:
    nearest = min(true_centers, key=lambda t: np.linalg.norm(c - t))
    errors.append(np.linalg.norm(c - nearest))
err_mean = np.mean(errors)

# =========================================================
# PYGAME INITIALIZATION
# =========================================================
pygame.init()
title = f"{ACTIVE_METHOD.upper()}_{DATA_MODE}"
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(title)
font = pygame.font.SysFont("Arial", 16)
clock = pygame.time.Clock()


def to_screen(p):
    return int(p[0] * SCALE), int((MAP_SIZE - p[1]) * SCALE)


COLORS = [(0, 120, 255), (255, 150, 0), (0, 200, 0), (200, 0, 200)]
start_time = time.time()

# =========================================================
# MAIN LOOP
# =========================================================
running = True
while running:
    clock.tick(60)

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

    # Slow animation progress
    elapsed = time.time() - start_time
    prog = min(elapsed / ANIM_DURATION, 1.0)

    screen.fill((255, 255, 255))

    # ---------------- DRAW NODES ----------------
    for i, p in enumerate(nodes):
        color = (255, 0, 0) if labels[i] == -1 else COLORS[labels[i] % 4]
        pygame.draw.circle(screen, color, to_screen(p), 3)

    # ---------------- TRUE CENTERS ----------------
    for tc in true_centers:
        pygame.draw.circle(screen, (0, 180, 0), to_screen(tc), 6)

    # ---------------- ESTIMATED CENTERS + ERROR ----------------
    for c in centers:
        pygame.draw.circle(screen, (0, 0, 0), to_screen(c), 8, 2)

        nearest = min(true_centers, key=lambda t: np.linalg.norm(c - t))

        # Error vector
        pygame.draw.line(screen, (255, 0, 0), to_screen(c), to_screen(nearest), 1)

        # Error circle with easing
        radius = np.linalg.norm(c - nearest) * SCALE * (prog**1.5)
        pygame.draw.circle(screen, (255, 0, 0), to_screen(c), int(radius), 1)

    # ---------------- METRICS PANEL ----------------
    metrics = [
        f"Silhouette Score        : {sil * prog:.2f}",
        f"Davies–Bouldin Index    : {dbi * prog:.2f}",
        f"Calinski–Harabasz Score : {ch * prog:.1f}",
        f"Center Recovery Error   : {err_mean * prog:.2f} m",
        f"Runtime                 : {runtime * prog:.1f} ms",
    ]

    for i, text in enumerate(metrics):
        screen.blit(font.render(text, True, (0, 0, 0)), (480, 20 + i * 25))

    pygame.display.flip()

pygame.quit()
