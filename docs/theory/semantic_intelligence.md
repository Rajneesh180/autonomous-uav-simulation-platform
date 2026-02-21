# Phase 4: Semantic Intelligence and Feature Space (Theory)

Current UAV routing algorithms heavily over-prioritize geometric proximity, solving traditional Traveling Salesperson Problems (TSP) while ignoring the underlying characteristics of the nodes themselves. 

The Semantic Intelligence engine upgrades the environment from a purely spatial network to a dense feature space, allowing the UAV algorithm to execute decisions based on urgency, risk, and data value.

## 1. Multi-Dimensional Node Feature Vectors
Each IoT node $i$ is expanded from a simple coordinate tuple $(x_i, y_i, z_i)$ into an operational feature vector $F_i$:
$$ F_i = [P_i, R_i, B_i, \tau_i] $$

Where:
*   **Priority ($P_i$):** Domain importance of the node (e.g., medical sensor = 0.9, agricultural moisture = 0.2).
*   **Risk Profile ($R_i$):** Inherent danger of the sector the node occupies, derived from the environmental risk map.
*   **Buffer Volume ($B_i$):** Accumulating data waiting for collection, measured in Megabits (Mb).
*   **Time-to-Deadline ($\tau_i$):** Remaining epochs before buffer overflow or data expiration.

## 2. Normalization and Scaling Engine
Before distance matrices can be evaluated against semantic features, $F_i$ must be normalized to prevent magnitude dominance (e.g., $B_i \in [0, 50]$ overshadowing $P_i \in [0, 1]$).

*   **Min-Max Scaling:** Applied to static bounds like Priority.
    $$ P'_{i} = \frac{P_i - P_{min}}{P_{max} - P_{min}} $$
*   **Z-Score Standardization:** Applied to volatile metrics like Buffer Volume.
    $$ B'_{i} = \frac{B_i - \mu_B}{\sigma_B} $$
*   **Exponential Time Decay:** Applied to Deadlines to sharply penalize expiring nodes.
    $$ \tau'_{i} = \exp(-\lambda \cdot \tau_i) $$

## 3. Semantic Weighted Clustering (Hybrid K-Means)
To scale to thousands of nodes, the system clusters geographically proximate nodes that share similar semantic profiles. The traditional Euclidean distance function $D(i, j)$ is replaced with a Semantic Distance $S(i, j)$:

$$ S(i, j) = \alpha \cdot || \vec{X_i} - \vec{X_j} ||_2 + \beta \cdot || F'_i - F'_j ||_2 $$

Where $\vec{X}$ is the 3D spatial coordinate, $F'$ is the scaled feature vector, and $\alpha, \beta$ are the tunable spatial-vs-semantic weighting coefficients. This results in the UAV prioritizing clusters of dense, high-value expiring nodes over nearby but empty, low-priority sectors.

## 4. Evaluation Metrics
To quantify the effectiveness of semantic routing, new metrics are tracked during experiments:
*   **Priority Satisfaction Rate:** Percentage of high-priority ($P_i > 0.8$) node buffers successfully drained.
*   **Semantic Purity Index:** Variance of the feature vectors within a generated cluster (lower variance = higher purity).
