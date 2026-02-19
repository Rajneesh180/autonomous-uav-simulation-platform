class ClusterManager:
    def __init__(self):
        self.last_node_count = 0
        self.recluster_count = 0

    def should_recluster(self, current_node_count, threshold=5):
        if abs(current_node_count - self.last_node_count) >= threshold:
            return True
        return False

    def mark_clustered(self, node_count):
        self.last_node_count = node_count
        self.recluster_count += 1

    def get_recluster_count(self):
        return self.recluster_count
