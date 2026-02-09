from core.dataset_generator import generate_nodes


def test_node_generation():
    nodes = generate_nodes("random", 10, 100, 100, seed=42)
    assert len(nodes) == 10


def test_seed_reproducibility():
    n1 = generate_nodes("random", 5, 100, 100, seed=42)
    n2 = generate_nodes("random", 5, 100, 100, seed=42)
    assert n1[0].x == n2[0].x and n1[0].y == n2[0].y
