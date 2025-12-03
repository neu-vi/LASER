import numpy as np
from scipy import ndimage


class DSU:
    def __init__(self, n):
        self.parent = np.arange(n)

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def merge_regions(labels, depth, threshold):
    """
    Merge adjacent regions in segmentation based on depth similarity.

    labels: (H, W) segmentation label map (int)
    depth:  (H, W) depth map (float)
    threshold: merging threshold (float)
    """
    unique_labels = np.unique(labels)
    n_regions = len(unique_labels)

    # Map original labels -> compact indices
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    # Compute mean depth per region
    mean_depths = ndimage.mean(depth, labels=labels, index=unique_labels)

    # Get adjacency: check right and bottom neighbors only (avoid duplicates)
    adj_pairs = set()
    for dy, dx in [(0, 1), (1, 0)]:
        neigh = np.zeros_like(labels)
        neigh[:-dy or None, :-dx or None] = labels[dy:, dx:]
        mask = (labels != neigh) & (labels != 0) & (neigh != 0)
        pairs = np.stack([labels[mask], neigh[mask]], axis=-1)
        for a, b in pairs:
            if a != b:
                ia, ib = label_to_idx[a], label_to_idx[b]
                adj_pairs.add(tuple(sorted((ia, ib))))

    # Disjoint set union merging
    dsu = DSU(n_regions)
    for ia, ib in adj_pairs:
        if abs(mean_depths[ia] - mean_depths[ib]) < threshold:
            dsu.union(ia, ib)

    # Relabel labels using merged groups
    new_labels = np.zeros_like(labels)
    for l in unique_labels:
        root = dsu.find(label_to_idx[l])
        new_labels[labels == l] = root + 1  # ensure 1-based

    return new_labels
