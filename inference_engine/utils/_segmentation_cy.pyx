# distutils: language=c++
# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

import numpy as np
cimport numpy as cnp
from scipy import ndimage

# Cython DSU implementation
cdef class DSU:
    cdef cnp.ndarray parent   # numpy array of int64
    cdef long[:] parent_view  # memoryview for fast access

    def __init__(self, int n):
        self.parent = np.arange(n, dtype=np.int64)
        self.parent_view = self.parent

    cdef int find(self, int x):
        cdef long[:] parent = self.parent_view
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    cpdef void union(self, int a, int b):
        cdef int ra = self.find(a)
        cdef int rb = self.find(b)
        if ra != rb:
            self.parent_view[rb] = ra


def merge_regions(cnp.ndarray[cnp.int64_t, ndim=2] labels,
                  cnp.ndarray[cnp.float32_t, ndim=2] depth,
                  float threshold):
    """
    Merge adjacent regions in segmentation based on depth similarity.

    labels: (H, W) segmentation label map (int64)
    depth:  (H, W) depth map (float32)
    threshold: merging threshold (float)
    """
    cdef cnp.ndarray[cnp.int64_t] unique_labels = np.unique(labels)
    cdef int n_regions = unique_labels.shape[0]

    # Map original labels -> compact indices
    label_to_idx = {int(l): i for i, l in enumerate(unique_labels)}

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
                ia, ib = label_to_idx[int(a)], label_to_idx[int(b)]
                adj_pairs.add(tuple(sorted((ia, ib))))

    # Disjoint set union merging (fast in Cython)
    dsu = DSU(n_regions)
    for ia, ib in adj_pairs:
        if abs(mean_depths[ia] - mean_depths[ib]) < threshold:
            dsu.union(ia, ib)

    # Relabel labels using merged groups
    new_labels = np.zeros_like(labels, dtype=np.int64)
    for l in unique_labels:
        root = dsu.find(label_to_idx[int(l)])
        new_labels[labels == l] = root + 1  # ensure 1-based

    return new_labels
