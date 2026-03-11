# distutils: language=c++
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as cnp

# Iterative DSU Find with path compression (prevents stack overflow)
cdef int find(int* parent, int i) nogil:
    cdef int root = i
    cdef int curr = i
    cdef int nxt

    # Find the root
    while parent[root] != root:
        root = parent[root]

    # Path compression: make all nodes point directly to the root
    while parent[curr] != root:
        nxt = parent[curr]
        parent[curr] = root
        curr = nxt

    return root

# DSU Union
cdef void union_sets(int* parent, int i, int j) nogil:
    cdef int root_i = find(parent, i)
    cdef int root_j = find(parent, j)
    if root_i != root_j:
        parent[root_i] = root_j

def _fast_graph_segmentation(cnp.float64_t[:, :, ::1] image, double threshold):
    """
    Simplified segmentation: merges adjacent pixels if their
    squared Euclidean color distance is below the threshold.
    """
    cdef int height = image.shape[0]
    cdef int width = image.shape[1]
    cdef int channels = image.shape[2]
    cdef int num_pixels = height * width

    # Initialize DSU: each pixel is its own parent
    cdef cnp.ndarray[cnp.int32_t, ndim=1] parent_arr = np.arange(num_pixels, dtype=np.int32)
    cdef int[::1] parent_view = parent_arr
    cdef int* parent = &parent_view[0]

    cdef int r, c, k
    cdef double dist_sq, diff
    cdef int curr_idx, neighbor_idx
    cdef double thresh_sq = threshold * threshold

    with nogil:
        for r in range(height):
            for c in range(width):
                curr_idx = r * width + c

                # Check Left Neighbor
                if c > 0:
                    neighbor_idx = curr_idx - 1
                    dist_sq = 0.0
                    for k in range(channels):
                        diff = image[r, c, k] - image[r, c - 1, k]
                        dist_sq += diff * diff

                    if dist_sq <= thresh_sq:
                        union_sets(parent, curr_idx, neighbor_idx)

                # Check Up Neighbor
                if r > 0:
                    neighbor_idx = curr_idx - width
                    dist_sq = 0.0
                    for k in range(channels):
                        diff = image[r, c, k] - image[r - 1, c, k]
                        dist_sq += diff * diff

                    if dist_sq <= thresh_sq:
                        union_sets(parent, curr_idx, neighbor_idx)

    # Flatten all trees so every node points directly to its root
    for r in range(num_pixels):
        parent_arr[r] = find(parent, r)

    # Fast, vectorized relabeling mapping arbitrary roots to [0, N-1]
    _, labels_flat = np.unique(parent_arr, return_inverse=True)

    return labels_flat.reshape((height, width)).astype(np.int32)


def fast_graph_segmentation(image, threshold):
    """
    Python wrapper to handle 2D (grayscale) and 3D (color) images safely.
    """
    image = np.asarray(image, dtype=np.float64)

    # Handle single-channel 2D images
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    elif image.ndim != 3:
        raise ValueError("Image must be 2D or 3D.")

    # Ensure memory is contiguous for Cython
    if not image.flags['C_CONTIGUOUS']:
        image = np.ascontiguousarray(image)

    return _fast_graph_segmentation(image, threshold)
