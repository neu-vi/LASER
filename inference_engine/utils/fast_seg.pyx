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

    while parent[root] != root:
        root = parent[root]

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
    Simplified segmentation with implicit depth-ramp prevention.
    Flying pixels (ramps) are assigned a label of -1.
    """
    cdef int height = image.shape[0]
    cdef int width = image.shape[1]
    cdef int channels = image.shape[2]
    cdef int num_pixels = height * width

    # Initialize DSU: each pixel is its own parent
    cdef cnp.ndarray[cnp.int32_t, ndim=1] parent_arr = np.arange(num_pixels, dtype=np.int32)
    cdef int[::1] parent_view = parent_arr
    cdef int* parent = &parent_view[0]

    # The implicit edge mask array
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] safe_mask_arr = np.ones((height, width), dtype=np.uint8)
    cdef cnp.uint8_t[:, ::1] safe_mask = safe_mask_arr

    cdef int r, c, k
    cdef double dist_sq, diff
    cdef int curr_idx, neighbor_idx

    cdef double thresh_sq = threshold * threshold
    cdef double max_span_sq = thresh_sq * 1.0

    with nogil:
        # --- PASS 1: Identify Edge/Ramp Pixels ---
        for r in range(height):
            for c in range(width):
                if c > 0 and c < width - 1:
                    dist_sq = 0.0
                    for k in range(channels):
                        diff = image[r, c+1, k] - image[r, c-1, k]
                        dist_sq += diff * diff
                    if dist_sq > max_span_sq:
                        safe_mask[r, c] = 0
                        continue

                if r > 0 and r < height - 1:
                    dist_sq = 0.0
                    for k in range(channels):
                        diff = image[r+1, c, k] - image[r-1, c, k]
                        dist_sq += diff * diff
                    if dist_sq > max_span_sq:
                        safe_mask[r, c] = 0

        # --- PASS 2: DSU Merging ---
        for r in range(height):
            for c in range(width):
                curr_idx = r * width + c

                if c > 0 and safe_mask[r, c] and safe_mask[r, c - 1]:
                    neighbor_idx = curr_idx - 1
                    dist_sq = 0.0
                    for k in range(channels):
                        diff = image[r, c, k] - image[r, c - 1, k]
                        dist_sq += diff * diff

                    if dist_sq <= thresh_sq:
                        union_sets(parent, curr_idx, neighbor_idx)

                if r > 0 and safe_mask[r, c] and safe_mask[r - 1, c]:
                    neighbor_idx = curr_idx - width
                    dist_sq = 0.0
                    for k in range(channels):
                        diff = image[r, c, k] - image[r - 1, c, k]
                        dist_sq += diff * diff

                    if dist_sq <= thresh_sq:
                        union_sets(parent, curr_idx, neighbor_idx)

    # Flatten all trees
    for r in range(num_pixels):
        parent_arr[r] = find(parent, r)

    # Fast relabeling mapping arbitrary roots to [0, N-1]
    _, labels_flat = np.unique(parent_arr, return_inverse=True)
    labels_2d = labels_flat.reshape((height, width)).astype(np.int32)

    # Assign -1 to all pixels that failed the safety mask
    labels_2d[safe_mask_arr == 0] = -1

    return labels_2d


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
