import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


def batched_image_op_wrapper(
        images,
        op_func=None,
        n_jobs=None,
        *args,
        **kwargs
):
    """
    Run segmentation on a batch of images.

    Parameters
    ----------
    images : ndarray, shape (B, M, N, [C])
        Batch of images. All must have the same shape (M, N, [C]).
    n_jobs : int or None
        Number of parallel workers. Default = os.cpu_count().
        Use 1 for sequential processing.
    op_func : callable
        Function that process a single image:
        `op_func(image, *args, **kwargs) -> (M, N)`.

    Returns
    -------
    labels : ndarray, shape (B, M, N)
        Segmentation masks for each image.
    """
    images = np.asarray(images)
    if op_func is None:
        raise ValueError("You must pass a op_func for single images.")

    B, H, W = images.shape[:3]
    if B == 0:
        return np.empty((0, H, W), dtype=np.intp)

    if n_jobs is None:
        n_jobs = min(os.cpu_count(), B)

    # sequential path
    if n_jobs == 1:
        return np.stack([
            op_func(images[i], *args, **kwargs)
            for i in range(B)
        ], axis=0).astype(np.intp, copy=False)

    # parallel path
    results = [None] * B
    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        promises = {
            ex.submit(op_func, images[i], *args, **kwargs): i
            for i in range(B)
        }
        for promise in as_completed(promises):
            idx = promises[promise]
            results[idx] = promise.result()

    return np.stack(results, axis=0).astype(np.intp, copy=False)
