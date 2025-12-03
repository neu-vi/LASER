import torch
import numpy as np

from .depth import segment_depth_felzenszwalb_rag, match_segmentation_seq, assign_overlap_window_depth_scale
from .batch_threading import batched_image_op_wrapper


def refine_depth_segments(
        src_pcd,
        tgt_pcd,
        src_sp_graphs,
        tgt_sp_graphs,
        overlap,
        corr_iou_thresh=0.4
):
    """
    src_pcd: previous window pcd
    tgt_pcd: current window pcd
    src_sp_graphs: previous window superpixel graph
    overlap: window overlap size
    conf_mask: confidence mask
    depth_merge_thresh: percentage confident depth range to be considered as smooth change
    corr_iou_thresh: IoU threshold for superpixels to be considered as corresponding
    """
    src_depth = src_pcd[..., -1]
    tgt_depth = tgt_pcd[..., -1]

    tgt_scale_mask = align_adjacent_windows_depth_segments(
        src_depth,
        tgt_depth,
        src_sp_graphs,
        tgt_sp_graphs,
        overlap,
        corr_iou_thresh
    )

    return torch.from_numpy(tgt_scale_mask[..., None])


def make_sp_graph(
        depth,
        conf_mask=None,
        depth_merge_thresh=0.1,
        corr_iou_thresh=0.3
):
    conf_mask = conf_mask if conf_mask is not None else np.ones_like(depth, dtype=bool)
    conf_depth = depth[conf_mask]
    merge_thresh = depth_merge_thresh * (np.max(conf_depth) - np.min(conf_depth))
    # labels = [segment_depth_felzenszwalb_rag(d, merge_thresh) for d in depth]
    labels = batched_image_op_wrapper(depth, segment_depth_felzenszwalb_rag, merge_thresh=merge_thresh)
    sp_graph = match_segmentation_seq(labels, iou_thresh=corr_iou_thresh)

    return sp_graph


def align_adjacent_windows_depth_segments(
        src_depth,  # N, H, W
        tgt_depth,  # N, H, W
        src_sp_graphs,
        tgt_sp_graphs,
        overlap,
        corr_iou_thresh=0.4
):
    """
    src_depth: previous window depth map
    tgt_depth: current window depth map
    src_sp_graphs: previous window superpixel graph (nested list of Vertex)
    tgt_sp_graphs: current window superpixel graph
    overlap: window overlap size
    corr_iou_thresh: IoU threshold for superpixels to be considered as corresponding

    Return:
        depth_scale_mask: N, H, W for current window pcd
    """

    def _propagate_scale_cache(parent, child, edge_wt):
        if len(parent.cache['scale']) > 0:
            iou_wts = np.asarray(parent.cache['iou'])
            prop_scale = np.dot(np.asarray(parent.cache['scale']), iou_wts / np.sum(iou_wts))
            child.cache['iou'].append(edge_wt)
            child.cache['scale'].append(prop_scale)

    def _get_scale_mask(mask, cache):
        mask = mask.astype(np.float32)
        if len(cache['scale']) > 0:
            iou_wts = np.asarray(cache['iou'])
            mu_scale = np.dot(np.asarray(cache['scale']), iou_wts / np.sum(iou_wts))
        else:
            mu_scale = 1.0
        return mask * mu_scale

    src_depth_overlap = src_depth[-overlap:]
    tgt_depth_overlap = tgt_depth[:overlap]
    src_sp_graphs_overlap = src_sp_graphs[-overlap:]
    tgt_sp_graphs_overlap = tgt_sp_graphs[:overlap]

    for sp_graph in src_sp_graphs_overlap:
        for v in sp_graph:
            v.remove_all_edges()

    # sptial scale initilaization
    assign_overlap_window_depth_scale(
        src_depth_overlap,
        tgt_depth_overlap,
        src_sp_graphs_overlap,
        tgt_sp_graphs_overlap,
        iou_thresh=corr_iou_thresh
    )
    # temporal scale propagation
    for tgt_graph_layer in tgt_sp_graphs:
        for v in tgt_graph_layer:
            v.propagate_data_once(_propagate_scale_cache)

    mask_seq = []
    for sp_graph in tgt_sp_graphs:
        mask_frame = sp_graph[0].data_cache_op(_get_scale_mask)
        for v in sp_graph[1:]:
            mask_frame += v.data_cache_op(_get_scale_mask)
        mask_seq.append(mask_frame)

    return np.stack(mask_seq)
