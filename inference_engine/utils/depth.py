import os
import numpy as np
from skimage.segmentation import felzenszwalb
from concurrent.futures import ThreadPoolExecutor, as_completed

from ._segmentation_cy import merge_regions
from pi3.utils.graph import Vertex


def align_depth_irls(
        src_depth,
        tgt_depth,
        mask=None,
        iters=10,
        eps=1e-8,
        stop_tol=0.05,
        clamp_min=1e-6
):
    if mask is not None:
        src_depth = src_depth[mask]
        tgt_depth = tgt_depth[mask]

    num = np.nanmean(tgt_depth)
    den = np.nanmean(src_depth)
    s_d = np.maximum(num / den, clamp_min)

    for _ in range(iters):
        d_res = s_d * src_depth - tgt_depth
        res = abs(d_res) + eps
        w = 1.0 / res

        num = (w * src_depth * tgt_depth).sum()
        den = (w * src_depth ** 2).sum()
        s_d_new = np.maximum(num / den, clamp_min)
        converged = abs(s_d_new - s_d) < stop_tol
        s_d = s_d_new

        if converged:
            break  # early stopping

    return s_d


def segment_depth_felzenszwalb_rag(
        depth_map,
        merge_thresh,
        seg_scale=300,
        seg_sigma=1.1,
        seg_min_size=500
):
    seg_mask = felzenszwalb(depth_map, scale=seg_scale, sigma=seg_sigma, min_size=seg_min_size)
    # depth_img = gray2rgb(depth_map)
    # rag = graph.rag_mean_color(depth_img, seg_mask, mode='distance')
    #
    # seg_mask_merged = graph.cut_threshold(seg_mask, rag, merge_thresh)
    seg_mask_merged = merge_regions(seg_mask, depth_map, merge_thresh)
    return seg_mask_merged


def pairwise_intersection_ratio(mask1, mask2):
    """
    Highest pairwise intersection ratio for assigning correspondence
    """
    N, H, W = mask1.shape
    M = mask2.shape[0]

    mask1_f = mask1.reshape(N, -1).astype(np.float32)
    mask2_f = mask2.reshape(M, -1).astype(np.float32)
    inter = mask1_f @ mask2_f.T  # pairwise intersection - N, M

    area1 = mask1_f.sum(axis=-1, keepdims=True)  # N, 1
    area2 = mask2_f.sum(axis=-1, keepdims=True).T  # 1, M
    area1 = np.maximum(area1, 1)
    area2 = np.maximum(area2, 1)

    ratios1 = inter / area1
    ratios2 = inter / area2
    min_rel_inter = np.minimum(ratios1, ratios2)
    max_rel_inter = np.maximum(ratios1, ratios2)

    return min_rel_inter, max_rel_inter  # N, M


def pairwise_iou(mask1, mask2):
    N, H, W = mask1.shape
    M = mask2.shape[0]

    mask1_f = mask1.reshape(N, -1).astype(np.float32)
    mask2_f = mask2.reshape(M, -1).astype(np.float32)
    inter = mask1_f @ mask2_f.T  # pairwise intersection - N, M

    area1 = mask1_f.sum(axis=-1, keepdims=True)  # N, 1
    area2 = mask2_f.sum(axis=-1, keepdims=True).T  # 1, M
    union = area1 + area2 - inter

    iou = inter / np.maximum(union, 1)
    return iou


def match_segmentation_seq(labels, iou_thresh=0.4):
    def get_seg_vertices(seg):
        seg_ids = np.unique(seg)
        masks = seg[None, :, :] == seg_ids[:, None, None]
        seg_vertices_ = [Vertex(data=m, default_cache={'iou': [], 'scale': []}) for m in masks]
        return seg_vertices_  # , masks

    # root = get_seg_vertices(labels[0])
    sp_graph = [get_seg_vertices(labels[0])]

    for seg_map in labels[1:]:
        seg_vertices = get_seg_vertices(seg_map)
        connect_bipartite_sp_graphs(sp_graph[-1], seg_vertices, iou_thresh=iou_thresh)
        sp_graph.append(seg_vertices)
        # prev_mask = cur_mask

    # for v in root:
    #     v.cut_edge_threshold(inter_thresh)
    return sp_graph


def connect_bipartite_sp_graphs(graph1, graph2, iou_thresh=0.3):
    masks1 = np.stack([v.data for v in graph1])
    masks2 = np.stack([v.data for v in graph2])

    iou = pairwise_iou(masks1, masks2)
    matchable = iou >= iou_thresh
    graph1_indices, graph2_indices = np.nonzero(matchable)

    for v1, v2 in zip(graph1_indices, graph2_indices):
        graph1[v1].add_edge(graph2[v2], iou[v1, v2])


def _edge_scale_worker(
        src_depth,
        tgt_depth,
        src_vertex
):
    src_mask = src_vertex.data
    for tgt_v, tgt_iou in zip(src_vertex.connectivity, src_vertex.edge_weights):
        tgt_mask = tgt_v.data
        inter_mask = src_mask & tgt_mask
        tgt2src_s = align_depth_irls(tgt_depth, src_depth, inter_mask)
        tgt_v.cache['iou'].append(tgt_iou)
        tgt_v.cache['scale'].append(tgt2src_s)


def assign_overlap_window_depth_scale(
        src_depth_overlap,
        tgt_depth_overlap,
        src_sp_graphs_overlap,
        tgt_sp_graphs_overlap,
        iou_thresh=0.4,
        n_jobs=1
):
    for src_sp_graph, tgt_sp_graph in zip(src_sp_graphs_overlap, tgt_sp_graphs_overlap):
        connect_bipartite_sp_graphs(src_sp_graph, tgt_sp_graph, iou_thresh=iou_thresh)

    for idx, src_graph in enumerate(src_sp_graphs_overlap):
        n_jobs = min(os.cpu_count(), len(src_graph)) if n_jobs is None else n_jobs
        if n_jobs == 1:
            for src_v in src_graph:
                _edge_scale_worker(src_depth_overlap[idx], tgt_depth_overlap[idx], src_v)
        else:
            with ThreadPoolExecutor(max_workers=n_jobs) as ex:
                promises = [
                    ex.submit(_edge_scale_worker, src_depth_overlap[idx], tgt_depth_overlap[idx], src_v)
                    for src_v in src_graph
                ]
                for promise in as_completed(promises):
                    promise.result()

        # for src_v in src_graph:
        #     src_mask = src_v.data
        #     for tgt_v, tgt_iou in zip(src_v.connectivity, src_v.edge_weights):
        #         tgt_mask = tgt_v.data
        #         inter_mask = src_mask & tgt_mask
        #         tgt2src_s = align_depth_irls(tgt_depth_overlap[idx], src_depth_overlap[idx], inter_mask)
        #         tgt_v.cache['iou'].append(tgt_iou)
        #         tgt_v.cache['scale'].append(tgt2src_s)

        # tgt_graph = tgt_sp_graphs_overlap[idx]
        # for v in tgt_graph:
        #     v.propagate_cache(merge_scale_cache)
