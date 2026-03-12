import torch
import torch.nn as nn

from collections import defaultdict
import time

from .streaming_window_engine import StreamingWindowEngine, STOP_SIGNAL
from .inference_utils import (
    register_adjacent_windows,
    estimate_pseudo_depth_and_intrinsics,
    unproject_depth_to_local_points,
    make_sp_graph,
    refine_depth_segments
)
from .utils.geometry import (
    homogenize_points,
    apply_sim3_to_pose,
    accumulate_sim3
)


class StreamingWindowEngineLC(StreamingWindowEngine):
    def __init__(
            self,
            delegate: nn.Module,
            inference_device: str,
            dtype: torch.dtype,
            process_device: str = 'cpu',
            top_conf_percentile: float = 0.5,
            window_size: int = 20,
            overlap: int = 5,
            depth_refine=False,
            cache_root: str = './cache'
    ):
        super().__init__(
            delegate=delegate.to(inference_device),
            inference_device=inference_device,
            dtype=dtype,
            process_device=process_device,
            top_conf_percentile=top_conf_percentile,
            window_size=window_size,
            overlap=overlap,
            depth_refine=depth_refine,
            cache_root=cache_root
        )

    def _registration_worker(self):
        ref_intrinsic = None
        tgt_sp_graph = None

        while True:
            item = self.registration_queue.get()
            if item is STOP_SIGNAL:
                return

            working_window, inference_duration = item
            t_start = time.perf_counter()

            for key in working_window.keys():
                if isinstance(working_window[key], torch.Tensor):
                    working_window[key] = working_window[key].squeeze(0)

            # camera pose registration
            conf_thresh = torch.quantile(working_window['conf'][:self.overlap], self.top_conf_percentile,
                                         interpolation='nearest')
            tgt_mask = working_window['conf'][:self.overlap] >= conf_thresh

            if self.prev_window_cache is not None:
                # fixed intrinsic enforce
                working_window['local_points'] = unproject_depth_to_local_points(
                    working_window.pop('local_points')[..., -1],
                    ref_intrinsic
                )
                # mutual conf mask
                prev_conf_thresh = torch.quantile(self.prev_window_cache['conf'][:self.overlap],
                                                  self.top_conf_percentile, interpolation='nearest')
                conf_mask = (self.prev_window_cache['conf'][:self.overlap] >= prev_conf_thresh) & tgt_mask

                # metric depth align
                prev_local_points = self.prev_window_cache['local_points'][-self.overlap:]
                cur_local_points = working_window['local_points'][:self.overlap]

                s_d, R, t = register_adjacent_windows(
                    prev_local_points,
                    cur_local_points,
                    self.prev_window_cache['camera_poses'][-self.overlap:],
                    working_window['camera_poses'][:self.overlap],
                    conf_mask
                )

                working_window['sim3'] = s_d, R, t
                # working_window['local_points'] = s_d * working_window.pop('local_points')
                # working_window['camera_poses'] = apply_sim3_to_pose(working_window.pop('camera_poses'), s_d, R, t)

                if self.depth_refine:
                    tgt_pcd = working_window['local_points'].cpu().numpy()
                    tgt_sp_graph = make_sp_graph(
                        tgt_pcd[..., -1],
                        conf_map=working_window['conf'].cpu().numpy(),
                        top_conf_percentile=self.top_conf_percentile
                    )
                    working_window['scale_mask'] = refine_depth_segments(
                        self.prev_window_cache['local_points'].cpu().numpy(),
                        tgt_pcd,
                        self.anchor_sp_graph,
                        tgt_sp_graph,
                        self.overlap
                    )
            else:
                _, intrinsic_ = estimate_pseudo_depth_and_intrinsics(working_window['local_points'])
                ref_intrinsic = intrinsic_[0]
                working_window['local_points'] = unproject_depth_to_local_points(
                    working_window.pop('local_points')[..., -1],
                    ref_intrinsic
                )
                working_window['sim3'] = (
                    1.0,
                    torch.eye(3, device=self.process_device),
                    torch.zeros(3, device=self.process_device)
                )

                if self.depth_refine:
                    tgt_sp_graph = make_sp_graph(
                        working_window['local_points'][..., -1].cpu().numpy(),
                        conf_map=working_window['conf'].cpu().numpy(),
                        top_conf_percentile=self.top_conf_percentile
                    )

            self._update_cache(working_window, tgt_sp_graph)
            self._save_cache()

            reg_duration = time.perf_counter() - t_start
            total_process_time = inference_duration + reg_duration
            self.latencies.append(total_process_time)

    @staticmethod
    def aggregate_caches(parsed_caches):
        aggregated_cache = defaultdict(list)
        ref_sim3 = (
            1.0,
            torch.eye(3, device='cpu'),
            torch.zeros(3, device='cpu')
        )
        for cache in parsed_caches:
            # apply local to world transformation
            cache_sim3 = cache['sim3']
            s_d, R, t = accumulate_sim3(ref_sim3, cache_sim3)
            if 'scale_mask' in cache.keys():
                cache['local_points'] = ref_sim3[0] * cache.pop('scale_mask') * cache.pop('local_points')
            else:
                cache['local_points'] = s_d * cache.pop('local_points')
            cache['camera_poses'] = apply_sim3_to_pose(cache.pop('camera_poses'), s_d, R, t)

            ref_sim3 = s_d, R, t

            for k, v in cache.items():
                if k == 'points':
                    continue
                aggregated_cache[k].append(v)

        for k in list(aggregated_cache.keys()):
            if isinstance(aggregated_cache[k][0], torch.Tensor):
                aggregated_cache[k] = torch.concat(aggregated_cache.pop(k), dim=0)[None]

        aggregated_cache['points'] = torch.einsum(
            'bnij, bnhwj -> bnhwi',
            aggregated_cache['camera_poses'],
            homogenize_points(aggregated_cache['local_points'])
        )[..., :3]
        return aggregated_cache
