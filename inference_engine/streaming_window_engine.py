import torch
import torch.nn as nn

import os
import threading
import queue
import pathlib
import gc
import tempfile
import shutil
import glob
from collections import defaultdict
import time

from . import VanillaEngine
from .inference_utils import (
    dict_to_device,
    register_adjacent_windows,
    estimate_pseudo_depth_and_intrinsics,
    unproject_depth_to_local_points,
    apply_sim3_to_pose,
    make_sp_graph,
    refine_depth_segments,
    sliding_window_t,
    sliding_window_l
)
from .utils.geometry import homogenize_points

STOP_SIGNAL = object()


class StreamingWindowEngine(VanillaEngine):
    def __init__(
            self,
            delegate: nn.Module,
            inference_device: str,
            dtype: torch.dtype,
            intermediate_device: str = 'cuda',
            process_device: str = 'cpu',
            top_conf_percentile: float = 0.5,
            window_size: int = 20,
            overlap: int = 5,
            depth_refine=True,
            cache_root: str = './cache',
            benchmark_latency=True
    ):
        super().__init__(
            delegate=delegate.to(inference_device)
        )
        self.window_size = window_size
        self.overlap = overlap
        self.intermediate_device = intermediate_device
        self.top_conf_percentile = 1 - top_conf_percentile if top_conf_percentile is not None else 0.0

        self.inference_device = inference_device
        self.process_device = process_device
        self.dtype = dtype
        self.depth_refine = depth_refine

        os.makedirs(cache_root, exist_ok=True)
        self.cache_dir = cache_root
        self.temp_cache_dir = None
        self.cache_id = 0
        self.inference_queue = queue.Queue()
        self.registration_queue = queue.Queue()

        self.prev_window_cache = None
        self.anchor_sp_graph = None

        self._inference_thread = None
        self._registration_thread = None

        self.running = False

        self.benchmark_latency = benchmark_latency
        self.latencies = []
        self.warmup_steps = 2

    def set_cache_dir(self, cache_dir):
        if self.running:
            raise RuntimeError('Cannot change cache directory while running')
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir

    def set_depth_refine(self, flag):
        if self.running:
            raise RuntimeError('Cannot change depth refinement mode while running')
        self.depth_refine = flag

    def _save_cache(self):
        torch.save(self.prev_window_cache, self.temp_cache_dir / f'window_cache_{self.cache_id}.pt')
        self.cache_id += 1

    def _update_cache(self, new_window_cache, sp_graph=None):
        self.prev_window_cache = new_window_cache
        self.anchor_sp_graph = sp_graph
        gc.collect()

    def _reset_state(self):
        self.cache_id = 0
        self.inference_queue = queue.Queue()
        self.registration_queue = queue.Queue()

        self.prev_window_cache = None
        self.anchor_sp_graph = None

        self._inference_thread = None
        self._registration_thread = None

        self.latencies = []

        gc.collect()

    @torch.no_grad()
    def _model_inference_worker(self):
        while True:
            sample_window = self.inference_queue.get()
            if sample_window is STOP_SIGNAL:
                return

            t_start = time.perf_counter()

            with torch.autocast(self.inference_device, dtype=self.dtype):
                prediction_window = self.delegate(sample_window)

            inference_duration = time.perf_counter() - t_start

            processed_window = dict_to_device(prediction_window, self.process_device)
            self.registration_queue.put((processed_window, inference_duration))
            if self.inference_device == 'cuda':
                torch.cuda.empty_cache()

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

                working_window['local_points'] = s_d * working_window.pop('local_points')
                working_window['camera_poses'] = apply_sim3_to_pose(working_window.pop('camera_poses'), s_d, R, t)

                if self.depth_refine:
                    tgt_pcd = working_window['local_points'].cpu().numpy()
                    tgt_sp_graph = make_sp_graph(
                        tgt_pcd[..., -1],
                        conf_map=working_window['conf'].cpu().numpy(),
                        top_conf_percentile=self.top_conf_percentile
                    )
                    working_window['local_points'] = working_window['local_points'] * refine_depth_segments(
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

    def begin(self):
        if self.running:
            raise RuntimeError('Cannot start a running inference engine')

        self.temp_cache_dir = pathlib.Path(tempfile.mkdtemp(dir=self.cache_dir))
        self._inference_thread = threading.Thread(target=self._model_inference_worker, daemon=True)
        self._registration_thread = threading.Thread(target=self._registration_worker, daemon=True)
        self._inference_thread.start()
        self._registration_thread.start()

        self.running = True

    def forward(self, sample, **kwargs):
        self.inference_queue.put(sample)

    def end(self):
        if not self.running:
            raise RuntimeError('Cannot terminate a stopped inference engine')

        self.inference_queue.put(STOP_SIGNAL)
        self._inference_thread.join()
        self.registration_queue.put(STOP_SIGNAL)
        self._registration_thread.join()

        if self.benchmark_latency:
            if self.latencies:
                print("\n" + "=" * 50)
                print("        INFERENCE PERFORMANCE SUMMARY        ")
                print("=" * 50)

                # Print list of all times
                latencies_ms = [t * 1000 for t in self.latencies]
                print(f"Raw Latencies (ms): {latencies_ms}")

                if len(self.latencies) > self.warmup_steps + 1:
                    steady_times = self.latencies[self.warmup_steps:-1]
                    avg_steady = sum(steady_times) / len(steady_times)
                    print("-" * 50)
                    print(f"Total Windows:     {len(self.latencies)}")
                    print(f"Warmup Windows:    {self.warmup_steps}")
                    print(f"Steady State Avg:  {avg_steady * 1000:.2f} ms")
                else:
                    avg_all = sum(self.latencies) / len(self.latencies)
                    print(f"Average (All):     {avg_all * 1000:.2f} ms")
                print("=" * 50 + "\n")

        self._reset_state()
        self.running = False

    def img_sliding_window(self, imgs):
        if isinstance(imgs, torch.Tensor):
            if len(imgs.shape) == 5:
                return sliding_window_t(imgs, self.window_size, self.overlap, dim=1)
            return sliding_window_t(imgs, self.window_size, self.overlap, dim=0)
        elif isinstance(imgs, list):
            return sliding_window_l(imgs, self.window_size, self.overlap)

    @staticmethod
    def parse_cache_file(cache_file, overlap=0):
        window_cache = torch.load(cache_file, map_location='cpu', weights_only=False)
        for key in window_cache.keys():
            if isinstance(window_cache[key], torch.Tensor):
                window_cache[key] = window_cache[key][overlap:]

        return window_cache

    @staticmethod
    def aggregate_caches(parsed_caches):
        aggregated_cache = defaultdict(list)
        for cache in parsed_caches:
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

    def parse_inference_cache_summary(self, remove_cache=True):
        assert self.temp_cache_dir is not None
        cache_files = sorted(glob.glob(str(self.temp_cache_dir / 'window_cache_*.pt')),
                             key=lambda p: int(p.split('_')[-1].split('.')[0]))

        parsed_caches = [self.parse_cache_file(cache_files[0])]
        for cache_fname in cache_files[1:]:
            parsed_caches.append(self.parse_cache_file(cache_fname, overlap=self.overlap))

        ret_dict = StreamingWindowEngine._post_process_pred(self.aggregate_caches(parsed_caches))

        if remove_cache:
            shutil.rmtree(self.temp_cache_dir)
        return ret_dict
