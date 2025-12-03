import torch
import torch.nn as nn

from .vanilla_engine import VanillaEngine
from .utils import dict_to_device, sliding_window, aggregate_windows, register_extrinsic_windows, \
    estimate_pseudo_depth_and_intrinsics


class SlidingWindowEngine(VanillaEngine):
    def __init__(
            self,
            delegate: nn.Module,
            top_conf_percentile: float = 0.5,
            window_size: int = 20,
            overlap: int = 5,
            intermediate_device: str = 'cuda'
    ):
        super().__init__(delegate)
        self.window_size = window_size
        self.overlap = overlap
        self.intermediate_device = intermediate_device
        self.top_conf_percentile = 1 - top_conf_percentile if top_conf_percentile is not None else 0.0
        # self.cam_intrinsic = None

    # def _post_process_pcd_window(self, pcd_window):
    #     if self.cam_intrinsic is None:
    #         depth, intrinsic = estimate_pseudo_depth_and_intrinsics(pcd_window)
    #         self.cam_intrinsic = torch.median(intrinsic, dim=0).values
    #     else:
    #         depth, intrinsic  = estimate_pseudo_depth_from_intrinsics(pcd_window, self.cam_intrinsic)
    #
    #     return depth, intrinsic

    def forward(self, images, depth_refine=True, debug=False, **kwargs):
        dim = 1 if len(images.shape) == 5 else 0
        image_windows = sliding_window(images, self.window_size, self.overlap, dim=dim)
        prediction_windows = []
        for sample in image_windows:
            prediction_windows.append(dict_to_device(self.delegate(sample), self.intermediate_device))

        for pred in prediction_windows:
            for key in pred.keys():
                if isinstance(pred[key], torch.Tensor):
                    pred[key] = pred[key].squeeze(0)

        images = aggregate_windows([pred['images'] for pred in prediction_windows], self.overlap, debug=debug)

        conf_windows = []
        mask_windows = []
        for pred in prediction_windows:
            conf_ = pred['conf']
            conf_windows.append(conf_)
            conf_thre = torch.quantile(conf_, self.top_conf_percentile, interpolation='nearest')
            mask_windows.append(conf_ >= conf_thre)
        conf = aggregate_windows(conf_windows, self.overlap, debug=debug)

        with torch.autocast('cuda', enabled=False):
            (
                camera_poses,
                local_points,
            ) = register_extrinsic_windows(
                [pred['local_points'] for pred in prediction_windows],
                [pred['camera_poses'] for pred in prediction_windows],
                mask_windows,
                overlap=self.overlap,
                depth_refine=depth_refine,
                debug=debug
            )

        predictions = {'images': images[None],
                       'conf': conf[None],
                       'camera_poses': camera_poses[None],
                       'local_points': local_points[None]}
        return self._post_process_pred(predictions)

        # depth = []
        # intrinsic = []
        # for pred in prediction_windows:
        #     _depth, _intrinsic = self._post_process_pcd_window(pred['local_points'].squeeze(0))
        #     depth.append(_depth)
        #     intrinsic.append(_intrinsic)
        # depth = aggregate_windows(depth, self.overlap, debug=debug)
        # intrinsic = aggregate_windows(intrinsic, self.overlap, debug=debug)

        # predictions = {'images': images[None],
        #                'depth_conf': conf[None],
        #                'extrinsic': camera_poses[None],
        #                'depth': depth[None],
        #                'intrinsic': intrinsic[None]}
        # return predictions
