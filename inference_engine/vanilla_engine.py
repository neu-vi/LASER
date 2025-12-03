import torch
import torch.nn as nn

from .utils import estimate_pseudo_depth_and_intrinsics


class VanillaEngine(nn.Module):
    def __init__(self, delegate: nn.Module):
        super().__init__()
        self.delegate = delegate

    @staticmethod
    def _post_process_pred(pred):
        if not all(k in pred for k in ('depth', 'intrinsic')):
            depth, intrinsic = estimate_pseudo_depth_and_intrinsics(pred['local_points'].squeeze(0))
            pred['depth'] = depth[None]
            pred['intrinsic'] = intrinsic[None]
        pred['depth_conf'] = pred['conf']
        pred['extrinsic'] = pred['camera_poses']

        return pred

    def forward(self, images, **kwargs):
        predictions = self.delegate(images, **kwargs)
        return self._post_process_pred(predictions)
