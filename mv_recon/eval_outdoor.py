import os
import json
from tqdm import tqdm

import torch
import numpy as np
import open3d as o3d
import os.path as osp
import hydra
import logging

from omegaconf import DictConfig
from pi3.models.pi3 import Pi3
from utils.interfaces import infer_mv_pointclouds, infer_streaming_mv_pointclouds
from mv_recon.utils import umeyama, accuracy, completion
from utils.messages import set_default_arg, write_csv
from utils.load_fn import load_and_preprocess_images

# Additional models
from inference_engine import StreamingWindowEngine

WINDOW_SIZE = 60
OVERLAP = 30
TOP_CONF_PERCENTILE = 0.6
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def create_pi3(cfg):
    pretrained_model_name_or_path: str = cfg.pi3.pretrained_model_name_or_path  # see configs/evaluation/videodepth.yaml
    model = Pi3.from_pretrained(pretrained_model_name_or_path).to(cfg.device).eval()
    return model


def create_streaming_pi3(cfg):
    pretrained_model_name_or_path: str = cfg.pi3.pretrained_model_name_or_path
    pi3 = Pi3.from_pretrained(pretrained_model_name_or_path)
    model = StreamingWindowEngine(
        pi3,
        inference_device=cfg.device,
        dtype=dtype,
        top_conf_percentile=0.3,
        window_size=WINDOW_SIZE,
        overlap=OVERLAP,
        cache_root='cache/',
        depth_refine=False
    ).eval()
    return model


@hydra.main(version_base="1.2", config_path="../configs", config_name="eval_mv_recon_outdoor")
def main(hydra_cfg: DictConfig):
    all_eval_datasets: DictConfig = hydra_cfg.eval_datasets  # see configs/evaluation/mv_recon.yaml
    all_data_info: DictConfig = hydra_cfg.data  # see configs/data

    # 0. create model
    # model = create_pi3(hydra_cfg)
    model = create_streaming_pi3(hydra_cfg)

    ##################################################
    logger = logging.getLogger("mv_recon-eval")
    logger.info(f"Loaded Pi3 from {hydra_cfg.pi3.pretrained_model_name_or_path}")
    output_dir = hydra_cfg.output_dir if hydra_cfg.dir_suffix is None else f'{hydra_cfg.output_dir}_{hydra_cfg.dir_suffix}'

    for idx_dataset, dataset_name in enumerate(all_eval_datasets, start=1):
        # 1.1 look up dataset config from configs/data, decide the dataset name, and load the dataset
        if dataset_name not in all_data_info:
            raise ValueError(f"Unknown dataset in global data information: {dataset_name}")
        dataset_info = all_data_info[dataset_name]
        dataset = hydra.utils.instantiate(dataset_info.cfg)

        # 1.2 ready for output directory & metrics
        output_root = osp.join(output_dir, dataset_name)
        os.makedirs(output_root, exist_ok=True)
        all_data_dict = {
            "Acc-mean": 0.0, "Acc-med": 0.0,
            "Comp-mean": 0.0, "Comp-med": 0.0,
            "CD": 0.0,
        }

        # 1.3 load define seq list
        seq_list = dataset.seq_list

        if osp.exists(osp.join(output_root, "_all_samples.csv")):
            os.remove(osp.join(output_root, "_all_samples.csv"))  # remove old csv file

        for seq_idx, seq_name in enumerate(tqdm(seq_list)):
            # 2. load data, choose specific ids of a sequence
            data = dataset.get_data(sequence_name=seq_name)
            filelist: list = data['image_paths']  # [str] * N
            # images: torch.Tensor   = data['images']       # (N, 3, H, W)
            images = load_and_preprocess_images(filelist)
            gt_pts: np.ndarray = data['pointclouds']  # (N, H, W, 3)
            valid_mask: np.ndarray = data['valid_mask']  # (N, H, W)

            # 3. real inference, predicted pointcloud aligned to ground truth (data_h, data_w)
            data_h, data_w = images.shape[-2:]
            # pred_pts: np.ndarray   = infer_mv_pointclouds(filelist, model, hydra_cfg, (data_h, data_w))  # (N, H, W, 3)
            pred_pts, conf = infer_streaming_mv_pointclouds(filelist, model, hydra_cfg,
                                                            (data_h, data_w))  # (N, H, W, 3)
            # VGGT-like output format
            # pred_pts, conf = infer_vggt_mv_pointclouds(filelist, model, hydra_cfg, (data_h, data_w))  # (N, H, W, 3)

            ##################################################
            # confidence thresholding
            cutoff = np.percentile(conf[valid_mask], 100 * (1 - TOP_CONF_PERCENTILE))
            conf_mask = conf[valid_mask] >= cutoff
            ##################################################

            ##################################################
            # CUT3R center crop
            # H, W = gt_pts.shape[1:3]
            # cx = W // 2
            # cy = H // 2
            # l, t = cx - 112, cy - 112
            # r, b = cx + 112, cy + 112

            # images = images[..., t:b, l:r]
            # gt_pts = gt_pts[:, t:b, l:r]
            # pred_pts = pred_pts[:, t:b, l:r]
            # valid_mask = valid_mask[:, t:b, l:r]
            ##################################################
            assert pred_pts.shape == gt_pts.shape, f"Predicted points shape {pred_pts.shape} does not match ground truth shape {gt_pts.shape}."

            # 4. save input images
            seq_name = seq_name.replace("/", "-")

            # 5. coarse align
            c, R, t = umeyama(pred_pts[valid_mask][conf_mask].T, gt_pts[valid_mask][conf_mask].T)
            pred_pts = c * np.einsum('nhwj, ij -> nhwi', pred_pts, R) + t.T

            # 6. filter invalid points
            pred_pts = pred_pts[valid_mask][conf_mask].reshape(-1, 3)
            gt_pts = gt_pts[valid_mask][conf_mask].reshape(-1, 3)

            # 7. save predicted & ground truth point clouds
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pred_pts)
            pcd_gt = o3d.geometry.PointCloud()
            pcd_gt.points = o3d.utility.Vector3dVector(gt_pts)

            # 8. ICP align refinement
            if "DTU" in dataset_name:
                threshold = 100
            else:
                threshold = 0.1

            trans_init = np.eye(4)
            reg_p2p = o3d.pipelines.registration.registration_icp(
                pcd,
                pcd_gt,
                threshold,
                trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            )

            transformation = reg_p2p.transformation
            pcd = pcd.transform(transformation)

            # 10. compute metrics
            acc, acc_med = accuracy(pcd_gt.points, pcd.points)
            comp, comp_med = completion(pcd_gt.points, pcd.points)
            cd = (acc + comp) / 2

            logger.info(
                f"[{dataset_name} {seq_idx}/{dataset.seq_list_len}] Seq: {seq_name}, Acc: {acc}, Comp: {comp}, CD: {cd}"
            )

            # 11. save metrics to csv
            write_csv(osp.join(output_root, f"_all_samples.csv"), {
                "seq": seq_name,
                "Acc-mean": acc,
                "Acc-med": acc_med,
                "Comp-mean": comp,
                "Comp-med": comp_med,
                "CD": cd,
            })
            all_data_dict["Acc-mean"] += acc
            all_data_dict["Acc-med"] += acc_med
            all_data_dict["Comp-mean"] += comp
            all_data_dict["Comp-med"] += comp_med
            all_data_dict["CD"] += cd

            # release cuda memory
            torch.cuda.empty_cache()

        num_samples = dataset.seq_list_len
        metric_dict = {
            metric: value / num_samples
            for metric, value in all_data_dict.items()
            if metric != "model"
        }

        statistics_file = osp.join(output_dir, f"{dataset_name}-metric")  # + ".csv"
        if getattr(hydra_cfg, "save_suffix", None) is not None:
            statistics_file += f"-{hydra_cfg.save_suffix}"
        statistics_file += ".csv"
        write_csv(statistics_file, metric_dict)

    del model
    torch.cuda.empty_cache()
    logger.info(f"Finished evaluating Pi3 on all datasets.")


if __name__ == "__main__":
    set_default_arg("evaluation", "mv_recon_outdoor")
    os.environ["HYDRA_FULL_ERROR"] = '1'
    with torch.no_grad():
        main()
