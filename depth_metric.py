import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from eval.depth_eval import depth_evaluation, group_by_directory
import numpy as np
import cv2
from tqdm import tqdm
import glob
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import json
from PIL import Image

import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('Depth metric evaluation', add_help=False)
    parser.add_argument('--eval_dataset', type=str, default='sintel',
                        choices=['davis', 'kitti', 'bonn', 'scannet', 'tum', 'nyu', 'sintel'],
                        help='choose dataset for pose evaluation')
    parser.add_argument('--result_dir', default='./results', type=str, help='stored inference results')
    parser.add_argument('--output_dir', default='./results', type=str, help='stored evaluation results')

    return parser


def eval_sintel(args):
    TAG_FLOAT = 202021.25

    def depth_read(filename):
        """ Read depth data from file, return as numpy array. """
        f = open(filename, 'rb')
        check = np.fromfile(f, dtype=np.float32, count=1)[0]
        assert check == TAG_FLOAT, ' depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(
            TAG_FLOAT, check)
        width = np.fromfile(f, dtype=np.int32, count=1)[0]
        height = np.fromfile(f, dtype=np.int32, count=1)[0]
        size = width * height
        assert width > 0 and height > 0 and size > 1 and size < 100000000, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(
            width, height)
        depth = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width))
        return depth

    pred_pathes = glob.glob(f"{args.result_dir}/*/frame_*.npy")  # TODO: update the path to your prediction
    pred_pathes = sorted(pred_pathes)

    if len(pred_pathes) > 643:
        full = True
    else:
        full = False

    if full:
        depth_pathes = glob.glob(f"/projects/vig/Datasets/sintel/training/depth/*/*.dpt")
        depth_pathes = sorted(depth_pathes)
    else:
        seq_list = ["alley_2", "ambush_4", "ambush_5", "ambush_6", "cave_2", "cave_4", "market_2",
                    "market_5", "market_6", "shaman_3", "sleeping_1", "sleeping_2", "temple_2", "temple_3"]
        depth_pathes_folder = [f"data/sintel/training/depth/{seq}" for seq in seq_list]
        depth_pathes = []
        for depth_pathes_folder_i in depth_pathes_folder:
            depth_pathes += glob.glob(depth_pathes_folder_i + '/*.dpt')
        depth_pathes = sorted(depth_pathes)

    def get_video_results():
        grouped_pred_depth = group_by_directory(pred_pathes)
        grouped_gt_depth = group_by_directory(depth_pathes)
        gathered_depth_metrics = []
        per_seq_metric = {}

        for key in tqdm(grouped_pred_depth.keys()):
            pd_pathes = grouped_pred_depth[key]
            gt_pathes = grouped_gt_depth[key.replace('_pred_depth', '')]

            gt_depth = np.stack([depth_read(gt_path) for gt_path in gt_pathes], axis=0)
            pr_depth = np.stack(
                [cv2.resize(np.load(pd_path), (gt_depth.shape[2], gt_depth.shape[1]), interpolation=cv2.INTER_CUBIC)
                 for pd_path in pd_pathes], axis=0)

            # for depth eval, set align_with_lad2=False to use median alignment; set align_with_lad2=True to use scale&shift alignment
            depth_results, error_map, depth_predict, depth_gt = depth_evaluation(
                pr_depth,
                gt_depth,
                max_depth=70,
                align_with_lad2=False,
                align_with_scale=True,
                use_gpu=True,
                post_clip_max=70
            )
            per_seq_metric[key] = {
                'Abs Rel': depth_results['Abs Rel'],
                'δ < 1.25': depth_results['δ < 1.25'],
                'δ < 1.25^2': depth_results['δ < 1.25^2'],
                'δ < 1.25^3': depth_results['δ < 1.25^3']
            }
            gathered_depth_metrics.append(depth_results)

        depth_log_path = f'{args.output_dir}/sintel_depth.json'
        average_metrics = {
            key: np.average(
                [metrics[key] for metrics in gathered_depth_metrics],
                weights=[metrics['valid_pixels'] for metrics in gathered_depth_metrics]
            )
            for key in gathered_depth_metrics[0].keys() if key != 'valid_pixels'
        }
        print('Average depth evaluation metrics:', average_metrics)
        with open(depth_log_path, 'w') as f:
            ret_json = {'per_seq_metrics': per_seq_metric, 'avg_metrics': average_metrics}
            f.write(json.dumps(ret_json, indent=2))

    get_video_results()


def eval_bonn(args):
    def depth_read(filename):
        # loads depth map D from png file
        # and returns it as a numpy array
        depth_png = np.asarray(Image.open(filename))
        # make sure we have a proper 16bit depth map here.. not 8bit!
        assert np.max(depth_png) > 255
        depth = depth_png.astype(np.float64) / 5000.0
        depth[depth_png == 0] = -1.0
        return depth

    seq_list = ["balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous"]

    img_pathes_folder = [f"/projects/vig/Datasets/bonn/rgbd_bonn_dataset/rgbd_bonn_{seq}/rgb_110/*.png" for seq in
                         seq_list]
    img_pathes = []
    for img_pathes_folder_i in img_pathes_folder:
        img_pathes += glob.glob(img_pathes_folder_i)
    img_pathes = sorted(img_pathes)
    depth_pathes_folder = [f"/projects/vig/Datasets/bonn/rgbd_bonn_dataset/rgbd_bonn_{seq}/depth_110/*.png" for seq in
                           seq_list]
    depth_pathes = []
    for depth_pathes_folder_i in depth_pathes_folder:
        depth_pathes += glob.glob(depth_pathes_folder_i)
    depth_pathes = sorted(depth_pathes)
    pred_pathes = glob.glob(f"{args.result_dir}/*/frame_*.npy")  # TODO: update the path to your prediction
    pred_pathes = sorted(pred_pathes)

    def get_video_results():
        grouped_pred_depth = group_by_directory(pred_pathes)
        grouped_gt_depth = group_by_directory(depth_pathes, idx=-2)
        gathered_depth_metrics = []
        per_seq_metric = {}

        for key in tqdm(grouped_pred_depth.keys()):
            pd_pathes = grouped_pred_depth[key]
            gt_pathes = grouped_gt_depth['rgbd_bonn_' + key]
            gt_depth = np.stack([depth_read(gt_path) for gt_path in gt_pathes], axis=0)
            pr_depth = np.stack(
                [cv2.resize(np.load(pd_path), (gt_depth.shape[2], gt_depth.shape[1]), interpolation=cv2.INTER_CUBIC)
                 for pd_path in pd_pathes], axis=0)

            # for depth eval, set align_with_lad2=False to use median alignment; set align_with_lad2=True to use scale&shift alignment
            depth_results, error_map, depth_predict, depth_gt = depth_evaluation(
                pr_depth,
                gt_depth,
                max_depth=70,
                align_with_lad2=False,
                align_with_scale=True,
                use_gpu=True
            )
            per_seq_metric[key] = {
                'Abs Rel': depth_results['Abs Rel'],
                'δ < 1.25': depth_results['δ < 1.25'],
                'δ < 1.25^2': depth_results['δ < 1.25^2'],
                'δ < 1.25^3': depth_results['δ < 1.25^3']
            }
            gathered_depth_metrics.append(depth_results)

        depth_log_path = f'{args.output_dir}/bonn_depth.json'
        average_metrics = {
            key: np.average(
                [metrics[key] for metrics in gathered_depth_metrics],
                weights=[metrics['valid_pixels'] for metrics in gathered_depth_metrics]
            )
            for key in gathered_depth_metrics[0].keys() if key != 'valid_pixels'
        }
        print('Average depth evaluation metrics:', average_metrics)
        with open(depth_log_path, 'w') as f:
            ret_json = {'per_seq_metrics': per_seq_metric, 'avg_metrics': average_metrics}
            f.write(json.dumps(ret_json, indent=2))

    get_video_results()


def eval_kitti(args):
    def depth_read(filename):
        # loads depth map D from png file
        # and returns it as a numpy array,
        # for details see readme.txt
        img_pil = Image.open(filename)
        depth_png = np.array(img_pil, dtype=int)
        # make sure we have a proper 16bit depth map here.. not 8bit!
        assert (np.max(depth_png) > 255)

        depth = depth_png.astype(float) / 256.
        depth[depth_png == 0] = -1.
        return depth

    depth_pathes = glob.glob("/projects/vig/Datasets/4D_dataset/kitti/depth_selection/val_selection_cropped/groundtruth_depth_gathered/*/*.png")
    depth_pathes = sorted(depth_pathes)
    pred_pathes = glob.glob(f"{args.result_dir}/*/frame_*.npy")  # TODO: update the path to your prediction
    pred_pathes = sorted(pred_pathes)

    def get_video_results():
        grouped_pred_depth = group_by_directory(pred_pathes)
        grouped_gt_depth = group_by_directory(depth_pathes)
        gathered_depth_metrics = []
        per_seq_metric = {}

        for key in tqdm(grouped_pred_depth.keys()):
            pd_pathes = grouped_pred_depth[key]
            gt_pathes = grouped_gt_depth[key]
            gt_depth = np.stack([depth_read(gt_path) for gt_path in gt_pathes], axis=0)
            pr_depth = np.stack(
                [cv2.resize(np.load(pd_path), (gt_depth.shape[2], gt_depth.shape[1]), interpolation=cv2.INTER_CUBIC)
                 for pd_path in pd_pathes], axis=0)

            # for depth eval, set align_with_lad2=False to use median alignment; set align_with_lad2=True to use scale&shift alignment
            depth_results, error_map, depth_predict, depth_gt = depth_evaluation(
                pr_depth,
                gt_depth,
                max_depth=None,
                align_with_lad2=False,
                align_with_scale=True,
                use_gpu=True
            )
            per_seq_metric[key] = {
                'Abs Rel': depth_results['Abs Rel'],
                'δ < 1.25': depth_results['δ < 1.25'],
                'δ < 1.25^2': depth_results['δ < 1.25^2'],
                'δ < 1.25^3': depth_results['δ < 1.25^3']
            }
            gathered_depth_metrics.append(depth_results)

        depth_log_path = f'{args.output_dir}/kitti_depth.json'
        average_metrics = {
            key: np.average(
                [metrics[key] for metrics in gathered_depth_metrics],
                weights=[metrics['valid_pixels'] for metrics in gathered_depth_metrics]
            )
            for key in gathered_depth_metrics[0].keys() if key != 'valid_pixels'
        }
        print('Average depth evaluation metrics:', average_metrics)
        with open(depth_log_path, 'w') as f:
            ret_json = {'per_seq_metrics': per_seq_metric, 'avg_metrics': average_metrics}
            f.write(json.dumps(ret_json, indent=2))

    get_video_results()


def eval_nyu_v2():
    depth_pathes = glob.glob("data/nyu-v2/val/nyu_depths/*.npy")
    depth_pathes = sorted(depth_pathes)
    pred_pathes = glob.glob(f"{args.result_dir}/*.npy")  # TODO: update the path to your prediction
    pred_pathes = sorted(pred_pathes)
    gathered_depth_metrics = []
    for idx in tqdm(range(len(depth_pathes))):
        pred_depth = np.load(pred_pathes[idx])
        gt_depth = np.load(depth_pathes[idx])
        pred_depth = cv2.resize(pred_depth, (gt_depth.shape[1], gt_depth.shape[0]), interpolation=cv2.INTER_CUBIC)

        depth_results, error_map, depth_predict, depth_gt = depth_evaluation(pred_depth, gt_depth, max_depth=None,
                                                                             lr=1e-3)
        gathered_depth_metrics.append(depth_results)

    depth_log_path = f'{args.output_dir}/nyu_depth.json'
    average_metrics = {
        key: np.average(
            [metrics[key] for metrics in gathered_depth_metrics],
            weights=[metrics['valid_pixels'] for metrics in gathered_depth_metrics]
        )
        for key in gathered_depth_metrics[0].keys() if key != 'valid_pixels'
    }
    print('Average depth evaluation metrics:', average_metrics)
    with open(depth_log_path, 'w') as f:
        f.write(json.dumps(average_metrics))


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()

    eval_dataset = args.eval_dataset
    if eval_dataset == 'sintel':
        eval_sintel(args)
    elif eval_dataset == 'bonn':
        eval_bonn(args)
    elif eval_dataset == 'kitti':
        eval_kitti(args)
    else:
        raise NotImplementedError
