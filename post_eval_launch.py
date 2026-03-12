# --------------------------------------------------------
# training executable for DUSt3R
# --------------------------------------------------------
from eval.post_pose_eval import eval_post_pose_estimation
import eval.misc as misc  # noqa
import torch
import numpy as np
import os
import argparse
import json


def get_args_parser():
    parser = argparse.ArgumentParser('Post evaluation launch', add_help=False)

    # training
    parser.add_argument('--seed', default=0, type=int, help="Random seed")
    parser.add_argument("--cudnn_benchmark", action='store_true', default=False,
                        help="set cudnn.benchmark = False")

    # switch mode for train / eval pose / eval depth
    parser.add_argument('--mode', default='train', type=str, help='train / eval_pose / eval_depth')

    # for pose eval
    parser.add_argument('--pose_eval_freq', default=0, type=int, help='pose evaluation frequency')
    parser.add_argument('--pose_eval_stride', default=1, type=int, help='stride for pose evaluation')
    parser.add_argument('--scene_graph_type', default='swinstride-5-noncyclic', type=str,
                        help='scene graph window size')
    parser.add_argument('--save_best_pose', action='store_true', default=False, help='save best pose')
    parser.add_argument('--n_iter', default=300, type=int, help='number of iterations for pose optimization')
    parser.add_argument('--save_pose_qualitative', action='store_true', default=False,
                        help='save qualitative pose results')
    parser.add_argument('--temporal_smoothing_weight', default=0.01, type=float,
                        help='temporal smoothing weight for pose optimization')
    parser.add_argument('--not_shared_focal', action='store_true', default=False,
                        help='use shared focal length for pose optimization')
    parser.add_argument('--use_gt_focal', action='store_true', default=False,
                        help='use ground truth focal length for pose optimization')
    parser.add_argument('--pose_schedule', default='linear', type=str, help='pose optimization schedule')

    parser.add_argument('--flow_loss_weight', default=0.01, type=float, help='flow loss weight for pose optimization')
    parser.add_argument('--flow_loss_fn', default='smooth_l1', type=str, help='flow loss type for pose optimization')
    parser.add_argument('--use_gt_mask', action='store_true', default=False,
                        help='use gt mask for pose optimization, for sintel/davis')
    parser.add_argument('--motion_mask_thre', default=0.35, type=float,
                        help='motion mask threshold for pose optimization')
    parser.add_argument('--sam2_mask_refine', action='store_true', default=False,
                        help='use sam2 mask refine for the motion for pose optimization')
    parser.add_argument('--flow_loss_start_epoch', default=0.1, type=float, help='start epoch for flow loss')
    parser.add_argument('--flow_loss_thre', default=20, type=float, help='threshold for flow loss')
    parser.add_argument('--pxl_thresh', default=50.0, type=float, help='threshold for flow loss')
    parser.add_argument('--depth_regularize_weight', default=0.0, type=float,
                        help='depth regularization weight for pose optimization')
    parser.add_argument('--translation_weight', default=1, type=float, help='translation weight for pose optimization')
    parser.add_argument('--silent', action='store_true', default=False, help='silent mode for pose evaluation')
    parser.add_argument('--full_seq', action='store_true', default=False, help='use full sequence for pose evaluation')
    parser.add_argument('--seq_list', nargs='+', default=None, help='list of sequences for pose evaluation')

    parser.add_argument('--eval_dataset', type=str, default='kitti_odometry',
                        choices=['kitti_odometry', 'vbr'],
                        help='choose dataset for pose evaluation')

    # for monocular depth eval
    parser.add_argument('--no_crop', action='store_true', default=False,
                        help='do not crop the image for monocular depth evaluation')

    parser.add_argument('--pose_dir', default='./exp', type=str, help="path where predicted poses are stored")
    parser.add_argument('--output_dir', default='./results/tmp', type=str, help="path where to save the output")
    return parser


def main(args):
    # fix the seed
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == 'eval_pose':
        ate_mean, rpe_trans_mean, rpe_rot_mean, seq_attr, outfile_list = eval_post_pose_estimation(
            args,
            pose_dir=args.pose_dir,
            save_dir=args.output_dir,
            inverse_extrinsic=False
        )
        print(f'ATE mean: {ate_mean}, RPE trans mean: {rpe_trans_mean}, RPE rot mean: {rpe_rot_mean}')
        result_dict = {
            'Seq Attributes': seq_attr,
            'ATE mean': ate_mean,
            'RPE trans mean': rpe_trans_mean,
            'RPE rot mean': rpe_rot_mean
        }
        with open(f'{args.output_dir}/{args.eval_dataset}_{args.mode}.json', 'w') as f:
            json.dump(result_dict, f, indent=2)

    exit(0)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    main(args)
