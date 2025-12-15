from eval.pose_eval import eval_pose_estimation
from eval.depth_eval import eval_mono_depth_estimation
from pi3.models.pi3 import Pi3
from inference_engine import VanillaEngine, StreamingWindowEngine, StreamingWindowEngineLC
from loop_closure.loop_closure import LoopClosureEngine
from loop_closure.utils.config_utils import load_config
from functools import partial
import eval.misc as misc  # noqa
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import os
import argparse
import json
from pathlib import Path
import glob
import shutil


def get_args_parser():
    parser = argparse.ArgumentParser('Evaluation launch', add_help=False)

    # training
    parser.add_argument('--seed', default=0, type=int, help="Random seed")
    parser.add_argument("--cudnn_benchmark", action='store_true', default=False,
                        help="set cudnn.benchmark = False")

    # others
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

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

    parser.add_argument('--eval_dataset', type=str, default='sintel',
                        choices=['davis', 'kitti', 'bonn', 'scannet', 'tum', 'nyu', 'sintel', 'kitti_odometry'],
                        help='choose dataset for pose evaluation')
    # model variant
    parser.add_argument('--model', type=str, required=True,
                        choices=['pi3', 'streaming_pi3', 'streaming_pi3_lc'],
                        help='choose model for pose evaluation')
    # checkpoint loading
    parser.add_argument('--ckpt_path', default=None, type=str, help='trained checkpoint for evaluation')

    # for monocular depth eval
    parser.add_argument('--no_crop', action='store_true', default=False,
                        help='do not crop the image for monocular depth evaluation')

    # output dir
    parser.add_argument('--output_dir', default='./results/tmp', type=str, help="path where to save the output")
    return parser


# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
device = "cuda" if torch.cuda.is_available() else "cpu"


def inference_streaming_model(model, imgs, *args, **kwargs):
    image_windows = model.img_sliding_window(imgs)

    model.begin()
    for sample in image_windows:
        model(sample)
    model.end()
    save_dict = model.parse_inference_cache_summary()
    return save_dict


def inference_streaming_model_lc(model, imgs, img_dir, *args, **kwargs):
    image_windows = model.img_sliding_window(imgs)

    model.begin()
    for sample in image_windows:
        model(sample)
    model.end()

    cache_path = Path(model.cache_dir)
    cache_path_lc = cache_path.parent / f'{cache_path.name}_lc'
    lc_engine = LoopClosureEngine(
        load_config('configs/loop_config.yaml'),
        img_dir,
        cache_path_lc,
        model.delegate,
        model.window_size,
        model.overlap,
    )

    cache_files = sorted(glob.glob(str(model.temp_cache_dir / 'window_cache_*.pt')),
                         key=lambda p: int(p.split('_')[-1].split('.')[0]))
    raw_predictions = [StreamingWindowEngine.parse_cache_file(cache_fname) for cache_fname in cache_files]
    sim3_list_lc = lc_engine.run(raw_predictions)
    sim3_list_lc.insert(0, raw_predictions[0]['sim3'])

    os.makedirs(str(cache_path_lc), exist_ok=True)
    for idx, (pred, sim3_lc) in enumerate(zip(raw_predictions, sim3_list_lc)):
        pred['sim3'] = sim3_lc
        torch.save(pred, str(cache_path_lc / f'window_cache_{idx}.pt'))

    cache_files_lc = sorted(glob.glob(str(cache_path_lc / 'window_cache_*.pt')),
                            key=lambda p: int(p.split('_')[-1].split('.')[0]))
    parsed_caches = [StreamingWindowEngine.parse_cache_file(cache_files_lc[0])]
    for cache_fname in cache_files_lc[1:]:
        parsed_caches.append(StreamingWindowEngine.parse_cache_file(cache_fname, overlap=model.overlap))

    ret_dict = StreamingWindowEngineLC._post_process_pred(StreamingWindowEngineLC.aggregate_caches(parsed_caches))
    shutil.rmtree(cache_path_lc)
    for key in ret_dict.keys():
        if isinstance(ret_dict[key], torch.Tensor):
            ret_dict[key] = ret_dict[key].cpu()
    return ret_dict


def pi3_main(args, engine_cls):
    print('Launching Pi3 eval')
    misc.init_distributed_mode(args)

    # fix the seed
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = args.cudnn_benchmark

    model = engine_cls(Pi3.from_pretrained("yyfz233/Pi3").to(device))
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.model == 'streaming_pi3':
        infer_func = partial(inference_streaming_model, model)
    else:
        infer_func = partial(inference_streaming_model_lc, model)
    if args.mode == 'eval_pose':
        ate_mean, rpe_trans_mean, rpe_rot_mean, seq_attr, outfile_list, bug = eval_pose_estimation(
            args,
            infer_func,
            device,
            dtype,
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
    if args.mode == 'eval_depth':
        eval_mono_depth_estimation(args, model, device, dtype)

    exit(0)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    model_variant = args.model
    if model_variant == 'pi3':
        pi3_main(args, VanillaEngine)
    elif model_variant == 'streaming_pi3':
        pi3_main(args, partial(StreamingWindowEngine, dtype=dtype, inference_device=device, window_size=20, overlap=5,
                               top_conf_percentile=0.5))
    elif model_variant == 'streaming_pi3_lc':
        pi3_main(args, partial(StreamingWindowEngineLC, dtype=dtype, inference_device=device, window_size=75, overlap=30,
                               top_conf_percentile=0.3))
    else:
        raise NotImplementedError
