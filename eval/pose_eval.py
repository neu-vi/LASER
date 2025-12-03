import os
import math
import numpy as np
import torch
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3
from eval.save_func import get_tum_poses, save_for_viser, get_se3_poses
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from eval.vo_eval import load_traj, eval_metrics, plot_trajectory, process_directory, calculate_averages
import eval.misc as misc
import torch.distributed as dist
from tqdm import tqdm
from eval.eval_metadata import dataset_metadata
from dycheck.emf import compute_angular_emf


def eval_pose_estimation(args, model, device, dtype, save_dir=None, inverse_extrinsic=True, fps=None):
    metadata = dataset_metadata.get(args.eval_dataset, dataset_metadata['sintel'])
    img_path = metadata['img_path']
    mask_path = metadata['mask_path']

    ate_mean, rpe_trans_mean, rpe_rot_mean, seq_attr, outfile_list, bug = eval_pose_estimation_dist(
        args, model, device, dtype, save_dir=save_dir, img_path=img_path, mask_path=mask_path,
        inverse_extrinsic=inverse_extrinsic, fps=fps
    )
    return ate_mean, rpe_trans_mean, rpe_rot_mean, seq_attr, outfile_list, bug


def eval_pose_estimation_dist(args, model, device, dtype, img_path, save_dir=None, mask_path=None,
                              inverse_extrinsic=True, fps=None):
    metadata = dataset_metadata.get(args.eval_dataset, dataset_metadata['sintel'])
    anno_path = metadata.get('anno_path', None)

    # silent = args.silent
    seq_list = args.seq_list
    if seq_list is None:
        if metadata.get('full_seq', False):
            args.full_seq = True
        else:
            seq_list = metadata.get('seq_list', [])
        if args.full_seq:
            seq_list = os.listdir(img_path)
            seq_list = [seq for seq in seq_list if os.path.isdir(os.path.join(img_path, seq))]
        seq_list = sorted(seq_list)

    if save_dir is None:
        save_dir = args.output_dir

    # Split seq_list across processes
    if misc.is_dist_avail_and_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    total_seqs = len(seq_list)
    seqs_per_proc = (total_seqs + world_size - 1) // world_size  # Ceiling division

    start_idx = rank * seqs_per_proc
    end_idx = min(start_idx + seqs_per_proc, total_seqs)

    seq_list = seq_list[start_idx:end_idx]

    ate_list = []
    rpe_trans_list = []
    rpe_rot_list = []
    outfile_list = []
    seq_attr = {}
    # load_img_size = 512

    error_log_path = f'{save_dir}/_error_log_{rank}.txt'  # Unique log file per process
    bug = False

    for seq in tqdm(seq_list):
        try:
            dir_path = metadata['dir_path_func'](img_path, seq)

            # Handle skip_condition
            skip_condition = metadata.get('skip_condition', None)
            if skip_condition is not None and skip_condition(save_dir, seq):
                continue

            # mask_path_seq_func = metadata.get('mask_path_seq_func', lambda mask_path, seq: None)
            # mask_path_seq = mask_path_seq_func(mask_path, seq)

            filelist = [os.path.join(dir_path, name) for name in os.listdir(dir_path)]
            filelist.sort()
            filelist = filelist[::args.pose_eval_stride]
            max_winsize = max(1, math.ceil((len(filelist) - 1) / 2))
            scene_graph_type = args.scene_graph_type
            if int(scene_graph_type.split('-')[1]) > max_winsize:
                scene_graph_type = f'{args.scene_graph_type.split("-")[0]}-{max_winsize}'
                if len(scene_graph_type.split("-")) > 2:
                    scene_graph_type += f'-{args.scene_graph_type.split("-")[2]}'

            # imgs = load_images(
            #     filelist, size=load_img_size, verbose=False,
            #     dynamic_mask_root=mask_path_seq, crop=not args.no_crop
            # )
            imgs = load_and_preprocess_images(filelist).to(device)

            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    # Predict attributes including cameras, depth maps, and point maps.
                    predictions = model(imgs)
                    # predictions = model(imgs, dir_path)

            if 'pose_enc' in predictions.keys():
                print("Converting pose encoding to extrinsic and intrinsic matrices...")
                extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], imgs.shape[-2:])
                predictions["extrinsic"] = extrinsic
                predictions["intrinsic"] = intrinsic

            print("Processing model outputs...")
            for key in predictions.keys():
                if isinstance(predictions[key], torch.Tensor):
                    predictions[key] = predictions[key].cpu().numpy().squeeze(
                        0)  # remove batch dimension and convert to numpy

            pred_traj = predictions["extrinsic"]
            if inverse_extrinsic:
                pred_traj = closed_form_inverse_se3(pred_traj)  # shape (S, 4, 4) typically
            # For convenience, we store only (3,4) portion, cam_to_world
            pred_traj = pred_traj[:, :3, :]
            pred_traj = get_tum_poses(pred_traj)

            os.makedirs(f'{save_dir}/{seq}', exist_ok=True)

            save_for_viser(predictions, seq, save_dir)

            gt_traj_file = metadata['gt_traj_func'](img_path, anno_path, seq)
            traj_format = metadata.get('traj_format', None)

            if args.eval_dataset == 'sintel':
                gt_traj = load_traj(gt_traj_file=gt_traj_file, stride=args.pose_eval_stride)
            elif traj_format is not None:
                gt_traj = load_traj(gt_traj_file=gt_traj_file, traj_format=traj_format)
            else:
                gt_traj = None

            if gt_traj is not None:
                ate, rpe_trans, rpe_rot = eval_metrics(
                    pred_traj, gt_traj, seq=seq, filename=f'{save_dir}/{seq}_eval_metric.txt'
                )
                plot_trajectory(
                    pred_traj, gt_traj, title=seq, filename=f'{save_dir}/{seq}.png'
                )
                gt_traj_se3 = get_se3_poses(gt_traj[0])
                angular_emf = compute_angular_emf(
                    gt_traj_se3[:, :3, :3].transpose(0, 2, 1),
                    gt_traj_se3[:, :3, 3],
                    fps=fps
                ) if fps is not None else 0
                translations = gt_traj_se3[:, :3, 3]
                delta_translations = translations[1:] - translations[:-1]
                displacements = np.linalg.norm(delta_translations, axis=-1)
                total_displacement = np.sum(displacements)
                avg_displacement = np.mean(displacements)

            else:
                ate, rpe_trans, rpe_rot, angular_emf = 0, 0, 0, 0
                total_displacement, avg_displacement = 0, 0
                outfile = None
                bug = True

            ate_list.append(ate)
            rpe_trans_list.append(rpe_trans)
            rpe_rot_list.append(rpe_rot)
            seq_attr[seq] = {
                'angular_emf': angular_emf,
                'total_displacement' : total_displacement,
                'avg_displacement': avg_displacement
            }
            # outfile_list.append(outfile)

            # Write to error log after each sequence
            with open(error_log_path, 'a') as f:
                f.write(
                    f'{args.eval_dataset}-{seq: <16} | ATE: {ate:.5f}, RPE trans: {rpe_trans:.5f}, RPE rot: {rpe_rot:.5f}\n')
                f.write(f'{ate:.5f}\n')
                f.write(f'{rpe_trans:.5f}\n')
                f.write(f'{rpe_rot:.5f}\n')

        except Exception as e:
            if 'out of memory' in str(e):
                # Handle OOM
                torch.cuda.empty_cache()  # Clear the CUDA memory
                with open(error_log_path, 'a') as f:
                    f.write(f'OOM error in sequence {seq}, skipping this sequence.\n')
                print(f'OOM error in sequence {seq}, skipping...')
            elif 'Degenerate covariance rank' in str(e) or 'Eigenvalues did not converge' in str(e):
                # Handle Degenerate covariance rank exception and Eigenvalues did not converge exception
                with open(error_log_path, 'a') as f:
                    f.write(f'Exception in sequence {seq}: {str(e)}\n')
                print(f'Traj evaluation error in sequence {seq}, skipping.')
            else:
                raise e  # Rethrow if it's not an expected exception

    # Aggregate results across all processes
    if misc.is_dist_avail_and_initialized():
        torch.distributed.barrier()

    bug_tensor = torch.tensor(int(bug), device=device)

    bug = bool(bug_tensor.item())

    # Handle outfile_list
    outfile_list_all = [None for _ in range(world_size)]

    outfile_list_combined = []
    for sublist in outfile_list_all:
        if sublist is not None:
            outfile_list_combined.extend(sublist)

    results = process_directory(save_dir)
    avg_ate, avg_rpe_trans, avg_rpe_rot = calculate_averages(results)

    # Write the averages to the error log (only on the main process)
    if rank == 0:
        with open(f'{save_dir}/_error_log.txt', 'a') as f:
            # Copy the error log from each process to the main error log
            for i in range(world_size):
                with open(f'{save_dir}/_error_log_{i}.txt', 'r') as f_sub:
                    f.write(f_sub.read())
            f.write(
                f'Average ATE: {avg_ate:.5f}, Average RPE trans: {avg_rpe_trans:.5f}, Average RPE rot: {avg_rpe_rot:.5f}\n')

    return avg_ate, avg_rpe_trans, avg_rpe_rot, seq_attr, outfile_list_combined, bug
