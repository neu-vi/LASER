import numpy as np
from evo.core.trajectory import PoseTrajectory3D
from copy import deepcopy
from pathlib import Path
import cv2
from scipy.spatial.transform import Rotation
from utils.geometry import closed_form_inverse_se3
import os


def c2w_to_tumpose(c2w):
    """
    Convert a camera-to-world matrix to a tuple of translation and rotation
    
    input: c2w: 4x4 matrix
    output: tuple of translation and rotation (x y z qw qx qy qz)
    """
    # convert input to numpy
    xyz = c2w[:3, -1]
    rot = Rotation.from_matrix(c2w[:3, :3])
    qx, qy, qz, qw = rot.as_quat()
    tum_pose = np.concatenate([xyz, [qw, qx, qy, qz]])
    return tum_pose


def tumpose_to_c2w(tum_pose):
    x, y, z, qw, qx, qy, qz = tum_pose
    C = np.array([x, y, z])

    # quaternion (scipy expects [x, y, z, w])
    rot = Rotation.from_quat([qx, qy, qz, qw])
    R_cw = rot.as_matrix()

    # build SE3
    T_cw = np.eye(4)
    T_cw[:3, :3] = R_cw
    T_cw[:3, 3] = C
    return T_cw


def save_intrinsics(K_raw, path):
    K = K_raw.reshape(-1, 9)
    np.savetxt(path, K, fmt='%.6f')
    return K_raw


def make_traj(args) -> PoseTrajectory3D:
    if isinstance(args, tuple) or isinstance(args, list):
        traj, tstamps = args
        return PoseTrajectory3D(
            positions_xyz=traj[:, :3],
            orientations_quat_wxyz=traj[:, 3:],
            timestamps=tstamps,
        )
    assert isinstance(args, PoseTrajectory3D), type(args)
    return deepcopy(args)


def get_tum_poses(poses):
    tt = np.arange(len(poses)).astype(float)
    tum_poses = [c2w_to_tumpose(p) for p in poses]
    tum_poses = np.stack(tum_poses, 0)
    return [tum_poses, tt]


def get_se3_poses(tum_poses):
    """
    Convert a batch of TUM poses (N, 7) into SE3 (N, 4, 4) world-to-camera matrices.
    """
    return np.stack([tumpose_to_c2w(p) for p in tum_poses], axis=0)


def save_trajectory_tum_format(poses, filename):
    tum_poses = get_tum_poses(poses)
    traj = make_traj(tum_poses)
    tostr = lambda a: " ".join(map(str, a))
    with Path(filename).open("w") as f:
        for i in range(traj.num_poses):
            f.write(
                f"{traj.timestamps[i]} {tostr(traj.positions_xyz[i])} {tostr(traj.orientations_quat_wxyz[i][[0, 1, 2, 3]])}\n"
            )


def save_depth_maps(depth_maps, path):
    for i, depth_map in enumerate(depth_maps):
        np.save(f'{path}/frame_{(i):04d}.npy', depth_map)


def save_rgb_imgs(imgs, path):
    imgs = imgs.transpose(0, 2, 3, 1)  # now (S, H, W, 3)
    for i, img in enumerate(imgs):
        # convert from rgb to bgr
        img = img[..., ::-1]
        cv2.imwrite(f'{path}/frame_{i:04d}.png', (img * 255).astype(np.uint8))


def save_conf_maps(conf, path):
    for i, c in enumerate(conf):
        np.save(f'{path}/conf_{i}.npy', c)
    return conf


def save_for_viser(pred_dict, scene_name, output_path="./output_dy", inverse_extrinsic=True):
    # Unpack prediction dict
    images = pred_dict["images"]  # (S, 3, H, W)
    # world_points_map = pred_dict["world_points"]  # (S, H, W, 3)
    # conf_map = pred_dict["world_points_conf"]  # (S, H, W)

    depth_map = pred_dict["depth"]  # (S, H, W, 1)
    try:
        depth_map = np.squeeze(depth_map, axis=-1)
    except:
        pass
    depth_conf = pred_dict["depth_conf"]  # (S, H, W)

    extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
    intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)
    if inverse_extrinsic:
        extrinsics_cam = closed_form_inverse_se3(extrinsics_cam)  # shape (S, 4, 4) typically
    # For convenience, we store only (3,4) portion, cam_to_world
    extrinsics_cam = extrinsics_cam[:, :3, :]

    os.makedirs(f'{output_path}/{scene_name}', exist_ok=True)

    save_intrinsics(intrinsics_cam, f"{output_path}/{scene_name}/pred_intrinsics.txt")
    save_trajectory_tum_format(extrinsics_cam, f'{output_path}/{scene_name}/pred_traj.txt')
    save_depth_maps(depth_map, f'{output_path}/{scene_name}')
    save_conf_maps(depth_conf, f'{output_path}/{scene_name}')
    save_rgb_imgs(images, f'{output_path}/{scene_name}')
