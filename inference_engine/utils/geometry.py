import torch
import numpy as np


def homogenize_points(
        points,
):
    """Convert batched points (xyz) to (xyz1)."""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def homogenize_points_np(
        points,
):
    """Convert batched points (xyz) to (xyz1)."""
    return np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)


def register_camera_poses_kabsch(src_cam_poses: np.ndarray, tgt_cam_poses: np.ndarray, scale=1.0):
    assert src_cam_poses.shape == tgt_cam_poses.shape
    src_cam_pos = src_cam_poses[:, :3, 3]
    src_cam_view = src_cam_poses[:, :3, :3] @ np.array([0., 0., -1.])
    src_cam_view_norm = src_cam_view / np.linalg.norm(src_cam_view, axis=-1, keepdims=True)

    tgt_cam_pos = tgt_cam_poses[:, :3, 3]
    tgt_cam_view = tgt_cam_poses[:, :3, :3] @ np.array([0., 0., -1.])
    tgt_cam_view_norm = tgt_cam_view / np.linalg.norm(tgt_cam_view, axis=-1, keepdims=True)

    src_centroid = np.mean(src_cam_pos, axis=0)
    tgt_centroid = np.mean(tgt_cam_pos, axis=0)

    # H = src_centered.T @ tgt_centered
    H = src_cam_view_norm.T @ tgt_cam_view_norm
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Fix improper rotation (reflection)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = tgt_centroid - scale * R @ src_centroid
    ret_se3 = np.eye(4)
    ret_se3[:3, :3] = scale * R
    ret_se3[:3, 3] = t
    return ret_se3


def register_camera_poses_kabsch_pytorch(
        src_cam_poses: torch.Tensor,
        tgt_cam_poses: torch.Tensor,
        scale=1.0
):
    assert src_cam_poses.shape == tgt_cam_poses.shape
    device = src_cam_poses.device

    src_cam_pos = src_cam_poses[:, :3, 3]
    tgt_cam_pos = tgt_cam_poses[:, :3, 3]

    view_direction = torch.tensor([0., 0., -1.], device=device)
    up_direction = torch.tensor([0., 1., 0.], device=device)

    src_cam_view = src_cam_poses[:, :3, :3] @ view_direction
    src_cam_view_norm = src_cam_view / torch.norm(src_cam_view, dim=-1, keepdim=True)
    src_cam_up = src_cam_poses[:, :3, :3] @ up_direction
    src_cam_up_norm = src_cam_up / torch.norm(src_cam_up, dim=-1, keepdim=True)

    tgt_cam_view = tgt_cam_poses[:, :3, :3] @ view_direction
    tgt_cam_view_norm = tgt_cam_view / torch.norm(tgt_cam_view, dim=-1, keepdim=True)
    tgt_cam_up = tgt_cam_poses[:, :3, :3] @ up_direction
    tgt_cam_up_norm = tgt_cam_up / torch.norm(tgt_cam_up, dim=-1, keepdim=True)

    src_corr = torch.vstack([scale * src_cam_pos,
                             scale * src_cam_pos + src_cam_view_norm,
                             scale * src_cam_pos + src_cam_up_norm])
    tgt_corr = torch.vstack([tgt_cam_pos,
                             tgt_cam_pos + tgt_cam_view_norm,
                             tgt_cam_pos + tgt_cam_up_norm])
    src_centroid = scale * src_cam_pos.mean(dim=0)
    tgt_centroid = tgt_cam_pos.mean(dim=0)
    src_corr_centered = src_corr - src_centroid
    tgt_corr_centered = tgt_corr - tgt_centroid

    # H = src_cam_view_norm.T @ tgt_cam_view_norm
    H = src_corr_centered.T @ tgt_corr_centered
    U, _, Vt = torch.linalg.svd(H)
    R = Vt.T @ U.T

    # Fix improper rotation (reflection)
    if torch.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = tgt_centroid - R @ src_centroid
    return R, t


def register_point_clouds_kabsch_pytorch(
        src_pcd: torch.Tensor,
        tgt_pcd: torch.Tensor,
        scale=1.0
):
    assert src_pcd.shape == tgt_pcd.shape

    src_corr = scale * src_pcd
    tgt_corr = tgt_pcd
    src_centroid = scale * src_pcd.mean(dim=0)
    tgt_centroid = tgt_pcd.mean(dim=0)
    src_corr_centered = src_corr - src_centroid
    tgt_corr_centered = tgt_corr - tgt_centroid

    # H = src_cam_view_norm.T @ tgt_cam_view_norm
    H = src_corr_centered.T @ tgt_corr_centered
    U, _, Vt = torch.linalg.svd(H)
    R = Vt.T @ U.T

    # Fix improper rotation (reflection)
    if torch.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # t = tgt_centroid - scale * R @ src_centroid
    t = tgt_centroid - R @ src_centroid
    return R, t


def apply_scale_with_so3(poses, R, scale):
    """
    Apply scale to camera poses in a rotated basis.

    Args:
        poses: (N, 4, 4) camera-to-world matrices
        R: (3, 3) rotation matrix (SO3)
        scale: float scalar

    Returns:
        poses_scaled: (N, 4, 4) scaled camera-to-world matrices
    """
    device = poses.device
    S = torch.eye(4, device=device)
    S[:3, :3] = scale * torch.eye(3, device=device)

    R_h = torch.eye(4, device=device)
    R_h[:3, :3] = R
    S_rot = R_h.T @ S @ R_h

    poses_scaled = S_rot @ poses
    return poses_scaled


def apply_sim3_to_pose(poses, scale, R, t):
    ret_pose = torch.eye(4, device=poses.device).repeat(poses.shape[0], 1, 1)
    R_c = poses[:, :3, :3]
    t_c = poses[:, :3, 3]

    R_new = R @ R_c
    t_new = scale * (R @ t_c.T).T + t
    ret_pose[:, :3, :3] = R_new
    ret_pose[:, :3, 3] = t_new

    return ret_pose


def closed_form_inverse_sim3(s, R, t):
    R_inv = R.T
    s_inv = 1.0 / s
    t_inv = -s_inv * (R_inv @ t)
    return s_inv, R_inv, t_inv


def accumulate_sim3(S1, S2):
    s1, R1, t1 = S1
    s2, R2, t2 = S2

    s = s1 * s2
    R = R1 @ R2
    t = s1 * R1 @ t2 + t1
    return s, R, t
