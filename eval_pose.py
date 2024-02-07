import argparse
import os

import numpy as np
import torch


def normalize(x):
    return x / np.linalg.norm(x)


np.set_printoptions(suppress=True)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w


def recenter_poses(poses):

    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.0], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses


def load_poses(pose_path: str):
    poses_arr = np.load(pose_path)
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])

    poses[:2, 4, :] = np.array((270, 480)).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1.0 / 2.0
    bds = poses_arr[:, -2:].transpose([1, 0])

    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)  # (N, 3, 5)
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    bd_factor = 0.9
    sc = 1.0 / (np.percentile(bds[:, 0], 5) * bd_factor)
    poses[:, :3, 3] *= sc
    bds *= sc

    poses = recenter_poses(poses)
    c2w = poses_avg(poses)

    identities = np.repeat(np.array([[[0, 0, 0, 1]]]), 12, axis=0)
    poses = poses[:, :3, :4]
    poses = np.concatenate([poses, identities], axis=1)
    return poses


def align(model, data):
    """Align two trajectories using the method of Horn (closed-form).

    Args:
        model -- first trajectory (3xn)
        data -- second trajectory (3xn)

    Returns:
        rot -- rotation matrix (3x3)
        trans -- translation vector (3x1)
        trans_error -- translational error per point (1xn)

    """
    np.set_printoptions(precision=3, suppress=True)
    model_zerocentered = model - model.mean(1).reshape((3, -1))
    data_zerocentered = data - data.mean(1).reshape((3, -1))

    W = np.zeros((3, 3))
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:, column], data_zerocentered[:, column])
    U, d, Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity(3))
    if np.linalg.det(U) * np.linalg.det(Vh) < 0:
        S[2, 2] = -1
    rot = U * S * Vh
    trans = data.mean(1).reshape((3, -1)) - rot * model.mean(1).reshape((3, -1))

    model_aligned = rot * model + trans
    alignment_error = model_aligned - data

    trans_error = np.sqrt(np.sum(np.multiply(alignment_error, alignment_error), 0)).A[0]

    return rot, trans, trans_error


def evaluate_ate(gt_traj, est_traj, is_plot_traj=True):
    """
    Input :
        gt_traj: list of 4x4 matrices
        est_traj: list of 4x4 matrices
        len(gt_traj) == len(est_traj)
    """
    gt_traj_pts = [torch.Tensor(gt_traj[idx][:3, 3]) for idx in range(len(gt_traj))]
    est_traj_pts = [torch.Tensor(est_traj[idx][:3, 3]) for idx in range(len(est_traj))]

    gt_traj_pts = torch.stack(gt_traj_pts).detach().cpu().numpy().T
    est_traj_pts = torch.stack(est_traj_pts).detach().cpu().numpy().T

    rot, trans, trans_error = align(gt_traj_pts, est_traj_pts)

    avg_trans_error = trans_error.mean()

    return avg_trans_error


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_folder", type=str, required=True)
parser.add_argument("-g", "--gt_folder", type=str, required=True)

args = parser.parse_args()
gt_pose = load_poses(os.path.join(args.gt_folder, "poses_bounds.npy"))
pred_pose = load_poses(os.path.join(args.input_folder, "poses_bounds_RoDynRF.npy"))

ate_rmse = evaluate_ate(gt_pose, pred_pose, is_plot_traj=False)
print("Final Average ATE RMSE: {:.2f} cm".format(ate_rmse * 100))
