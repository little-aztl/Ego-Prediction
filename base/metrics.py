import torch
import torch.nn as nn
from base.utils import yaw_pitch_to_norm_vector, quat_inv, quat_mult

class Average_Gaze_Angular_Error(nn.Module):
    def __init__(self):
        super(Average_Gaze_Angular_Error, self).__init__()
    def forward(self, pred_eye_gaze : torch.Tensor, gt_clip, device):
        gt_eye_gaze = gt_clip['eye_gaze'].to(device)
        pred_vector = yaw_pitch_to_norm_vector(pred_eye_gaze[:, :, 0], pred_eye_gaze[:, :, 1])
        gt_vector = yaw_pitch_to_norm_vector(gt_eye_gaze[:, :, 0], gt_eye_gaze[:, :, 1])

        dot_product = torch.sum(pred_vector * gt_vector, dim=2)
        angular_error = torch.acos(dot_product) * 180 / torch.pi
        return torch.mean(angular_error)


class Average_Traj_Error(nn.Module):
    def __init__(self):
        super(Average_Traj_Error, self).__init__()

    def forward(self, pred_traj : torch.Tensor, gt_clip, device):
        pred_coord = pred_traj[0]
        pred_quat = pred_traj[1]

        gt_coord = (-gt_clip['cam_scene_matrix'][:, :, :3, 3]).to(device)
        gt_quat = gt_clip['cam_pose_quat'].to(device)

        coord_error = torch.mean(torch.norm(pred_coord - gt_coord, dim=2))
        
        error_quat = quat_mult(quat_inv(pred_quat), gt_quat)
        angular_error = torch.mean(torch.acos(error_quat[:, :, 0]) * 180 / torch.pi)

        return coord_error, angular_error