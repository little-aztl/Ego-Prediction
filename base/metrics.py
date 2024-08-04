import torch
from base.utils import yaw_pitch_to_norm_vector

def Average_Angular_Error(pred_eye_gaze : torch.Tensor, gt_eye_gaze : torch.Tensor):
    pred_vector = yaw_pitch_to_norm_vector(pred_eye_gaze[:, :, 0], pred_eye_gaze[:, :, 1])
    gt_vector = yaw_pitch_to_norm_vector(gt_eye_gaze[:, :, 0], gt_eye_gaze[:, :, 1])

    dot_product = torch.sum(pred_vector * gt_vector, dim=2)
    angular_error = torch.acos(dot_product) * 180 / torch.pi
    return torch.mean(angular_error).item()