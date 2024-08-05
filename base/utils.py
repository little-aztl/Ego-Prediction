import yaml
import torch

def load_config(config_path):
    with open(config_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return
        
def yaw_pitch_to_norm_vector(yaw : torch.Tensor, pitch : torch.Tensor):
    vector = torch.stack((torch.tan(yaw), torch.tan(pitch), torch.ones_like(yaw)), dim=2)
    vector = vector / torch.linalg.norm(vector, dim=2)[:, :, None]
    return vector

def yaw_pitch_depth_to_vector(yaw : torch.Tensor, pitch : torch.Tensor, depth : torch.Tensor):
    vector = yaw_pitch_to_norm_vector(yaw, pitch)
    vector = vector * depth[:, :, None]
    return vector

def quat_inv(quat : torch.Tensor):
    '''
    quat: (batch_size * seq_len * 4)
    '''
    real_part = quat[:, :, 0]
    imag_part = quat[:, :, 1:]
    inv_quat = torch.cat((real_part[:, :, None], -imag_part), dim=2)
    return inv_quat

def quat_mult(q1 : torch.Tensor, q2 : torch.Tensor):
    w1, x1, y1, z1 = q1[:, :, 0], q1[:, :, 1], q1[:, :, 2], q1[:, :, 3]
    w2, x2, y2, z2 = q2[:, :, 0], q2[:, :, 1], q2[:, :, 2], q2[:, :, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return torch.stack((w, x, y, z), dim=2)