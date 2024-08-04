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