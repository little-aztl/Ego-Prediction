import yaml
import torch

def load_config(config_path):
    with open(config_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return
        
def yaw_pitch_depth_to_vector(yaw : torch.Tensor, pitch : torch.Tensor, depth : torch.Tensor):
    yaw = yaw.to(torch.float64)
    pitch = pitch.to(torch.float64)
    depth = depth.to(torch.float64)
    vector = torch.hstack((torch.tan(yaw), torch.tan(pitch), torch.ones_like(yaw)))
    norms = torch.linalg.norm(vector, axis=1)
    vector /= norms[:, None]
    vector *= depth
    return vector