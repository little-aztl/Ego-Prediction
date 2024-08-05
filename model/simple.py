import torch
import torch.nn as nn
from base.utils import yaw_pitch_depth_to_vector

class Simple_Eye_Gaze_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Simple_Eye_Gaze_MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, input_clip, device):
        # unfold along the time dimension
        x = input_clip['eye_gaze'].to(device)
        x = x.reshape(x.shape[0], -1)

        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)

        # fold along the time dimension
        out = out.reshape(out.shape[0], -1, 3)
        return out
    
class Simple_Eye_Gaze_Loss(nn.Module):
    def __init__(self):
        super(Simple_Eye_Gaze_Loss, self).__init__()
        
    def forward(self, y_pred, gt_clip, device):
        # print("y_pred.shape: ", y_pred.shape)
        y_true = gt_clip['eye_gaze'].to(device)
        angle_pred = yaw_pitch_depth_to_vector(y_pred[:, :, 0], y_pred[:, :, 1], y_pred[:, :, 2])
        angle_true = yaw_pitch_depth_to_vector(y_true[:, :, 0], y_true[:, :, 1], y_true[:, :, 2])
        return torch.mean(torch.sum((angle_pred - angle_true) ** 2, dim=2))
        

class Simple_Trajectory_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Simple_Trajectory_MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, input_clip, device):
        eye_gaze = input_clip['eye_gaze'].to(device)
        cam_scene_matrix = input_clip['cam_scene_matrix'].to(device)
        '''
        eye_gaze: (batch_size, len_per_input_seq, 3)
        scene_cam_matrix: (batch_size, len_per_input_seq, 4, 4)
        '''
        cam_scene_matrix = cam_scene_matrix.reshape(cam_scene_matrix.shape[0], cam_scene_matrix.shape[1], -1)
        x = torch.cat((eye_gaze, cam_scene_matrix), dim=2)
        # x: (batch_size, len_per_input_seq, 3 + 16)

        # unfold along the time dimension
        x = x.reshape(x.shape[0], -1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)

        # fold along the time dimension
        out = out.reshape(out.shape[0], -1, 3 + 4)
        out_coord = out[:, :, :3]
        out_quat = out[:, :, 3:]
        out_quat = torch.nn.functional.normalize(out_quat, p=2, dim=2)

        return out_coord, out_quat
    
class Simple_Trajectory_Loss(nn.Module):
    def __init__(self):
        super(Simple_Trajectory_Loss, self).__init__()
        
    def forward(self, y_pred, gt_clip, device):
        coord_pred = y_pred[0]
        quat_pred = y_pred[1]
        
        coord_true = (-gt_clip['cam_scene_matrix'][:, :, :3, 3]).to(device)
        quat_true = gt_clip['cam_pose_quat'].to(device)
        
        coord_loss = torch.mean(torch.sum((coord_pred - coord_true) ** 2, dim=2))
        quat_loss = torch.mean(torch.sum((quat_pred - quat_true) ** 2, dim=2))

        # print("coord_loss: {:.4f}, quat_loss: {:.4f}".format(coord_loss.item(), quat_loss.item()))

        return coord_loss + quat_loss