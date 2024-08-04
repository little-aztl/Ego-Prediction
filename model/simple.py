import torch
import torch.nn as nn
from base.utils import yaw_pitch_depth_to_vector

class Simple_Eye_Gaze_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Simple_Eye_Gaze_MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        # unfold along the time dimension
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
        
    def forward(self, y_pred, y_true):
        # print("y_pred.shape: ", y_pred.shape)
        angle_pred = yaw_pitch_depth_to_vector(y_pred[:, :, 0], y_pred[:, :, 1], y_pred[:, :, 2])
        angle_true = yaw_pitch_depth_to_vector(y_true[:, :, 0], y_true[:, :, 1], y_true[:, :, 2])
        return torch.mean(torch.sum((angle_pred - angle_true) ** 2, dim=2))
        

        