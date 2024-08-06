from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core import calibration
from projectaria_tools.projects.adt import (
    AriaDigitalTwinDataProvider,
    AriaDigitalTwinSkeletonProvider,
    AriaDigitalTwinDataPathsProvider,
    bbox3d_to_line_coordinates,
    bbox2d_to_image_coordinates,
    utils as adt_utils,
)
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import h5py, time
import torch

class ADT_Dataset(Dataset):
    def __init__(
        self, 
        h5py_path="dataset/data.h5py", 
        len_per_input_seq=100,
        len_per_output_seq=30,
        interval=50, 
        stride=5,
        train=True,
        dataset_path="dataset/"
        ):
        
        self.whole_data = h5py.File(h5py_path, "r")
        self.len_per_input_seq = len_per_input_seq
        self.len_per_output_seq = len_per_output_seq
        self.interval = interval
        self.dataset_path = dataset_path

        self.sequence_count = 0
        self.piece_count = 0
        self.frame_count = 0

        self.sequence_names = []
        self.h5py_paths = []
        self.intervals = []

        print("Loading data...")
        for sequence_name in tqdm(self.whole_data.keys()):
            if train and not self.whole_data[sequence_name].attrs['train']:
                continue
            if not train and self.whole_data[sequence_name].attrs['train']:
                continue
            cur_seq = self.whole_data[sequence_name]
            self.sequence_count += 1
            for device_idx in cur_seq.keys():
                cur_piece = cur_seq[device_idx]
                self.piece_count += 1

                cur_len = cur_piece['timestamps'].shape[0]
                self.frame_count += cur_len

                l = 0
                r = self.len_per_input_seq + self.len_per_output_seq - 1
                while r < cur_len:
                    self.sequence_names.append(sequence_name)
                    self.h5py_paths.append(sequence_name + '/' + device_idx)
                    self.intervals.append((l, r, stride))
                    l += self.interval
                    r += self.interval

        self.sequence_names = np.array(self.sequence_names)
        self.h5py_paths = np.array(self.h5py_paths)
        self.intervals = np.array(self.intervals)

    def __len__(self):
        return self.sequence_names.shape[0]
    
    def __getitem__(self, index):
        cur_piece = self.whole_data[self.h5py_paths[index]]
        res_input = dict()
        res_output = dict()
        l, r, stride = self.intervals[index]
        res_input['timestamps'] = torch.from_numpy(cur_piece['timestamps'][l : l + self.len_per_input_seq : stride])
        res_output['timestamps'] = torch.from_numpy(cur_piece['timestamps'][l + self.len_per_input_seq : r + 1 : stride])
        res_input['video'] = self.get_video(cur_piece['img_path'][l : l + self.len_per_input_seq : stride])
        res_output['video'] = self.get_video(cur_piece['img_path'][l + self.len_per_input_seq : r + 1 : stride])
        res_input['cam_scene_matrix'] = torch.from_numpy(cur_piece['cam_scene_matrix'][l : l + self.len_per_input_seq : stride])
        res_output['cam_scene_matrix'] = torch.from_numpy(cur_piece['cam_scene_matrix'][l + self.len_per_input_seq : r + 1 : stride])
        res_input['scene_cam_matrix'] = torch.from_numpy(cur_piece['scene_cam_matrix'][l : l + self.len_per_input_seq : stride])
        res_output['scene_cam_matrix'] = torch.from_numpy(cur_piece['scene_cam_matrix'][l + self.len_per_input_seq : r + 1 : stride])
        res_input['3d_boundingboxes_aabb'] = torch.from_numpy(cur_piece['3d_boundingboxes_aabb'][l : l + self.len_per_input_seq : stride])
        res_output['3d_boundingboxes_aabb'] = torch.from_numpy(cur_piece['3d_boundingboxes_aabb'][l + self.len_per_input_seq : r + 1 : stride])
        res_input['3d_boundingboxes_transform_scene_object_matrix'] = torch.from_numpy(cur_piece['3d_boundingboxes_transform_scene_object_matrix'][l : l + self.len_per_input_seq : stride])
        res_output['3d_boundingboxes_transform_scene_object_matrix'] = torch.from_numpy(cur_piece['3d_boundingboxes_transform_scene_object_matrix'][l + self.len_per_input_seq : r + 1 : stride])
        res_input['2d_boundingboxes_aabb'] = torch.from_numpy(cur_piece['2d_boundingboxes_aabb'][l : l + self.len_per_input_seq : stride])
        res_output['2d_boundingboxes_aabb'] = torch.from_numpy(cur_piece['2d_boundingboxes_aabb'][l + self.len_per_input_seq : r + 1 : stride])
        res_input['eye_gaze'] = torch.from_numpy(cur_piece['eye_gaze'][l : l + self.len_per_input_seq : stride])
        res_output['eye_gaze'] = torch.from_numpy(cur_piece['eye_gaze'][l + self.len_per_input_seq : r + 1 : stride])
        res_input['cam_pose_quat'] = torch.from_numpy(cur_piece['cam_pose_quat'][l : l + self.len_per_input_seq : stride])
        res_output['cam_pose_quat'] = torch.from_numpy(cur_piece['cam_pose_quat'][l + self.len_per_input_seq : r + 1 : stride])
        return res_input, res_output

    def get_video(self, img_paths):
        frames = []
        for img_path in img_paths:
            frame = plt.imread(self.dataset_path + img_path.decode('utf-8'))
            frames.append(frame)
        return torch.from_numpy(np.stack(frames))

class Sanity_Check_Model(nn.Module):
    def __init__(self):
        super(Sanity_Check_Model, self).__init__()
    def forward(self, piece):
        print("The shape of timestamps:", piece['timestamps'].shape)
        print("The shape of cam_scene_matrix:", piece['cam_scene_matrix'].shape)
        print("The shape of 3d_boundingboxes_aabb:", piece['3d_boundingboxes_aabb'].shape)
        print("The shape of 3d_boundingboxes_transform_scene_object_matrix:", piece['3d_boundingboxes_transform_scene_object_matrix'].shape)
        print("The shape of 2d_boundingboxes_aabb:", piece['2d_boundingboxes_aabb'].shape)
        print("The shape of video:", piece['video'].shape)
        print("The shape of eye_gaze:", piece['eye_gaze'].shape)
        print("The shape of cam_pose_quat:", piece['cam_pose_quat'].shape)
        print("---------------")
    
if __name__ == "__main__":
    my_dataset = ADT_Dataset()
    print("sequence_count:", my_dataset.sequence_count)
    print("piece_count:", my_dataset.piece_count)
    print("total_frame_count", my_dataset.frame_count)
    print("count of my_dataset", len(my_dataset))

    dataloader = DataLoader(my_dataset, batch_size=8, shuffle=True, num_workers=36)

    my_model = Sanity_Check_Model()
    for input, output in dataloader:
        print("\033[1m" + "Input" + "\033[0m")
        my_model(input)
        print("\033[1m" + "Output" + "\033[0m")
        my_model(output)