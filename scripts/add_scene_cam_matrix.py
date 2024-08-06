import h5py 
import numpy as np
import quaternion
from tqdm import tqdm

data = h5py.File("dataset/data.h5py", 'r+')

for seq in tqdm(data.values()):
    for piece in seq.values():
        # piece.create_dataset("cam_pose_quat", piece['timestamps'].shape[0], dtype=np.float32)
        cam_scene_matrix = piece['cam_scene_matrix']
        scene_cam_matrix = np.linalg.inv(cam_scene_matrix)
        piece.create_dataset("scene_cam_matrix", data=scene_cam_matrix)

del data
        

        