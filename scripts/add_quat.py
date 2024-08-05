import h5py 
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

data = h5py.File("dataset/data.h5py", 'r+')

for seq in tqdm(data.values()):
    for piece in seq.values():
        # piece.create_dataset("cam_pose_quat", piece['timestamps'].shape[0], dtype=np.float32)
        cam_pose_quat = np.zeros((piece['timestamps'].shape[0], 4), dtype=np.float32)
        cam_scene_matrix = piece['cam_scene_matrix']
        for idx, matrix in enumerate(cam_scene_matrix):
            rotation = R.from_matrix(matrix[:3, :3]).inv()
            quat = rotation.as_quat()
            cam_pose_quat[idx] = quat
        if "cam_pose_quat" in piece:
            del piece['cam_pose_quat']
        piece.create_dataset("cam_pose_quat", data=cam_pose_quat)

del data
        

        