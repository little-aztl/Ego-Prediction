import h5py 
import numpy as np
import quaternion
from tqdm import tqdm

data = h5py.File("dataset/data.h5py", 'r+')

for seq in tqdm(data.values()):
    for piece in seq.values():
        # piece.create_dataset("cam_pose_quat", piece['timestamps'].shape[0], dtype=np.float32)
        cam_pose_quat = np.zeros((piece['timestamps'].shape[0], 4), dtype=np.float32)
        cam_scene_matrix = piece['cam_scene_matrix']
        for idx, matrix in enumerate(cam_scene_matrix):
            matrix_inv = np.linalg.inv(matrix)
            quat = quaternion.from_rotation_matrix(matrix_inv[:3, :3])
            cam_pose_quat[idx] = quaternion.as_float_array(quat)
        if "cam_pose_quat" in piece:
            del piece['cam_pose_quat']
        piece.create_dataset("cam_pose_quat", data=cam_pose_quat)

del data
        

        