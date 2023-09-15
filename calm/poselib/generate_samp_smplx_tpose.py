import torch
import numpy as np

from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion

# fbx_file = "D:/project/pytorch/MotionGeneration/SAMP_tpose.fbx"
# motion = SkeletonMotion.from_fbx(
#     fbx_file_path=fbx_file,
#     root_joint="pelvis",
#     fps=60
# )
# motion.to_file("D:/project/pytorch/MotionGeneration/AMP/SAMP_dataset/smplx_tpose.npy")

cmu_tpose = SkeletonState.from_file('data/cmu_tpose.npy')
smplx_motion = np.load('D:/project/pytorch/MotionGeneration/AMP/SAMP_dataset/smplx/armchair.npy', allow_pickle = True)

smplx_motion.item()['skeleton_tree']['local_translation']['arr'] = smplx_motion.item()['skeleton_tree']['local_translation']['arr'] * 0.01
cmu_tpose._skeleton_tree = SkeletonTree.from_dict(smplx_motion.item()['skeleton_tree'])
cmu_tpose._root_translation = torch.from_numpy(np.array([0.00116772, 0.99210936, 0.01266907], dtype=np.float32))
rotation = np.zeros_like(smplx_motion.item()['rotation']['arr'][0], dtype = np.float32)
rotation[:, -1] = 1.0
cmu_tpose._rotation = torch.from_numpy(rotation)
cmu_tpose.to_file("data/smplx_tpose.npy")