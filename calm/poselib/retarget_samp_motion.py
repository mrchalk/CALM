import os
import os.path as osp
import torch
import json

from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import SkeletonState, SkeletonMotion

from retarget_motion import project_joints

def main():
    retarget_data_path = "data/configs/retarget_smplx_to_amp.json"
    with open(retarget_data_path) as f:
        retarget_data = json.load(f)

    # load and visualize t-pose files
    source_tpose = SkeletonState.from_file(retarget_data["source_tpose"])

    target_tpose = SkeletonState.from_file(retarget_data["target_tpose"])

    joint_mapping = retarget_data["joint_mapping"]
    rotation_to_target_skeleton = torch.tensor(retarget_data["rotation"])
    
    fbx_dir = "D:/project/pytorch/MotionGeneration/SAMP/SAMP"
    fbx_files = os.listdir(fbx_dir)
    fbx_files = [file for file in fbx_files if file.endswith(".fbx") or file.endswith(".FBX")]
    target_motion_dir = "D:/project/pytorch/MotionGeneration/AMP/SAMP_dataset/npy"
    
    for fbx_file in fbx_files:
        if fbx_file in ["lie_down_5.fbx", "lie_down_6.fbx"] or osp.isfile(osp.join(target_motion_dir, fbx_file[:-4] + '.npy')):
            continue
        print(fbx_file)

        source_motion = SkeletonMotion.from_fbx(
            fbx_file_path=osp.join(fbx_dir, fbx_file),
            root_joint="pelvis",
            fps=60
        )

        # run retargeting
        target_motion = source_motion.retarget_to_by_tpose(
        joint_mapping=joint_mapping,
        source_tpose=source_tpose,
        target_tpose=target_tpose,
        rotation_to_target_skeleton=rotation_to_target_skeleton,
        scale_to_target_skeleton=retarget_data["scale"]
        )

        # keep frames between [trim_frame_beg, trim_frame_end - 1]
        frame_beg = retarget_data["trim_frame_beg"]
        frame_end = retarget_data["trim_frame_end"]
        if (frame_beg == -1):
            frame_beg = 0
            
        if (frame_end == -1):
            frame_end = target_motion.local_rotation.shape[0]
            
        local_rotation = target_motion.local_rotation
        root_translation = target_motion.root_translation
        local_rotation = local_rotation[frame_beg:frame_end, ...]
        root_translation = root_translation[frame_beg:frame_end, ...]
        
        new_sk_state = SkeletonState.from_rotation_and_root_translation(target_motion.skeleton_tree, local_rotation, root_translation, is_local=True)
        target_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=target_motion.fps)

        # need to convert some joints from 3D to 1D (e.g. elbows and knees)
        target_motion = project_joints(target_motion)

        # move the root so that the feet are on the ground
        local_rotation = target_motion.local_rotation
        root_translation = target_motion.root_translation
        tar_global_pos = target_motion.global_translation
        min_h = torch.min(tar_global_pos[..., 2])
        root_translation[:, 2] += -min_h
        
        # adjust the height of the root to avoid ground penetration
        root_height_offset = retarget_data["root_height_offset"]
        root_translation[:, 2] += root_height_offset
        
        new_sk_state = SkeletonState.from_rotation_and_root_translation(target_motion.skeleton_tree, local_rotation, root_translation, is_local=True)
        target_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=target_motion.fps)

        # save retargeted motion
        target_motion.to_file(osp.join(target_motion_dir, fbx_file[:-4] + '.npy'))
    
    return

if __name__ == '__main__':
    main()