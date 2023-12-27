import os
import os.path as osp
import torch
import json
import shutil

from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive

from retarget_motion import project_joints

def main():
    retarget_data_path = "data/configs/retarget_smplx_to_amp.json"
    with open(retarget_data_path) as f:
        retarget_data = json.load(f)

    # load and visualize t-pose files
    target_tpose = SkeletonState.from_file("data/amp_humanoid_tpose.npy")

    joint_mapping = {
        "Pelvis": "pelvis",
        "L_Hip": "left_thigh",
        "L_Knee": "left_shin",
        "L_Ankle": "left_foot",
        "R_Hip": "right_thigh",
        "R_Knee": "right_shin",
        "R_Ankle": "right_foot",
        "Spine1": "torso",
        "Head": "head",
        "L_Shoulder": "left_upper_arm",
        "L_Elbow": "left_lower_arm",
        "L_Wrist": "left_hand",
        "R_Shoulder": "right_upper_arm",
        "R_Elbow": "right_lower_arm",
        "R_Wrist": "right_hand"
    }
    rotation_to_target_skeleton = torch.tensor([0.5, 0.5, 0.5, 0.5])
    
    amass_motion_dir = "D:/project/pytorch/MotionGeneration/HumanML3D_UE/HumanML3D_UE/bvh_data/"

    for root, dirs, files in os.walk(amass_motion_dir):
        folder = root.replace("HumanML3D_UE/HumanML3D_UE/bvh_data", "AMASS/npy")
        os.makedirs(folder, exist_ok=True)
        for file_name in files:
            output_path = osp.join(folder, file_name[:-4] + ".npy")
            if osp.isfile(output_path):
                continue

            # make subject tpose
            source_tpose = SkeletonState.from_file('data/cmu_tpose.npy')
            source_motion = SkeletonMotion.from_bvh(
                bvh_file_path=osp.join(root, file_name),
                fps=30,
                strip_root=True
            )
            source_tpose._skeleton_tree = source_motion.skeleton_tree
            root_translation = source_motion.root_translation[0]
            source_tpose._root_translation = root_translation
            source_tpose._rotation = torch.zeros_like(source_motion.rotation[0])
            source_tpose._rotation[:, -1] = 1.0

            # plot_skeleton_state(source_tpose)

            print(output_path[45:])

            # run retargeting
            target_motion = source_motion.retarget_to_by_tpose(
                joint_mapping=joint_mapping,
                source_tpose=source_tpose,
                target_tpose=target_tpose,
                rotation_to_target_skeleton=rotation_to_target_skeleton,
                scale_to_target_skeleton= 1.0 / root_translation[1]
            )
            
            # keep frames between [trim_frame_beg, trim_frame_end]
            frame_beg = 1
            frame_end = target_motion.local_rotation.shape[0]

            if frame_end - frame_beg - 1 < 15:
                print("Less than 0.5s, too short!")
                continue
            
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

            # plot_skeleton_motion_interactive(target_motion)

            # save retargeted motion
            target_motion.to_file(output_path)

            # break

    return

if __name__ == '__main__':
    main()