import os
import os.path as osp
import torch

from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive

from retarget_motion import project_joints

import pandas as pd
from tqdm import tqdm


def main():
    dataset_root = "D:/project/pytorch/MotionGeneration/100STYLE/"

    # make subject tpose
    source_tpose = SkeletonState.from_file('data/cmu_tpose.npy')
    source_tpose_motion = SkeletonMotion.from_bvh(
        bvh_file_path=osp.join(dataset_root, "tpose.bvh"),
        fps=60
    )
    source_tpose._skeleton_tree = source_tpose_motion.skeleton_tree
    root_translation = source_tpose_motion.root_translation[0]
    source_tpose._root_translation = root_translation
    source_tpose._rotation = torch.zeros_like(source_tpose_motion.rotation[0])
    source_tpose._rotation[:, -1] = 1.0

    # plot_skeleton_state(source_tpose)

    # load and visualize t-pose files
    target_tpose = SkeletonState.from_file("data/amp_humanoid_tpose.npy")

    joint_mapping = {
        "Hips": "pelvis",
        "LeftHip": "left_thigh",
        "LeftKnee": "left_shin",
        "LeftAnkle": "left_foot",
        "RightHip": "right_thigh",
        "RightKnee": "right_shin",
        "RightAnkle": "right_foot",
        "Chest2": "torso",
        "Head": "head",
        "LeftShoulder": "left_upper_arm",
        "LeftElbow": "left_lower_arm",
        "LeftWrist": "left_hand",
        "RightShoulder": "right_upper_arm",
        "RightElbow": "right_lower_arm",
        "RightWrist": "right_hand"
    }
    rotation_to_target_skeleton = torch.tensor([0.5, 0.5, 0.5, 0.5])
    
    target_motion_dir = osp.join(dataset_root, "100STYLE_npy")

    bvh_dir = osp.join(dataset_root, "100STYLE_bvh")
    styles = os.listdir(bvh_dir)
    
    index_file = pd.read_csv(osp.join(dataset_root, "Frame_Cuts.csv"))

    frame_cuts = {}
    for i in tqdm(range(index_file.shape[0])):
        if index_file.loc[i]["STYLE_NAME"] not in styles:
            print(f"Style {index_file.loc[i]['STYLE_NAME']} not exists in {bvh_dir}")
            return
        frame_cuts[index_file.loc[i]["STYLE_NAME"]] = index_file.loc[i]

    for style in styles:
        if style not in frame_cuts.keys():
            print(f"Style {style} has no frame cuts info")
            continue
        
        style_dir = osp.join(bvh_dir, style)
        bvh_files = os.listdir(style_dir)
        for bvh_file in bvh_files:

            output_path = osp.join(target_motion_dir, bvh_file[:-4] + ".npy")
            if osp.isfile(output_path):
                continue

            print(bvh_file[:-4])

            source_motion = SkeletonMotion.from_bvh(
                bvh_file_path=osp.join(style_dir, bvh_file),
                fps=60
            )

            # run retargeting
            target_motion = source_motion.retarget_to_by_tpose(
                joint_mapping = joint_mapping,
                source_tpose = source_tpose,
                target_tpose = target_tpose,
                rotation_to_target_skeleton = rotation_to_target_skeleton,
                scale_to_target_skeleton = 1.0 / root_translation[1]
            )

            movement_type = bvh_file[:-4].split("_")[1]
            frame_beg = frame_cuts[style][f"{movement_type}_START"]
            frame_end = frame_cuts[style][f"{movement_type}_STOP"]
            if pd.isna(frame_beg):
                frame_beg = 0
            if pd.isna(frame_end):
                frame_end = target_motion.local_rotation.shape[0]
            frame_beg = int(frame_beg)
            frame_end = int(frame_end)
            
            local_rotation = target_motion.local_rotation
            root_translation = target_motion.root_translation
            # trim frames from start to stop, and downsample from 60fps to 30fps
            local_rotation = local_rotation[frame_beg:frame_end:2, ...]
            root_translation = root_translation[frame_beg:frame_end:2, ...]
            
            new_sk_state = SkeletonState.from_rotation_and_root_translation(target_motion.skeleton_tree, local_rotation, root_translation, is_local=True)
            target_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=30)

            # need to convert some joints from 3D to 1D (e.g. elbows and knees)
            target_motion = project_joints(target_motion)

            # move the root so that the feet are on the ground
            local_rotation = target_motion.local_rotation
            root_translation = target_motion.root_translation
            tar_global_pos = target_motion.global_translation
            min_h = torch.min(tar_global_pos[..., 2])
            root_translation[:, 2] += -min_h
            
            # adjust the height of the root to avoid ground penetration
            root_height_offset = 0.05
            root_translation[:, 2] += root_height_offset
            
            new_sk_state = SkeletonState.from_rotation_and_root_translation(target_motion.skeleton_tree, local_rotation, root_translation, is_local=True)
            target_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=target_motion.fps)

            # plot_skeleton_motion_interactive(target_motion)

            # save retargeted motion
            target_motion.to_file(output_path)

            # break
        # break

    return

if __name__ == '__main__':
    main()