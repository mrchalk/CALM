import os
import os.path as osp
import torch
import json

from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_motion_interactive

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
    
    fbx_dir = "D:/project/pytorch/MotionGeneration/SAMP/chair_mo/"
    fbx_files = os.listdir(fbx_dir)
    fbx_files = [file for file in fbx_files if file.endswith(".fbx") or file.endswith(".FBX")]
    fbx_files.sort()
    # print(fbx_files)

    # walk from 1m away then sit
    # frame_begs = [658, 625, 338, 673, 713, 775, 720, 668, 478, 480, 500, 938, 328, 715, 340, 550, 568, 435, 870, 532]
    # sit only begin
    frame_begs = [802, 768, 525, 793, 805, 870, 850, 800, 570, 570, 575, 1075, 475, 843, 475, 630, 670, 550, 1038, 620]
    frame_ends = [1055, 1025, 825, 1063, 1040, 1158, 1190, 992, 750, 790, 895, 1380, 750, 1190, 820, 1138, 950, 865, 1282, 1042]

    target_motion_dir = "D:/project/pytorch/MotionGeneration/SAMP/npy/sit_only/"
    
    for index, fbx_file in enumerate(fbx_files):
        output_path = osp.join(target_motion_dir, fbx_file[:-4] + ".npy")

        # if fbx_file in ["lie_down_5.fbx", "lie_down_6.fbx"] or osp.isfile(output_path):
        #     continue
        print(fbx_file)

        source_motion = SkeletonMotion.from_fbx(
            fbx_file_path=osp.join(fbx_dir, fbx_file),
            root_joint="pelvis",
            fps=30
        )

        # run retargeting
        target_motion = source_motion.retarget_to_by_tpose(
        joint_mapping=joint_mapping,
        source_tpose=source_tpose,
        target_tpose=target_tpose,
        rotation_to_target_skeleton=rotation_to_target_skeleton,
        scale_to_target_skeleton=retarget_data["scale"]
        )

        # keep frames between [trim_frame_beg, trim_frame_end]
        frame_beg = round(frame_begs[index] * 0.5)
        frame_end = round(frame_ends[index] * 0.5)

        # motion is at least 1s long
        # frame_beg = 15
        # frame_end = -15
        # if frame_end >= 0:
        #     frame_end = target_motion.local_rotation.shape[0]
        #     if frame_end - frame_beg < 30:
        #         continue
        # else:
        #     if target_motion.local_rotation.shape[0] - (frame_beg - frame_end) < 30:
        #         continue
        
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