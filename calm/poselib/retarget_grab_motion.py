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
    target_tpose = SkeletonState.from_file(retarget_data["target_tpose"])

    joint_mapping = retarget_data["joint_mapping"]
    rotation_to_target_skeleton = torch.tensor(retarget_data["rotation"])
    
    fbx_dir = "D:/project/pytorch/MotionGeneration/GRAB/grab_fbx/"
    target_motion_dir = "D:/project/pytorch/MotionGeneration/GRAB/grab_npy/"
    lift_motion_dir = "D:/project/pytorch/MotionGeneration/GRAB/grab_lift/"

    for subject_index in range(1, 11):

        fbx_files = os.listdir(osp.join(fbx_dir, f"s{subject_index}"))
        fbx_files = [file for file in fbx_files if file.endswith(".fbx") or file.endswith(".FBX")]
        fbx_files.sort()

        for fbx_file in fbx_files:

            output_path = osp.join(target_motion_dir, f"s{subject_index}_{fbx_file[:-4]}.npy")
            if osp.isfile(output_path):
                continue

            print(f"s{subject_index}_{fbx_file[:-4]}")

            # make subject tpose
            source_tpose = SkeletonState.from_file('data/cmu_tpose.npy')
            source_tpose_motion = SkeletonMotion.from_fbx(
                fbx_file_path=osp.join(fbx_dir, f"s{subject_index}_tpose.fbx"),
                root_joint="pelvis",
                fps=24
            )
            source_tpose._skeleton_tree = source_tpose_motion.skeleton_tree
            root_translation = source_tpose_motion.root_translation[0]
            source_tpose._root_translation = root_translation
            source_tpose._rotation = torch.zeros_like(source_tpose_motion.rotation[0])
            source_tpose._rotation[:, -1] = 1.0

            source_motion = SkeletonMotion.from_fbx(
                fbx_file_path=osp.join(fbx_dir, f"s{subject_index}", fbx_file),
                root_joint="pelvis",
                fps=30
            )

            # run retargeting
            target_motion = source_motion.retarget_to_by_tpose(
                joint_mapping=joint_mapping,
                source_tpose=source_tpose,
                target_tpose=target_tpose,
                rotation_to_target_skeleton=rotation_to_target_skeleton,
                scale_to_target_skeleton= 1.0 / root_translation[1]
            )

            # keep frames between [trim_frame_beg, trim_frame_end]
            frame_beg = 15
            frame_end = -15

            # motion is at least 1s long
            if frame_end >= 0:
                frame_end = target_motion.local_rotation.shape[0]
                if frame_end - frame_beg < 30:
                    continue
            else:
                if target_motion.local_rotation.shape[0] - (frame_beg - frame_end) < 30:
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

            if "lift" in fbx_file[:-4]:
                shutil.copy(output_path, osp.join(lift_motion_dir, f"s{subject_index}_{fbx_file[:-4]}.npy"))

        #     break
        # break

    return

if __name__ == '__main__':
    main()