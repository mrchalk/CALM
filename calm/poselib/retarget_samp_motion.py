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
    
    fbx_dir = "D:/project/pytorch/MotionGeneration/SAMP/lie_down_13/"
    fbx_files = os.listdir(fbx_dir)
    fbx_files = [file for file in fbx_files if file.endswith(".fbx") or file.endswith(".FBX")]
    fbx_files.sort()
    # print(fbx_files)

    # sit: walk from 1m away then sit
    # frame_begs = [658, 625, 338, 673, 713, 775, 720, 668, 478, 480, 500, 938, 328, 715, 340, 550, 568, 435, 870, 532]
    # sit only
    #frame_begs = [802, 768, 525, 793, 805, 870, 850, 800, 570, 570, 575, 1075, 475, 843, 475, 630, 670, 550, 1038, 620]
    # sit end
    #frame_ends = [1055, 1025, 825, 1063, 1040, 1158, 1190, 992, 750, 790, 895, 1380, 750, 1190, 820, 1138, 950, 865, 1282, 1042]

    # lie down
    # frame_begs = [167, 329, 265, 129, 0, # 1-5
    #               0, 252, 225, 208, 94, # 6-10
    #               229, 153, 265, 179, 141, # 11-15
    #               129, 254, 170, 138, 202, # 16-20
    #               100, 146, 120, 225, 230, # 21-25
    #               100, 115, 197, 200, 135, # 26-30
    #               126, 211, 143, 215, # 31-34
    #               ]
    # frame_ends = [370, 513, 446, 380, 0,
    #               0, 473, 405, 404, 296,
    #               437, 355, 504, 385, 316,
    #               384, 452, 383, 366, 416,
    #               265, 373, 312, 452, 440,
    #               270, 284, 360, 370, 323,
    #               270, 385, 316, 378,
    #               ]
    
    # arm chair
    # frame_begs = [230, 285, 186, 291, 272, 272, # 0-5
    #               184, 277, 152, 152, 141, # 6-10
    #               193, 223, 284, 315, 320, # 11-15
    #               358, 400, 252, 215, # 16-19
    #               ]
    # frame_ends = [400, 456, 338, 480, 505, 460,
    #               430, 450, 300, 289, 308,
    #               368, 425, 413, 488, 490,
    #               535, 600, 394, 368,
    #               ]

    # standup then walk for two steps, from chair_mo
    # frame_begs = [674, 574, 522, 614, 591, 744, # 0-5
    #               1049, 649, 625, 644, 691, # 6-10
    #               785, 543, 782, 671, 604, # 11-15
    #               645, 551, 706, 513 # 16-19
    #               ]
    # frame_ends = [800, 697, 707, 753, 726, 910,
    #               1143, 840, 794, 760, 875,
    #               970, 701, 876, 774, 728,
    #               837, 709, 825, 635
    #               ]

    # getup then walk for two steps, from lie_down
    frame_begs = [688, 575, 684, 807, 575, # 1-5
                  460, 622, 462, 663, 571, # 6-10
                  618, 492, 611, # 11-13
                  ]
    frame_ends = [872, 787, 873, 982, 794,
                  606, 804, 664, 829, 753,
                  779, 663, 777,
                  ]
    lie_down_index = [2, 3, 7, 8, 12, 21, 25, 27, 28, 29, 32, 33, 34]

    # rescale from 24fps to 60 fps
    frame_begs = [frame_num * 2.5 for frame_num in frame_begs]
    frame_ends = [frame_num * 2.5 for frame_num in frame_ends]

    target_motion_dir = "D:/project/pytorch/MotionGeneration/SAMP/npy/getup/"
    
    for index, fbx_file in enumerate(fbx_files):
        output_path = osp.join(target_motion_dir, fbx_file[:-4] + ".npy")

        if fbx_file in ["lie_down_5.fbx", "lie_down_6.fbx"] or osp.isfile(output_path):
            continue
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

        # lie down
        index = fbx_file[:-4].split("_")[-1]
        if index == "down":
            index = 0
        else:
            index = int(index) - 1
        index = lie_down_index.index(index + 1)
        
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