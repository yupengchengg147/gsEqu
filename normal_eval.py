import os
import glob
import json
from argparse import ArgumentParser

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from tqdm import tqdm


def get_mae(gt_normal_stack: np.ndarray, render_normal_stack: np.ndarray) -> float:
    # compute mean angular error
    MAE = np.mean(
        np.arccos(np.clip(np.sum(gt_normal_stack * render_normal_stack, axis=-1), -1, 1))
        * 180
        / np.pi
    )
    return MAE.item()


if __name__ == "__main__":
    parser = ArgumentParser(description="TensoIR convert script parameters")
    parser.add_argument("--output_dir", type=str, help="The path to the output directory that stores the predicted normal results.")
    parser.add_argument("--gt_dir", type=str, help="The path to the output directory that stores the normal ground truth.")
    parser.add_argument("--result_file", type=str, help="The path where the MAE results should be saved as a JSON file.")
    args = parser.parse_args()

    parts = args.output_dir.split('/')
    scene_method = f"{parts[-2]}_{parts[-1]}"

    output_dir = args.output_dir
    result_file = args.result_file

    test_dirs = glob.glob(os.path.join(args.gt_dir, "test_*"))
    test_dirs.sort()

    normal_gt_stack = []
    normal_gs_stack = []
    normal_from_depth_stack = []
    normal_bg = np.array([0.0, 0.0, 1.0])

    for test_dir in tqdm(test_dirs):
        test_id = int(test_dir.split("_")[-1])
        normal_gt_path = os.path.join(test_dir, "normal.png")
        normal_gt_img = Image.open(normal_gt_path)
        normal_gt = np.array(normal_gt_img)[..., :3] / 255  # [H, W, 3] in range [0, 1]
        normal_gt = (normal_gt - 0.5) * 2.0  # [H, W, 3] in range (-1, 1)
        alpha_mask = np.array(normal_gt_img)[..., [-1]] / 255  # [H, W, 1] in range [0, 1]
        normal_gt = normal_gt * alpha_mask + normal_bg * (1.0 - alpha_mask)  # [H, W, 3]
        normal_gt = normal_gt / np.linalg.norm(normal_gt, axis=-1, ord=2, keepdims=True)
        normal_gt_stack.append(normal_gt)

        # gs normal
        normal_gs_path = os.path.join(output_dir, "test", "ours_45000", "n_Render", f"{test_id:05d}.png")
        normal_gs_img = Image.open(normal_gs_path)
        normal_gs = np.array(normal_gs_img)[..., :3] / 255  # [H, W, 3] in range [0, 1]
        normal_gs = (normal_gs - 0.5) * 2.0  # [H, W, 3] in range (-1, 1)
        mask = (np.array(normal_gs_img)[..., :3] == np.array([128, 128, 255], dtype=np.uint8)).all(-1)
        normal_gs[mask] = np.array([0.0, 0.0, 1.0])
        normal_gs = normal_gs / np.linalg.norm(normal_gs, axis=-1, ord=2, keepdims=True)
        normal_gs_stack.append(normal_gs)

        # normal from depth
        normal_from_depth_path = os.path.join(output_dir, "test", "ours_45000", "n_Depth", f"{test_id:05d}.png")
        normal_from_depth_img = Image.open(normal_from_depth_path)
        normal_from_depth = np.array(normal_from_depth_img)[..., :3] / 255  # [H, W, 3] in range [0, 1]
        normal_from_depth = (normal_from_depth - 0.5) * 2.0  # [H, W, 3] in range (-1, 1)
        mask = (np.array(normal_from_depth_img)[..., :3] == np.array([128, 128, 255], dtype=np.uint8)).all(-1)
        normal_from_depth[mask] = np.array([0.0, 0.0, 1.0])
        normal_from_depth = normal_from_depth / np.linalg.norm(normal_from_depth, axis=-1, ord=2, keepdims=True)
        normal_from_depth_stack.append(normal_from_depth)

    # MAE
    normal_gt_stack = np.stack(normal_gt_stack)
    normal_gs_stack = np.stack(normal_gs_stack)
    normal_from_depth_stack = np.stack(normal_from_depth_stack)
    mae_gs = get_mae(normal_gt_stack, normal_gs_stack)
    mae_from_depth = get_mae(normal_gt_stack, normal_from_depth_stack)
    print(f"MAE: gs={mae_gs}; from_depth={mae_from_depth}")

   
    # Load existing results if the file exists
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = {}

    # Update existing data with new results
    new_results = {
        "mae_gs": mae_gs,
        "mae_from_depth": mae_from_depth
    }

    if scene_method in existing_data.keys():
        if existing_data[scene_method] is not None:
            existing_data[scene_method].append(new_results)
    else:
        existing_data[scene_method] = new_results

    
    # Save updated data back to the JSON file
    with open(result_file, 'w') as f:
        json.dump(existing_data, f, indent=4)

    print(f"MAE results appended to {result_file}")