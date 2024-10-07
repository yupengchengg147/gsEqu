import torch
import numpy as np
from tqdm import tqdm, trange
from PIL import Image
import os
import json
from utils.image_utils import psnr as get_psnr
from utils.loss_utils import ssim as get_ssim
from lpips import LPIPS
from argparse import ArgumentParser

lpips_fn = LPIPS(net="vgg").cuda()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--output_dir", type=str, help="The path to the output directory that stores the relighting results.")
    parser.add_argument("--gt_dir", type=str, help="The path to the output directory that stores the relighting ground truth.")
    parser.add_argument("--result_file", type=str, default="results.json", help="The path to save the results in a JSON file.")
    args = parser.parse_args()

    # light_name_list = ["bridge", "city", "fireplace", "forest", "night"]
    light_name_list = ["fireplace", "snow", "night"]

    # Initialize a list to store results for each light_name
    results = []

    parts = args.output_dir.split('/')
    scene_method = f"{parts[-2]}_{parts[-1]}"

    # Check if the result file already exists
    if os.path.exists(args.result_file):
        with open(args.result_file, 'r') as json_file:
            existing_results = json.load(json_file)  # Load existing data
    else:
        existing_results = {}  # If file doesn't exist, start with an empty list

    for light_name in light_name_list:
        print(f"evaluation {light_name}")
        num_test = 200
        psnr_avg = 0.0
        ssim_avg = 0.0
        lpips_avg = 0.0
        for idx in trange(num_test):
            with torch.no_grad():
                prediction = np.array(Image.open(os.path.join(args.output_dir, light_name, "test", "ours_45000", "renders",  f"{idx:05}.png")))[..., :3]  # [H, W, 3]
                prediction = torch.from_numpy(prediction).cuda().permute(2, 0, 1) / 255.0  # [3, H, W]
                gt_img = np.array(Image.open(os.path.join(args.gt_dir, f"test_{idx:03}", f"rgba_{light_name}.png")))[..., :3]  # [H, W, 3]
                gt_img = torch.from_numpy(gt_img).cuda().permute(2, 0, 1) / 255.0  # [3, H, W]
                
                # Calculate metrics
                psnr_avg += get_psnr(gt_img, prediction).mean().double()
                ssim_avg += get_ssim(gt_img, prediction).mean().double()
                lpips_avg += lpips_fn(gt_img, prediction).mean().double()

        # Compute the average for the current light_name
        psnr_avg /= num_test
        ssim_avg /= num_test
        lpips_avg /= num_test

        print(f"{light_name} psnr_avg: {psnr_avg}")
        print(f"{light_name} ssim_avg: {ssim_avg}")
        print(f"{light_name} lpips_avg: {lpips_avg}")

        # Store the results for the current light_name in a dictionary
        result = {
            "light_name": light_name,
            "psnr_avg": psnr_avg.item(),  # Convert to Python float
            "ssim_avg": ssim_avg.item(),  # Convert to Python float
            "lpips_avg": lpips_avg.item()  # Convert to Python float
        }
        results.append(result)

    # Append new results to the existing data
    existing_results[scene_method] = results

    # Save the updated results back to the JSON file
    with open(args.result_file, "w") as json_file:
        json.dump(existing_results, json_file, indent=4)
    
    print(f"Results appended to {args.result_file}")
