#!/bin/bash
export LD_LIBRARY_PATH=/is/software/nvidia/cuda-11.8/lib64
export PATH=$PATH:/is/software/nvidia/cuda-11.8/bin
export CUDA_HOME=/is/software/nvidia/cuda-11.8

source /home/pyu/miniforge3/etc/profile.d/conda.sh
conda activate 2dgs

cd /home/pyu/local_code/gsEqu

objs=("armadillo" "ficus" "hotdog" "lego")
methods=("df_alpha_0.01" "fw_alpha_0.01" "mixxed_alpha_0.01" "mixxed_reversed_alpha_0.01")

for obj in ${objs[@]}; do
    for method in ${methods[@]}; do
        python eval_albedo_ir.py --output_dir /is/cluster/fast/pyu/results_refined/${obj}/${method} --gt_dir /is/cluster/fast/pyu/data/tensorir/${obj} --result_file ./ir_refined_albedo.json
    done
done