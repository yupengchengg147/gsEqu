#!/bin/bash
module load cuda/11.8

export TORCH_EXTENSIONS_DIR="/tmp/.cache"
export PATH=$PATH
export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1
export LD_LIBRARY_PATH=/is/software/nvidia/cuda-11.8/lib64:$LD_LIBRARY_PATH
export PATH=/is/software/nvidia/cuda-11.8/bin:$PATH
export CUDA_HOME=/is/software/nvidia/cuda-11.8
export C_INCLUDE_PATH=/is/software/nvidia/cudnn-8.7.0-cu11.x/include
export CPLUS_INCLUDE_PATH=$C_INCLUDE_PATH
export LD_LIBRARY_PATH=/is/software/nvidia/cudnn-8.7.0-cu11.x/lib64:$LD_LIBRARY_PATH


# /is/cluster/pyu/miniforge3/envs/2dgs/bin/python pbr_train.py -s /is/cluster/fast/pyu/data/refnerf/helmet -m /is/cluster/fast/pyu/refnerf_results/helmet/iter_1_1 -w --eval --warmup_iterations 1 --lambda_dist 100 --lambda_normal 0.01 --fw_iter 1 --df_iter 1 --mode iterative --gamma --tone

models=("ball" "car" "coffee" "helmet" "teapot" "toaster")
fw_iter_intervals=(1 1 1 1 5 25 100)
df_iter_intervals=(1 5 25 100 1 1 1)

for model in "${models[@]}"; do
    for i in "${!fw_iter_intervals[@]}"; do
        fw_iter_interval=${fw_iter_intervals[$i]}
        df_iter_interval=${df_iter_intervals[$i]}

        output_dir="/is/cluster/fast/pyu/refnerf_results/${model}/iter_${fw_iter_interval}_${df_iter_interval}"

        /is/cluster/pyu/miniforge3/envs/2dgs/bin/python pbr_train.py \
            -s "/is/cluster/fast/pyu/data/refnerf/${model}/" \
            -m "${output_dir}" \
            -w \
            --eval \
            --warmup_iterations 1 \
            --lambda_dist 100 \
            --lambda_normal 0.01 \
            --fw_iter "${fw_iter_interval}" \
            --df_iter "${df_iter_interval}" \
            --mode "iterative" \
            --gamma \
            --tone

        checkpoint="${output_dir}/chkpnt45000.pth"

        /is/cluster/pyu/miniforge3/envs/2dgs/bin/python pbr_render.py \
            -s "/is/cluster/fast/pyu/data/refnerf/${model}/" \
            -m "${output_dir}" \
            -w \
            --eval \
            --checkpoint "${checkpoint}" \
            --mode "iterative" \
            --gamma \
            --tone
    done

done

#fw
for model in "${models[@]}"; do
    output_dir="/is/cluster/fast/pyu/refnerf_results/${model}/fw"

    /is/cluster/pyu/miniforge3/envs/2dgs/bin/python pbr_train.py \
        -s "/is/cluster/fast/pyu/data/refnerf/${model}/" \
        -m "${output_dir}" \
        -w \
        --eval \
        --warmup_iterations 1 \
        --lambda_dist 100 \
        --lambda_normal 0.01 \
        --mode "fw" \
        --gamma \
        --tone

    checkpoint="${output_dir}/chkpnt45000.pth"

    /is/cluster/pyu/miniforge3/envs/2dgs/bin/python pbr_render.py \
        -s "/is/cluster/fast/pyu/data/refnerf/${model}/" \
        -m "${output_dir}" \
        -w \
        --eval \
        --checkpoint "${checkpoint}" \
        --mode "fw" \
        --gamma \
        --tone
done

#df
for model in "${models[@]}"; do
    output_dir="/is/cluster/fast/pyu/refnerf_results/${model}/df"

    /is/cluster/pyu/miniforge3/envs/2dgs/bin/python pbr_train.py \
        -s "/is/cluster/fast/pyu/data/refnerf/${model}/" \
        -m "${output_dir}" \
        -w \
        --eval \
        --warmup_iterations 1 \
        --lambda_dist 100 \
        --lambda_normal 0.01 \
        --mode "df" \
        --gamma \
        --tone

    checkpoint="${output_dir}/chkpnt45000.pth"

    /is/cluster/pyu/miniforge3/envs/2dgs/bin/python pbr_render.py \
        -s "/is/cluster/fast/pyu/data/refnerf/${model}/" \
        -m "${output_dir}" \
        -w \
        --eval \
        --checkpoint "${checkpoint}" \
        --mode "df" \
        --gamma \
        --tone
done

# exit

# Loading extension module renderutils_plugin...
# Lock file exists in build directory: '/lustre/home/pyu/.cache/torch_extensions/py39_cu118/nvdiffrast_plugin/lock'
