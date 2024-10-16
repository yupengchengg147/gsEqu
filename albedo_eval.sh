export LD_LIBRARY_PATH=/is/software/nvidia/cuda-11.8/lib64
export PATH=$PATH:/is/software/nvidia/cuda-11.8/bin
export CUDA_HOME=/is/software/nvidia/cuda-11.8
conda activate 2dgs

cd /home/pyu/local_code/gsEqu

# methods=("df" "fw" "st_fwrate_0.1" "st_fwrate_0.5" "st_fwrate_0.9" "iter_1_1" "iter_1_5" "iter_5_1")
methods=("mixxed")
objects=("armadillo" "ficus" "lego" "hotdog")

for obj in ${objects[@]}; do
    for method in ${methods[@]}; do
        python albedo_eval.py --output_dir /is/cluster/fast/pyu/results_mixxed/${obj}/${method} --gt_dir /is/cluster/fast/pyu/data/tensorir/${obj} --result_file ./tensorIR_albedo.json
    done
done
# python albedo_eval.py --output_dir /is/cluster/fast/pyu/results_tensorir/armadillo/df --gt_dir /is/cluster/fast/pyu/data/tensorir/armadillo --result_file ./test_al.json
# /is/cluster/fast/pyu/results_mixxed/armadillo/mixxed/test/ours_45000/brdf