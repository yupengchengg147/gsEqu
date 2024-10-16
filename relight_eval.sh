export LD_LIBRARY_PATH=/is/software/nvidia/cuda-11.8/lib64
export PATH=$PATH:/is/software/nvidia/cuda-11.8/bin
export CUDA_HOME=/is/software/nvidia/cuda-11.8
conda activate 2dgs

cd /home/pyu/local_code/gsEqu

methods=("df" "fw" "st_fwrate_0.1" "st_fwrate_0.5" "st_fwrate_0.9" "iter_1_1" "iter_1_5" "iter_5_1" "mixxed")
# methods=("mixxed")
# objects=("armadillo" "ficus" "lego" "hotdog")
objects=("cat" "bell" "horse")

for obj in ${objects[@]}; do
    for method in ${methods[@]}; do
        python relight_eval.py --output_dir /is/cluster/fast/pyu/results_glossy/${obj}/${method} --gt_dir /is/cluster/fast/pyu/data/nero/relight_gt --obj ${obj} --result_file ./NeRO_relight.json
    done
done