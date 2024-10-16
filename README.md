BRANCH DESCRIPTION:
for now donot use normal_visible, main, and master

branch stochastic: 2dgs representation + diff modes:
stochastic with fw_rate as parameter
iterative as fw_iter, df_iter as parameter
also df and fw

branch 3dgs_n: 3dgs + shortest axis as normal plus delta_n as residual
with diff modes:
stochastic with fw_rate as parameter
iterative as fw_iter, df_iter as parameter
also df and fw

branch mixxed shading: based on 3dgs_n with additional mode mixxed,
where df shading for diffuse_rgb and fw shading for specular_rgb.

branch mixxed refined:
3dgs_n + additional mode.
fw without mask
df with warm-up
for mixxed: also with warmup, and without mask

git checkout mixxed_refined
python pbr_train.py -s ... -m ... 
