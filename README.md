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
