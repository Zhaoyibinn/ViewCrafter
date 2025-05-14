#!/bin/bash

# 定义扫描编号数组
# scans=(24 37 40 55 63 65 69 83 97 105 106 110 114 118 122)
scans=(110)
# 定义基础路径和通用参数
base_dir="/root/ViewCrafter"
conda_env="/root/miniconda3/envs/viewcrafter/bin/python"
ckpt_path="./checkpoints/model_sparse.ckpt"
config="./configs/inference_pvd_1024.yaml"
model_path="./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
out_dir="./output/DTU_3_2"
mode="sparse_view_interp"
bg_trd="0.2"
seed="123"
ddim_steps="50"
video_length="25"
device="cuda:0"
height="576"
width="1024"

# 进入基础目录
cd $base_dir

# 循环执行推理脚本
for scan in "${scans[@]}"
do
    image_dir="test/DTU_3_2/scan$scan/images"
    dtu_path="test/DTU_3_2/scan$scan/sparse/0"
    exp_name="scan$scan"

    $conda_env inference.py --image_dir $image_dir --dtu_path $dtu_path --exp_name $exp_name --out_dir $out_dir --mode $mode --bg_trd $bg_trd --seed $seed --ckpt_path $ckpt_path --config $config --ddim_steps $ddim_steps --video_length $video_length --device $device --height $height --width $width --model_path $model_path
done