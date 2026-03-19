#!/bin/bash

source /home/zhang/anaconda3/etc/profile.d/conda.sh
conda activate zxc_das

configs=(
    "/home/zhang/zxc/STFT_3DDL/DATASETS/raw_data/DAS1K_cutout /home/zhang/zxc/STFT_3DDL/DATASETS/preprocessed_data/DAS1K_cutout"
    "/home/zhang/zxc/STFT_3DDL/DATASETS/raw_data/DAS1K_noise /home/zhang/zxc/STFT_3DDL/DATASETS/preprocessed_data/DAS1K_noise"
    "/home/zhang/zxc/STFT_3DDL/DATASETS/raw_data/DAS1K_scale /home/zhang/zxc/STFT_3DDL/DATASETS/preprocessed_data/DAS1K_scale"
    "/home/zhang/zxc/STFT_3DDL/DATASETS/raw_data/DAS1K_shift /home/zhang/zxc/STFT_3DDL/DATASETS/preprocessed_data/DAS1K_shift"
    "/home/zhang/zxc/STFT_3DDL/DATASETS/raw_data/DAS1K_SNSCS1 /home/zhang/zxc/STFT_3DDL/DATASETS/preprocessed_data/DAS1K_SNSCS1"
    "/home/zhang/zxc/STFT_3DDL/DATASETS/raw_data/DAS1K_SNSCS3 /home/zhang/zxc/STFT_3DDL/DATASETS/preprocessed_data/DAS1K_SNSCS3"
    "/home/zhang/zxc/STFT_3DDL/DATASETS/raw_data/DAS1K_stretch /home/zhang/zxc/STFT_3DDL/DATASETS/preprocessed_data/DAS1K_stretch"
)

for config in "${configs[@]}"; do
    IFS=' ' read -r dataset_path save_path <<< "$config"

    echo "Command Arguments: --dataset_path ${dataset_path} --save_path ${save_path}"
    
    python ../prepreprocessing/1d_data_augmentation/das_data_augmentation.py \
        --dataset_path "${dataset_path}" \
        --save_path "${save_path}" &

    echo "Started process for ${dataset_path}"
    echo "-------------------------"
done

# 等待所有后台进程完成
wait

echo "All tasks completed."
