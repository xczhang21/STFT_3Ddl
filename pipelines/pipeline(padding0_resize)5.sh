#!/bin/bash

source /home/zhang/anaconda3/etc/profile.d/conda.sh
conda activate zxc_das

# 配置参数列表
configs=(
    "unet_padding0_ss64_train phase"
    "unet_padding0_ss128_train phase"
    "unet_padding0_ss256_train phase"
    "unet_padding0_ss64_train intensity"
    "unet_padding0_ss128_train intensity"
    "unet_padding0_ss256_train intensity"
    "unet_resize_ss64_train phase"
    "unet_resize_ss128_train phase"
    "unet_resize_ss256_train phase"
    "unet_resize_ss64_train intensity"
    "unet_resize_ss128_train intensity"
    "unet_resize_ss256_train intensity"
)

# 外层循环，跑 5 遍
for i in {1..5}; do
    echo "-------------------- Start Run $i --------------------"
    
    # 遍历所有配置组合
    for config in "${configs[@]}"; do
        # 将配置拆分成两个部分
        IFS=' ' read -r train_config prepro_method <<< "$config"
        
        # 构建任务名称
        task_name="${train_config}_${prepro_method}_train"
        
        # 清理任务名称中的多余下划线
        task_name=$(echo "$task_name" | sed 's/^_//;s/_$//;s/__/_/g')
        
        # 打印任务信息
        echo "Running Task: $task_name"
        echo "Command Arguments: --train_config ${train_config} --prepro_method ${prepro_method}"
        
        # 示例执行命令，替换为实际任务（如调用 Python 脚本）
        # python ./stft_3ddl/train.py --train_config "${train_config}" --prepro_method "${prepro_method}"
        # 指定第1张显卡
        CUDA_VISIBLE_DEVICES=1 python ./stft_3ddl/train.py --train_config "${train_config}" --prepro_method "${prepro_method}"

        
        # 模拟任务完成
        echo "Task $task_name completed."
        echo "------------------------"
    done

    echo "-------------------- End Run $i --------------------"
done

echo "All tasks completed."
