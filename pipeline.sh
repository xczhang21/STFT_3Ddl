#!/bin/bash

source /home/zhang/anaconda3/etc/profile.d/conda.sh
conda activate zxc_das

# 配置参数列表
configs=(
    "mfunet_ss64_train pi"
    "mfunet_ss128_train pi"
    "mfunet_ss256_train pi"
    "msunet_ssms_train phase"
    "msunet_ssms_train intensity"
    "unet_gadf_ss64_train phase"
    "unet_gadf_ss128_train phase"
    "unet_gadf_ss256_train phase"
    "unet_gadf_ss64_train intensity"
    "unet_gadf_ss128_train intensity"
    "unet_gadf_ss256_train intensity"
    "unet_gasf_ss64_train phase"
    "unet_gasf_ss128_train phase"
    "unet_gasf_ss256_train phase"
    "unet_gasf_ss64_train intensity"
    "unet_gasf_ss128_train intensity"
    "unet_gasf_ss256_train intensity"
    "unet_ss64_train pi"
    "unet_ss128_train pi"
    "unet_ss256_train pi"
    "unet_ss64_train intensity"
    "unet_ss128_train intensity"
    "unet_ss256_train intensity"
    "unet_ss64_train phase"
    "unet_ss128_train phase"
    "unet_ss256_train phase"
    "ss64_train pi"
    "ss128_train pi"
    "ss256_train pi"
    "ss64_train intensity"
    "ss128_train intensity"
    "ss256_train intensity"
    "ss64_train phase"
    "ss128_train phase"
    "ss256_train phase"
)

# 外层循环，跑 10 遍
for i in {1..2}; do
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
        python ./stft_3ddl/train.py --train_config "${train_config}" --prepro_method "${prepro_method}"
        
        # 模拟任务完成
        echo "Task $task_name completed."
        echo "------------------------"
    done

    echo "-------------------- End Run $i --------------------"
done

echo "All tasks completed."
