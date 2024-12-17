#!/bin/bash

source /home/zhang/anaconda3/etc/profile.d/conda.sh
conda activate zxc_das

ssizes=("64" "128" "256")
nets=("" "unet")
prepros=("" "intensity" "pi")

# 三层循环
for ssize in "${ssizes[@]}"; do
    for net in "${nets[@]}"; do
        for prepro in "${prepros[@]}"; do

            # 构建任务名称
            task_name="${net}_${prepro}_ss${ssize}_train"

            # 清理多余的下划线
            task_name=$(echo "$task_name" | sed 's/^_//;s/_$//;s/__/_/g')

            # 构建命令参数
            args="--train_config ss${ssize}_train"
            if [ -n "$net" ]; then
                args="--train_config ${net}_ss${ssize}_train"
            fi
            if [ -n "$prepro" ]; then
                args="${args} --prepro_method ${prepro}"
            fi

            # 打印任务信息
            echo "Running Task: $task_name"
            echo "Command Arguments: $args"

            # 示例执行命令，替换为实际任务（如调用 Python 脚本）
            python ./stft_3ddl/train.py $args

            # 模拟任务完成
            echo "Task $task_name completed."
            echo "------------------------"

        done
    done
done

echo "All tasks completed."