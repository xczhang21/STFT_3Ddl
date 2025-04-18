# 该文件将tensorboardX存储在log中实验结果保存到csv中，用于在origin中绘制箱线图等
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from multiprocessing import Pool



def get_scalar_value(log_dir: str, scalar_name: str, step: int) -> Optional[float]:
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    assert scalar_name in ea.Tags().get("scalars", []), f"[Error] Scalar '{scalar_name}' not found in log."

    events = ea.Scalars(scalar_name)
    for e in events:
        if e.step == step:
            return e.value

def get_log_dir_paths(base_dir):
    base_path = Path(base_dir)
    log_paths = []

    assert base_path.exists(), f"[Error] base_path '{str(base_path)}' not exists."

    # 遍历 base_dir 下的所有模型目录
    for model_dir in sorted(base_path.iterdir()):
        if model_dir.is_dir():
            # 遍历每个模型目录下的 01、02、03 等子目录
            for run_dir in sorted(model_dir.iterdir()):
                if run_dir.is_dir():
                    log_dir = run_dir / "log"
                    if log_dir.is_dir():
                        log_paths.append(log_dir)  # 或用 log_dir.resolve() 获取绝对路径

    return log_paths

def process_scalar(scalar_name):
    df = pd.DataFrame()
    for train_dir in TRAIN_DIRS:
        train_dir = Path(train_dir)
        key = train_dir.parts[-1]
        base_dir = Path(ROOT_PATH) / train_dir
        log_paths = get_log_dir_paths(base_dir)

        # 对log_paths进行排序，按照01,02,03,...
        log_paths = sorted(log_paths, key=lambda p: int(p.parent.name))

        values = []
        for log_path in log_paths:
            value = get_scalar_value(str(log_path), scalar_name, STEP)
            values.append(value)
        

        # 对齐行数
        num_rows = len(df)
        num_values = len(values)
        max_len = max(num_rows, num_values)
        # 扩展df行数（自动补NaN）
        if num_rows < max_len:
            df = df.reindex(range(max_len))
        # 补全values
        if num_values < max_len:
            values.extend([np.nan] * (max_len - num_values))

        df[key] = values
    df.to_csv(f"{save_path}/{scalar_name.split('/')[0]}.csv", index=False)



if __name__ == '__main__':
    # 下面内容是通过find . -maxdepth 2 -type d得到的
    # 经过手动排序
    # 缺少实验：STFT_3Ddl_das1k_resize/phase_unet_resize...
    TRAIN_DIRS = [
        "./STFT_3Ddl_das1k/phase_unet_ss64_train",
        "./STFT_3Ddl_das1k/phase_unet_ss128_train",
        "./STFT_3Ddl_das1k/phase_unet_ss256_train",
        "./STFT_3Ddl_das1k/intensity_unet_ss64_train",
        "./STFT_3Ddl_das1k/intensity_unet_ss128_train",
        "./STFT_3Ddl_das1k/intensity_unet_ss256_train",
        "./STFT_3Ddl_das1k/pi_unet_ss64_train",
        "./STFT_3Ddl_das1k/pi_unet_ss128_train",
        "./STFT_3Ddl_das1k/pi_unet_ss256_train",
        "./STFT_3Ddl_das1k/phase_resnet_ss64_train",
        "./STFT_3Ddl_das1k/phase_resnet_ss128_train",
        "./STFT_3Ddl_das1k/phase_resnet_ss256_train",
        "./STFT_3Ddl_das1k/intensity_resnet_ss64_train",
        "./STFT_3Ddl_das1k/intensity_resnet_ss128_train",
        "./STFT_3Ddl_das1k/intensity_resnet_ss256_train",
        "./STFT_3Ddl_das1k/pi_resnet_ss64_train",
        "./STFT_3Ddl_das1k/pi_resnet_ss128_train",
        "./STFT_3Ddl_das1k/pi_resnet_ss256_train",
        "./STFT_3Ddl_das1k_padding0/phase_unet_padding0_ss64_train",
        "./STFT_3Ddl_das1k_padding0/phase_unet_padding0_ss128_train",
        "./STFT_3Ddl_das1k_padding0/phase_unet_padding0_ss256_train",
        "./STFT_3Ddl_das1k_padding0/intensity_unet_padding0_ss64_train",
        "./STFT_3Ddl_das1k_padding0/intensity_unet_padding0_ss128_train",
        "./STFT_3Ddl_das1k_padding0/intensity_unet_padding0_ss256_train",
        "./STFT_3Ddl_das1k_resize/phase_unet_resize_ss64_train",
        "./STFT_3Ddl_das1k_resize/phase_unet_resize_ss128_train",
        "./STFT_3Ddl_das1k_resize/phase_unet_resize_ss256_train",
        "./STFT_3Ddl_das1k_resize/intensity_unet_resize_ss64_train",
        "./STFT_3Ddl_das1k_resize/intensity_unet_resize_ss128_train",
        "./STFT_3Ddl_das1k_resize/intensity_unet_resize_ss256_train",
        "./STFT_3Ddl_das1k_gadf/phase_unet_gadf_ss64_train",
        "./STFT_3Ddl_das1k_gadf/phase_unet_gadf_ss128_train",
        "./STFT_3Ddl_das1k_gadf/phase_unet_gadf_ss256_train",
        "./STFT_3Ddl_das1k_gadf/intensity_unet_gadf_ss64_train",
        "./STFT_3Ddl_das1k_gadf/intensity_unet_gadf_ss128_train",
        "./STFT_3Ddl_das1k_gadf/intensity_unet_gadf_ss256_train",
        "./STFT_3Ddl_das1k_gasf/phase_unet_gasf_ss64_train",
        "./STFT_3Ddl_das1k_gasf/phase_unet_gasf_ss128_train",
        "./STFT_3Ddl_das1k_gasf/phase_unet_gasf_ss256_train",
        "./STFT_3Ddl_das1k_gasf/intensity_unet_gasf_ss64_train",
        "./STFT_3Ddl_das1k_gasf/intensity_unet_gasf_ss128_train",
        "./STFT_3Ddl_das1k_gasf/intensity_unet_gasf_ss256_train",
        "./STFT_3Ddl_mfdas1k/pi_mfunet_ss64_train",
        "./STFT_3Ddl_mfdas1k/pi_mfunet_ss128_train",
        "./STFT_3Ddl_mfdas1k/pi_mfunet_ss256_train",
        "./STFT_3Ddl_msdas1k/phase_msunet_ssms_train",
        "./STFT_3Ddl_msdas1k/intensity_msunet_ssms_train",
    ]
    SCALAR_NAMES = [
        '1-Loss/val',
        '2-Accuracy/val',
        '3-Precision/val',
        '4-Recall/val',
        '5-F1_score/val',
        '6-mAP/val'
    ]
    STEP = 149

    ROOT_PATH = "/home/zhang/zxc/STFT_3DDL/model"
    save_path = "/home/zhang/zxc/STFT_3DDL/results"
    Path(save_path).mkdir(parents=True, exist_ok=True)

    with Pool(processes=32) as pool:
        list(tqdm(pool.imap_unordered(process_scalar, SCALAR_NAMES), total=len(SCALAR_NAMES), desc="Processing"))

    # with ThreadPoolExecutor() as executor:
    #     futures = [executor.submit(process_scalar, scalar) for scalar in SCALAR_NAMES]
    #     for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
    #         pass

# event_path = "/home/zhang/zxc/STFT_3DDL/model/STFT_3Ddl_das1k/phase_unet_ss64_train/UNet_epo150_bs32_lr0.01_ssize64/01/log"

# value = get_scalar_value(event_path, SCALAR_NAMES[0], STEP)

# base_dir = Path("/home/zhang/zxc/STFT_3DDL/model") / "./STFT_3Ddl_das1k/phase_unet_ss64_train"

# log_paths = get_log_dir_paths(base_dir)