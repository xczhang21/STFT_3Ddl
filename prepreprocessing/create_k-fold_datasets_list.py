"""
功能：
- 读取 DAS1K 原始数据集
- 执行 K-fold 分层划分（每个类别均匀分布到 K 个 fold）
- 保存路径格式为：
    DAS1K_K-Fold/10-Fold/1/train.txt
    DAS1K_K-Fold/10-Fold/1/test.txt
    DAS1K_K-Fold/10-Fold/2/train.txt
    DAS1K_K-Fold/10-Fold/2/test.txt
    ...

每行格式：
    <sample_id> <class_id>
"""

import os
import random
from pathlib import Path
from tqdm import tqdm


# ===================== 配置 =====================

dataset_path = Path('/home/zhang/zxc/STFT_3DDL/DATASETS/raw_data/DAS1K/')

K = 10   # K-fold

output_root = Path("/home/zhang/zxc/STFT_3DDL/STFT_3Ddl/stft_3ddl/lists/DAS1K_K-Fold/10-Fold/")

class_id = {
    'CARHORN':0,
    'DRILLING':1,
    'FOOTSTEPS':2,
    'HANDHAMMER':3,
    'HANDSAW':4,
    'JACKHAMMER':5,
    'RAIN':6,
    'SHOVELING':7,
    'THUNDERSTORM':8,
    'WELDING':9
}

RANDOM_SEED = 2025

# =================================================


def get_class_names(directory_path: Path):
    return [p.name for p in directory_path.iterdir() if p.is_dir()]


def get_iterfile_names(directory_path: Path):
    return [p.stem for p in directory_path.iterdir() if p.is_file() and p.suffix == '.mat']


def build_class_dict(dataset_path: Path):
    """
    扫描数据集：得到
        class_dict = {
            'CARHORN': ['carhorn1', 'carhorn2', ...],
            ...
        }
    """
    class_dict = {}
    for cname in get_class_names(dataset_path):
        ids = get_iterfile_names(dataset_path / cname)
        class_dict[cname] = ids
    return class_dict


def assign_folds(class_dict, k: int, seed: int):
    """
    每个类别打乱 → 轮转分配到 K 个 fold
    """
    random.seed(seed)
    fold_assignments = {}

    for cname, ids in class_dict.items():
        ids = ids.copy()
        random.shuffle(ids)
        fold_assignments[cname] = [(sample_id, i % k) for i, sample_id in enumerate(ids)]

    return fold_assignments


def write_fold_lists(fold_assignments, k: int, output_root: Path, class_id_map: dict):
    """
    保存每个 fold 的 train/test.txt
    路径格式：
        10-Fold/1/train.txt
        10-Fold/1/test.txt
        ...
    """

    for fold in range(k):
        # 为该 fold 创建子目录
        fold_dir = output_root / str(fold + 1)
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_path = fold_dir / "train.txt"
        test_path = fold_dir / "test.txt"

        train_lines = []
        test_lines = []

        # 遍历所有类别
        for cname, samples in fold_assignments.items():
            label = class_id_map[cname]
            for sample_id, fold_idx in samples:
                line = f"{sample_id} {label}\n"
                if fold_idx == fold:
                    test_lines.append(line)
                else:
                    train_lines.append(line)

        # 写入文件
        with open(train_path, "w") as f:
            f.writelines(train_lines)

        with open(test_path, "w") as f:
            f.writelines(test_lines)

        print(f"[Fold {fold+1}] train={len(train_lines)}  test={len(test_lines)}")


def main():
    print("=== 构建 class -> IDs 映射 ===")
    class_dict = build_class_dict(dataset_path)

    for cname, ids in class_dict.items():
        print(f"{cname}: {len(ids)} samples")

    print("\n=== 分配 K-Fold ===")
    fold_assignments = assign_folds(class_dict, K, RANDOM_SEED)

    print("\n=== 写入 train/test 文件 ===")
    write_fold_lists(fold_assignments, K, output_root, class_id)

    print("\n完成！列表文件已保存到：")
    print(str(output_root))


if __name__ == "__main__":
    main()
