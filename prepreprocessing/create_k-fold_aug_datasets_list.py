"""
    为保证测试集的纯洁，且生成用于调取数据增强后的数据，在create_k-fold_datasets_list.py生成的datasets_list的基础上，只将训练集的数据进行重新映射
    如/home/zhang/zxc/STFT_3DDL/STFT_3Ddl/stft_3ddl/lists/DAS1K_K-Fold/10-Fold/1/train.txt
    将该train.txt中的
        "welding28 9"
    改成
        “welding28 9"
        “welding28_aug1 9"
        “welding28_aug2 9"
        “welding28_aug3 9"
    并保存到
    /home/zhang/zxc/STFT_3DDL/STFT_3Ddl/stft_3ddl/lists/DAS1K_K-Fold_augmentations/10-Fold/DAS1K_SNSCS3
"""

import os
from pathlib import Path
import shutil

def augment_line(line: str) -> list:
    """
    将一行 '<name> <label>' 变成：
    ['name label', 'name_aug1 label', 'name_aug2 label', 'name_aug3 label']
    保留 label，忽略空行和仅空白行。
    """
    s = line.strip()
    if not s:
        return []  # 跳过空行
    # 兼容任意空白分隔（空格、Tab 等），只切两段：名字 与 其余（标签）
    parts = s.split()
    if len(parts) < 2:
        # 如果行格式异常，就原样返回（也可选择 raise）
        return [s]
    name, label = parts[0], parts[-1]
    out = [f"{name} {label}"]
    for k in (1, 2, 3):
        out.append(f"{name}_aug{k} {label}")
    return out

def main(
    src_root="/home/zhang/zxc/STFT_3DDL/STFT_3Ddl/stft_3ddl/lists/DAS1K_K-Fold/10-Fold",
    dst_root="/home/zhang/zxc/STFT_3DDL/STFT_3Ddl/stft_3ddl/lists/DAS1K_K-Fold_augmentations/10-Fold/DAS1K_SNSCS3",
):
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    # 找到源里所有数字命名的折目录（1..10），并排序
    fold_dirs = sorted([p for p in src_root.iterdir() if p.is_dir()],
                       key=lambda p: (len(p.name), p.name))

    for fold_dir in fold_dirs:
        fold_name = fold_dir.name  # 例如 "1", "2", ..., "10"
        src_train = fold_dir / "train.txt"
        src_test  = fold_dir / "test.txt"

        # 目标路径
        dst_fold  = dst_root / fold_name
        dst_fold.mkdir(parents=True, exist_ok=True)
        dst_train = dst_fold / "train.txt"
        dst_test  = dst_fold / "test.txt"

        # 复制 test.txt（不存在则跳过或报错）
        if src_test.exists():
            shutil.copy2(src_test, dst_test)
        else:
            print(f"[WARN] 缺少 test.txt: {src_test}")

        # 读取并增强 train.txt
        if src_train.exists():
            lines_out = []
            with src_train.open("r", encoding="utf-8") as f:
                for line in f:
                    lines_out.extend(augment_line(line))
            # 写回新的 train.txt
            with dst_train.open("w", encoding="utf-8", newline="\n") as f:
                for i, l in enumerate(lines_out):
                    f.write(l)
                    if i != len(lines_out) - 1:
                        f.write("\n")
        else:
            print(f"[WARN] 缺少 train.txt: {src_train}")

    print("完成：目录构造、test 复制与 train 增强。")
    
if __name__ == "__main__":
    # 如需自定义路径，可改 main() 参数：
    # main(src_root="/path/to/DAS1K_K-Fold/10-Fold",
    #      dst_root="/path/to/DAS1K_K-Fold_augmentations/10-Fold/DAS1K_SNSCS3")
    main()
