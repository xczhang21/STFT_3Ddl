import re
import os
import argparse
import numpy as np
from scipy.stats import ttest_rel


def parse_kfold_txt(file_path):
    results = {}
    current_model = None

    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        if line.startswith("MODEL:"):
            current_model = line.split("MODEL:")[1].strip()
            results[current_model] = {
                'accuracy': [],
                'f1': [],
                'precision': [],
                'recall': [],
                'map': []
            }

        if "accuracy: mean=" in line:
            val = float(re.search(r"mean=([0-9.]+)", line).group(1))
            results[current_model]['accuracy'].append(val)

        if "f1: mean=" in line:
            val = float(re.search(r"mean=([0-9.]+)", line).group(1))
            results[current_model]['f1'].append(val)

        if "precision: mean=" in line:
            val = float(re.search(r"mean=([0-9.]+)", line).group(1))
            results[current_model]['precision'].append(val)

        if "recall: mean=" in line:
            val = float(re.search(r"mean=([0-9.]+)", line).group(1))
            results[current_model]['recall'].append(val)

        if "map: mean=" in line:
            val = float(re.search(r"mean=([0-9.]+)", line).group(1))
            results[current_model]['map'].append(val)

    return results


def paired_t_test(model_a, model_b, metric_name):
    a = np.array(model_a[metric_name])
    b = np.array(model_b[metric_name])
    t_stat, p_value = ttest_rel(a, b)
    return t_stat, p_value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_txt", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()

    data = parse_kfold_txt(args.input_txt)

    ours_name = "pi_mfofunet_learnableadd_aug_ss256_kfold_train"
    ours = data[ours_name]

    baselines = [
        "pi_resnet_aug_ss256_kfold_train",
        "pi_unet_aug_ss256_kfold_train",
        "pi_ipcnn_aug_ss256_kfold_train",
        "pi_convnext_aug_ss256_kfold_train"
    ]

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "paired_ttest_results.txt")

    with open(save_path, "w", encoding="utf-8") as f:

        header = f"Ours model: {ours_name}\n\n"
        print(header)
        f.write(header)

        for base in baselines:
            title = f"--- Ours vs {base} ---\n"
            print(title)
            f.write(title)

            for metric in ["accuracy", "f1", "map"]:
                t, p = paired_t_test(ours, data[base], metric)
                sig = "YES" if p < 0.05 else "NO"

                line = f"{metric}: p={p:.6f}  significant={sig}\n"
                print(line)
                f.write(line)

            sep = "-" * 50 + "\n"
            print(sep)
            f.write(sep)

    print(f"\n Results saved to: {save_path}")


if __name__ == "__main__":
    main()