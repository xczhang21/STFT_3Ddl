import os
import re
import argparse
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def find_event_files(root_dir):
    event_files = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.startswith("events.out.tfevents"):
                event_files.append(os.path.join(root, f))
    return sorted(event_files)


def get_last_scalar(event_file, tag):
    ea = EventAccumulator(event_file)
    ea.Reload()
    try:
        vals = ea.Scalars(tag)
        if len(vals) > 0:
            return vals[-1].value
    except Exception:
        return None
    return None


def parse_fold_run(event_path):

    parts = event_path.split(os.sep)

    run_id = None
    exp_name = None
    fold_id = None

    for i, p in enumerate(parts):
        if re.fullmatch(r"\d{2}", p):
            run_id = p
            if i - 1 >= 0:
                exp_name = parts[i - 1]
            break

    if exp_name is not None:
        m = re.search(r"SNSCS3-(\d+)_10", exp_name)
        if m:
            fold_id = m.group(1)

    return fold_id, run_id


def summarize_one_model(grouped, metrics):

    fold_summary = {}
    overall_summary = {}

    for fold_id in sorted(grouped.keys(), key=lambda x: int(x)):

        fold_summary[fold_id] = {}

        for metric in metrics:

            vals = []

            for run_id in sorted(grouped[fold_id].keys()):

                if metric in grouped[fold_id][run_id]:
                    vals.append(grouped[fold_id][run_id][metric])

            if len(vals) > 0:

                vals = np.array(vals)

                fold_summary[fold_id][metric] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    "runs": vals.tolist(),
                }

    for metric in metrics:

        fold_means = []

        for fold_id in fold_summary:

            if metric in fold_summary[fold_id]:
                fold_means.append(fold_summary[fold_id][metric]["mean"])

        arr = np.array(fold_means)

        overall_summary[metric] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)),
            "fold_means": arr.tolist(),
        }

    return fold_summary, overall_summary


def read_one_model(root_dir):

    tag_map = {
        "loss": "1-Loss/val",
        "accuracy": "2-Accuracy/val",
        "precision": "3-Precision/val",
        "recall": "4-Recall/val",
        "f1": "5-F1_score/val",
        "map": "6-mAP/val",
    }

    event_files = find_event_files(root_dir)

    grouped = defaultdict(lambda: defaultdict(dict))

    for ef in tqdm(event_files, desc="Reading TensorBoard"):

        fold_id, run_id = parse_fold_run(ef)

        if fold_id is None:
            continue

        for metric, tag in tag_map.items():

            value = get_last_scalar(ef, tag)

            if value is not None:
                grouped[fold_id][run_id][metric] = value

    metrics = list(tag_map.keys())

    return summarize_one_model(grouped, metrics)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_root",
        type=str,
        required=True,
        help="例如 /home/zhang/zxc/STFT_3DDL/model",
    )

    args = parser.parse_args()

    model_dirs = [
        "STFT_3Ddl_das1k_aug/pi_resnet_aug_ss256_kfold_train",
        "STFT_3Ddl_das1k_aug/pi_convnext_aug_ss256_kfold_train",
        "STFT_3Ddl_das1k_aug/pi_unet_aug_ss256_kfold_train",
        "STFT_3Ddl_mfdas1k_aug/pi_ipcnn_aug_ss256_kfold_train",
        "STFT_3Ddl_mfdas1k_aug/pi_mfofunet_learnableadd_aug_ss256_kfold_train",
    ]

    save_dir = os.path.join(args.model_root, "kfold_statistics")
    os.makedirs(save_dir, exist_ok=True)

    txt_file = open(os.path.join(save_dir, "kfold_results.txt"), "w")
    table_file = open(os.path.join(save_dir, "kfold_table.txt"), "w")

    all_results = {}

    for rel_dir in tqdm(model_dirs, desc="Models"):

        abs_dir = os.path.join(args.model_root, rel_dir)

        if not os.path.exists(abs_dir):
            continue

        print("\nProcessing:", abs_dir)

        fold_summary, overall_summary = read_one_model(abs_dir)

        model_name = os.path.basename(abs_dir)

        all_results[model_name] = overall_summary

        txt_file.write("\n" + "=" * 80 + "\n")
        txt_file.write(f"MODEL: {model_name}\n")
        txt_file.write("=" * 80 + "\n")

        for fold_id in fold_summary:

            txt_file.write(f"\nFold {fold_id}\n")

            for metric, info in fold_summary[fold_id].items():

                txt_file.write(
                    f"{metric}: mean={info['mean']:.6f} std={info['std']:.6f}\n"
                )

        txt_file.write("\nOverall:\n")

        for metric, info in overall_summary.items():

            txt_file.write(
                f"{metric}: {info['mean']:.4f} ± {info['std']:.4f}\n"
            )

    txt_file.close()

    table_file.write("Model\tAccuracy\tPrecision\tRecall\tF1\tmAP\n")

    for model_name, res in all_results.items():

        table_file.write(
            f"{model_name}\t"
            f"{res['accuracy']['mean']:.4f}±{res['accuracy']['std']:.4f}\t"
            f"{res['precision']['mean']:.4f}±{res['precision']['std']:.4f}\t"
            f"{res['recall']['mean']:.4f}±{res['recall']['std']:.4f}\t"
            f"{res['f1']['mean']:.4f}±{res['f1']['std']:.4f}\t"
            f"{res['map']['mean']:.4f}±{res['map']['std']:.4f}\n"
        )

    table_file.close()

    print("\nResults saved to:", save_dir)


if __name__ == "__main__":
    main()