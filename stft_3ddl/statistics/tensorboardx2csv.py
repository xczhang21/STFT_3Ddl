import argparse
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
from pathlib import Path
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm


def get_scalar_value(log_dir: str, scalar_name: str, step: int):
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    assert scalar_name in ea.Tags().get("scalars", []), f"[Error] Scalar '{scalar_name}' not found in log: {log_dir}"

    events = ea.Scalars(scalar_name)
    for e in events:
        if e.step == step:
            return e.value
    return None


def get_log_dir_paths(base_dir: Path):
    log_paths = []

    assert base_dir.exists(), f"[Error] Base directory '{str(base_dir)}' does not exist."

    for model_dir in sorted(base_dir.iterdir()):
        if model_dir.is_dir():
            for run_dir in sorted(model_dir.iterdir()):
                if run_dir.is_dir():
                    log_dir = run_dir / "log"
                    if log_dir.is_dir():
                        log_paths.append(log_dir)

    return log_paths


def process_scalar(scalar_name: str):
    df = pd.DataFrame()

    key = input_dir.name
    base_dir = input_dir
    log_paths = get_log_dir_paths(base_dir)
    log_paths = sorted(log_paths, key=lambda p: int(p.parent.name))  # 01, 02, ...

    values = []
    for log_path in log_paths:
        value = get_scalar_value(str(log_path), scalar_name, STEP)
        values.append(value)

    df[key] = values
    df.to_csv(save_path / f"{scalar_name.split('/')[0]}.csv", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process TensorBoard logs in one experiment folder.")
    parser.add_argument("train_dir", type=str, help="The training directory path.")
    args = parser.parse_args()

    input_dir = Path(args.train_dir).resolve()
    assert input_dir.exists(), f"[Error] The provided path does not exist: {input_dir}"

    save_path = input_dir / "results"
    save_path.mkdir(parents=True, exist_ok=True)

    SCALAR_NAMES = [
        '1-Loss/val',
        '2-Accuracy/val',
        '3-Precision/val',
        '4-Recall/val',
        '5-F1_score/val',
        '6-mAP/val'
    ]
    STEP = 149

    with Pool(processes=6) as pool:
        list(tqdm(pool.imap_unordered(process_scalar, SCALAR_NAMES), total=len(SCALAR_NAMES), desc="Processing"))
