"""
将
phase(32, 32),phase(128, 128),phase(256, 256),
intensity(32, 32),intensity(128, 128),intensity(256, 256),
合并成
phase+intensity(2, 64, 64)、phase+intensity(2, 128, 128)、phase+intensity(2, 256, 256)
并保存到
/home/zhang03/zxc/STFT_3DDL/DATASETS/preprocessed_data/DAS1K/phase+intensity中
"""

from pathlib import Path
import numpy as np
from tqdm import tqdm
import shutil

base_path = Path("/home/zhang03/zxc/STFT_3DDL/DATASETS/preprocessed_data/DAS1K")

phase_dir = base_path / "phase" / "matrixs"
intensity_dir = base_path / "intensity" / "matrixs"


SCALES = [64, 128, 256]

phase_ids = [file.name for file in (phase_dir/"scale_256").iterdir() if file.is_dir()]
intensity_ids = [file.name for file in (intensity_dir/"scale_256").iterdir() if file.is_dir()]

assert set(phase_ids) == set(intensity_ids), "phase和intensity中的样本名不相同"
ids = phase_ids


for scale in tqdm(SCALES, desc="数据处理进度(SCALES)"):
    for id in tqdm(ids, desc="数据处理进度(ids)"):
        save_path = base_path / "pi" / "matrixs" 
        phase_data_path = phase_dir / f"scale_{str(scale)}" / id / f"{id}.npz"
        phase_data = np.load(phase_data_path, allow_pickle=True)
        phase_data = phase_data[phase_data.files[0]].tolist()

        intensity_data_path = intensity_dir / f"scale_{str(scale)}" / id / f"{id}.npz"
        intensity_data = np.load(intensity_data_path, allow_pickle=True)
        intensity_data = intensity_data[intensity_data.files[0]].tolist()
        
        assert phase_data['data_class'] == intensity_data['data_class'], "phase和intensity的data_class不相同"
        assert phase_data['data_id'] == intensity_data['data_id'], "phase和intensity的data_id不相同"
        assert phase_data['faxis'].all() == intensity_data['faxis'].all(), "phase和intensity的faxis不相同"
        assert phase_data['taxis'].all() == intensity_data['taxis'].all(), "phase和intensity的taxis不相同"

        new_data = {
            'data_class': phase_data['data_class'],
            'data_id': id,
            'faxis': phase_data['faxis'],
            'taxis': phase_data['taxis'],
            'spectrum': np.stack((phase_data['spectrum'], intensity_data['spectrum']), axis=0),
            }
        
        # 保存(2, X, X)的spectrum，并将phase和intensity对应的image复制到相对应的文件夹下，将image重名名为spectrum类型(pahse或intensity)
        save_path = save_path / f"scale_{scale}" / id
        if not save_path.exists():
            save_path.mkdir(parents=True)
        
        if not (save_path/f"{id}.npz").exists():
            np.savez(save_path/f"{id}.npz", new_data)
        
        # 将phase和intensity中生成的image复制到save_path中
        if not (save_path/"phase.png").exists():
            shutil.copy((phase_dir / f"scale_{str(scale)}" / id / f"{id}.png"), save_path/"phase.png")
        if not (save_path/"intensity.png").exists():
            shutil.copy((intensity_dir / f"scale_{str(scale)}" / id / f"{id}.png"), save_path/"intensity.png")
print("处理完成!")