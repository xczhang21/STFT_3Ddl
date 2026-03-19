import argparse
import matplotlib.pylab as plt
import numpy as np
import os
import sys
from pathlib import Path
from hdf5storage import loadmat
from scipy import signal
from tqdm import tqdm
import random

def parse_args():
    parser = argparse.ArgumentParser(description="DAS1K STFT Spectrogram Generator")
    parser.add_argument('--dataset_path', type=str, required=True, help='原始 DAS1K 数据集路径')
    parser.add_argument('--save_path', type=str, required=True, help='保存 STFT 矩阵和图像的路径')
    return parser.parse_args()

args = parse_args()

dataset_path = Path(args.dataset_path)
save_base_path = Path(args.save_path)

def get_class_names(directory_path):
    directory = Path(directory_path)
    if not directory.is_dir():
        raise ValueError(f"{directory_path} 不是一个有效的目录路径")
    category_names = set()
    for subfolder in directory.iterdir():
        if subfolder.is_dir():
            category_names.add(subfolder.stem)
    return list(category_names)

def get_iterfile_names(directory_path):
    directory = Path(directory_path)
    if not directory.is_dir():
        raise ValueError(f"{directory_path} 不是一个有效的目录路径")
    file_names = set()
    for subfolder in directory.iterdir():
        if subfolder.is_file():
            file_names.add(subfolder.stem)
    return list(file_names)

def read_raw_signal_file(dataset_path, class_name, iterfile_name):
    iterfile_path = Path(dataset_path) / class_name / f"{iterfile_name}.mat"
    mat = loadmat(str(iterfile_path))
    phase = mat[iterfile_name][0]
    intensity = mat[iterfile_name][1]
    return phase, intensity

def get_nperseg_noverlap(len_signal, lfaxis, ltaxis):
    nperseg = 2 * (lfaxis - 1)
    noverlap = int((ltaxis * nperseg - len_signal)/(ltaxis-2))-1
    return nperseg, noverlap

def temp_check_get_nperseg_noverlap(signal_data, fs, window, nperseg, noverlap, lfaxis, ltaxis, added_flag):
    faxis, taxis, spectrum = signal.stft(signal_data, fs, window=window, nperseg=nperseg, noverlap=noverlap, boundary=None, padded=True)
    crop = False
    flag = True
    if len(faxis) != lfaxis:
        print("len(faxis) != lfaxis")
        exit()

    if len(taxis) == ltaxis:
        flag = False
        return nperseg, noverlap, crop, flag, added_flag
    elif len(taxis) < ltaxis:
        noverlap = noverlap + 1
        if added_flag == True:
            crop = True
            flag = False
            return nperseg, noverlap, crop, flag, added_flag
        else:
            flag = True
            return nperseg, noverlap, crop, flag, added_flag
    else:
        added_flag = True
        noverlap = noverlap - 1
        flag = True
        return nperseg, noverlap, crop, flag, added_flag

def get_spectrum_corp(signal_data, fs, window, nperseg, noverlap, lfaxis, ltaxis, crop):
    faxis, taxis, spectrum = signal.stft(signal_data, fs, window, nperseg, noverlap, boundary=None, padded=True)
    if len(faxis)<lfaxis or len(taxis)<ltaxis:
        print('get_matrix_image error')
        exit()
    if crop:
        spectrum = spectrum[:lfaxis, :ltaxis]
    return faxis[:lfaxis], taxis[:ltaxis], np.abs(spectrum)

def save_spectrum_matrix(save_path, spectrum, class_name, data_id, faxis, taxis):
    data = {'data_class':class_name, 'data_id':data_id, 'faxis':faxis, 'taxis':taxis, 'spectrum':spectrum}
    save_path = Path(save_path)
    np.savez(save_path/f"{data_id}.npz", data)

def save_spectrum_image(save_path, spectrum, class_name, data_id, faxis, taxis):
    plt.pcolormesh(taxis, faxis, np.abs(spectrum))
    plt.title(f'STFT {data_id} ltaxis*lfaxis={len(taxis)}*{len(faxis)}')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.savefig(save_path/f"{data_id}.png", dpi=len(faxis))

window = 'hamming'
fs = 10000
scales = [256, 128, 64]
datas = []

class_names = get_class_names(dataset_path)

for class_name in class_names:
    iterfile_names = get_iterfile_names(dataset_path/class_name)
    for iterfile_name in iterfile_names:
        phase, intensity = read_raw_signal_file(dataset_path, class_name, iterfile_name)
        for scale in scales:
            nperseg, noverlap = get_nperseg_noverlap(len(phase), lfaxis=scale, ltaxis=scale)
            flag = True
            added_flag = False
            while flag:
                nperseg, noverlap, crop, flag, added_flag = temp_check_get_nperseg_noverlap(
                    phase, fs, window, nperseg=nperseg, noverlap=noverlap, lfaxis=scale, ltaxis=scale, added_flag=added_flag)
            datas.append({'id':iterfile_name, 'class':class_name, 'scale':scale,
                          'nperseg':nperseg, 'noverlap':noverlap,
                          'phase':phase, 'intensity':intensity, 'crop':crop})

# 处理并保存数据
for data in tqdm(datas, desc="数据处理进度"):
    phase_save_path = save_base_path / "phase" / "matrixs" / f"scale_{data['scale']}" / data['id']
    phase_save_path.mkdir(parents=True, exist_ok=True)
    intensity_save_path = save_base_path / "intensity" / "matrixs" / f"scale_{data['scale']}" / data['id']
    intensity_save_path.mkdir(parents=True, exist_ok=True)

    faxis, taxis, phase = get_spectrum_corp(data['phase'], fs=fs, window=window,
                                               nperseg=data['nperseg'], noverlap=data['noverlap'],
                                               lfaxis=data['scale'], ltaxis=data['scale'], crop=data['crop'])
    save_spectrum_matrix(phase_save_path, phase, data['class'], data['id'], faxis=faxis, taxis=taxis)
    save_spectrum_image(phase_save_path, phase, data['class'], data['id'], faxis=faxis, taxis=taxis)

    faxis, taxis, intensity = get_spectrum_corp(data['intensity'], fs=fs, window=window,
                                               nperseg=data['nperseg'], noverlap=data['noverlap'],
                                               lfaxis=data['scale'], ltaxis=data['scale'], crop=data['crop'])
    save_spectrum_matrix(intensity_save_path, intensity, data['class'], data['id'], faxis=faxis, taxis=taxis)
    save_spectrum_image(intensity_save_path, intensity, data['class'], data['id'], faxis=faxis, taxis=taxis)
