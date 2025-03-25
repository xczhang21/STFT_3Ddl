# 将DAS1K数据集使用默认matplotlib默认STFT配置进行STFT，然后resize成64、128、256

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
from hdf5storage import loadmat
from scipy import signal
from tqdm import tqdm
from skimage.transform import resize

def get_class_names(directory_path):
    # 转换为Path对象
    directory = Path(directory_path)
    # 确保路径存在且为目录
    if not directory.is_dir():
        raise ValueError(f"{directory_path} 不是一个有效的目录路径")
    
    # 创建一个集合来存储类别名，避免重复
    category_names = set()
    
    # 遍历主文件夹下的所有子文件夹
    for subfolder in directory.iterdir():
        if subfolder.is_dir():  # 检查是否为子文件夹
            # 遍历子文件夹中的文件
            category_names.add(subfolder.stem)
    
    # 返回类别名列表
    return list(category_names)

def get_iterfile_names(directory_path):
    # 转换为Path对象
    directory = Path(directory_path)
    # 确保路径存在且为目录
    if not directory.is_dir():
        raise ValueError(f"{directory_path} 不是一个有效的目录路径")
    
    # 创建一个集合来存储文件名，避免重复
    file_names = set()
    
    # 遍历主文件夹下的所有子文件夹
    for subfolder in directory.iterdir():
        if subfolder.is_file():  # 检查是否为子文件
            file_names.add(subfolder.stem)
    
    # 返回文件名列表
    return list(file_names)

def read_raw_signal_file(dataset_path, class_name, iterfile_name):
    iterfile_path = Path(dataset_path) / class_name / f"{iterfile_name}.mat"
    # mat = loadmat('/home/zhang/zxc/STFT_3DDL/DATASETS/raw_data/DAS1K/CARHORN/carhorn1.mat')
    mat = loadmat(str(iterfile_path))
    phase = mat[iterfile_name][0]
    intensity = mat[iterfile_name][1]
    # taxis = np.arange(len(phase))/10000
    # plt.figure(figsize=(15, 15))
    # plt.plot(taxis, phase)
    # plt.figure(figsize=(15, 15))
    # plt.plot(taxis, intensity)
    # print('test')
    return phase, intensity


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

def get_spectrum(signal_data, fs, window, nperseg, noverlap):
    faxis, taxis, spectrum = signal.stft(signal_data, fs, window=window, nperseg=nperseg, noverlap=noverlap, boundary=None, padded=True)
    return faxis, taxis, spectrum

def resize_spectrum(spectrum, faxis, taxis, scale):
    # spectrum是已经np.abs过的
    spectrum_resized = resize(
        spectrum,
        (scale, scale),
        mode='reflect',
        anti_aliasing=True, # 平滑处理
        preserve_range=True, # 保留原始数值范围，避免自动归一化
    )
    
    # resize frequency和time axes
    new_freq_bins = scale
    new_time_bins = scale
    faxis_resized = np.linspace(faxis[0], faxis[-1], new_freq_bins)
    taxis_resized = np.linspace(taxis[0], taxis[-1], new_time_bins)
    return faxis_resized, taxis_resized, spectrum_resized




if __name__ == '__main__':
    
    # matplotlib中的参数设置如下
    NFFT = 256
    NOVERLAP = NFFT // 2
    WINDOW = 'hann' # ”动态调节方法”使用的窗口是hamming窗口
    fs = 10000
    scales = [256, 128, 64]
    datas = []

    dataset_path = Path('/home/zhang/zxc/STFT_3DDL/DATASETS/raw_data/DAS1K')

    class_names = get_class_names(dataset_path)

    for class_name in class_names:
        iterfile_names = get_iterfile_names(dataset_path/class_name)
        for iterfile_name in iterfile_names:
            phase, intensity = read_raw_signal_file(dataset_path, class_name, iterfile_name)
            phase_faxis, phase_taxis, phase_spectrum = get_spectrum(phase, fs=fs, window=WINDOW, nperseg=NFFT, noverlap=NOVERLAP)
            intensity_faxis, intensity_taxis, intensity_spectrum = get_spectrum(phase, fs=fs, window=WINDOW, nperseg=NFFT, noverlap=NOVERLAP)
            for scale in scales:
                phase_faxis, phase_taxis, phase_spectrum = resize_spectrum(np.abs(phase_spectrum),
                                                                           phase_faxis,
                                                                           phase_taxis,
                                                                           scale)
                intensity_faxis, intensity_taxis, intensity_spectrum = resize_spectrum(np.abs(intensity_spectrum),
                                                                                       intensity_faxis,
                                                                                       intensity_taxis,
                                                                                       scale)
                datas.append({'id':iterfile_name, 'class':class_name, 'scale':scale,
                              'faxis':{'phase':phase_faxis, 'intensity':intensity_faxis},
                              'taxis':{'phase':phase_taxis, 'intensity':intensity_taxis},
                              'signal_data':{'phase':phase_spectrum, 'intensity':intensity_spectrum}})
    # 数据生成
    base_save_path = Path('/home/zhang/zxc/STFT_3DDL/DATASETS/preprocessed_data/DAS1K_resize/')
    data_types = ['phase', 'intensity']
    for data in tqdm(datas, desc="数据处理进度"):
        for data_type in data_types:
            spectrum = data['signal_data'][data_type]
            faxis = data['faxis'][data_type]
            taxis = data['taxis'][data_type]
            save_path = base_save_path/f"{data_type}"/f"scale_{data['scale']}"/f"{data['id']}/"
            # 检查save_path是否存在
            if not save_path.exists():
                save_path.mkdir(parents=True)
            
            save_spectrum_matrix(save_path, spectrum, data['class'], data['id'], faxis=faxis, taxis=taxis)
            save_spectrum_image(save_path, spectrum, data['class'], data['id'], faxis=faxis, taxis=taxis)