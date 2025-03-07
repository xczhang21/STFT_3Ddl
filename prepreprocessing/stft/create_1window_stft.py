import matplotlib.pylab as plt
import numpy as np
import os
import sys
from pathlib import Path
from hdf5storage import loadmat
from scipy import signal


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
    # mat = loadmat('/home/zhang03/zxc/STFT_3DDL/DATASETS/raw_data/DAS1K/CARHORN/carhorn1.mat')
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


def get_stft_spectrogram(signal_data, fs, window, nperseg, noverlap):
    """
    signal_data : 信号数据
    fs : 采样频率
    window : 窗口函数名
    nperseg : 窗口大小
    noverlap : 窗口重叠大小
    """
    faxis, taxis, spectrum = signal.stft(signal_data, fs, window=window, nperseg=nperseg, noverlap=noverlap)

    # # 绘制短时傅里叶变换
    # plt.figure(figsize=(15,15))
    # plt.pcolormesh(taxis, faxis, np.abs(spectrum))
    # # plt.colorbar(label='Magnitude')
    # plt.title('Time-Frequency Analysis (STFT) using pcolormesh')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Frequency (Hz)')
    # plt.xlim([taxis[0], taxis[-1]])
    return len(faxis), len(taxis), np.abs(spectrum)


dataset_path = Path('/home/zhang03/zxc/STFT_3DDL/DATASETS/raw_data/DAS1K/')

window = 'hamming'
nperseg = 266
noverlap = 0
fs = 10000

experimet_id = f"w_{window}_np_{nperseg}_no_{noverlap}"
save_path = Path(f'/home/zhang03/zxc/STFT_3DDL/DATASETS/preprocessed_data/DAS1K/phase/{experimet_id}/')
lists_save_path = Path(f"/home/zhang03/zxc/STFT_3DDL/STFT_3Ddl/stft_3ddl/lists/DAS1K/phase/{experimet_id}/")

# 检查save_path是否存在
if not save_path.exists():
    save_path.mkdir(parents=True)
    print(f"preprocessed 文件夹'{experimet_id}'\t 已创建。")
else:
    print(f"preprocessed 文件夹'{experimet_id}'\t 已存在。")

# 检查save_path是否存在
if not lists_save_path.exists():
    lists_save_path.mkdir(parents=True)
    print(f"lists 文件夹'{experimet_id}'\t 已创建。")
else:
    print(f"lists 文件夹'{experimet_id}'\t 已存在。")

class_names = get_class_names(dataset_path)

datas = []
min_faxis = 100000
max_faxis = 0
min_taxis = 100000
max_taxis = 0
for class_name in class_names:
    iterfile_names = get_iterfile_names(dataset_path/class_name)
    for iterfile_name in iterfile_names:
        phase, _ = read_raw_signal_file(dataset_path, class_name, iterfile_name)
        faxis, taxis, spectrum = get_stft_spectrogram(phase, fs, window, nperseg, noverlap)
        if faxis < min_faxis:
            min_faxis = faxis
        if taxis < min_taxis:
            min_taxis = taxis
        if faxis > max_faxis:
            max_faxis = faxis
        if taxis > max_taxis:
            max_taxis = taxis
        data = {'class':class_name, 'id':iterfile_name, 'faxis':faxis, 'taxis':taxis, 'spectrum':spectrum}
        datas.append(data)
        # np.savez(save_path/f"{iterfile_name}.npz", data)
    # print(f"class_name:{class_name}\t phase \t num:{len(iterfile_names)}\t STFT({experimet_id})完成。")

datas_dict = {class_name:[] for class_name in class_names}
for data in datas:
    datas_dict[data['class']].append(data)

print('test')
# for data in datas:
    