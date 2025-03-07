import matplotlib.pylab as plt
import numpy as np
from pathlib import Path
from hdf5storage import loadmat
from scipy import signal
from pyts.image import GramianAngularField
from tqdm import tqdm

dataset_path = Path('/home/zhang/zxc/STFT_3DDL/DATASETS/raw_data/DAS1K/')
list_path = Path("/home/zhang/zxc/STFT_3DDL/STFT_3Ddl/stft_3ddl/lists/DAS1K/phase_intensity/")
# GAF实验不保存train list和test list，沿用stft的train list和test list
# 所以只进行数据生成，不进行列表生成

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
    taxis = np.arange(len(phase))/10000
    # plt.figure(figsize=(15, 15))
    # plt.plot(taxis, phase)
    # plt.figure(figsize=(15, 15))
    # plt.plot(taxis, intensity)
    # print('test')
    return phase, intensity


def signal_resample_1(raw_signal, num_resampled):
    # 只resample
    resampled_signal = signal.resample(raw_signal, num=num_resampled)
    return resampled_signal

def signal_resample_2(raw_signal, num_resampled):
    # 先归一化到[0,1],再重采样
    signal_data = (raw_signal - np.min(raw_signal)) / (np.max(raw_signal) - np.min(raw_signal)) # 归一化到[0,1]
    resampled_signal = signal.resample(signal_data, num=num_resampled)
    return resampled_signal

def signal_resample_3(raw_signal, num_resampled):
    # 先重采样，再归一化到[0,1]
    resampled_signal = signal.resample(raw_signal, num=num_resampled)
    resampled_signal = (resampled_signal - np.min(resampled_signal)) / (np.max(resampled_signal) - np.min(resampled_signal)) # 归一化到[0,1]
    return resampled_signal

def signal_resample_4(raw_signal, num_resampled):
    # 先归一化到[-1,1]，再重采样
    signal_data = 2 * (raw_signal - np.min(raw_signal)) / (np.max(raw_signal) - np.min(raw_signal) + 1e-10) - 1 # 归一化到[-1,1]
    resampled_signal = signal.resample(signal_data, num=num_resampled)
    return resampled_signal


def get_gasf(signal):
    gaf = GramianAngularField(method="summation") # GASF
    gasf = gaf.fit_transform(signal.reshape(1, -1))[0]
    return gasf

def get_gadf(signal):
    gaf = GramianAngularField(method="difference") # GADF
    gadf = gaf.fit_transform(signal.reshape(1, -1))[0]
    return gadf

def save_gaf_matrix(save_path, gasf, gadf, scale, class_name, data_id):
    data = {
        'data_class': class_name,
        'data_id': data_id,
        'scale': scale,
        'gasf': gasf,
        'gadf': gadf
    }
    save_path = Path(save_path)
    np.savez(save_path/f"{data_id}.npz", data)

def save_gaf_image(save_path, gasf, gadf, scale, class_name, data_id):
    plt.figure()
    plt.imshow(gasf, cmap='hot', interpolation='nearest')
    plt.title(f"GASF {data_id} scale={scale}")
    plt.colorbar()
    plt.savefig(save_path/"GASF.png")

    plt.figure()
    plt.imshow(gadf, cmap='hot', interpolation='nearest')
    plt.title(f"GADF {data_id} scale={scale}")
    plt.colorbar()
    plt.savefig(save_path/"GADF.png")
    plt.figure()

fs = 10000
scales = [256, 128, 64]

datas = []

class_names = get_class_names(dataset_path)

for class_name in class_names:
    iterfile_names = get_iterfile_names(dataset_path/class_name)
    for iterfile_name in iterfile_names:
        phase, _ = read_raw_signal_file(dataset_path, class_name, iterfile_name)
        for scale in scales:
            datas.append({
                'id':iterfile_name,
                'class':class_name,
                'scale':scale,
                'signal_data':phase
            })

# 数据生成
for data in tqdm(datas, desc="数据处理进度"):
    save_path = Path(f'/home/zhang/zxc/STFT_3DDL/DATASETS/preprocessed_data/DAS1K_GAF/phase/')
    signal_data = signal_resample_4(raw_signal=data['signal_data'], num_resampled=data['scale'])
    gasf = get_gasf(signal_data)
    gadf = get_gadf(signal_data)

    save_path = save_path/f"scale_{data['scale']}/{data['id']}/"
    # 检查save_path是否存在
    if not save_path.exists():
        save_path.mkdir(parents=True)
    
    scale = data['scale']
    class_name = data['class']
    id = data['id']
    
    save_gaf_matrix(save_path, gasf=gasf, gadf=gadf, scale=scale, class_name=class_name, data_id=id)
    save_gaf_image(save_path, gasf=gasf, gadf=gadf, scale=scale, class_name=class_name, data_id=id)
    