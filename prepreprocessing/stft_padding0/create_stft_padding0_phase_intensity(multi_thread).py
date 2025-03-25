import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from hdf5storage import loadmat
from scipy import signal
from tqdm import tqdm
import multiprocessing


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


def pad_array(arr, final_length, pad_value):
    """
    对一维数组进行 padding，使其长度达到 final_length
    :param arr: 输入的一维数组 (numpy array or list)
    :param final_length: 目标长度
    :param pad_value: 用于填充的数值
    :return: 填充后的 numpy 数组
    """
    arr = np.array(arr) # 确保输入是numpy数组
    current_length = len(arr)

    if current_length >= final_length:
        return arr[:final_length] # 如果长度大于等于目标长度，截断返回
    else:
        padding = np.full(final_length - current_length, pad_value) # 生成填充数组
        return np.concatenate((arr, padding)) # 连接原数组和填充部分


def get_nperseg_noverlap(len_signal, lfaxis, ltaxis):
    nperseg = 2 * (lfaxis - 1)
    # nperseg = 2 * lfaxis
    # noverlap = int((len_signal - nperseg * (ltaxis - 1))/ltaxis)
    # noverlap = int((len_signal - nperseg * ltaxis)/ltaxis)
    noverlap = int((ltaxis * nperseg - len_signal)/(ltaxis-2))-1
    return nperseg, noverlap

def temp_check_get_nperseg_noverlap(signal_data, fs, window, nperseg, noverlap, lfaxis, ltaxis, added_flag):
    faxis, taxis, spectrum = signal.stft(signal_data, fs, window=window, nperseg=nperseg, noverlap=noverlap, boundary=None, padded=True)
    crop = False # 不用crop
    flag = True # True：继续循环
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
    if crop == True:
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

def process_file(args):
    dataset_path, class_name, iterfile_name, max_lengths, scales, window, fs = args
    phase, intensity = read_raw_signal_file(dataset_path, class_name, iterfile_name)
    phase = pad_array(phase, max_lengths, 0)
    intensity = pad_array(intensity, max_lengths, 0)
    results = []
    
    for scale in scales:
        nperseg, noverlap = get_nperseg_noverlap(len(phase), lfaxis=scale, ltaxis=scale)
        flag = True
        added_flag = False
        while(flag):
            nperseg, noverlap, crop, flag, added_flag = temp_check_get_nperseg_noverlap(phase, fs, window, nperseg, noverlap, lfaxis=scale, ltaxis=scale, added_flag=added_flag)
        
        nperseg_i, noverlap_i = get_nperseg_noverlap(len(intensity), lfaxis=scale, ltaxis=scale)
        flag = True
        added_flag = False
        while(flag):
            nperseg_i, noverlap_i, crop_i, flag, added_flag = temp_check_get_nperseg_noverlap(intensity, fs, window, nperseg=nperseg, noverlap=noverlap, lfaxis=scale, ltaxis=scale, added_flag=added_flag)
        
        assert nperseg == nperseg_i and noverlap == noverlap_i and crop == crop_i, "phase与intensity计算结果不一致"
        results.append({'id': iterfile_name, 'class': class_name, 'scale': scale, 'nperseg': nperseg, 'noverlap': noverlap, 'signal_data': {'phase': phase, 'intensity': intensity}, 'crop': crop})
    return results

def process_data(data):
    base_save_path = Path('/home/zhang/zxc/STFT_3DDL/DATASETS/preprocessed_data/DAS1K_padding0/')
    data_types = ['phase', 'intensity']
    for data_type in data_types:
        faxis, taxis, spectrum = get_spectrum_corp(data['signal_data'][data_type], fs=10000, window="hamming", nperseg=data['nperseg'], noverlap=data['noverlap'], lfaxis=data['scale'], ltaxis=data['scale'], crop=data['crop'])
        save_path = base_save_path / f"{data_type}" / f"scale_{data['scale']}" / f"{data['id']}"
        save_path.mkdir(parents=True, exist_ok=True)
        save_spectrum_matrix(save_path, spectrum, data['class'], data['id'], faxis=faxis, taxis=taxis)
        save_spectrum_image(save_path, spectrum, data['class'], data['id'], faxis=faxis, taxis=taxis)

if __name__ == '__main__':
    window = "hamming"
    fs = 10000
    scales = [256, 128, 64]
    max_lengths = 49450
    dataset_path = Path('/home/zhang/zxc/STFT_3DDL/DATASETS/raw_data/DAS1K/')
    class_names = get_class_names(dataset_path)
    
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    args_list = [(dataset_path, class_name, iterfile_name, max_lengths, scales, window, fs) 
                 for class_name in class_names 
                 for iterfile_name in get_iterfile_names(dataset_path / class_name)]
    
    all_data = []
    for result in tqdm(pool.imap_unordered(process_file, args_list), total=len(args_list), desc="数据读取进度"):
        all_data.extend(result)
    
    pool.close()
    pool.join()
    
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    for _ in tqdm(pool.imap_unordered(process_data, all_data), total=len(all_data), desc="数据处理进度"):
        pass
    pool.close()
    pool.join()