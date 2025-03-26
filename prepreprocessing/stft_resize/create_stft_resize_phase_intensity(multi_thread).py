import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from hdf5storage import loadmat
from scipy import signal
from tqdm import tqdm
from skimage.transform import resize
from concurrent.futures import ThreadPoolExecutor

def get_class_names(directory_path):
    directory = Path(directory_path)
    if not directory.is_dir():
        raise ValueError(f"{directory_path} 不是一个有效的目录路径")
    return [subfolder.stem for subfolder in directory.iterdir() if subfolder.is_dir()]

def get_iterfile_names(directory_path):
    directory = Path(directory_path)
    if not directory.is_dir():
        raise ValueError(f"{directory_path} 不是一个有效的目录路径")
    return [subfolder.stem for subfolder in directory.iterdir() if subfolder.is_file()]

def read_raw_signal_file(dataset_path, class_name, iterfile_name):
    iterfile_path = Path(dataset_path) / class_name / f"{iterfile_name}.mat"
    mat = loadmat(str(iterfile_path))
    phase = mat[iterfile_name][0]
    intensity = mat[iterfile_name][1]
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
    plt.close()

def get_spectrum(signal_data, fs, window, nperseg, noverlap):
    faxis, taxis, spectrum = signal.stft(signal_data, fs, window=window, nperseg=nperseg, noverlap=noverlap, boundary=None, padded=True)
    return faxis, taxis, spectrum

def resize_spectrum(spectrum, faxis, taxis, scale):
    spectrum_resized = resize(
        spectrum,
        (scale, scale),
        mode='reflect',
        anti_aliasing=True,
        preserve_range=True,
    )
    faxis_resized = np.linspace(faxis[0], faxis[-1], scale)
    taxis_resized = np.linspace(taxis[0], taxis[-1], scale)
    return faxis_resized, taxis_resized, spectrum_resized

def process_single_file(args):
    dataset_path, class_name, iterfile_name, scales, fs, NFFT, NOVERLAP, WINDOW = args
    try:
        phase, intensity = read_raw_signal_file(dataset_path, class_name, iterfile_name)
        phase_faxis, phase_taxis, phase_spectrum = get_spectrum(phase, fs=fs, window=WINDOW, nperseg=NFFT, noverlap=NOVERLAP)
        intensity_faxis, intensity_taxis, intensity_spectrum = get_spectrum(intensity, fs=fs, window=WINDOW, nperseg=NFFT, noverlap=NOVERLAP)

        results = []
        for scale in scales:
            pf, pt, ps = resize_spectrum(np.abs(phase_spectrum), phase_faxis, phase_taxis, scale)
            inf, int_, ins = resize_spectrum(np.abs(intensity_spectrum), intensity_faxis, intensity_taxis, scale)
            results.append({'id':iterfile_name, 'class':class_name, 'scale':scale,
                            'faxis':{'phase':pf, 'intensity':inf},
                            'taxis':{'phase':pt, 'intensity':int_},
                            'signal_data':{'phase':ps, 'intensity':ins}})
        return results
    except Exception as e:
        print(f"[ERROR] {class_name}/{iterfile_name}: {e}")
        return []

if __name__ == '__main__':
    NFFT = 256
    NOVERLAP = NFFT // 2
    WINDOW = 'hann'
    fs = 10000
    scales = [256, 128, 64]
    dataset_path = Path('/home/zhang/zxc/STFT_3DDL/DATASETS/raw_data/DAS1K')
    base_save_path = Path('/home/zhang/zxc/STFT_3DDL/DATASETS/preprocessed_data/DAS1K_resize/')
    data_types = ['phase', 'intensity']

    class_names = get_class_names(dataset_path)
    args_list = []

    for class_name in class_names:
        iterfile_names = get_iterfile_names(dataset_path/class_name)
        for iterfile_name in iterfile_names:
            args_list.append((dataset_path, class_name, iterfile_name, scales, fs, NFFT, NOVERLAP, WINDOW))

    # 多线程并行处理数据
    all_datas = []
    with ThreadPoolExecutor() as executor:
        futures = list(executor.map(process_single_file, args_list))
        for data_list in tqdm(futures, desc="数据处理进度"):
            all_datas.extend(data_list)

    # 保存数据（仍然串行，可以改成线程池）
    for data in tqdm(all_datas, desc="保存数据中"):
        for data_type in data_types:
            spectrum = data['signal_data'][data_type]
            faxis = data['faxis'][data_type]
            taxis = data['taxis'][data_type]
            save_path = base_save_path/f"{data_type}"/f"scale_{data['scale']}"/f"{data['id']}/"
            save_path.mkdir(parents=True, exist_ok=True)
            save_spectrum_matrix(save_path, spectrum, data['class'], data['id'], faxis=faxis, taxis=taxis)
            save_spectrum_image(save_path, spectrum, data['class'], data['id'], faxis=faxis, taxis=taxis)