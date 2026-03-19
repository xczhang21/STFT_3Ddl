import matplotlib.pylab as plt
import numpy as np
import os
import sys
from pathlib import Path
from hdf5storage import loadmat, savemat
from scipy import signal


class DASDataAugmentor:
    def __init__(self, dataset_path, output_path, frequency=10000):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.frequency = frequency
    
    def get_class_names(self):
        directory = self.dataset_path
        if not directory.is_dir():
            raise ValueError(f"{directory} 不是一个有效的目录路径")
        category_names = set()
        for subfolder in directory.iterdir():
            if subfolder.is_dir():
                category_names.add(subfolder.stem)
        return list(category_names)

    def get_iterfile_names(self, class_name):
        directory = self.dataset_path / class_name
        if not directory.is_dir():
            raise ValueError(f"{directory} 不是一个有效的目录路径")
        file_names = set()
        for file in directory.iterdir():
            if file.is_file():
                file_names.add(file.stem)
        return list(file_names)

    def read_raw_signal_file(self, class_name, iterfile_name):
        iterfile_path = self.dataset_path / class_name / f"{iterfile_name}.mat"
        mat = loadmat(str(iterfile_path))
        phase = mat[iterfile_name][0]
        intensity = mat[iterfile_name][1]
        return phase, intensity

    def print_signal(self, signal, title="Signal"):
        taxis = np.arange(len(signal)) / self.frequency
        plt.figure(figsize=(15, 5))
        plt.plot(taxis, signal)
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.show()
    
    # =================== 增强方法 ===================
    def time_shift(self, signal, shift_ratio=0.05):
        """
        时间平移：将信号在时间轴上平移一定比例。
        参数：
            shift_ratio: 平移的最大比例，正负随机。
        """
        shift_len = np.random.randint(-int(shift_ratio * len(signal)), int(shift_ratio * len(signal)))
        return np.roll(signal, shift_len)

    def add_noise(self, signal, noise_std=0.01):
        """
        加性高斯噪声：为信号加入随机噪声，模拟环境干扰。
        参数：
            noise_std: 噪声标准差。
        """
        noise = np.random.normal(0, noise_std, size=signal.shape)
        return signal + noise

    def amplitude_scaling(self, signal, scale_range=(0.8, 1.2)):
        """
        振幅缩放：对信号整体进行放大或缩小。
        参数：
            scale_range: 缩放比例范围（最小值, 最大值）。
        """
        scale = np.random.uniform(*scale_range)
        return signal * scale

    def random_cutout(self, signal, cut_ratio=0.05):
        """
        随机遮挡：将信号中的一段连续片段置零，模拟丢失。
        参数：
            cut_ratio: 遮挡的信号比例。
        """
        cut_len = int(cut_ratio * len(signal))
        start = np.random.randint(0, len(signal) - cut_len)
        signal_copy = signal.copy()
        signal_copy[start:start + cut_len] = 0
        return signal_copy

    def time_stretch(self, signal, rate_choices=(0.9, 1.1)):
        """
        时间拉伸：模拟信号加速或减慢播放，改变时间尺度。
        参数：
            rate_choices: 拉伸比例的可选集合。
        """
        from scipy.interpolate import interp1d
        x = np.arange(len(signal))
        f = interp1d(x, signal, kind='linear')
        rate = np.random.choice(rate_choices)
        new_len = int(len(signal) * rate)
        x_new = np.linspace(0, len(signal) - 1, new_len)
        stretched = f(x_new)
        return np.interp(np.linspace(0, new_len - 1, len(signal)), np.arange(new_len), stretched)
    
    def apply_augmentation(self, signal, method):
        """
        根据方法名选择对应的增强方法。
        参数：
            signal: 输入信号。
            method: 增强方法名。
        返回：增强后的信号。
        """
        if method == 'shift':
            return self.time_shift(signal)
        elif method == 'noise':
            return self.add_noise(signal)
        elif method == 'scale':
            return self.amplitude_scaling(signal)
        elif method == 'cutout':
            return self.random_cutout(signal)
        elif method == 'stretch':
            return self.time_stretch(signal)
        else:
            raise ValueError(f"未知的增强方法：{method}")
    
    def save_single_augmentation(self, methods=None):
        if methods is None:
            methods = ['shift', 'noise', 'scale', 'cutout', 'stretch']

        class_names = self.get_class_names()
        for class_name in class_names:
            iterfile_names = self.get_iterfile_names(class_name)

            for iterfile_name in iterfile_names:
                phase, intensity = self.read_raw_signal_file(class_name, iterfile_name)

                for method in methods:
                    # 每种增强方法单独建文件夹
                    save_folder = self.output_path / f"DAS1K_{method}" / class_name
                    save_folder.mkdir(parents=True, exist_ok=True)

                    # ⭐保存原始信号 carhorn1.mat（如果还没保存）
                    ori_save_path = save_folder / f"{iterfile_name}.mat"
                    if not ori_save_path.exists():
                        savemat(str(ori_save_path), {f"{iterfile_name}": np.vstack((phase, intensity))})

                    # 保存增强版 carhorn1_shift.mat
                    phase_aug = self.apply_augmentation(phase, method)
                    intensity_aug = self.apply_augmentation(intensity, method)

                    aug_save_name = f"{iterfile_name}_aug.mat"
                    aug_save_path = save_folder / aug_save_name

                    savemat(str(aug_save_path), {f"{iterfile_name}_aug": np.vstack((phase_aug, intensity_aug))})
   

    def save_mixed_augmentation(self, methods=None, num_augmented_samples=1):
        if methods is None:
            methods = ['shift', 'noise', 'scale', 'cutout', 'stretch']

        class_names = self.get_class_names()
        for class_name in class_names:
            iterfile_names = self.get_iterfile_names(class_name)

            for iterfile_name in iterfile_names:
                phase, intensity = self.read_raw_signal_file(class_name, iterfile_name)

                save_folder = self.output_path / f"DAS1K_SNSCS{num_augmented_samples}" / class_name # SNSCS:'shift', 'noise', 'scale', 'cutout', 'stretch'
                save_folder.mkdir(parents=True, exist_ok=True)

                # 保存原始信号
                ori_save_path = save_folder / f"{iterfile_name}.mat"
                if not ori_save_path.exists():
                    savemat(str(ori_save_path), {f"{iterfile_name}": np.vstack((phase, intensity))})

                for i in range(num_augmented_samples):
                    phase_aug = phase.copy()
                    intensity_aug = intensity.copy()
                    for method in methods:
                        phase_aug = self.apply_augmentation(phase_aug, method)
                        intensity_aug = self.apply_augmentation(intensity_aug, method)

                    save_name = f"{iterfile_name}_aug{i+1}.mat"
                    save_path = save_folder / save_name

                    savemat(str(save_path), {f"{iterfile_name}_aug{i+1}": np.vstack((phase_aug, intensity_aug))})



if __name__ == "__main__":
    dataset_path = '/home/zhang/zxc/STFT_3DDL/DATASETS/raw_data/DAS1K/'
    output_path = '/home/zhang/zxc/STFT_3DDL/DATASETS/raw_data/'

    data_augmentor = DASDataAugmentor(dataset_path=dataset_path, output_path=output_path)
    # data_augmentor.save_single_augmentation()
    data_augmentor.save_mixed_augmentation(num_augmented_samples=7)
    # data_augmentor.save_mixed_augmentation(num_augmented_samples=1)