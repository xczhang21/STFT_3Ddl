"""
Wu H, Zhou B, Zhu K, et al. Pattern recognition in distributed fiber-optic acoustic sensor using an intensity and phase stacked convolutional neural network with data augmentation[J]. Optics express, 2021, 29(3): 3269-3283.
"""
"""
已复现IPCNN，但对模型不做修改则模型不收敛，
对模型做修改(加上BN)后，模型收敛，但还是达不到作者论文中的88.2%
"""
import ml_collections

def get_IPCNN_config():
    config = ml_collections.ConfigDict()
    config.in_channels = 1         # Mel-spectrogram is single-channel
    config.out_channels = 10       # 10 classes
    config.init_features = 16      # Base number of filters
    return config


# 模型测试
if __name__ == '__main__':
    from ipcnn_modeling import IPCNN
    import torch

    config = get_IPCNN_config()
    model = IPCNN(config)
    model.cuda()
    dummy_input1 = torch.randn(32, 1, 64, 64).cuda()  # intensity
    dummy_input2 = torch.randn(32, 1, 64, 64).cuda()  # phase

    output = model(dummy_input1, dummy_input2)
    print(output.shape)
