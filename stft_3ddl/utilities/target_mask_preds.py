import torch
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import numpy as np

def visualize_tmmp_fast(target, masked_input, mask, pred):
    """
    使用 PIL 和 numpy 拼接图像，不依赖 matplotlib，生成 [3, H, W] 格式图像
    """
    def to_uint8_heatmap(img):
        """
        转为 numpy uint8 热度图，用红色通道表示强度
        """
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().squeeze().numpy()
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        heatmap = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        heatmap[:, :, 0] = img  # 红色通道
        return heatmap

    images = [
        to_uint8_heatmap(target),
        to_uint8_heatmap(mask),
        to_uint8_heatmap(masked_input),
        to_uint8_heatmap(pred)
    ]
    
    # 拼接图像
    concat_img = np.concatenate(images, axis=1)  # 横向拼接
    image = torch.from_numpy(concat_img).permute(2, 0, 1)  # HWC -> CHW

    return image

def visualize_tmmp_matplotlib(target, masked_input, mask, pred):
    """
    使用 Matplotlib 可视化 TMMP 并将其转换为 TensorBoard 可接收的格式 (CHW, RGB)
    返回：numpy ndarray 格式图像 [3, H, W]
    """
    def to_numpy(img):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().squeeze().numpy()
        return img

    # 获取图像尺寸以自动调整画布大小
    H, W = to_numpy(target).shape
    scale_factor = 20  # 调整此比例可控制显示大小
    figsize = (W * 4 / scale_factor, H / scale_factor)  # 横向4张图

    titles = ["Target", "Mask", "Masked Input", "Prediction"]
    images = [to_numpy(target), to_numpy(mask), to_numpy(masked_input), to_numpy(pred)]

    fig, axs = plt.subplots(1, 4, figsize=figsize)
    for i, (img, title) in enumerate(zip(images, titles)):
        axs[i].imshow(img, cmap='hot')
        axs[i].set_title(title)
        axs[i].axis('off')

    plt.tight_layout()

    # 将图像保存到内存中再转为TensorBoard格式
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    image = Image.open(buf).convert("RGB")
    image = np.array(image).transpose(2, 0, 1)  # HWC -> CHW
    image = torch.from_numpy(image)
    return image



def save_TMMPs(epoch_num, all_TMMPs, writer, save_TMMP_interval=10, save_TMMP_num=None):
    """
    保存 TMMP 可视化图到 TensorBoard
    """
    if epoch_num != 0 and (epoch_num + 1) % save_TMMP_interval != 0:
        return

    tmmp_list = all_TMMPs if save_TMMP_num is None else [all_TMMPs[save_TMMP_num]]

    for tmmp in tmmp_list:
        tmmp_img = visualize_tmmp_matplotlib(
            tmmp['target'], tmmp['masked_target'], tmmp['mask'], tmmp['pred']
        )
        writer.add_image(f"TMMP/{tmmp['id']}", tmmp_img, epoch_num)


import torch
import numpy as np
import os
from PIL import Image
if __name__ == "__main__":
    # 创建伪数据
    H, W = 64, 64
    target = torch.rand(1, H, W)
    masked_input = target.clone()
    mask = torch.zeros_like(target)
    mask[:, 20:40, 20:40] = 1.0  # 中心打一个 mask
    masked_input[mask.bool()] = 0.0
    pred = target + 0.1 * torch.randn_like(target)

    # 可视化并保存图像
    image_tensor = visualize_tmmp_fast(target, masked_input, mask, pred)
    save_path = "./tmmp_example_output.png"

    # 转换为PIL图像并保存
    def tensor_to_pil(tensor_img):
        array = (tensor_img.detach().cpu().numpy() * 255).astype(np.uint8)
        array = np.transpose(array, (1, 2, 0))  # [H, W, C]
        return Image.fromarray(array)

    image_pil = tensor_to_pil(image_tensor)
    image_pil.save(save_path)

    print(f"Saved TMMP visualization to {save_path}")