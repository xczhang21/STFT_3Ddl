import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import models
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 注册前向钩子
        self.target_layer.register_forward_hook(self.save_activations)
        # 注册反向钩子
        # self.target_layer.register_backward_hook(self.save_gradients) # 会触发过时警告
        self.target_layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor, target_class=None, target_batch=0):
        # 为多输入模型改cam
        if isinstance(input_tensor, list):
            for i in range(len(input_tensor)):
                # 判断目标batch是否超过batch size
                assert target_batch < input_tensor[i].shape[0], f"target_batch:{target_batch} exceeds the range of th batch:{input_tensor.shape[0]}"
            # 前向传播
                output = self.model(*input_tensor)
            # if len(input_tensor) == 3:
            #     output = self.model(input_tensor[0], input_tensor[1], input_tensor[2])
            # elif len(input_tensor) == 2:
            #     output = self.model(input_tensor[0], input_tensor[1])

        
        else:                   
            # 判断目标batch是否超过batch size
            assert target_batch < input_tensor.shape[0], f"target_batch:{target_batch} exceeds the range of th batch:{input_tensor.shape[0]}"
            # 前向传播
            output = self.model(input_tensor)

        # batch_size = input_tensor.size(0) # 获取批次大小

        # 如果未制定 target_class, 则默认使用最高分的类别
        if target_class is None:
            target_class = torch.argmax(output, dim=1).cpu().numpy() # [batch_size]
        
        # 对batch是target_batch的样本单独计算 Grad-CAM
        self.model.zero_grad()

        # 选取第 i 个样本的目标类别       
        class_score = output[target_batch, target_class[target_batch]]
        class_score.backward(retain_graph=True)

        # 获取当前样本的梯度和激活
        gradients = self.gradients[target_batch].cpu().data.numpy()
        activations = self.activations[target_batch].cpu().data.numpy()

        # 权重计算(全局平均池化)
        weigths = np.mean(gradients, axis=(1, 2)) # 对每个通道计算权重

        # 生成CAM
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for j, w in enumerate(weigths):
            cam += w * activations[j]
        
        # ReLU 激活
        cam = np.maximum(cam, 0)

        # 归一化
        cam = cam - np.min(cam)
        cam = cam / np.max(cam) if np.max(cam) != 0 else cam

        return cam


    def generate_cam_multi(self, input_tensors, target_class=None, target_batch=0):
        assert isinstance(input_tensors, list), "使用generate_cam_multi时，input_tensor必须是[(B,C,H,W)，(B,C,H,W),...]"

        # 判断目标batch是否超过batch size
        assert all(target_batch < tensor.shape[0] for tensor in input_tensors),(
                f"target_batch:{target_batch} exceeds the batch size of at least one tensor: " 
                f"{[tensor.shape[0] for tensor in input_tensors]}"
        )

        # 前向传播
        output = self.model(*input_tensors)

        # batch_size = input_tensors[0].size(0) # 获取批次大小

        # 如果未制定 target_class, 则默认使用最高分的类别
        if target_class is None:
            target_class = torch.argmax(output, dim=1).cpu().numpy() # [batch_size]
        
        # 对batch是target_batch的样本单独计算 Grad-CAM
        self.model.zero_grad()

        # 选取第 i 个样本的目标类别
        class_score = output[target_batch, target_class[target_batch]]
        class_score.backward(retain_graph=True)

        # 获取当前样本的梯度和激活
        gradients = self.gradients[target_batch].cpu().data.numpy()
        activations = self.activations[target_batch].cpu().data.numpy()

        # 权重计算(全局平均池化)
        weigths = np.mean(gradients, axis=(1, 2)) # 对每个通道计算权重

        # 生成CAM
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for j, w in enumerate(weigths):
            cam += w * activations[j]
        
        # RelU 激活
        cam = np.maximum(cam, 0)

        # 归一化
        cam = cam - np.min(cam)
        cam = cam / np.max(cam) if np.max(cam) != 0 else cam

        return cam

    

# 可视化热力图
def visualize_cam(cam, image):
    # 将cam上下颠倒，因为生成的CAM图上下是错位的
    cam = np.flipud(cam)

    # 如果 image 是 Pytorch 张量，调整为numpy格式
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy() # 转换为(H, W, C)
    
    # 调整热力图大小
    cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR) # (宽，高)

    # 应用颜色映射
    cam_colored = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)

    # 确保 cam_colored 和 image 的形状相同
    assert cam_colored.shape[:2] == image.shape[:2], f"cam_colored.shape:{cam_colored.shape} not equal to image.shape{image.shape}"

    # 叠加图像
    superimposed_image = np.float32(cam_colored) / 255 + np.float32(image) / 255
    superimposed_image = superimposed_image / np.max(superimposed_image)

    return superimposed_image

# 2个输入的可视化热力图
def visualize_cam_multi(cam, images, single_image=False):
    """
    可视化多个图像融合后的 Grad-CAM 结果。

    :param cam: np.ndarray, 2 张图像的 Grad-CAM 融合特征图 (H, W)
    :param images: list 或 torch.Tensor, 包含两张图像 (2, C, H, W) 或 [img1, img2]
    :single_image: 只用images[0]进行融合，因为多尺度任务时，多个image的尺度不同，但是表示的内容相同
    :return: superimposed_image, 叠加的 Grad-CAM 可视化图
    """
    # 将cam上下颠倒，因为生成的CAM图上下时错位的
    cam = np.flipud(cam)

    # 处理输入图像
    if isinstance(images, torch.Tensor):
        assert images.ndim == 4, "输入 images 必须是形状为(N,C,H,W)的torch.Tensor"
        images = [img.permute(1,2,0).numpy() for img in images] # 转换为(H,W,C)格式
    elif isinstance(images, list) and all(isinstance(img, (np.ndarray, torch.Tensor)) for img in images):
        images = [img if isinstance(img, np.ndarray) else img.permute(1,2,0).numpy() for img in images]
    else:
        raise ValueError("iamges 应该包含多张图像的list或形状为(N,C,H,W)的torch.Tensor")
    
    # 确保至少有一张图像
    assert len(images) > 0, "images 不能为空"

    """
    # 处理输入图像
    if isinstance(images, torch.Tensor):
        assert images.shape[0] == 2, "输入 images 必须包含2张图像"
        images = [img.permute(1, 2, 0).numpy() for img in images] # 转换为(H,W,C)格式

    elif isinstance(images, list) and len(images) == 2:
        images = [img if isinstance(img, np.ndarray) else img.permute(1, 2, 0).numpy() for img in images]
    
    else:
        raise ValueError("images 应该是包含两张图的 list 或形状为(2, C, H, W) 的 torch.Tensor")
    """

    # 获得最终输出图像大小
    h, w = images[0].shape[:2]

    # 调整 Grad-CAM 大小以匹配目标图像
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)

    # 应用颜色映射
    cam_colored = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)

    # 确保 Grad-CAM 形状与目标图像匹配
    assert cam_colored.shape[:2] == (h, w), f"cam_colored.shape {cam_colored.shape} != traget size {(h,w)}"

    # 计算所有图像的平均值（融合）或单图
    if single_image:
        merged_image = images[0]
    else:
        merged_image = np.mean([np.float32(img) for img in images], axis=0)

    """
    # 计算两个图像的平均值（融合）
    merged_image = (np.float32(images[0]) / 2 + np.float32(images[1]) / 2)
    """
    # 叠加 Grad-CAM
    superimposed_image = np.float32(cam_colored) / 255 + merged_image / 255
    superimposed_image = superimposed_image / np.max(superimposed_image)

    return superimposed_image

def save_grad_cam(epoch_num, val_spectrums, model,target_layer, writer, save_CAM_interval=5, save_CAM_num=None):
    """
    保存指定或所有val_spectrums的Grad-CAM结果到TensorBoard。
    
    参数:
    - epoch_num: 当前训练轮数。
    - val_spectrums: 验证集频谱数据。
    - model: 用于生成Grad-CAM的模型。
    - writer: TensorBoard的SummaryWriter。
    - save_CAM_interval: Grad-CAM生成周期。
    - save_CAM_num: 指定val_spectrums中的某一张进行Grad-CAM计算（默认为None，即全部执行）。
    """
    spectrum_list = val_spectrums if save_CAM_num is None else [val_spectrums[save_CAM_num]]
    
    if epoch_num == 0:
        for spectrum_data in spectrum_list:
            spectrum_id = spectrum_data['id']
            if 'spectrum' in spectrum_data.keys():
                spectrum = spectrum_data['spectrum']
            elif 'image' in spectrum_data.keys():
                spectrum = spectrum_data['image']
            else:
                assert False, "数据集中的key中既没有spectrum也没有image"
            writer.add_image(f'grad_cam-{spectrum_id}', torch.flip(spectrum, dims=[1]), 0, dataformats='CHW')

    if (epoch_num + 1) % save_CAM_interval == 0:
        for spectrum_data in spectrum_list:
            spectrum_id = spectrum_data['id']
            if 'spectrum' in spectrum_data.keys():
                spectrum = spectrum_data['spectrum']
            elif 'image' in spectrum_data.keys():
                spectrum = spectrum_data['image']
            else:
                assert False, "数据集中的key中既没有spectrum也没有image"

            grad_cam = GradCAM(model, target_layer=target_layer)
            cam = grad_cam.generate_cam(spectrum.unsqueeze(0).cuda(), target_class=None, target_batch=0)
            
            if spectrum.shape[0] == 0:
                superimposed_image = visualize_cam(cam, spectrum)
            else:
                superimposed_image = visualize_cam(cam, spectrum[:1])
            
            writer.add_image(f'grad_cam-{spectrum_id}', superimposed_image, epoch_num, dataformats='HWC')


def save_grad_cam_multi(epoch_num, val_spectrums, model, target_layers, writer, save_CAM_interval=5, save_CAM_num=None, single_image=False):
    """
    相较于save_grad_cam，save_grad_cams实现了多输入的grad_cam
    """

    # 确保val_spectrums中的key是spectrums不是spectrum
    assert "spectrums" in val_spectrums[0].keys(), "'spectrums' not in keys()"


    # # 先只考虑两个输入的grad_cam
    # assert len(val_spectrums[0]['spectrums']) == 2, "输入的特征数不为2"

    # # target_layers得有3个,分别是特征1的、特征2的、特征1融合特征2的
    # assert len(target_layers) == 3, "输入的target_layer不为3"

    spectrum_list = val_spectrums if save_CAM_num is None else [val_spectrums[save_CAM_num]]

    if epoch_num == 0:
        for spectrum_data in spectrum_list:
            
            inputs_num = len(spectrum_data['spectrums'])
            # target_layers的长度应该比模型输入的数量多1
            assert inputs_num+1 == len(target_layers), "模型的输入与grad CAM的目标层数不匹配"
            
            spectrum_id = spectrum_data['id']
            spectrums = spectrum_data['spectrums']
            for input_id in range(inputs_num):
                writer.add_image(f"grad_cam-{spectrum_id}/input{str(input_id+1)}", torch.flip(spectrums[input_id], dims=[1]), 0, dataformats='CHW')
            for input_id in range(inputs_num):
                writer.add_image(f"grad_cam-{spectrum_id}/fusion", torch.flip(spectrums[input_id], dims=[1]), input_id, dataformats='CHW')
            """
            下面是2输入的情况
            # writer.add_image(f'grad_cam-{spectrum_id}/input1', torch.flip(spectrums[0], dims=[1]), 0, dataformats='CHW')
            # writer.add_image(f'grad_cam-{spectrum_id}/input2', torch.flip(spectrums[1], dims=[1]), 0, dataformats='CHW')
            # writer.add_image(f'grad_cam-{spectrum_id}/fusion', torch.flip(spectrums[0], dims=[1]), 0, dataformats='CHW')
            # writer.add_image(f'grad_cam-{spectrum_id}/fusion', torch.flip(spectrums[1], dims=[1]), 1, dataformats='CHW')
            """
    if (epoch_num + 1) % save_CAM_interval == 0:
        for spectrum_data in spectrum_list:

            inputs_num = len(spectrum_data['spectrums'])
            # target_layers的长度应该比模型输入的数量多1
            assert inputs_num+1 == len(target_layers), "模型的输入与grad CAM的目标层数不匹配"
            
            spectrum_id = spectrum_data['id']
            spectrums = spectrum_data['spectrums']

            for input_id in range(inputs_num):
                # 模型输入_input_id的grad_cam
                grad_cam = GradCAM(model, target_layer=target_layers[input_id])
                cam = grad_cam.generate_cam_multi([spectrum.unsqueeze(0).cuda() for spectrum in spectrums], target_class=None)
                # 模型输入_input_id的superimposed_image
                if spectrums[input_id].shape[0] == 0:
                    superimposed_image = visualize_cam(cam, spectrums[input_id])
                else:
                    superimposed_image = visualize_cam(cam, spectrums[input_id][:1])
                writer.add_image(f"grad_cam-{spectrum_id}/input{str(input_id+1)}", superimposed_image, epoch_num, dataformats='HWC')
            
            # 融合的grad_cam
            grad_cam_fusion = GradCAM(model, target_layer=target_layers[-1])
            cam_fusion = grad_cam_fusion.generate_cam_multi([spectrum.unsqueeze(0).cuda() for spectrum in spectrums], target_class=None)

            # 融合的superimposed_image
            if all(spectrum.shape[0] == 0 for spectrum in spectrums):
                superimposed_image_fusion = visualize_cam_multi(cam_fusion, spectrums, single_image=single_image)
            elif all(spectrum.shape[0] != 0 for spectrum in spectrums):
                superimposed_image_fusion = visualize_cam_multi(cam_fusion, [spectrum[:1] for spectrum in spectrums], single_image=single_image)
            else:
                raise ValueError("spectrums的格式错误")
            writer.add_image(f"grad_cam-{spectrum_id}/fusion", superimposed_image_fusion, epoch_num+inputs_num, dataformats='HWC')

            """
            下面是2输入的情况
            # 输入1的grad_cam
            grad_cam_1 = GradCAM(model, target_layer=target_layers[0])
            cam_1 = grad_cam_1.generate_cam_multi([spectrums[0].unsqueeze(0).cuda(), spectrums[1].unsqueeze(0).cuda()], target_class=None)

            # 输入2的grad_cam
            grad_cam_2 = GradCAM(model, target_layer=target_layers[1])
            cam_2 = grad_cam_2.generate_cam_multi([spectrums[0].unsqueeze(0).cuda(), spectrums[1].unsqueeze(0).cuda()], target_class=None)

            # 融合的grad_cam
            grad_cam_3 = GradCAM(model, target_layer=target_layers[2])
            cam_3 = grad_cam_3.generate_cam_multi([spectrums[0].unsqueeze(0).cuda(), spectrums[1].unsqueeze(0).cuda()], target_class=None)

            # 输入1的superimposed_image
            if spectrums[0].shape[0] == 0:
                superimposed_image_1 = visualize_cam(cam_1, spectrums[0])
            else:
                superimposed_image_1 = visualize_cam(cam_1, spectrums[0][:1])
            writer.add_image(f'grad_cam-{spectrum_id}/input1', superimposed_image_1, epoch_num, dataformats='HWC')

            # 输入2的superimposed_image
            if spectrums[1].shape[0] == 0:
                superimposed_image_2 = visualize_cam(cam_2, spectrums[1])
            else:
                superimposed_image_2 = visualize_cam(cam_2, spectrums[1][:1])
            writer.add_image(f'grad_cam-{spectrum_id}/input2', superimposed_image_2, epoch_num, dataformats='HWC')

            # 融合的superimposed_image
            if spectrums[0].shape[0] == 0 and spectrums[1].shape[0] == 0:
                superimposed_image_3 = visualize_cam_multi(cam_3, spectrums)
            elif spectrums[0].shape[0] !=0 and spectrums[1].shape[0] != 0:
                superimposed_image_3 = visualize_cam_multi(cam_3, [spectrums[0][:1], spectrums[1][:1]])
            else:
                raise ValueError("spectrums的格式错误")
            writer.add_image(f'grad_cam-{spectrum_id}/fusion', superimposed_image_3, epoch_num+1, dataformats='HWC')
            """            
            
            


# 模块测试
if __name__ == '__main__':
    import torch.nn as nn

    # 简单的卷积神经网络
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
            self.fc = nn.Linear(32 * 32 * 32, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    # 模型实例化
    model = SimpleCNN()

    # 随机输入数据
    input_tensor = torch.randn(1, 1, 32, 32)
    target_batch = 0

    # 测试 GradCAM
    grad_cam = GradCAM(model, target_layer=model.conv2)
    # cams = grad_cam.generate_cams(input_tensor)
    cam = grad_cam.generate_cam(input_tensor, target_class=None, target_batch=target_batch)
    
    # 生成RGB热力图
    superimposed_image = visualize_cam(cam, input_tensor[target_batch])
    print("input shape:", input_tensor.shape)

    plt.imshow(superimposed_image)
    plt.savefig('./utilities/cam_test.png' )
    

    # print("CAM shape:", cam.shape)
    # print("CAM Min:", cam.min())
    # print("CAM MAx:", cam.max())


