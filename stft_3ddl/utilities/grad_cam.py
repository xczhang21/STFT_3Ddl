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
            output = self.model(input_tensor[0], input_tensor[1], input_tensor[2])
        
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


    def generate_cams(self, input_tensor, target_class=None):
        # 前向传播
        output = self.model(input_tensor)

        batch_size = input_tensor.size(0) # 获取批次大小
        cams = [] # 存储每个样本的 Grad-CAM

        # 如果未制定 target_class, 则默认使用最高分的类别
        if target_class is None:
            target_class = torch.argmax(output, dim=1).cpu().numpy() # [batch_size]
        
        # 对每个样本单独计算 Grad-CAM
        for i in range(batch_size):
            self.model.zero_grad()

            # 选取第 i 个样本的目标类别       
            class_score = output[i, target_class[i]]
            class_score.backward(retain_graph=True)

            # 获取当前样本的梯度和激活
            gradients = self.gradients[i].cpu().data.numpy()
            activations = self.activations[i].cpu().data.numpy()

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

            cams.append(cam)
        return np.array(cams) # 返回所有样本的CAM
    

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


