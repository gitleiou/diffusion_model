"""
DDPM模型测试脚本
用于测试扩散步骤和生成样本
主要功能：
1. 加载预训练模型
2. 测试扩散过程
3. 生成样本图像
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from ddpm_pytorch import UNet, DiffusionScheduler
import os

def test_diffusion_step(model, scheduler, image_path, device):
    """
    测试单个扩散步骤
    Args:
        model (nn.Module): 预训练的UNet模型
        scheduler (DiffusionScheduler): 扩散调度器
        image_path (str): 测试图像的路径
        device (torch.device): 计算设备
    """
    # 加载并预处理图像
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    # 选择时间步
    t = torch.tensor([500], device=device)
    
    # 添加噪声
    noisy_image, noise = scheduler.add_noise(image, t)
    
    # 预测噪声
    with torch.no_grad():
        predicted_noise = model(noisy_image, t)
    
    # 可视化结果
    plt.figure(figsize=(15, 5))
    
    # 原始图像
    plt.subplot(131)
    plt.imshow(image[0].cpu().permute(1, 2, 0).numpy())
    plt.title('Original Image')
    
    # 带噪声图像
    plt.subplot(132)
    plt.imshow(noisy_image[0].cpu().permute(1, 2, 0).numpy())
    plt.title('Noisy Image')
    
    # 预测的噪声
    plt.subplot(133)
    plt.imshow(predicted_noise[0].cpu().permute(1, 2, 0).numpy())
    plt.title('Predicted Noise')
    
    plt.savefig('diffusion_step_test.png')
    plt.close()

def generate_samples(model, scheduler, num_samples, device):
    """
    生成样本图像
    Args:
        model (nn.Module): 预训练的UNet模型
        scheduler (DiffusionScheduler): 扩散调度器
        num_samples (int): 要生成的样本数量
        device (torch.device): 计算设备
    """
    # 创建输出目录
    os.makedirs('generated_samples', exist_ok=True)
    
    # 生成样本
    with torch.no_grad():
        samples = scheduler.sample(model, num_samples, device)
    
    # 保存生成的样本
    for i, sample in enumerate(samples):
        # 将图像转换到CPU并调整值范围
        sample = sample.cpu().permute(1, 2, 0).numpy()
        sample = (sample + 1) / 2.0  # 从[-1,1]转换到[0,1]
        
        # 保存图像
        plt.imsave(f'generated_samples/sample_{i}.png', sample)

def main():
    """
    主函数，用于运行测试
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化模型和调度器
    model = UNet().to(device)
    scheduler = DiffusionScheduler()
    
    # 加载预训练模型
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    
    # 测试扩散步骤
    test_diffusion_step(model, scheduler, 'test_image.png', device)
    
    # 生成样本
    generate_samples(model, scheduler, 4, device)

if __name__ == '__main__':
    main() 