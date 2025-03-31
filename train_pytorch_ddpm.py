"""
DDPM模型训练脚本
用于训练扩散模型
主要功能：
1. 数据加载和预处理
2. 模型训练
3. 检查点保存
4. 训练过程可视化
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from ddpm_pytorch import (
    UNet, GaussianDiffusion, DiffusionModel, get_transform, CustomImageDataset,
    img_size, img_channels, widths, has_attention, num_res_blocks, norm_groups, 
    batch_size, total_timesteps, learning_rate
)
from torch.utils.data import DataLoader

def denormalize(x):
    """将 [-1, 1] 范围的张量转换为 [0, 1] 范围以便可视化"""
    return (x + 1) / 2

def show_tensor_images(images, num_images=25, size=(3, 64, 64), save_path=None):
    """
    展示一批图像，并可选择保存
    """
    image_tensor = images[:num_images]
    image_grid = make_grid(denormalize(image_tensor), nrow=5)
    
    if save_path:
        # 如果提供了保存路径，则保存图像
        save_image(image_grid, save_path)
    
    # 可视化
    plt.figure(figsize=(8, 8))
    plt.imshow(image_grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.show()

def save_samples(epoch, model, sample_dir, num_images=16):
    """
    生成并保存样本图像
    """
    # 生成样本
    samples = model.generate_images(num_images)
    
    # 取最后一个时间步的样本（完全去噪后的样本）
    if samples:
        final_samples = torch.from_numpy(samples[-1])
        # 保存图像
        os.makedirs(sample_dir, exist_ok=True)
        save_path = os.path.join(sample_dir, f"samples_epoch_{epoch}.png")
        image_grid = make_grid(denormalize(final_samples), nrow=4)
        save_image(image_grid, save_path)
        print(f"保存了样本到 {save_path}")

def parse_args():
    """
    解析命令行参数
    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(description='Train DDPM model')
    parser.add_argument('--data_dir', type=str, required=True, help='训练数据目录')
    parser.add_argument('--output_dir', type=str, default='output', help='输出目录')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--log_interval', type=int, default=10, help='每多少批次打印日志')
    parser.add_argument('--sample_interval', type=int, default=10, help='每多少轮次生成样本')
    parser.add_argument('--save_interval', type=int, default=20, help='每多少轮次保存模型')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    return parser.parse_args()

def setup_training(args):
    """
    设置训练环境
    Args:
        args (argparse.Namespace): 命令行参数
    Returns:
        tuple: (模型, 优化器, 数据加载器, 调度器, 设备)
    """
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    sample_dir = os.path.join(args.output_dir, "samples")
    model_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 选择设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 准备数据集
    transform = get_transform()
    
    train_dataset = CustomImageDataset(
        img_dir=args.data_dir,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    
    print(f"数据集大小: {len(train_dataset)}张图像")
    
    # 创建扩散模型工具
    gdf = GaussianDiffusion(
        timesteps=total_timesteps,
        device=device
    )
    
    # 创建 UNet 模型
    unet = UNet(
        img_size=img_size,
        img_channels=img_channels,
        widths=widths,
        has_attention=has_attention,
        num_res_blocks=num_res_blocks,
        norm_groups=norm_groups
    ).to(device)
    
    # 创建 DDPM 模型
    model = DiffusionModel(
        unet=unet,
        timesteps=total_timesteps,
        gdf_util=gdf
    ).to(device)
    
    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 加载预训练模型（如果有）
    start_epoch = 0
    if args.resume:
        try:
            checkpoint_path = args.resume
            checkpoint = torch.load(checkpoint_path)
            model.load_model(checkpoint_path)
            start_epoch = checkpoint.get('epoch', 0)
            optimizer.load_state_dict(checkpoint.get('optimizer_state_dict', optimizer.state_dict()))
            print(f"从 {checkpoint_path} 恢复训练，起始轮次: {start_epoch}")
        except Exception as e:
            print(f"加载预训练模型失败: {e}")
    
    return model, optimizer, train_loader, gdf, device

def train_epoch(model, dataloader, scheduler, optimizer, device):
    """
    训练一个轮次
    Args:
        model (nn.Module): UNet模型
        dataloader (DataLoader): 数据加载器
        scheduler (GaussianDiffusion): 扩散调度器
        optimizer (torch.optim.Optimizer): 优化器
        device (torch.device): 计算设备
    Returns:
        float: 平均损失
    """
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # 移动数据到设备
        real_images = batch.to(device)
        
        # 前向传播
        loss = model(real_images)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 更新 EMA 模型
        model.update_ema_model()
        
        # 记录进度
        total_loss += loss.item()
        if batch_idx % args.log_interval == 0:
            print(f"轮次 {epoch+1}/{args.epochs}, 批次 {batch_idx}/{len(dataloader)}, "
                  f"损失: {loss.item():.4f}")
    
    return total_loss / len(dataloader)

def save_checkpoint(model, optimizer, epoch, save_dir):
    """
    保存检查点
    Args:
        model (nn.Module): UNet模型
        optimizer (torch.optim.Optimizer): 优化器
        epoch (int): 当前轮数
        save_dir (str): 保存目录
    """
    checkpoint_path = os.path.join(save_dir, f'model_checkpoint_epoch_{epoch+1}.pt')
    torch.save({
        'epoch': epoch + 1,
        'unet_state_dict': model.unet.state_dict(),
        'ema_state_dict': model.ema_model.state_dict() if model.ema_model is not None else None,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss,
    }, checkpoint_path)
    print(f"保存检查点到 {checkpoint_path}")

def plot_loss(losses):
    """
    绘制损失曲线
    Args:
        losses (list): 损失值列表
    """
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(args.output_dir, 'training_loss.png'))
    plt.close()

def main():
    """
    主函数，用于运行训练
    """
    # 解析参数
    args = parse_args()
    
    # 设置训练环境
    model, optimizer, dataloader, scheduler, device = setup_training(args)
    
    # 训练循环
    losses = []
    for epoch in range(args.epochs):
        # 训练一个轮次
        loss = train_epoch(model, dataloader, scheduler, optimizer, device)
        losses.append(loss)
        
        # 打印训练信息
        print(f'Epoch {epoch+1}/{args.epochs}, Loss: {loss:.4f}')
        
        # 生成并保存样本图像
        if (epoch + 1) % args.sample_interval == 0 or epoch == 0:
            model.eval()
            save_samples(epoch + 1, model, args.output_dir, num_images=16)
            model.train()
            
        # 保存检查点
        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
            save_checkpoint(model, optimizer, epoch, args.output_dir)
    
    # 绘制损失曲线
    plot_loss(losses)
    
    # 保存最终模型
    torch.save(model.unet.state_dict(), os.path.join(args.output_dir, 'final_model.pt'))

if __name__ == '__main__':
    main() 