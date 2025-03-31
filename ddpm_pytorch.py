"""
DDPM (Denoising Diffusion Probabilistic Models) 的 PyTorch 实现
主要功能：实现扩散模型的前向过程（添加噪声）和反向过程（去噪）

核心思想：
1. 前向过程：逐步向图像添加高斯噪声，直到图像变成纯噪声
2. 反向过程：训练神经网络逐步去除噪声，从纯噪声恢复出图像
3. 使用马尔可夫链结构，每个时间步只依赖于前一个时间步
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm

"""
## Hyperparameters
"""

batch_size = 32
num_epochs = 100  # Just for the sake of demonstration
total_timesteps = 300
norm_groups = 8  # Number of groups used in GroupNorm layer
learning_rate = 2e-4

img_size = 64
img_channels = 3
clip_min = -1.0
clip_max = 1.0

first_conv_channels = 64
channel_multiplier = [1, 2, 4, 8]
widths = [first_conv_channels * mult for mult in channel_multiplier]
has_attention = [False, False, True, True]
num_res_blocks = 2  # Number of residual blocks

"""
## Dataset
"""

class CustomImageDataset(Dataset):
    """
    自定义数据集类，用于加载和处理图像数据
    继承自 torch.utils.data.Dataset，这是 PyTorch 中所有数据集的基类
    """
    def __init__(self, img_dir, transform=None):
        """
        初始化数据集
        Args:
            img_dir (str): 图像文件所在的根目录路径
            transform (callable, optional): 应用于图像的转换操作
        """
        self.img_dir = img_dir
        self.transform = transform
        self.images = [f for f in os.listdir(img_dir) if f.endswith('.png')]

    def __len__(self):
        """
        返回数据集中的样本数量
        Returns:
            int: 数据集中的图像数量
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        获取指定索引的图像样本
        Args:
            idx (int): 样本索引
        Returns:
            tuple: (图像张量, 图像文件名)
        """
        img_name = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image

def get_transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(min(img_size, img_size)),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # Scales to [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Scales to [-1, 1]
    ])

# Setup dataset and data loader
train_dataset = CustomImageDataset(
    img_dir='dataset/64x64/train/nolabel',
    transform=get_transform()
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    drop_last=True
)

"""
## Diffusion Schedules
"""

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

"""
## Gaussian diffusion utilities
We define the forward process and the reverse process
as a separate utility. This is similar to the TensorFlow implementation
but adapted to PyTorch.
"""

class GaussianDiffusion:
    """Gaussian diffusion utility.
    Args:
        beta_start: Start value of the scheduled variance
        beta_end: End value of the scheduled variance
        timesteps: Number of time steps in the forward process
    """

    def __init__(
        self,
        beta_start=1e-4,
        beta_end=0.02,
        timesteps=300,
        clip_min=-1.0,
        clip_max=1.0,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.device = device

        # Define the beta schedule
        self.betas = betas = cosine_beta_schedule(timesteps)
        
        self.num_timesteps = int(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        self.betas = betas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.alphas_cumprod_prev = alphas_cumprod_prev.to(device)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - alphas_cumprod).to(device)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod).to(device)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1).to(device)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_variance = posterior_variance.to(device)
        
        # Log calculation clipped because the posterior variance is 0 at the beginning
        # of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(
            torch.max(posterior_variance, torch.tensor(1e-20))
        ).to(device)
        
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).to(device)
        
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        ).to(device)

    def _extract(self, a, t, x_shape):
        """Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1] for broadcasting purposes.
        Args:
            a: Tensor to extract from
            t: Timestep for which the coefficients are to be extracted
            x_shape: Shape of the current batched samples
        """
        batch_size = x_shape[0]
        out = a.gather(-1, t.cpu()).to(t.device)
        return out.reshape(batch_size, *([1] * (len(x_shape) - 1)))

    def q_mean_variance(self, x_start, t):
        """Extracts the mean, and the variance at current timestep.
        Args:
            x_start: Initial sample (before the first diffusion step)
            t: Current timestep
        """
        x_start_shape = x_start.shape
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start_shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start_shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start_shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """Diffuse the data.
        Args:
            x_start: Initial sample (before the first diffusion step)
            t: Current timestep
            noise: Gaussian noise to be added at the current timestep
        Returns:
            Diffused samples at timestep `t`
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        x_start_shape = x_start.shape
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x_start_shape) * x_start
            + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start_shape) * noise
        )

    def predict_start_from_noise(self, x_t, t, noise):
        """
        Computes the starting point x_0 from noise
        """
        x_t_shape = x_t.shape
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t_shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t_shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        """Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        x_t_shape = x_t.shape
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t_shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t_shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t_shape)
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped, t, x_t_shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, pred_noise, x, t, clip_denoised=True):
        """
        The model outputs predicted noise, we convert it to the 
        predicted x_0 and then the mean and variance of p(x_{t-1} | x_t)
        """
        x_recon = self.predict_start_from_noise(x, t, pred_noise)
        
        if clip_denoised:
            x_recon = torch.clamp(x_recon, self.clip_min, self.clip_max)
            
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, pred_noise, x, t, clip_denoised=True):
        """
        Sample from the model p(x_{t-1} | x_t)
        """
        model_mean, _, model_log_variance = self.p_mean_variance(
            pred_noise, x, t, clip_denoised=clip_denoised
        )
        noise = torch.randn_like(x) if any(t > 0) else torch.zeros_like(x)
        
        # Compute x_{t-1} based on the mean, log variance, and noise
        return model_mean + noise * (0.5 * model_log_variance).exp()
    
    def p_sample_loop(self, model, shape):
        """
        Sample images by looping through the model
        """
        device = next(model.parameters()).device
        
        # Start with pure noise
        batch_size = shape[0]
        img = torch.randn(shape, device=device)
        imgs = []

        for i in reversed(range(0, self.num_timesteps)):
            img = self.p_sample(
                model(img, torch.full((batch_size,), i, device=device, dtype=torch.long)),
                img,
                torch.full((batch_size,), i, device=device, dtype=torch.long)
            )
            imgs.append(img.cpu().numpy())
        return imgs

    def sample(self, model, image_size, batch_size=16, channels=3):
        """
        Generate samples from the model
        """
        return self.p_sample_loop(
            model, 
            shape=(batch_size, channels, image_size, image_size)
        )

"""
## Neural Network Model Components
"""

class AttentionBlock(nn.Module):
    """Applies self-attention.
    Args:
        channels: Number of input/output channels
        groups: Number of groups for GroupNorm
    """
    def __init__(self, channels, groups=8):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(groups, channels)
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x):
        batch, channel, height, width = x.shape
        norm_x = self.norm(x)
        
        # Flatten spatial dimensions
        q = self.query(norm_x).view(batch, channel, -1)
        k = self.key(norm_x).view(batch, channel, -1)
        v = self.value(norm_x).view(batch, channel, -1)
        
        # Compute attention
        attn = torch.einsum('bci,bcj->bij', q, k) * self.scale
        attn = F.softmax(attn, dim=2)
        
        # Apply attention to value and reshape
        out = torch.einsum('bij,bcj->bci', attn, v)
        out = out.view(batch, channel, height, width)
        out = self.proj_out(out)
        
        return x + out

class TimeEmbedding(nn.Module):
    """
    时间嵌入层，将时间步转换为高维特征向量
    使用正弦和余弦函数进行位置编码
    """
    def __init__(self, dim):
        """
        初始化时间嵌入层
        Args:
            dim (int): 嵌入维度
        """
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        # Magic number 10000 from the transformer literature
        emb = math.log(10000) / (half_dim - 1)
        self.register_buffer('emb', torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb))

    def forward(self, timesteps):
        """
        前向传播
        Args:
            timesteps (torch.Tensor): 时间步
        Returns:
            torch.Tensor: 时间嵌入向量
        """
        emb = timesteps.float()[:, None] * self.emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class TimeMLP(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.time_embedding = TimeEmbedding(embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, timesteps):
        return self.net(self.time_embedding(timesteps))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
            
        self.time_mlp = nn.Linear(time_channels, out_channels)
    
    def forward(self, x, time_emb):
        h = self.act1(self.norm1(x))
        h = self.conv1(h)
        
        h = h + self.time_mlp(time_emb)[:, :, None, None]
        
        h = self.act2(self.norm2(h))
        h = self.conv2(h)
        
        return h + self.shortcut(x)

class DownSample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)

class UpSample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

class UNet(nn.Module):
    """
    UNet网络结构，用于预测噪声
    包含下采样路径和上采样路径，使用跳跃连接
    """
    def __init__(
        self,
        img_size,
        img_channels,
        widths,
        has_attention,
        num_res_blocks=2,
        norm_groups=8,
        activation_fn=F.silu
    ):
        super().__init__()
        self.img_size = img_size
        self.img_channels = img_channels
        
        block_out = widths[0]
        time_emb_dim = block_out * 4
        
        # Time embedding
        self.time_mlp = TimeMLP(block_out, block_out * 4, time_emb_dim)
        
        # Initial convolution to get from image channels to first width
        self.init_conv = nn.Conv2d(img_channels, block_out, 3, padding=1)
        
        # Down blocks
        self.downs = nn.ModuleList()
        for i, (width, use_attn) in enumerate(zip(widths, has_attention)):
            # Add residual blocks with attention
            res_blocks = []
            for j in range(num_res_blocks):
                res_blocks.append(ResidualBlock(block_out, width, time_emb_dim, norm_groups))
                block_out = width
                if use_attn:
                    res_blocks.append(AttentionBlock(width, norm_groups))
                    
            self.downs.append(nn.ModuleList(res_blocks))
            
            # Add a downsample layer except for the last block
            if i < len(widths) - 1:
                self.downs.append(DownSample(block_out))
        
        # Middle block (lowest resolution)
        mid_blocks = []
        mid_blocks.append(ResidualBlock(block_out, block_out, time_emb_dim, norm_groups))
        mid_blocks.append(AttentionBlock(block_out, norm_groups))
        mid_blocks.append(ResidualBlock(block_out, block_out, time_emb_dim, norm_groups))
        self.middle = nn.ModuleList(mid_blocks)
        
        # Up blocks
        self.ups = nn.ModuleList()
        reversed_widths = list(reversed(widths))
        reversed_attn = list(reversed(has_attention))
        
        for i, (width, use_attn) in enumerate(zip(reversed_widths, reversed_attn)):
            # Add residual blocks with attention and skip connections
            res_blocks = []
            for j in range(num_res_blocks + 1):
                # If not the last layer, add a skip connection from the corresponding down layer
                if i > 0 or j < num_res_blocks:
                    res_blocks.append(ResidualBlock(block_out * 2, width, time_emb_dim, norm_groups))
                else:
                    res_blocks.append(ResidualBlock(block_out, width, time_emb_dim, norm_groups))
                
                block_out = width
                if use_attn:
                    res_blocks.append(AttentionBlock(width, norm_groups))
                    
            self.ups.append(nn.ModuleList(res_blocks))
            
            # Add an upsample layer except for the last block
            if i < len(reversed_widths) - 1:
                self.ups.append(UpSample(block_out))
        
        # Final layers
        self.final_norm = nn.GroupNorm(norm_groups, block_out)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(block_out, img_channels, 3, padding=1)
        # 将卷积层的权重初始化为0
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)
        
    def forward(self, x, timesteps):
        """
        前向传播
        Args:
            x (torch.Tensor): 输入图像
            timesteps (torch.Tensor): 时间步
        Returns:
            torch.Tensor: 预测的噪声
        """
        # Time embedding
        t_emb = self.time_mlp(timesteps)
        
        # Initial convolution
        h = self.init_conv(x)
        
        # Store the outputs of each down layer for skip connections
        outs = [h]
        
        # Down blocks
        for i, layer in enumerate(self.downs):
            if isinstance(layer, DownSample):
                h = layer(h)
                outs.append(h)
            else:
                for block in layer:
                    if isinstance(block, ResidualBlock):
                        h = block(h, t_emb)
                    else:
                        h = block(h)
                outs.append(h)
        
        # Middle
        for block in self.middle:
            if isinstance(block, ResidualBlock):
                h = block(h, t_emb)
            else:
                h = block(h)
        
        # Up blocks with skip connections
        for i, layer in enumerate(self.ups):
            if isinstance(layer, UpSample):
                h = layer(h)
            else:
                # Get the skip connection from the down path
                skip_idx = len(outs) - i - 1
                if skip_idx >= 0:
                    h = torch.cat([h, outs[skip_idx]], dim=1)
                
                for block in layer:
                    if isinstance(block, ResidualBlock):
                        h = block(h, t_emb)
                    else:
                        h = block(h)
        
        # Final layers
        h = self.final_act(self.final_norm(h))
        h = self.final_conv(h)
        
        return h

"""
## Diffusion Model Trainer
"""

class DiffusionModel(nn.Module):
    def __init__(self, unet, timesteps, gdf_util):
        super().__init__()
        self.unet = unet
        self.gdf = gdf_util
        self.timesteps = timesteps
        
        # Optional: EMA setup
        self.ema_model = None
        self.ema_rate = 0.999
        
    def forward(self, x_0):
        # Sample noise
        noise = torch.randn_like(x_0)
        
        # Sample a random timestep for each image
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=x_0.device, dtype=torch.long)
        
        # Add noise to the images according to the timestep
        x_t = self.gdf.q_sample(x_0, t, noise)
        
        # Predict the noise using the UNet
        pred_noise = self.unet(x_t, t)
        
        # Return the loss (mean squared error between actual and predicted noise)
        return F.mse_loss(noise, pred_noise)
    
    def update_ema_model(self):
        if self.ema_model is None:
            # Initialize the EMA model as a copy of the current model
            self.ema_model = type(self.unet)(**vars(self.unet))
            self.ema_model.load_state_dict(self.unet.state_dict())
            self.ema_model.eval()  # Put EMA model in evaluation mode
            # Move to the same device as the main model
            self.ema_model.to(next(self.unet.parameters()).device)
        
        # Update the EMA model parameters
        with torch.no_grad():
            for param_ema, param_model in zip(self.ema_model.parameters(), self.unet.parameters()):
                param_ema.data = self.ema_rate * param_ema.data + (1 - self.ema_rate) * param_model.data
    
    def generate_images(self, num_images=16):
        # Use the EMA model for generation if available, otherwise use the main model
        model_to_use = self.ema_model if self.ema_model is not None else self.unet
        
        # Sample images using the diffusion process
        samples = self.gdf.sample(
            model_to_use, 
            image_size=self.unet.img_size, 
            batch_size=num_images, 
            channels=self.unet.img_channels
        )
        return samples
    
    def save_model(self, path):
        torch.save({
            'unet_state_dict': self.unet.state_dict(),
            'ema_state_dict': self.ema_model.state_dict() if self.ema_model is not None else None,
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.unet.load_state_dict(checkpoint['unet_state_dict'])
        if checkpoint['ema_state_dict'] is not None:
            if self.ema_model is None:
                self.ema_model = type(self.unet)(**vars(self.unet))
                self.ema_model.eval()
                self.ema_model.to(next(self.unet.parameters()).device)
            self.ema_model.load_state_dict(checkpoint['ema_state_dict'])

"""
## Training Loop
"""

def train(model, train_loader, optimizer, device, num_epochs):
    """
    训练DDPM模型
    Args:
        model (nn.Module): UNet模型
        train_loader (DataLoader): 数据加载器
        optimizer (torch.optim.Optimizer): 优化器
        device (torch.device): 计算设备
        num_epochs (int): 训练轮数
    """
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            real_images = batch.to(device)
            
            # Forward pass
            loss = model(real_images)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update EMA model
            model.update_ema_model()
            
            # Log progress
            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}")
        
        # Log epoch results
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Generate and save sample images
        if (epoch + 1) % 10 == 0 or epoch == 0:
            samples = model.generate_images(num_images=4)
            # Here you would save or display the generated images
            
        # Save model checkpoint
        if (epoch + 1) % 20 == 0 or epoch == num_epochs - 1:
            model.save_model(f"model_checkpoint_epoch_{epoch+1}.pt")

"""
## Main execution
"""

def main():
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create diffusion utility
    gdf = GaussianDiffusion(
        timesteps=total_timesteps,
        device=device
    )
    
    # Create UNet model
    unet = UNet(
        img_size=img_size,
        img_channels=img_channels,
        widths=widths,
        has_attention=has_attention,
        num_res_blocks=num_res_blocks,
        norm_groups=norm_groups
    ).to(device)
    
    # Create diffusion model trainer
    model = DiffusionModel(
        unet=unet,
        timesteps=total_timesteps,
        gdf_util=gdf
    ).to(device)
    
    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    train(model, train_loader, optimizer, device, num_epochs)

if __name__ == "__main__":
    main() 