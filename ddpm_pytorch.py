"""
DDPM (Denoising Diffusion Probabilistic Models) 的 PyTorch 实现
这是一个用于图像生成的深度学习模型，其核心思想是：
1. 前向过程：逐步向图像添加高斯噪声，直到图像变成纯噪声
2. 反向过程：训练神经网络逐步去除噪声，从纯噪声恢复出图像
3. 使用马尔可夫链结构，每个时间步只依赖于前一个时间步

主要组件说明：
1. 数据集类 (CustomImageDataset)：用于加载和处理图像数据
2. 扩散过程类 (GaussianDiffusion)：实现添加和去除噪声的数学过程
3. 神经网络组件：
   - UNet：主要的网络结构，用于预测噪声
   - ResidualBlock：残差块，用于特征提取
   - AttentionBlock：注意力机制，用于关注重要特征
   - TimeEmbedding：时间嵌入，将时间步转换为特征向量
4. 训练循环：实现模型的训练过程

使用的主要库：
- torch：PyTorch深度学习框架
- torch.nn：神经网络模块
- torch.utils.data：数据加载工具
- torchvision：计算机视觉工具
- PIL：图像处理库
- numpy：数值计算库
- matplotlib：绘图库
"""

# 导入必要的库
import math  # 数学计算
import numpy as np  # 数值计算库
import matplotlib.pyplot as plt  # 绘图库
import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 神经网络函数
from torch.utils.data import Dataset, DataLoader  # 数据加载工具
from torchvision import transforms  # 图像转换工具
from PIL import Image  # 图像处理
import os  # 操作系统接口
from tqdm import tqdm  # 进度条显示

"""
## 超参数设置
这些参数控制模型的结构和训练过程
"""
batch_size = 32  # 每批处理的图像数量
num_epochs = 100  # 训练的总轮数
total_timesteps = 300  # 扩散过程的时间步数
norm_groups = 8  # GroupNorm层的分组数
learning_rate = 2e-4  # 学习率，控制参数更新的步长

# 图像相关参数
img_size = 64  # 输入图像的大小（64x64像素）
img_channels = 3  # 图像的通道数（RGB三通道）
clip_min = -1.0  # 像素值的最小值
clip_max = 1.0  # 像素值的最大值

# 网络结构参数
first_conv_channels = 64  # 第一个卷积层的通道数
channel_multiplier = [1, 2, 4, 8]  # 通道数的倍增因子
widths = [first_conv_channels * mult for mult in channel_multiplier]  # 计算每一层的通道数
has_attention = [False, False, True, True]  # 是否在对应层使用注意力机制
num_res_blocks = 2  # 每个分辨率下的残差块数量

"""
## 数据集类
用于加载和处理训练数据
"""
class CustomImageDataset(Dataset):
    """
    自定义数据集类，用于加载和处理图像数据
    继承自 torch.utils.data.Dataset，这是 PyTorch 中所有数据集的基类
    
    主要功能：
    1. 加载指定目录下的图像文件
    2. 对图像进行预处理和转换
    3. 提供数据访问接口
    """
    def __init__(self, img_dir, transform=None):
        """
        初始化数据集
        Args:
            img_dir (str): 图像文件所在的根目录路径
            transform (callable, optional): 应用于图像的转换操作
        """
        self.img_dir = img_dir  # 图像目录
        self.transform = transform  # 图像转换函数
        # 获取目录下所有的PNG图像文件
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
        # 构建完整的图像文件路径
        # os.path.join() 会自动修改、添加、去除分隔符，保证当前os正确的路径 (/  \)
        img_name = os.path.join(self.img_dir, self.images[idx])
        # 打开图像并转换为RGB模式。好的编程习惯，可以防止潜在的格式问题
        image = Image.open(img_name).convert('RGB')
        
        # 如果定义了转换函数，则应用转换
        if self.transform:
            image = self.transform(image)
            
        return image

def get_transform():
    """
    定义图像转换操作
    返回一个转换操作序列，包括：
    1. 随机水平翻转
    2. 中心裁剪
    3. 调整大小
    4. 转换为张量
    5. 标准化
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.CenterCrop(min(img_size, img_size)),  # 中心裁剪
        transforms.Resize((img_size, img_size)),  # 调整大小
        transforms.ToTensor(),  # 转换为张量，并缩放到[0, 1]，从(H, W, C)到(C, H, W)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化到[-1, 1]normalized = (original - mean) / std
    ])

# 设置数据集和数据加载器
train_dataset = CustomImageDataset(
    img_dir='dataset/64x64/train/nolabel',  # 训练数据目录
    transform=get_transform()  # 应用图像转换
)

# 创建数据加载器，用于批量加载数据
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,  # 每批加载的图像数量  32
    shuffle=True,  # 随机打乱数据
    num_workers=4,  # 数据加载的线程数
    drop_last=True  # 丢弃最后不完整的批次
)

"""
## 扩散过程调度
定义如何随时间步增加噪声的数学函数
"""

def cosine_beta_schedule(timesteps, s=0.008):  #timesteps=300 readme.md
    """
    余弦调度函数，用于计算每个时间步的噪声水平
    基于论文 https://arxiv.org/abs/2102.09672 提出的方法
    
    Args:
        timesteps (int): 总时间步数
        s (float): 控制调度曲线形状的参数
    Returns:
        torch.Tensor: 每个时间步的beta值（噪声水平）
    """
    steps = timesteps + 1 #301
    # 创建从0到timesteps的线性序列 
    # torch.linspace(start, end, steps) [start，end]，均匀分成steps-1份
    # x = torch.linspace(0, 4, 5)    # 结果: tensor([0., 1., 2., 3., 4.])
    x = torch.linspace(0, timesteps, steps) # 0,1,2,...,300
    # 计算累积乘积的alpha值
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    # 归一化
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    # 计算beta值 切片 [start:end] 包含start，不包含end
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1]) 
    # 限制beta值的范围
    return torch.clamp(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """
    线性调度函数，用于计算每个时间步的噪声水平
    使用线性插值在beta_start和beta_end之间生成值
    
    Args:
        timesteps (int): 总时间步数
        beta_start (float): 起始噪声水平
        beta_end (float): 结束噪声水平
    Returns:
        torch.Tensor: 每个时间步的beta值
    """
    # [0.0001, 0.02]分成timesteps-1份
    return torch.linspace(beta_start, beta_end, timesteps) 

"""
## 高斯扩散工具类
实现扩散模型的前向过程（添加噪声）和反向过程（去噪）
"""

class GaussianDiffusion:
    """
    高斯扩散工具类，实现扩散模型的核心数学过程
    
    主要功能：
    1. 前向过程：逐步向图像添加高斯噪声
    2. 反向过程：从噪声中恢复图像
    3. 计算各种扩散过程所需的系数
    
    Args:
        beta_start (float): 起始噪声水平
        beta_end (float): 结束噪声水平
        timesteps (int): 总时间步数
        clip_min (float): 像素值的最小值
        clip_max (float): 像素值的最大值
        device (str): 计算设备（'cuda'或'cpu'）
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
        # 保存基本参数
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.device = device

        # 定义beta调度
        self.betas = betas = cosine_beta_schedule(timesteps)
        self.num_timesteps = int(timesteps) #300

        # 计算扩散过程所需的各种系数
        alphas = 1.0 - betas  # alpha值
        alphas_cumprod = torch.cumprod(alphas, dim=0)  # alpha的累积乘积
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])  # 前一个时间步的累积乘积

        # 将系数移动到指定设备（GPU或CPU）
        self.betas = betas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.alphas_cumprod_prev = alphas_cumprod_prev.to(device)

        # 计算扩散过程q(x_t | x_{t-1})所需的其他系数
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - alphas_cumprod).to(device)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod).to(device)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1).to(device)

        # 计算后验分布q(x_{t-1} | x_t, x_0)所需的系数
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_variance = posterior_variance.to(device)
        
        # 计算后验分布的对数方差（添加小值避免数值不稳定）
        self.posterior_log_variance_clipped = torch.log(
            torch.max(posterior_variance, torch.tensor(1e-20))
        ).to(device)
        
        # 计算后验均值的系数
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).to(device)
        
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        ).to(device)

    def _extract(self, a, t, x_shape):
        """
        从张量中提取指定时间步的系数，并重塑为适合广播的形状
        
        Args:
            a (torch.Tensor): 要提取系数的张量
            t (torch.Tensor): 时间步
            x_shape (tuple): 输入张量的形状
        Returns:
            torch.Tensor: 重塑后的系数张量
        """
        batch_size = x_shape[0]
        out = a.gather(-1, t.cpu()).to(t.device)
        return out.reshape(batch_size, *([1] * (len(x_shape) - 1)))

    def q_mean_variance(self, x_start, t):
        """
        计算扩散过程q(x_t | x_{t-1})的均值和方差
        
        Args:
            x_start (torch.Tensor): 初始样本
            t (torch.Tensor): 当前时间步
        Returns:
            tuple: (均值, 方差, 对数方差)
        """
        x_start_shape = x_start.shape
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start_shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start_shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start_shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        执行扩散过程，向图像添加噪声
        
        Args:
            x_start (torch.Tensor): 初始样本
            t (torch.Tensor): 当前时间步
            noise (torch.Tensor, optional): 要添加的噪声
        Returns:
            torch.Tensor: 添加噪声后的样本
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
        从噪声预测初始图像x_0
        
        Args:
            x_t (torch.Tensor): 当前时间步的噪声图像
            t (torch.Tensor): 当前时间步
            noise (torch.Tensor): 预测的噪声
        Returns:
            torch.Tensor: 预测的初始图像
        """
        x_t_shape = x_t.shape
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t_shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t_shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        """
        计算后验分布q(x_{t-1} | x_t, x_0)的均值和方差
        
        Args:
            x_start (torch.Tensor): 初始样本
            x_t (torch.Tensor): 当前时间步的样本
            t (torch.Tensor): 当前时间步
        Returns:
            tuple: (后验均值, 后验方差, 后验对数方差)
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
        计算模型预测的均值和方差
        
        Args:
            pred_noise (torch.Tensor): 模型预测的噪声
            x (torch.Tensor): 当前样本
            t (torch.Tensor): 当前时间步
            clip_denoised (bool): 是否裁剪去噪后的值
        Returns:
            tuple: (模型均值, 后验方差, 后验对数方差)
        """
        # 从预测的噪声计算重建的x_0
        x_recon = self.predict_start_from_noise(x, t, pred_noise)
        
        # 如果需要，裁剪重建的值
        if clip_denoised:
            x_recon = torch.clamp(x_recon, self.clip_min, self.clip_max)
            
        # 计算后验分布的参数
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, pred_noise, x, t, clip_denoised=True):
        """
        从模型p(x_{t-1} | x_t)采样
        
        Args:
            pred_noise (torch.Tensor): 模型预测的噪声
            x (torch.Tensor): 当前样本
            t (torch.Tensor): 当前时间步
            clip_denoised (bool): 是否裁剪去噪后的值
        Returns:
            torch.Tensor: 采样得到的样本
        """
        # 计算均值和方差
        model_mean, _, model_log_variance = self.p_mean_variance(
            pred_noise, x, t, clip_denoised=clip_denoised
        )
        # 生成随机噪声
        noise = torch.randn_like(x) if any(t > 0) else torch.zeros_like(x)
        
        # 基于均值和方差采样
        return model_mean + noise * (0.5 * model_log_variance).exp()
    
    def p_sample_loop(self, model, shape):
        """
        通过循环模型生成图像
        
        Args:
            model (nn.Module): 训练好的模型
            shape (tuple): 生成图像的形状
        Returns:
            list: 生成过程中的所有中间结果
        """
        device = next(model.parameters()).device
        
        # 从纯噪声开始
        batch_size = shape[0]
        img = torch.randn(shape, device=device)
        imgs = []

        # 逐步去噪
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
        从模型生成样本图像
        
        Args:
            model (nn.Module): 训练好的模型
            image_size (int): 生成图像的大小
            batch_size (int): 每批生成的图像数量
            channels (int): 图像的通道数
        Returns:
            list: 生成过程中的所有中间结果
        """
        return self.p_sample_loop(
            model, 
            shape=(batch_size, channels, image_size, image_size)
        )

"""
## 神经网络模型组件
实现DDPM模型所需的各个神经网络层和模块
"""

class AttentionBlock(nn.Module):
    """
    注意力机制模块，用于关注输入特征中的重要部分
    
    工作原理：
    1. 将输入特征转换为查询(Q)、键(K)和值(V)矩阵
    2. 计算注意力权重
    3. 将注意力权重应用到值矩阵上
    
    Args:
        channels (int): 输入/输出通道数
        groups (int): GroupNorm层的分组数
    """
    def __init__(self, channels, groups=8):
        super().__init__()
        self.channels = channels
        # 归一化层
        self.norm = nn.GroupNorm(groups, channels)
        # 查询、键、值转换层
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        # 输出投影层
        self.proj_out = nn.Conv2d(channels, channels, 1)
        # 缩放因子
        self.scale = channels ** -0.5

    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入特征图
        Returns:
            torch.Tensor: 添加注意力后的特征图
        """
        batch, channel, height, width = x.shape
        # 归一化输入
        norm_x = self.norm(x)
        
        # 将空间维度展平
        q = self.query(norm_x).view(batch, channel, -1)
        k = self.key(norm_x).view(batch, channel, -1)
        v = self.value(norm_x).view(batch, channel, -1)
        
        # 计算注意力权重
        attn = torch.einsum('bci,bcj->bij', q, k) * self.scale
        attn = F.softmax(attn, dim=2)
        
        # 应用注意力到值并重塑
        out = torch.einsum('bij,bcj->bci', attn, v)
        out = out.view(batch, channel, height, width)
        out = self.proj_out(out)
        
        # 残差连接
        return x + out

class TimeEmbedding(nn.Module):
    """
    时间嵌入层，将时间步转换为高维特征向量
    
    使用正弦和余弦函数进行位置编码，类似于Transformer中的位置编码
    
    Args:
        dim (int): 嵌入维度
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        # 从Transformer文献中使用的魔法数字10000
        emb = math.log(10000) / (half_dim - 1)
        # 注册缓冲区，用于存储位置编码
        self.register_buffer('emb', torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb))

    def forward(self, timesteps):
        """
        前向传播
        
        Args:
            timesteps (torch.Tensor): 时间步
        Returns:
            torch.Tensor: 时间嵌入向量
        """
        # 计算位置编码
        emb = timesteps.float()[:, None] * self.emb[None, :]
        # 连接正弦和余弦编码
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class TimeMLP(nn.Module):
    """
    时间MLP网络，用于处理时间嵌入
    
    将时间嵌入转换为网络可用的特征
    
    Args:
        embedding_dim (int): 嵌入维度
        hidden_dim (int): 隐藏层维度
        output_dim (int): 输出维度
    """
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.time_embedding = TimeEmbedding(embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),  # SiLU激活函数
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, timesteps):
        """
        前向传播
        
        Args:
            timesteps (torch.Tensor): 时间步
        Returns:
            torch.Tensor: 处理后的时间特征
        """
        return self.net(self.time_embedding(timesteps))

class ResidualBlock(nn.Module):
    """
    残差块，用于特征提取
    
    包含两个卷积层，每个卷积层前都有归一化和激活函数
    使用残差连接（跳跃连接）来帮助梯度流动
    
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        time_channels (int): 时间嵌入的通道数
        groups (int): GroupNorm层的分组数
    """
    def __init__(self, in_channels, out_channels, time_channels, groups=8):
        super().__init__()
        # 第一个卷积块
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        # 第二个卷积块
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # 如果输入输出通道数不同，添加1x1卷积作为捷径连接
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
            
        # 时间嵌入处理层
        self.time_mlp = nn.Linear(time_channels, out_channels)
        
        # 初始化权重为0
        nn.init.zeros_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        if isinstance(self.shortcut, nn.Conv2d):
            nn.init.zeros_(self.shortcut.weight)
            nn.init.zeros_(self.shortcut.bias)
    
    def forward(self, x, time_emb):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入特征
            time_emb (torch.Tensor): 时间嵌入
        Returns:
            torch.Tensor: 处理后的特征
        """
        # 第一个卷积块
        h = self.act1(self.norm1(x))
        h = self.conv1(h)
        
        # 添加时间信息
        h = h + self.time_mlp(time_emb)[:, :, None, None]
        
        # 第二个卷积块
        h = self.act2(self.norm2(h))
        h = self.conv2(h)
        
        # 残差连接
        return h + self.shortcut(x)

class DownSample(nn.Module):
    """
    下采样层，用于降低特征图的空间分辨率
    
    Args:
        channels (int): 输入/输出通道数
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入特征
        Returns:
            torch.Tensor: 下采样后的特征
        """
        return self.conv(x)

class UpSample(nn.Module):
    """
    上采样层，用于提高特征图的空间分辨率
    
    Args:
        channels (int): 输入/输出通道数
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入特征
        Returns:
            torch.Tensor: 上采样后的特征
        """
        # 使用最近邻插值进行上采样
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

class UNet(nn.Module):
    """
    UNet网络结构，用于预测噪声
    
    包含下采样路径和上采样路径，使用跳跃连接
    在特定层使用注意力机制
    
    Args:
        img_size (int): 输入图像大小
        img_channels (int): 输入图像通道数
        widths (list): 每一层的通道数
        has_attention (list): 是否在对应层使用注意力机制
        num_res_blocks (int): 每个分辨率下的残差块数量
        norm_groups (int): GroupNorm层的分组数
        activation_fn (callable): 激活函数
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
        
        # 设置初始通道数和时间嵌入维度
        block_out = widths[0]
        time_emb_dim = block_out * 4
        
        # 时间嵌入层
        self.time_mlp = TimeMLP(block_out, block_out * 4, time_emb_dim)
        
        # 初始卷积层，将图像通道转换为第一个宽度
        self.init_conv = nn.Conv2d(img_channels, block_out, 3, padding=1)
        
        # 下采样路径
        self.downs = nn.ModuleList()
        for i, (width, use_attn) in enumerate(zip(widths, has_attention)):
            # 添加残差块和注意力层
            res_blocks = []
            for j in range(num_res_blocks):
                res_blocks.append(ResidualBlock(block_out, width, time_emb_dim, norm_groups))
                block_out = width
                if use_attn:
                    res_blocks.append(AttentionBlock(width, norm_groups))
                    
            self.downs.append(nn.ModuleList(res_blocks))
            
            # 添加下采样层（除了最后一层）
            if i < len(widths) - 1:
                self.downs.append(DownSample(block_out))
        
        # 中间块（最低分辨率）
        mid_blocks = []
        mid_blocks.append(ResidualBlock(block_out, block_out, time_emb_dim, norm_groups))
        mid_blocks.append(AttentionBlock(block_out, norm_groups))
        mid_blocks.append(ResidualBlock(block_out, block_out, time_emb_dim, norm_groups))
        self.middle = nn.ModuleList(mid_blocks)
        
        # 上采样路径
        self.ups = nn.ModuleList()
        reversed_widths = list(reversed(widths))
        reversed_attn = list(reversed(has_attention))
        
        for i, (width, use_attn) in enumerate(zip(reversed_widths, reversed_attn)):
            # 添加残差块和注意力层，使用跳跃连接
            res_blocks = []
            for j in range(num_res_blocks + 1):
                # 如果不是最后一层，添加跳跃连接
                if i > 0 or j < num_res_blocks:
                    res_blocks.append(ResidualBlock(block_out * 2, width, time_emb_dim, norm_groups))
                else:
                    res_blocks.append(ResidualBlock(block_out, width, time_emb_dim, norm_groups))
                
                block_out = width
                if use_attn:
                    res_blocks.append(AttentionBlock(width, norm_groups))
                    
            self.ups.append(nn.ModuleList(res_blocks))
            
            # 添加上采样层（除了最后一层）
            if i < len(reversed_widths) - 1:
                self.ups.append(UpSample(block_out))
        
        # 最终层
        self.final_norm = nn.GroupNorm(norm_groups, block_out)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(block_out, img_channels, 3, padding=1)
        # 初始化最终卷积层的权重为0
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
        # 时间嵌入
        t_emb = self.time_mlp(timesteps)
        
        # 初始卷积
        h = self.init_conv(x)
        
        # 存储下采样路径的输出用于跳跃连接
        outs = [h]
        
        # 下采样路径
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
        
        # 中间块
        for block in self.middle:
            if isinstance(block, ResidualBlock):
                h = block(h, t_emb)
            else:
                h = block(h)
        
        # 上采样路径（使用跳跃连接）
        for i, layer in enumerate(self.ups):
            if isinstance(layer, UpSample):
                h = layer(h)
            else:
                # 获取下采样路径的跳跃连接
                skip_idx = len(outs) - i - 1
                if skip_idx >= 0:
                    h = torch.cat([h, outs[skip_idx]], dim=1)
                
                for block in layer:
                    if isinstance(block, ResidualBlock):
                        h = block(h, t_emb)
                    else:
                        h = block(h)
        
        # 最终层
        h = self.final_act(self.final_norm(h))
        h = self.final_conv(h)
        
        return h

"""
## 扩散模型训练器
实现DDPM模型的训练和生成功能
"""

class DiffusionModel(nn.Module):
    """
    扩散模型训练器，整合UNet和扩散过程
    
    主要功能：
    1. 训练模型预测噪声
    2. 使用EMA（指数移动平均）提高生成质量
    3. 生成新的图像样本
    
    Args:
        unet (nn.Module): UNet模型
        timesteps (int): 扩散过程的时间步数
        gdf_util (GaussianDiffusion): 高斯扩散工具类
    """
    def __init__(self, unet, timesteps, gdf_util):
        super().__init__()
        self.unet = unet  # UNet模型
        self.gdf = gdf_util  # 扩散工具类
        self.timesteps = timesteps  # 时间步数
        
        # EMA（指数移动平均）设置
        self.ema_model = None  # EMA模型
        self.ema_rate = 0.999  # EMA更新率
        
    def forward(self, x_0):
        """
        前向传播，计算训练损失
        
        Args:
            x_0 (torch.Tensor): 初始图像
        Returns:
            torch.Tensor: 预测噪声和实际噪声之间的均方误差
        """
        # 生成随机噪声
        noise = torch.randn_like(x_0)
        
        # 为每个图像随机采样一个时间步
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=x_0.device, dtype=torch.long)
        
        # 根据时间步向图像添加噪声
        x_t = self.gdf.q_sample(x_0, t, noise)
        
        # 使用UNet预测噪声
        pred_noise = self.unet(x_t, t)
        
        # 返回预测噪声和实际噪声之间的均方误差
        return F.mse_loss(noise, pred_noise)
    
    def update_ema_model(self):
        """
        更新EMA模型
        
        EMA（指数移动平均）是一种技术，用于创建模型参数的平滑版本，
        通常能提供更稳定的生成结果
        """
        if self.ema_model is None:
            # 初始化EMA模型为当前模型的副本
            self.ema_model = type(self.unet)(**vars(self.unet))
            self.ema_model.load_state_dict(self.unet.state_dict())
            self.ema_model.eval()  # 将EMA模型设置为评估模式
            # 将EMA模型移动到与主模型相同的设备
            self.ema_model.to(next(self.unet.parameters()).device)
        
        # 更新EMA模型的参数
        with torch.no_grad():
            for param_ema, param_model in zip(self.ema_model.parameters(), self.unet.parameters()):
                param_ema.data = self.ema_rate * param_ema.data + (1 - self.ema_rate) * param_model.data
    
    def generate_images(self, num_images=16):
        """
        生成新的图像样本
        
        Args:
            num_images (int): 要生成的图像数量
        Returns:
            list: 生成过程中的所有中间结果
        """
        # 如果有EMA模型，使用EMA模型进行生成，否则使用主模型
        model_to_use = self.ema_model if self.ema_model is not None else self.unet
        
        # 使用扩散过程生成样本
        samples = self.gdf.sample(
            model_to_use, 
            image_size=self.unet.img_size, 
            batch_size=num_images, 
            channels=self.unet.img_channels
        )
        return samples
    
    def save_model(self, path):
        """
        保存模型检查点
        
        Args:
            path (str): 保存路径
        """
        torch.save({
            'unet_state_dict': self.unet.state_dict(),
            'ema_state_dict': self.ema_model.state_dict() if self.ema_model is not None else None,
        }, path)
    
    def load_model(self, path):
        """
        加载模型检查点
        
        Args:
            path (str): 加载路径
        """
        checkpoint = torch.load(path)
        self.unet.load_state_dict(checkpoint['unet_state_dict'])
        if checkpoint['ema_state_dict'] is not None:
            if self.ema_model is None:
                self.ema_model = type(self.unet)(**vars(self.unet))
                self.ema_model.eval()
                self.ema_model.to(next(self.unet.parameters()).device)
            self.ema_model.load_state_dict(checkpoint['ema_state_dict'])

"""
## 训练循环
实现模型的训练过程
"""

def train(model, train_loader, optimizer, device, num_epochs):
    """
    训练DDPM模型
    
    Args:
        model (nn.Module): 扩散模型
        train_loader (DataLoader): 数据加载器
        optimizer (torch.optim.Optimizer): 优化器
        device (torch.device): 计算设备
        num_epochs (int): 训练轮数
    """
    model.train()  # 将模型设置为训练模式
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            # 将数据移动到指定设备
            real_images = batch.to(device)
            
            # 前向传播，计算损失
            loss = model(real_images)
            
            # 反向传播
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新参数
            
            # 更新EMA模型
            model.update_ema_model()
            
            # 记录训练进度
            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}")
        
        # 记录每个epoch的平均损失
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # 生成和保存样本图像
        if (epoch + 1) % 10 == 0 or epoch == 0:
            samples = model.generate_images(num_images=4)
            # 这里可以添加保存或显示生成图像的代码
            
        # 保存模型检查点
        if (epoch + 1) % 20 == 0 or epoch == num_epochs - 1:
            model.save_model(f"model_checkpoint_epoch_{epoch+1}.pt")

"""
## 主执行函数
设置和启动训练过程
"""

def main():
    """
    主函数，设置训练环境并启动训练
    """
    # 设置计算设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 创建扩散工具类
    gdf = GaussianDiffusion(
        timesteps=total_timesteps,
        device=device
    )
    
    # 创建UNet模型
    unet = UNet(
        img_size=img_size,
        img_channels=img_channels,
        widths=widths,
        has_attention=has_attention,
        num_res_blocks=num_res_blocks,
        norm_groups=norm_groups
    ).to(device)
    
    # 创建扩散模型训练器
    model = DiffusionModel(
        unet=unet,
        timesteps=total_timesteps,
        gdf_util=gdf
    ).to(device)
    
    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 开始训练
    train(model, train_loader, optimizer, device, num_epochs)

if __name__ == "__main__":
    main() 