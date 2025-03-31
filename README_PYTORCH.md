# PyTorch 版本 DDPM 模型

这是一个基于 PyTorch 实现的去噪扩散概率模型 (DDPM)。该项目是对原始 TensorFlow DDPM 实现的转换。

## 项目结构

- `ddpm_pytorch.py`: PyTorch 版本的 DDPM 模型实现
- `test_pytorch_ddpm.py`: 用于测试扩散步骤和生成样本的简单脚本
- `train_pytorch_ddpm.py`: 完整的训练脚本，包含参数解析、日志记录和模型保存功能

## 环境要求

- Python 3.7+
- PyTorch 1.8+
- torchvision
- matplotlib
- numpy
- pillow

安装依赖：

```bash
pip install torch torchvision matplotlib numpy pillow
```

## 数据集

该模型使用位于 `dataset/64x64/train/nolabel` 目录下的图像数据集进行训练。

## 使用方法

### 训练模型

```bash
python train_pytorch_ddpm.py --data_dir dataset/64x64/train/nolabel --output_dir output --epochs 100
```

可选参数：

- `--data_dir`: 训练数据目录
- `--output_dir`: 输出目录
- `--epochs`: 训练轮次
- `--log_interval`: 每多少批次打印日志
- `--sample_interval`: 每多少轮次生成样本
- `--save_interval`: 每多少轮次保存模型
- `--resume`: 恢复训练的检查点路径
- `--seed`: 随机种子
- `--num_workers`: 数据加载线程数

### 测试模型

```bash
python test_pytorch_ddpm.py
```

这将测试扩散步骤并尝试生成样本。如果有训练好的模型可用，它将加载并使用该模型生成样本。

## 模型架构

该实现使用了类似于原始 DDPM 论文中的 U-Net 架构，具有以下特点：

1. 使用 ResNet 块作为基本构建块
2. 在适当的分辨率下使用注意力机制
3. 时间步嵌入用于条件生成
4. 通过指数移动平均 (EMA) 进行模型平滑

## 参考文献

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (Ho et al., 2020)
- [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) (Nichol et al., 2021) 