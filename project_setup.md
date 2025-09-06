# PyTorch 版本自动选择和源管理指南

## 🎯 配置说明

基于您的示例，我们已经配置了自动选择GPU/CPU版本PyTorch的功能，并支持手动控制包源。

## 📦 安装方式

### 自动选择版本（推荐）

```bash
# GPU版本（自动从PyTorch官方GPU源安装）
uv pip install -e .[gpu]

# CPU版本（自动从PyTorch官方CPU源安装）
uv pip install -e .[cpu]

# 开发环境
uv pip install -e .[dev]

# 完整安装
uv pip install -e .[all]
```

### 手动控制源

```bash
# 使用阿里云镜像
uv pip install -i https://mirrors.aliyun.com/pypi/simple -e .[gpu]

# 使用清华镜像
uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e .[gpu]

# 使用中科大镜像
uv pip install -i https://pypi.mirrors.ustc.edu.cn/simple -e .[gpu]

# 使用华为云镜像
uv pip install -i https://repo.huaweicloud.com/repository/pypi/simple -e .[gpu]
```

## ⚙️ 配置详解

### 1. 依赖冲突配置

```toml
[tool.uv]
conflicts = [
    [
        { extra = "cpu" },
        { extra = "gpu" },
    ],
]
```

- 防止同时安装CPU和GPU版本

### 2. 包源配置

```toml
[tool.uv.sources]
torch = [
    { index = "pytorch-cpu", extra = "cpu" },
    { index = "pytorch-gpu", extra = "gpu" },
]
```

- CPU版本：自动使用 `pytorch-cpu` 索引
- GPU版本：自动使用 `pytorch-gpu` 索引

### 3. 索引源配置

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[[tool.uv.index]]
name = "pytorch-gpu"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```

- PyTorch官方CPU源
- PyTorch官方GPU源（CUDA 12.8）

### 4. 中文镜像源

配置了4个常用的中文镜像源，支持手动切换。

## 🚀 使用场景

### 场景1：自动安装（推荐）

```bash
# 在有GPU的环境中
uv sync --extra gpu  # 自动下载GPU版本

# 在无GPU的环境中
uv sync --extra cpu  # 自动下载CPU版本
```

### 场景2：网络环境差时手动选择源

```bash
# 如果自动安装慢，可以手动指定镜像
uv sync --extra gpu --index https://mirrors.aliyun.com/pypi/simple/
```

### 场景3：离线环境

```bash
# 先下载到本地
uv pip download -i https://mirrors.aliyun.com/pypi/simple -e .[gpu] -d ./packages

# 离线安装
uv pip install --no-index --find-links=./packages -e .
```

## 🔧 故障排除

### 如果安装失败

1. 检查网络连接
2. 尝试更换镜像源
3. 确认CUDA版本兼容性

### 如果版本冲突

1. 卸载现有版本：`uv remove torch torchvision torchaudio`
2. 重新安装指定版本：`uv sync --extra gpu`

### 如果需要特定CUDA版本

修改 `pyproject.toml` 中的GPU索引URL：

```toml
[[tool.uv.index]]
name = "pytorch-gpu"
url = "https://download.pytorch.org/whl/cu121"  # CUDA 12.1
explicit = true
```

## 📚 更多资源

- [PyTorch 安装指南](https://pytorch.org/get-started/locally/)
- [uv 官方文档](https://docs.astral.sh/uv/)
- [CUDA 兼容性](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
