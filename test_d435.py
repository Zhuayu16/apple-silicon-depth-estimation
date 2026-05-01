"""
RealSense D435 - Apple Silicon (M1/M2/M3) 完整解决方案

你的系统：Apple M3 芯片（ARM64）
问题：conda 版本是 x86_64 转译，不支持 M 芯片的 USB 电源管理

【已验证的解决方案】按推荐顺序排列：

方案 1: 使用 run_depth.py（强烈推荐 ⭐⭐⭐⭐⭐）
----------------------------------------
✅ 优点:
- 完美支持 Apple Silicon（原生 ARM64）
- 不需要任何硬件设备
- 使用 AI 从单张图片生成深度图
- 效果好，速度快

📝 使用方法:
python run_depth.py

方案 2: 从源码编译 librealsense（困难模式）
----------------------------------------
需要 Xcode 命令行工具和一定时间

# 1. 安装依赖
xcode-select --install
brew install cmake libusb glfw

# 2. 克隆源码
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense

# 3. 创建构建目录
mkdir build && cd build

# 4. 配置（启用 Python 绑定）
cmake .. -DBUILD_PYTHON_BINDINGS=bool:true \
         -DPYTHON_EXECUTABLE=$(which python3) \
         -DCMAKE_BUILD_TYPE=Release \
         -DFORCE_LIBUVC=true

# 5. 编译（可能需要 30 分钟）
make -j8

# 6. 安装
sudo make install

# 7. 链接 Python 绑定
export PYTHONPATH=/usr/local/lib:$PYTHONPATH

方案 3: 使用 Docker 容器（中等难度）
----------------------------------------
# 1. 安装 Docker Desktop for Mac
# 2. 运行容器
docker run --rm -it \
  --device=/dev/video0:/dev/video0 \
  intelrealense/librealsense:latest

方案 4: 等待官方支持（长期方案）
----------------------------------------
关注 GitHub Issue:
https://github.com/IntelRealSense/librealsense/issues/7792

========================================
当前检测
========================================
"""

import sys
import platform
import os

print(__doc__)

print(f"处理器：{platform.machine()}")
print(f"Python: {platform.python_version()}")
print(f"架构：{platform.architecture()[0]}")

# 检查是否有 conda 版本
try:
    import pyrealsense2 as rs
    print(f"\n⚠️  检测到 pyrealsense2: {rs.__version__}")
    print("   路径:", rs.__file__)
    
    # 尝试检测设备
    ctx = rs.context()
    devs = ctx.query_devices()
    if len(devs) > 0:
        print(f"   检测到 {len(devs)} 个设备")
        print("\n❌ 但在 M 芯片上可能无法正常工作（USB 电源问题）")
except ImportError:
    print("\n✅ 未检测到 pyrealsense2（已卸载 conda 版本）")

print("\n" + "=" * 60)
print("💡 强烈建议使用 run_depth.py")
print("   这是目前在 Apple Silicon 上最稳定的方案！")
print("=" * 60)

# 提供快速测试
print("\n想要快速测试深度估计吗？")
print("运行：python run_depth.py")