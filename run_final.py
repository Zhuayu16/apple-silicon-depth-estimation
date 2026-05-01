"""
实时深度估计 - 使用摄像头或视频文件
基于 Intel DPT 大模型，原生支持 Apple Silicon M1/M2/M3

功能:
- 从 Mac 内置摄像头或外接 USB 摄像头获取实时视频
- 使用 AI 实时生成深度图
- 支持保存视频
- 原生 ARM64 优化，M3 芯片流畅运行

使用方法:
python run_final.py

按键说明:
- q: 退出
- s: 保存当前帧
- d: 切换显示模式 (原图/深度图/并排)
运行正常！！！！！！！！！！！！！！！
cd /Users/zzzhy/Desktop/depth && python run_final.py  
"""

import cv2
import torch
from transformers import DPTForDepthEstimation, DPTImageProcessor
import numpy as np
from datetime import datetime
import platform
import time

print("=" * 60)
print("实时深度估计系统 - Apple Silicon 优化版")
print("=" * 60)
print(f"系统：{platform.system()} {platform.machine()}")
print(f"Python: {platform.python_version()}")

# 检测设备
device_name = "mps" if torch.backends.mps.is_available() else "cpu"
if device_name == "mps":
    print("✅ 检测到 Apple Silicon GPU，使用 MPS 加速")
else:
    print("ℹ️  使用 CPU 模式")
device = torch.device(device_name)

# 加载模型
print("\n正在加载深度估计模型...")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(device)
model.eval()
processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
print("✅ 模型加载完成\n")

# 打开摄像头 - 优先使用 AVFoundation 后端
print("正在打开摄像头...")

# 方法 1: 尝试使用 AVFoundation 后端（macOS 最优）
cap = None
for i in range(3):  # 尝试设备 0, 1, 2
    print(f"  尝试摄像头设备 {i} (AVFoundation)...")
    temp_cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
    if temp_cap.isOpened():
        cap = temp_cap
        print(f"✅ 找到摄像头：设备 {i}")
        break
    else:
        temp_cap.release()

# 如果都失败，尝试默认后端
if not cap or not cap.isOpened():
    print("  AVFoundation 失败，尝试默认后端...")
    cap = cv2.VideoCapture(0)
    
if not cap.isOpened():
    print("\n❌ 无法打开摄像头")
    print("\n【重要】这是 macOS 摄像头权限问题")
    print("\n解决步骤:")
    print("1. 打开「系统偏好设置」→「安全性与隐私」→「隐私」→「摄像头」")
    print("2. 在列表中找到并勾选:")
    print("   • 终端 (Terminal)")
    print("   • 或你的 IDE (VSCode/PyCharm)")
    print("3. 重启终端/IDE")
    print("4. 重新运行此程序")
    print("\n或者:")
    print("• 如果是外接摄像头，请检查 USB 连接")
    print("• 尝试使用 Photo Booth 测试摄像头是否正常工作")
    exit(1)

# 设置摄像头参数 - 使用更高的分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 从 320 提升到 640
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 从 240 提升到 480
cap.set(cv2.CAP_PROP_FPS, 30)

# 等待摄像头预热
time.sleep(0.3)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"✅ 摄像头已打开：{width}x{height} @ {fps}fps\n")

# 显示模式：0=并排，1=原图，2=深度图
display_mode = 0
frame_count = 0
success_count = 0
start_time = time.time()

# 窗口缩放比例（用于放大显示）
zoom_factor = 1.5  # 放大 1.5 倍

print("操作说明:")
print("  [q] 退出")
print("  [s] 保存当前帧")
print("  [d] 切换显示模式")
print("  [+/-] 调整窗口大小")
print("=" * 60)
print("正在启动...\n")

while True:
    ret, frame = cap.read()
    
    if not ret:
        frame_count += 1
        if frame_count > 10 and success_count == 0:
            print("❌ 无法读取帧")
            print("\n可能原因:")
            print("1. 摄像头被其他应用占用（如 Zoom、微信、FaceTime）")
            print("2. 摄像头权限未授予")
            print("3. 摄像头硬件故障")
            print("\n请关闭其他使用摄像头的应用后重试")
            break
        continue
    
    success_count += 1
    
    # 计算实际 FPS
    current_time = time.time()
    elapsed = current_time - start_time
    actual_fps = success_count / elapsed if elapsed > 0 else 0
    
    # 转换颜色空间 BGR -> RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 预处理
    inputs = processor(images=rgb_frame, return_tensors="pt").to(device)
    
    # 推理
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    
    # 后处理
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(height, width),
        mode="bicubic",
        align_corners=False,
    )
    depth_map = prediction.squeeze().cpu().numpy()
    
    # 归一化深度图到 0-255
    depth_normalized = cv2.normalize(
        depth_map, 
        None, 
        0, 255, 
        cv2.NORM_MINMAX
    )
    depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_PLASMA)
    
    # 根据显示模式准备输出
    if display_mode == 0:
        # 并排显示
        output = np.hstack([frame, depth_colored])
        title = f"FPS: {actual_fps:.1f} | [D]isplay: Side-by-Side | [Q]uit"
    elif display_mode == 1:
        # 只显示原图
        output = frame
        title = f"FPS: {actual_fps:.1f} | [D]isplay: Original | [Q]uit"
    else:
        # 只显示深度图
        output = depth_colored
        title = f"FPS: {actual_fps:.1f} | [D]isplay: Depth | [Q]uit"
    
    # 添加标题栏
    cv2.putText(output, title, (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # 如果设置了缩放，放大显示
    if zoom_factor != 1.0:
        h, w = output.shape[:2]
        new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
        output = cv2.resize(output, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 显示
    cv2.imshow('Depth Estimation', output)
    
    # 按键处理 - 使用 waitKey(10) 而不是 waitKey(1) 提高兼容性
    key = cv2.waitKey(10) & 0xFF
    
    # 检查每个按键
    if key == ord('q'):
        print("\n✅ 退出程序")
        break
    elif key == ord('s'):
        # 保存当前帧
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"frame_{timestamp}.png", output)
        print(f"✅ 已保存：frame_{timestamp}.png")
    elif key == ord('d'):
        display_mode = (display_mode + 1) % 3
        modes = ["并排显示", "仅原图", "仅深度图"]
        print(f"📺 显示模式：{modes[display_mode]}")
    elif key == ord('+') or key == ord('='):  # + 键放大
        zoom_factor = min(2.0, zoom_factor + 0.1)
        print(f"🔍 窗口缩放：{zoom_factor:.1f}x")
    elif key == ord('-') or key == ord('_'):  # - 键缩小
        zoom_factor = max(0.5, zoom_factor - 0.1)
        print(f"🔍 窗口缩放：{zoom_factor:.1f}x")

# 清理
cap.release()
cv2.destroyAllWindows()
print("✅ 程序已结束")