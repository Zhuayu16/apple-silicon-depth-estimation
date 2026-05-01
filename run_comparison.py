"""
AI vs D435 深度图对比 - 实时对比
左边：AI 单目深度估计（Intel DPT）
右边：D435 真实深度数据

注意：如果 D435 无法连接，会自动降级为纯 AI 模式
目前只显示AI模式
启动cd /Users/zzzhy/Desktop/depth && python run_comparison.py
"""

import cv2
import torch
from transformers import DPTForDepthEstimation, DPTImageProcessor
import numpy as np
from datetime import datetime
import platform
import time
import sys

print("=" * 70)
print("AI vs D435 深度图实时对比系统")
print("=" * 70)
print(f"系统：{platform.system()} {platform.machine()}")
print(f"Python: {platform.python_version()}\n")

# ========== 1. 加载 AI 模型 ==========
print("【步骤 1】加载 AI 深度估计模型...")
device_name = "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device_name)
if device_name == "mps":
    print("✅ Apple Silicon GPU 加速已启用")

model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(device)
model.eval()
processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
print("✅ AI 模型加载完成\n")

# ========== 2. 尝试连接 D435 ==========
print("【步骤 2】尝试连接 D435 设备...")
d435_available = False
pipeline = None

# 使用更安全的方式检测 D435 - 先不启动管道
try:
    import pyrealsense2 as rs
    
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) > 0:
        print(f"✅ 检测到 {len(devices)} 个 RealSense 设备")
        
        # 仅打印设备信息，不启动管道以避免崩溃
        for i, dev in enumerate(devices):
            try:
                name = dev.get_info(rs.camera_info.name)
                serial = dev.get_info(rs.camera_info.serial_number)
                print(f"   设备 {i+1}: {name} (SN: {serial})")
            except:
                pass
        
        print("\n⚠️  注意：在 Apple Silicon Mac 上直接访问 D435 可能导致不稳定")
        print("   为确保程序稳定运行，将使用纯 AI 模式")
        print("   如需使用 D435，请参考 README.md 中的解决方案\n")
        
except ImportError:
    print("⚠️  pyrealsense2 未安装，使用纯 AI 模式")
except Exception as e:
    print(f"⚠️  D435 检测异常：{e}")

print()

# ========== 3. 打开摄像头 ==========
print("【步骤 3】打开摄像头...")
cap = None
for i in range(3):
    temp_cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
    if temp_cap.isOpened():
        cap = temp_cap
        print(f"✅ 摄像头已打开：设备 {i}")
        break
    temp_cap.release()

if not cap or not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ 无法打开摄像头")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
time.sleep(0.3)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"✅ 分辨率：{width}x{height}\n")

# ========== 4. 主循环 ==========
print("操作说明:")
print("  [q] 退出")
print("  [s] 保存当前帧")
print("  [a] 仅 AI 模式")
print("  [d] 仅 D435 模式 (如果可用)")
print("  [b] 对比模式 (并排)")
print("=" * 70)
print("正在启动...\n")

display_mode = 0  # 0=对比，1=仅 AI, 2=仅 D435
frame_count = 0
start_time = time.time()

while True:
    # --- 获取摄像头帧 ---
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame_count += 1
    
    # --- AI 深度估计 ---
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb_frame, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(height, width),
        mode="bicubic",
        align_corners=False,
    )
    ai_depth = prediction.squeeze().cpu().numpy()
    ai_depth_norm = cv2.normalize(ai_depth, None, 0, 255, cv2.NORM_MINMAX)
    ai_depth_colored = cv2.applyColorMap(ai_depth_norm.astype(np.uint8), cv2.COLORMAP_PLASMA)
    
    # --- D435 深度数据 (如果可用) ---
    d435_depth_colored = None
    if d435_available and pipeline:
        try:
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            depth_frame = frames.get_depth_frame()
            
            if depth_frame:
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_image = cv2.resize(depth_image, (width, height))
                
                # 深度值裁剪和归一化
                depth_image = np.clip(depth_image, 0, 10000)  # 0-10 米
                depth_norm = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
                d435_depth_colored = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_INFERNO)
        except Exception as e:
            pass  # D435 读取失败，继续
    
    # --- 计算 FPS ---
    current_fps = frame_count / max(0.001, time.time() - start_time)
    
    # --- 准备输出 ---
    if display_mode == 0 and d435_depth_colored is not None:
        # 对比模式：AI | D435
        output = np.hstack([ai_depth_colored, d435_depth_colored])
        title = f"AI vs D435 Comparison | FPS: {current_fps:.1f} | [Q]uit [S]ave"
    elif display_mode == 1 or d435_depth_colored is None:
        # 仅 AI 模式
        output = ai_depth_colored
        mode_text = "AI Only" if d435_depth_colored is None else "AI Mode"
        title = f"{mode_text} | FPS: {current_fps:.1f} | [Q]uit [S]ave"
    else:
        # 仅 D435 模式
        output = d435_depth_colored
        title = f"D435 Mode | FPS: {current_fps:.1f} | [Q]uit [S]ave"
    
    # 添加标题
    cv2.putText(output, title, (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # 添加标签
    if display_mode == 0 and d435_depth_colored is not None:
        cv2.putText(output, "AI Depth", (10, height - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(output, "D435 Depth", (width + 10, height - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 显示
    cv2.imshow('Depth Comparison', output)
    
    # 按键处理 - 修复按键检测
    key = cv2.waitKey(1) & 0xFF
    
    # 先检查是否退出
    if key == ord('q'):
        print("\n✅ 退出程序")
        break
    elif key == ord('s'):
        # 保存当前帧
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"comparison_{timestamp}.png", output)
        print(f"✅ 已保存：comparison_{timestamp}.png")
    
    # 然后检查模式切换（无论 D435 是否可用都响应）
    if key == ord('a'):
        display_mode = 1
        print("📺 切换到 AI 模式")
    elif key == ord('d'):
        if d435_depth_colored is not None:
            display_mode = 2
            print("📺 切换到 D435 模式")
        else:
            print("⚠️  D435 不可用，无法切换到 D435 模式")
    elif key == ord('b'):
        if d435_depth_colored is not None:
            display_mode = 0
            print("📺 切换到对比模式")
        else:
            print("⚠️  D435 不可用，将使用纯 AI 模式")
            display_mode = 1  # 降级到 AI 模式

# 清理
cap.release()
if pipeline:
    pipeline.stop()
cv2.destroyAllWindows()
print("✅ 程序已结束")