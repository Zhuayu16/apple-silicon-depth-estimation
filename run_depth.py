import torch
import os
# 强制离线模式，跳过所有在线检查、更新、拉取
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import torch
from transformers import DPTForDepthEstimation, DPTImageProcessor
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# 使用 Hugging Face transformers 库的 DPT 模型（离线模式）
print("正在加载深度估计模型（离线模式）...")
try:
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large", local_files_only=True)
    processor = DPTImageProcessor.from_pretrained("Intel/dpt-large", local_files_only=True)
    print("✅ 模型加载成功（使用本地缓存）")
except Exception as e:
    print(f"❌ 模型加载失败：{e}")
    print("\n请确保已下载模型到本地缓存")
    print("首次运行需要联网下载，之后可以离线使用")
    exit(1)

model.eval()

# 检测设备
device_name = "mps" if torch.backends.mps.is_available() else "cpu"
if device_name == "mps":
    print("✅ 检测到 Apple Silicon GPU，使用 MPS 加速")
else:
    print("ℹ️  使用 CPU 模式")
device = torch.device(device_name)
model.to(device)

# 打开一张图片
img_path = "test.jpg"  # 把你自己的图片放这里
try:
    image = Image.open(img_path).convert("RGB")
except FileNotFoundError:
    print(f"错误：找不到图片文件 {img_path}")
    print("请确保 test.jpg 文件存在于当前目录下")
    exit(1)

# 预处理
inputs = processor(images=image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

# 推理深度
with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

# 后处理
prediction = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=image.size[::-1],
    mode="bicubic",
    align_corners=False,
)
output = prediction.squeeze().cpu().numpy()

# 显示结果
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# 原图
ax1.imshow(image)
ax1.set_title("Original Image", fontsize=14)
ax1.axis('off')

# 深度图
im = ax2.imshow(output, cmap='plasma')
ax2.set_title("Depth Map", fontsize=14)
ax2.axis('off')
plt.colorbar(im, ax=ax2)

plt.tight_layout()
plt.show()

print("深度图生成完成！")