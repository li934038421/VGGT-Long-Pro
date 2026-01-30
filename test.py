import torch
from base_models.vggt.models.vggt import VGGT

import torch
import os
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF


def get_vggt_pose_only(image_folder = './images', ckpt_path='./weights/model.pt', merge_r=0.7):
    # 1. 初始化模型：开启 Token Merging，仅开启 Camera Head
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGGT(
        enable_camera=True,
        enable_depth=False,  # 禁用深度头以提速
        enable_point=False,  # 禁用点云头
        merging=1,           # 开启合并逻辑
        merge_ratio=merge_r  # 合并比例
    )

    # 2. 加载权重
    print(f"正在加载权重: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model = model.to(device).eval().to(torch.bfloat16)

    # 3. 读取并处理图片
    image_paths = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    processed_images = []
    target_size = 518 # VGGT 默认尺寸
    
    print(f"正在处理 {len(image_paths)} 张图片...")
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        # 简单的 Resize 和对齐（这里采用最基础的缩放至 518x518）
        img = img.resize((target_size, target_size), Image.Resampling.BICUBIC)
        img_tensor = TF.to_tensor(img) # 自动归一化到 [0, 1] 并且维度变为 [3, H, W]
        processed_images.append(img_tensor)

    # 堆叠成 Batch Tensor: [1, Sequence_Length, 3, 518, 518]
    # VGGT 的 forward 期望输入带 Batch 维度
    input_tensor = torch.stack(processed_images).unsqueeze(0).to(device).to(torch.bfloat16)

    # 4. 执行推理
    print("🚀 正在通过 VGGT 提取位姿 (已启用 Token Merging)...")
    with torch.no_grad():
        # 更新 Patch 维度（518/14 = 37）
        model.update_patch_dimensions(37, 37)
        predictions = model(input_tensor)
        
    # 5. 获取结果
    pose_enc = predictions["pose_enc"] # [1, S, 9]
    return pose_enc

# --- 使用示例 ---
# ckpt = "你的权重路径.pt"
# img_dir = "你的图片目录"
# poses = get_vggt_pose_only(img_dir, ckpt, merge_r=0.7)
# print("位姿提取完成，形状为:", poses.shape)