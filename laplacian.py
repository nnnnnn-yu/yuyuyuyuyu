#!/usr/bin/env python3
"""替代MATLAB的laplacian.m脚本
使用拉普拉斯金字塔技术将处理后的图像放大回原始尺寸
"""

import os
import sys
import numpy as np
from PIL import Image
import argparse

def gaussian_pyramid_reduce(image):
    """高斯金字塔 - 下一层"""
    return image.resize((image.width // 2, image.height // 2), Image.LANCZOS)

def laplacian_pyramid_reconstruct(original_path, processed_path, output_path):
    """使用拉普拉斯金字塔重建图像"""
    # 读取原始和处理后的图像
    original = Image.open(original_path)
    processed = Image.open(processed_path)
    
    # 转换为numpy数组
    A = np.array(original, dtype=np.float32)
    C = np.array(processed, dtype=np.float32)
    
    # 创建高斯金字塔
    Ad1 = np.array(gaussian_pyramid_reduce(original), dtype=np.float32)
    Ad2 = np.array(gaussian_pyramid_reduce(Image.fromarray(Ad1.astype(np.uint8))), dtype=np.float32)
    Ad3 = np.array(gaussian_pyramid_reduce(Image.fromarray(Ad2.astype(np.uint8))), dtype=np.float32)
    Ad4 = np.array(gaussian_pyramid_reduce(Image.fromarray(Ad3.astype(np.uint8))), dtype=np.float32)
    
    # 创建拉普拉斯金字塔
    L1 = A - np.array(Image.fromarray(Ad1.astype(np.uint8)).resize((A.shape[1], A.shape[0]), Image.LANCZOS), dtype=np.float32)
    L2 = Ad1 - np.array(Image.fromarray(Ad2.astype(np.uint8)).resize((Ad1.shape[1], Ad1.shape[0]), Image.LANCZOS), dtype=np.float32)
    L3 = Ad2 - np.array(Image.fromarray(Ad3.astype(np.uint8)).resize((Ad2.shape[1], Ad2.shape[0]), Image.LANCZOS), dtype=np.float32)
    L4 = Ad3 - np.array(Image.fromarray(Ad4.astype(np.uint8)).resize((Ad3.shape[1], Ad3.shape[0]), Image.LANCZOS), dtype=np.float32)
    
    # 重建图像
    Cu1 = np.array(Image.fromarray(C.astype(np.uint8)).resize((Ad3.shape[1], Ad3.shape[0]), Image.LANCZOS), dtype=np.float32) + L4
    Cu2 = np.array(Image.fromarray(Cu1.astype(np.uint8)).resize((Ad2.shape[1], Ad2.shape[0]), Image.LANCZOS), dtype=np.float32) + L3
    Cu3 = np.array(Image.fromarray(Cu2.astype(np.uint8)).resize((Ad1.shape[1], Ad1.shape[0]), Image.LANCZOS), dtype=np.float32) + L2
    Cu4 = np.array(Image.fromarray(Cu3.astype(np.uint8)).resize((A.shape[1], A.shape[0]), Image.LANCZOS), dtype=np.float32) + L1
    
    # 保存结果
    result = Image.fromarray(Cu4.astype(np.uint8))
    result.save(output_path, quality=100)
    print(f"重建完成: {output_path}")

def laplacian_reconstruction(input_dir, original_dir, output_dir):
    """批量使用拉普拉斯金字塔重建图像"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取输入文件夹中的所有图像文件
    file_list = os.listdir(input_dir)
    image_files = [f for f in file_list if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    for i, filename in enumerate(image_files):
        try:
            input_path = os.path.join(input_dir, filename)
            original_path = os.path.join(original_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            laplacian_pyramid_reconstruct(original_path, input_path, output_path)
            print(f"处理完成 ({i+1}/{len(image_files)}): {filename}")
        except Exception as e:
            print(f"处理 {filename} 时出错: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='使用拉普拉斯金字塔重建图像')
    parser.add_argument('input_dir', help='处理后的图像目录')
    parser.add_argument('original_dir', help='原始图像目录')
    parser.add_argument('output_dir', help='输出图像目录')
    args = parser.parse_args()
    
    laplacian_reconstruction(args.input_dir, args.original_dir, args.output_dir)
