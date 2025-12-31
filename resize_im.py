#!/usr/bin/env python3
"""替代MATLAB的resize_im.m脚本
将输入文件夹中的所有图像调整为256x256像素并保存到输出文件夹
"""

import os
import sys
from PIL import Image
import argparse

def resize_images(input_dir, output_dir, size=(256, 256)):
    """将输入文件夹中的所有图像调整为指定尺寸"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取输入文件夹中的所有图像文件
    file_list = os.listdir(input_dir)
    image_files = [f for f in file_list if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    for i, filename in enumerate(image_files):
        try:
            # 读取图像
            input_path = os.path.join(input_dir, filename)
            img = Image.open(input_path)
            
            # 调整图像尺寸
            resized_img = img.resize(size, Image.LANCZOS)
            
            # 保存图像
            output_path = os.path.join(output_dir, filename)
            resized_img.save(output_path, quality=100)
            
            print(f"处理完成 ({i+1}/{len(image_files)}): {filename}")
        except Exception as e:
            print(f"处理 {filename} 时出错: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='调整图像尺寸')
    parser.add_argument('input_dir', help='输入图像目录')
    parser.add_argument('output_dir', help='输出图像目录')
    args = parser.parse_args()
    
    resize_images(args.input_dir, args.output_dir)
