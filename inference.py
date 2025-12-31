"""Translate an image to another image
An example of command-line usage is:
python inference.py --model pretrained/Hazy2GT_15_12_2025_11:56:53.pb \
                 --input input_sample.jpg \
                 --output output_sample.jpg \
                 --image_size 256
"""

import tensorflow as tf
import os
import argparse
import numpy as np
from model import CycleGAN
import utils

def parse_args():
    parser = argparse.ArgumentParser(description='图像去雾推理')
    parser.add_argument('--model', required=True, help='模型路径 (.pb)')
    parser.add_argument('--input', default='input_sample.jpg', help='输入图像路径 (.jpg)')
    parser.add_argument('--output', default='output_sample.jpg', help='输出图像路径 (.jpg)')
    parser.add_argument('--image_size', type=int, default=256, help='图像尺寸，默认256')
    return parser.parse_args()

def inference():
    args = parse_args()
    
    # 首先读取模型文件（在图外）
    with open(args.model, 'rb') as model_file:
        model_data = model_file.read()
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(model_data)
    
    graph = tf.Graph()
    
    with graph.as_default():
        # 读取输入图像
        image_data = tf.io.read_file(args.input)
        input_image = tf.image.decode_jpeg(image_data, channels=3)
        input_image = tf.image.resize(input_image, size=(args.image_size, args.image_size))
        input_image = utils.convert2float(input_image)
        input_image = tf.reshape(input_image, [args.image_size, args.image_size, 3])
        
        # 导入图定义
        output_tensor = tf.import_graph_def(graph_def,
                              input_map={'input_image': input_image},
                              return_elements=['output_image:0'],
                              name='output')
        # 获取实际输出张量
        output_image = output_tensor[0]
    
    with tf.compat.v1.Session(graph=graph) as sess:
        # 初始化变量
        sess.run(tf.compat.v1.global_variables_initializer())
        
        # 获取生成的图像
        generated = sess.run(output_image)
        
        # 检查输出类型和形状
        print(f"Generated image type: {type(generated)}")
        if hasattr(generated, 'shape'):
            print(f"Generated image shape: {generated.shape}")
        
        # 处理不同类型的输出
        if isinstance(generated, bytes):
            # 如果输出已经是字节格式（JPEG编码），直接写入文件
            with open(args.output, 'wb') as f:
                f.write(generated)
        elif isinstance(generated, np.ndarray):
            # 如果输出是numpy数组，需要编码为JPEG
            encoded_image = tf.image.encode_jpeg(tf.cast(generated, tf.uint8))
            encoded_data = sess.run(encoded_image)
            
            with open(args.output, 'wb') as f:
                f.write(encoded_data)
        else:
            # 其他情况，尝试转换
            try:
                # 转换为numpy数组
                if not isinstance(generated, np.ndarray):
                    generated = np.array(generated)
                    
                encoded_image = tf.image.encode_jpeg(tf.cast(generated, tf.uint8))
                encoded_data = sess.run(encoded_image)
                
                with open(args.output, 'wb') as f:
                    f.write(encoded_data)
            except Exception as e:
                print(f"Error processing output: {e}")

def main():
    inference()

if __name__ == '__main__':
    main()
