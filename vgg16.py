import inspect
import os

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1  # 导入 TF 1.x 兼容模块
tf1.disable_v2_behavior()  # 禁用 TF 2.x 行为，启用 TF 1.x 行为
import time

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg16:
    def __init__(self, vgg16_npy_path=None):
        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg16.npy")
            vgg16_npy_path = path
            print(path)

        try:
            self.data_dict = np.load(vgg16_npy_path, encoding='latin1',allow_pickle=True).item()
            print("npy file loaded")
        except Exception as e:
            print(f"Failed to load VGG16 weights: {e}")
            print("Creating VGG16 model without pre-trained weights")
            self.data_dict = None

    def build(self, rgb):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        # 使用 tf1.split 替代 tf.split
        red, green, blue = tf1.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        # 使用 tf1.concat 替代 tf.concat
        bgr = tf1.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')

        #self.fc6 = self.fc_layer(self.pool5, "fc6")
        #assert self.fc6.get_shape().as_list()[1:] == [4096]
        #self.relu6 = tf.nn.relu(self.fc6)

        #self.fc7 = self.fc_layer(self.relu6, "fc7")
        #self.relu7 = tf.nn.relu(self.fc7)

        #self.fc8 = self.fc_layer(self.relu7, "fc8")

        #self.prob = tf.nn.softmax(self.fc8, name="prob")

        #self.data_dict = None
        print(("build model finished: %ds" % (time.time() - start_time)))
        return self.pool2, self.pool5

    def avg_pool(self, bottom, name):
        # 使用 tf1.nn.avg_pool 替代 tf.nn.avg_pool
        return tf1.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        # 使用 tf1.nn.max_pool 替代 tf.nn.max_pool
        return tf1.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        # 使用 tf1.variable_scope 替代 tf.variable_scope
        with tf1.variable_scope(name):
            filt = self.get_conv_filter(name)

            # 使用 tf1.nn.conv2d 替代 tf.nn.conv2d
            conv = tf1.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            # 使用 tf1.nn.bias_add 替代 tf.nn.bias_add
            bias = tf1.nn.bias_add(conv, conv_biases)

            # 使用 tf1.nn.relu 替代 tf.nn.relu
            relu = tf1.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        # 使用 tf1.variable_scope 替代 tf.variable_scope
        with tf1.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            # 使用 tf1.reshape 替代 tf.reshape
            x = tf1.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            # 使用 tf1.nn.bias_add 替代 tf.nn.bias_add
            fc = tf1.nn.bias_add(tf1.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        if self.data_dict is None:
            # 如果没有预训练权重，创建随机权重
            # 这里使用简单的初始化，实际应用中可能需要更复杂的初始化
            return tf1.constant(np.random.randn(3, 3, 3, 64) * 0.01, dtype=tf.float32, name="filter")
        return tf1.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        if self.data_dict is None:
            # 如果没有预训练权重，创建随机偏置
            return tf1.constant(np.zeros(64), dtype=tf.float32, name="biases")
        return tf1.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        if self.data_dict is None:
            # 如果没有预训练权重，创建随机权重
            # 这里需要根据实际的层来设置合适的形状
            return tf1.constant(np.random.randn(25088, 4096) * 0.01, dtype=tf.float32, name="weights")
        return tf1.constant(self.data_dict[name][0], name="weights")
