import tensorflow as tf
import tensorflow.compat.v1 as tf1  # 导入 TF 1.x 兼容模块
tf1.disable_v2_behavior()  # 禁用 TF 2.x 行为，启用 TF 1.x 行为

## Layers: follow by naming convention used in the original paper
### Generator layers
def c7s1_k(input, k, reuse=False, norm='instance', activation='relu', is_training=True, name='c7s1_k'):
  """ A 7x7 Convolution-BatchNorm-ReLU layer with k filters and stride 1
  Args:
    input: 4D tensor
    k: integer, number of filters (output depth)
    norm: 'instance' or 'batch' or None
    activation: 'relu' or 'tanh'
    name: string, e.g. 'c7sk-32'
    is_training: boolean or BoolTensor
    name: string
    reuse: boolean
  Returns:
    4D tensor
  """
  # 使用 tf1.variable_scope 替代 tf.variable_scope
  with tf1.variable_scope(name, reuse=reuse):
    weights = _weights("weights",
      shape=[7, 7, input.get_shape()[3], k])

    # 使用 tf1.pad 替代 tf.pad
    padded = tf1.pad(input, [[0,0],[3,3],[3,3],[0,0]], 'REFLECT')
    # 使用 tf1.nn.conv2d 替代 tf.nn.conv2d
    conv = tf1.nn.conv2d(padded, weights,
        strides=[1, 1, 1, 1], padding='VALID')

    normalized = _norm(conv, is_training, norm)

    if activation == 'relu':
      # 使用 tf1.nn.relu 替代 tf.nn.relu
      output = tf1.nn.relu(normalized)
    if activation == 'tanh':
      # 使用 tf1.nn.tanh 替代 tf.nn.tanh
      output = tf1.nn.tanh(normalized)
    return output

def dk(input, k, reuse=False, norm='instance', is_training=True, name=None):
  """ A 3x3 Convolution-BatchNorm-ReLU layer with k filters and stride 2
  Args:
    input: 4D tensor
    k: integer, number of filters (output depth)
    norm: 'instance' or 'batch' or None
    is_training: boolean or BoolTensor
    name: string
    reuse: boolean
    name: string, e.g. 'd64'
  Returns:
    4D tensor
  """
  # 使用 tf1.variable_scope 替代 tf.variable_scope
  with tf1.variable_scope(name, reuse=reuse):
    weights = _weights("weights",
      shape=[3, 3, input.get_shape()[3], k])

    # 使用 tf1.nn.conv2d 替代 tf.nn.conv2d
    conv = tf1.nn.conv2d(input, weights,
        strides=[1, 2, 2, 1], padding='SAME')
    normalized = _norm(conv, is_training, norm)
    # 使用 tf1.nn.relu 替代 tf.nn.relu
    output = tf1.nn.relu(normalized)
    return output


def Rk(input, k,  reuse=False, norm='instance', is_training=True, name=None):
  """ A residual block that contains two 3x3 convolutional layers
      with the same number of filters on both layer
  Args:
    input: 4D Tensor
    k: integer, number of filters (output depth)
    reuse: boolean
    name: string
  Returns:
    4D tensor (same shape as input)
  """
  # 使用 tf1.variable_scope 替代 tf.variable_scope
  with tf1.variable_scope(name, reuse=reuse):
    # 使用 tf1.variable_scope 替代 tf.variable_scope
    with tf1.variable_scope('layer1', reuse=reuse):
      weights1 = _weights("weights1",
        shape=[3, 3, input.get_shape()[3], k])
      # 使用 tf1.pad 替代 tf.pad
      padded1 = tf1.pad(input, [[0,0],[1,1],[1,1],[0,0]], 'REFLECT')
      # 使用 tf1.nn.conv2d 替代 tf.nn.conv2d
      conv1 = tf1.nn.conv2d(padded1, weights1,
          strides=[1, 1, 1, 1], padding='VALID')
      normalized1 = _norm(conv1, is_training, norm)
      # 使用 tf1.nn.relu 替代 tf.nn.relu
      relu1 = tf1.nn.relu(normalized1)

    # 使用 tf1.variable_scope 替代 tf.variable_scope
    with tf1.variable_scope('layer2', reuse=reuse):
      weights2 = _weights("weights2",
        shape=[3, 3, relu1.get_shape()[3], k])

      # 使用 tf1.pad 替代 tf.pad
      padded2 = tf1.pad(relu1, [[0,0],[1,1],[1,1],[0,0]], 'REFLECT')
      # 使用 tf1.nn.conv2d 替代 tf.nn.conv2d
      conv2 = tf1.nn.conv2d(padded2, weights2,
          strides=[1, 1, 1, 1], padding='VALID')
      normalized2 = _norm(conv2, is_training, norm)
    output = input+normalized2
    return output

def n_res_blocks(input, reuse, norm='instance', is_training=True, n=6):
  depth = input.get_shape()[3]
  for i in range(1,n+1):
    output = Rk(input, depth, reuse, norm, is_training, 'R{}_{}'.format(depth, i))
    input = output
  return output

def uk(input, k, reuse=False, norm='instance', is_training=True, name=None, output_size1=None, output_size2=None):
  """ A 3x3 fractional-strided-Convolution-BatchNorm-ReLU layer
      with k filters, stride 1/2
  Args:
    input: 4D tensor
    k: integer, number of filters (output depth)
    norm: 'instance' or 'batch' or None
    is_training: boolean or BoolTensor
    reuse: boolean
    name: string, e.g. 'c7sk-32'
    output_size: integer, desired output size of layer
  Returns:
    4D tensor
  """
  # 使用 tf1.variable_scope 替代 tf.variable_scope
  with tf1.variable_scope(name, reuse=reuse):
    input_shape = input.get_shape().as_list()

    weights = _weights("weights",
      shape=[3, 3, k, input_shape[3]])

    if not output_size1:
      output_size1 = input_shape[1]*2
      output_size2 = input_shape[2]*2
    output_shape = [input_shape[0], output_size1, output_size2, k]
    # 使用 tf1.nn.conv2d_transpose 替代 tf.nn.conv2d_transpose
    fsconv = tf1.nn.conv2d_transpose(input, weights,
        output_shape=output_shape,
        strides=[1, 2, 2, 1], padding='SAME')
    normalized = _norm(fsconv, is_training, norm)
    # 使用 tf1.nn.relu 替代 tf.nn.relu
    output = tf1.nn.relu(normalized)
    return output


### Discriminator layers
def Ck(input, k, slope=0.2, stride=2, reuse=False, norm='instance', is_training=True, name=None):
  """ A 4x4 Convolution-BatchNorm-LeakyReLU layer with k filters and stride 2
  Args:
    input: 4D tensor
    k: integer, number of filters (output depth)
    slope: LeakyReLU's slope
    stride: integer
    norm: 'instance' or 'batch' or None
    is_training: boolean or BoolTensor
    reuse: boolean
    name: string, e.g. 'C64'
  Returns:
    4D tensor
  """
  # 使用 tf1.variable_scope 替代 tf.variable_scope
  with tf1.variable_scope(name, reuse=reuse):
    weights = _weights("weights",
      shape=[4, 4, input.get_shape()[3], k])

    # 使用 tf1.nn.conv2d 替代 tf.nn.conv2d
    conv = tf1.nn.conv2d(input, weights,
        strides=[1, stride, stride, 1], padding='SAME')

    normalized = _norm(conv, is_training, norm)
    output = _leaky_relu(normalized, slope)
    return output


def last_conv(input, reuse=False, use_sigmoid=False, name=None):
  """ Last convolutional layer of discriminator network
      (1 filter with size 4x4, stride 1)
  Args:
    input: 4D tensor
    reuse: boolean
    use_sigmoid: boolean (False if use lsgan)
    name: string, e.g. 'C64'
  """
  # 使用 tf1.variable_scope 替代 tf.variable_scope
  with tf1.variable_scope(name, reuse=reuse):
    weights = _weights("weights",
      shape=[4, 4, input.get_shape()[3], 1])
    biases = _biases("biases", [1])

    # 使用 tf1.nn.conv2d 替代 tf.nn.conv2d
    conv = tf1.nn.conv2d(input, weights,
        strides=[1, 1, 1, 1], padding='SAME')
    output = conv + biases
    if use_sigmoid:
      # 使用 tf1.sigmoid 替代 tf.sigmoid
      output = tf1.sigmoid(output)
    return output

### Helpers
def _weights(name, shape, mean=0.0, stddev=0.02):
  """ Helper to create an initialized Variable
  Args:
    name: name of the variable
    shape: list of ints
    mean: mean of a Gaussian
    stddev: standard deviation of a Gaussian
  Returns:
    A trainable variable
  """
  # 使用 tf1.get_variable 替代 tf.get_variable
  # 使用 tf1.random_normal_initializer 替代 tf.random_normal_initializer
  var = tf1.get_variable(
    name, shape,
    initializer=tf1.random_normal_initializer(
      mean=mean, stddev=stddev, dtype=tf1.float32))
  return var

def _biases(name, shape, constant=0.0):
  """ Helper to create an initialized Bias with constant
  """
  # 使用 tf1.get_variable 替代 tf.get_variable
  # 使用 tf1.constant_initializer 替代 tf.constant_initializer
  return tf1.get_variable(name, shape,
            initializer=tf1.constant_initializer(constant))

def _leaky_relu(input, slope):
  # 使用 tf1.maximum 替代 tf.maximum
  return tf1.maximum(slope*input, input)

def _norm(input, is_training, norm='instance'):
  """ Use Instance Normalization or Batch Normalization or None
  """
  if norm == 'instance':
    return _instance_norm(input)
  elif norm == 'batch':
    return _batch_norm(input, is_training)
  else:
    return input

def _batch_norm(input, is_training):
  """ Batch Normalization
  """
  # 使用 tf1.variable_scope 替代 tf.variable_scope
  with tf1.variable_scope("batch_norm"):
    # 使用 tf1.layers.batch_normalization 替代 tf.contrib.layers.batch_norm
    return tf1.layers.batch_normalization(input,
                                        axis=3,
                                        momentum=0.99,
                                        epsilon=0.001,
                                        center=True,
                                        scale=True,
                                        training=is_training)

def _instance_norm(input):
  """ Instance Normalization
  """
  # 使用 tf1.variable_scope 替代 tf.variable_scope
  with tf1.variable_scope("instance_norm"):
    depth = input.get_shape()[3]
    scale = _weights("scale", [depth], mean=1.0)
    offset = _biases("offset", [depth])
    # 使用 tf1.nn.moments 替代 tf.nn.moments
    mean, variance = tf1.nn.moments(input, axes=[1,2], keep_dims=True)
    epsilon = 1e-5
    # 使用 tf1.rsqrt 替代 tf.rsqrt
    inv = tf1.rsqrt(variance + epsilon)
    normalized = (input-mean)*inv
    return scale*normalized + offset

def safe_log(x, eps=1e-12):
  # 使用 tf1.log 替代 tf.log
  return tf1.log(x + eps)
