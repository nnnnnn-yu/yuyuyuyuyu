import tensorflow as tf
import tensorflow.compat.v1 as tf1  # 导入 TF 1.x 兼容模块
tf1.disable_v2_behavior()  # 禁用 TF 2.x 行为，启用 TF 1.x 行为
import utils

class Reader():
  def __init__(self, tfrecords_file, image_size1=256, image_size2=256, min_queue_examples=1000, batch_size=1, num_threads=8, name=''):
    """
    Args:
      tfrecords_file: string, tfrecords file path
      min_queue_examples: integer, minimum number of samples to retain in the queue that provides of batches of examples
      batch_size: integer, number of images per batch
      num_threads: integer, number of preprocess threads
    """
    self.tfrecords_file = tfrecords_file
    self.image_size1 = image_size1
    self.image_size2 = image_size2
    self.min_queue_examples = min_queue_examples
    self.batch_size = batch_size
    self.num_threads = num_threads
    # 使用 tf1.TFRecordReader 替代 tf.TFRecordReader
    self.reader = tf1.TFRecordReader()
    self.name = name

  def feed(self):
    """
    Returns:
      images: 4D tensor [batch_size, image_width, image_height, image_depth]
    """
    # 使用 tf1.name_scope 替代 tf.name_scope
    with tf1.name_scope(self.name):
      # 使用 tf1.train.string_input_producer 替代 tf.train.string_input_producer
      filename_queue = tf1.train.string_input_producer([self.tfrecords_file])

      _, serialized_example = self.reader.read(filename_queue)
      # 使用 tf1.parse_single_example 替代 tf.parse_single_example
      features = tf1.parse_single_example(
          serialized_example,
          features={
            # 使用 tf1.FixedLenFeature 替代 tf.FixedLenFeature
            'image/file_name': tf1.FixedLenFeature([], tf.string),
            'image/encoded_image': tf1.FixedLenFeature([], tf.string),
          })

      image_buffer = features['image/encoded_image']
      # 使用 tf1.image.decode_jpeg 替代 tf.image.decode_jpeg
      image = tf1.image.decode_jpeg(image_buffer, channels=3)
      image = self._preprocess(image)
      # 使用 tf1.train.shuffle_batch 替代 tf.train.shuffle_batch
      images = tf1.train.shuffle_batch(
            [image], batch_size=self.batch_size, num_threads=self.num_threads,
            capacity=self.min_queue_examples + 3*self.batch_size,
            min_after_dequeue=self.min_queue_examples
          )

      # 使用 tf1.summary.image 替代 tf.summary.image
      tf1.summary.image('_input', images)
    return images

  def _preprocess(self, image):
    # 使用 tf1.image.resize_images 替代 tf.image.resize_images
    image = tf1.image.resize_images(image, size=(self.image_size1, self.image_size2))
    image = utils.convert2float(image)
    image.set_shape([self.image_size1, self.image_size2, 3])
    return image

def test_reader():
  TRAIN_FILE_1 = 'data/tfrecords/X.tfrecords'  # 更新文件名
  TRAIN_FILE_2 = 'data/tfrecords/Y.tfrecords'  # 更新文件名

  # 使用 tf1.Graph 替代 tf.Graph
  with tf1.Graph().as_default():
    reader1 = Reader(TRAIN_FILE_1, batch_size=2)
    reader2 = Reader(TRAIN_FILE_2, batch_size=2)
    images_op1 = reader1.feed()
    images_op2 = reader2.feed()

    # 使用 tf1.Session 替代 tf.Session
    sess = tf1.Session()
    # 使用 tf1.global_variables_initializer 替代 tf.global_variables_initializer
    init = tf1.global_variables_initializer()
    sess.run(init)

    # 使用 tf1.train.Coordinator 替代 tf.train.Coordinator
    coord = tf1.train.Coordinator()
    # 使用 tf1.train.start_queue_runners 替代 tf.train.start_queue_runners
    threads = tf1.train.start_queue_runners(sess=sess, coord=coord)

    try:
      step = 0
      while not coord.should_stop():
        batch_images1, batch_images2 = sess.run([images_op1, images_op2])
        print("image shape: {}".format(batch_images1))
        print("image shape: {}".format(batch_images2))
        print("="*10)
        step += 1
    except KeyboardInterrupt:
      print('Interrupted')
      coord.request_stop()
    except Exception as e:
      coord.request_stop(e)
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()
      coord.join(threads)

if __name__ == '__main__':
  test_reader()
