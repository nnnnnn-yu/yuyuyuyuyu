import tensorflow as tf
import random
import os
from PIL import Image
import argparse  # 替代 tf.flags

try:
  from os import scandir
except ImportError:
  # Python 2 polyfill module
  from scandir import scandir

# 使用 argparse 替代 tf.flags
parser = argparse.ArgumentParser(description='Convert image data to tfrecords')
parser.add_argument('--X_input_dir', type=str, default='data/trainA',
                    help='X input directory, default: data/trainA')
parser.add_argument('--Y_input_dir', type=str, default='data/trainB',
                    help='Y input directory, default: data/trainB')
parser.add_argument('--X_output_file', type=str, default='data/tfrecords/X.tfrecords',
                    help='X output tfrecords file, default: data/tfrecords/X.tfrecords')
parser.add_argument('--Y_output_file', type=str, default='data/tfrecords/Y.tfrecords',
                    help='Y output tfrecords file, default: data/tfrecords/Y.tfrecords')
FLAGS, unparsed = parser.parse_known_args()


def data_reader(input_dir, shuffle=True):
  """Read images from input_dir then shuffle them
  Args:
    input_dir: string, path of input dir, e.g., /path/to/dir
  Returns:
    file_paths: list of strings
  """
  file_paths = []

  for img_file in scandir(input_dir):
    if img_file.name.endswith('.jpg') and img_file.is_file():
      file_paths.append(img_file.path)

  if shuffle:
    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to label in the
    # saved TFRecord files. Make the randomization repeatable.
    shuffled_index = list(range(len(file_paths)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    file_paths = [file_paths[i] for i in shuffled_index]

  return file_paths


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(file_path, image_buffer):
  """Build an Example proto for an example.
  Args:
    file_path: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
  Returns:
    Example proto
  """
  file_name = file_path.split('/')[-1]

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/file_name': _bytes_feature(tf.compat.as_bytes(os.path.basename(file_name))),
      'image/encoded_image': _bytes_feature((image_buffer))
    }))
  return example

def data_writer(input_dir, output_file):
  """Write data to tfrecords
  """
  file_paths = data_reader(input_dir)

  # create tfrecords dir if not exists
  output_dir = os.path.dirname(output_file)
  try:
    os.makedirs(output_dir)
  except (os.error):
    pass

  images_num = len(file_paths)

  # 使用 tf.io.TFRecordWriter 替代 tf.python_io.TFRecordWriter
  writer = tf.io.TFRecordWriter(output_file)

  for i in range(len(file_paths)):
    file_path = file_paths[i]

    # 使用普通的文件读取替代 tf.gfile.FastGFile
    with open(file_path, 'rb') as f:
      image_data = f.read()

    example = _convert_to_example(file_path, image_data)
    writer.write(example.SerializeToString())

    if i % 500 == 0:
      print("Processed {}/{}.".format(i, images_num))
  print("Done.")
  writer.close()

def main(unused_argv):
  print("Convert X data to tfrecords...")
  data_writer(FLAGS.X_input_dir, FLAGS.X_output_file)
  print("Convert Y data to tfrecords...")
  data_writer(FLAGS.Y_input_dir, FLAGS.Y_output_file)

if __name__ == '__main__':
  # 直接调用 main 函数，替代 tf.app.run()
  main([])
