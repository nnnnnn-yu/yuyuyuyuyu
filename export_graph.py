""" Freeze variables and convert 2 generator networks to 2 GraphDef files.
This makes file size smaller and can be used for inference in production.
An example of command-line usage is:
python export_graph.py --checkpoint_dir checkpoints/20170424-1152 \
                       --XtoY_model apple2orange.pb \
                       --YtoX_model orange2apple.pb \
                       --image_size 256
"""

import tensorflow as tf
import tensorflow.compat.v1 as tf1  # 导入 TF 1.x 兼容模块
tf1.disable_v2_behavior()  # 禁用 TF 2.x 行为，启用 TF 1.x 行为
import os
from tensorflow.python.tools.freeze_graph import freeze_graph
from model import CycleGAN
import utils

FLAGS = tf1.flags.FLAGS  # 使用 tf1.flags

tf1.flags.DEFINE_string('checkpoint_dir', '', 'checkpoints directory path')
tf1.flags.DEFINE_string('XtoY_model', 'apple2orange.pb', 'XtoY model name, default: apple2orange.pb')
tf1.flags.DEFINE_string('YtoX_model', 'orange2apple.pb', 'YtoX model name, default: orange2apple.pb')
tf1.flags.DEFINE_integer('image_size1', '256', 'image size, default: 256')
tf1.flags.DEFINE_integer('image_size2', '256', 'image size, default: 256')
tf1.flags.DEFINE_integer('ngf', 64,
                        'number of gen filters in first conv layer, default: 64')
tf1.flags.DEFINE_string('norm', 'instance',
                       '[instance, batch] use instance norm or batch norm, default: instance')

def export_graph(model_name, XtoY=True):
  graph = tf1.Graph()  # 使用 tf1.Graph

  with graph.as_default():
    cycle_gan = CycleGAN(ngf=FLAGS.ngf, norm=FLAGS.norm, image_size1=FLAGS.image_size1, image_size2=FLAGS.image_size2)

    input_image = tf1.placeholder(tf1.float32, shape=[FLAGS.image_size1, FLAGS.image_size2, 3], name='input_image')
    cycle_gan.model()
    if XtoY:
      output_image = cycle_gan.G.sample(tf1.expand_dims(input_image, 0))
    else:
      output_image = cycle_gan.F.sample(tf1.expand_dims(input_image, 0))

    output_image = tf1.identity(output_image, name='output_image')
    restore_saver = tf1.train.Saver()  # 使用 tf1.train.Saver
    export_saver = tf1.train.Saver()

  with tf1.Session(graph=graph) as sess:  # 使用 tf1.Session
    sess.run(tf1.global_variables_initializer())  # 使用 tf1.global_variables_initializer
    latest_ckpt = tf1.train.latest_checkpoint(FLAGS.checkpoint_dir)
    restore_saver.restore(sess, latest_ckpt)
    output_graph_def = tf1.graph_util.convert_variables_to_constants(  # 使用 tf1.graph_util
        sess, graph.as_graph_def(), [output_image.op.name])

    tf1.train.write_graph(output_graph_def, 'pretrained', model_name, as_text=False)  # 使用 tf1.train.write_graph

def main(unused_argv):
  print('Export XtoY model...')
  export_graph(FLAGS.XtoY_model, XtoY=True)
  print('Export YtoX model...')
  export_graph(FLAGS.YtoX_model, XtoY=False)

if __name__ == '__main__':
  tf1.app.run()  # 使用 tf1.app.run
