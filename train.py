import tensorflow as tf
import tensorflow.compat.v1 as tf1  # 导入 TF 1.x 兼容模块
tf1.disable_v2_behavior()  # 禁用 TF 2.x 行为，启用 TF 1.x 行为

from model import CycleGAN
from reader import Reader
from datetime import datetime
import os
import logging
import subprocess
from utils import ImagePool
import argparse  # 替代 tf.flags

# 使用 argparse 替代 tf.flags
parser = argparse.ArgumentParser(description='Train CycleGAN model')
parser.add_argument('--batch_size', type=int, default=1, help='batch size, default: 1')
parser.add_argument('--image_size1', type=int, default=256, help='image size, default: 256')
parser.add_argument('--image_size2', type=int, default=256, help='image size, default: 256')
parser.add_argument('--use_lsgan', type=bool, default=True, help='use lsgan (mean squared error) or cross entropy loss, default: True')
parser.add_argument('--norm', type=str, default='instance', choices=['instance', 'batch'], help='[instance, batch] use instance norm or batch norm, default: instance')
parser.add_argument('--lambda1', type=float, default=10.0, help='weight for forward cycle loss (X->Y->X), default: 10.0')
parser.add_argument('--lambda2', type=float, default=10.0, help='weight for backward cycle loss (Y->X->Y), default: 10.0')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='initial learning rate for Adam, default: 0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of Adam, default: 0.5')
parser.add_argument('--pool_size', type=float, default=50, help='size of image buffer that stores previously generated images, default: 50')
parser.add_argument('--ngf', type=int, default=64, help='number of gen filters in first conv layer, default: 64')
parser.add_argument('--X', type=str, default='data/tfrecords/X.tfrecords', help='X tfrecords file for training, default: data/tfrecords/X.tfrecords')
parser.add_argument('--Y', type=str, default='data/tfrecords/Y.tfrecords', help='Y tfrecords file for training, default: data/tfrecords/Y.tfrecords')
parser.add_argument('--load_model', type=str, default=None, help='folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')
FLAGS, unparsed = parser.parse_known_args()


def train():
  if FLAGS.load_model is not None:
    checkpoints_dir = "checkpoints/" + FLAGS.load_model
  else:
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    #checkpoints_dir = "checkpoints/{}".format(current_time)
    checkpoints_dir="checkpoints/Hazy2GT"
    try:
      os.makedirs(checkpoints_dir)
    except os.error:
      pass

  graph = tf.Graph()
  with graph.as_default():
    cycle_gan = CycleGAN(
        X_train_file=FLAGS.X,
        Y_train_file=FLAGS.Y,
        batch_size=FLAGS.batch_size,
        image_size1=FLAGS.image_size1,
        image_size2=FLAGS.image_size2,
        use_lsgan=FLAGS.use_lsgan,
        norm=FLAGS.norm,
        lambda1=FLAGS.lambda1,
        lambda2=FLAGS.lambda2,
        learning_rate=FLAGS.learning_rate,
        beta1=FLAGS.beta1,
        ngf=FLAGS.ngf
    )
    G_loss, D_Y_loss, F_loss, D_X_loss, fake_y, fake_x  = cycle_gan.model()
    optimizers = cycle_gan.optimize(G_loss, D_Y_loss, F_loss, D_X_loss)

    summary_op = tf1.summary.merge_all()
    train_writer = tf1.summary.FileWriter(checkpoints_dir, graph)
    saver = tf1.train.Saver()

  config = tf1.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction = 0.5
  with tf1.Session(config=config, graph=graph) as sess:
    if FLAGS.load_model is not None:
      checkpoint = tf1.train.get_checkpoint_state(checkpoints_dir)
      meta_graph_path = "checkpoints/Hazy2GT/model.ckpt-200000.meta"
      print(tf1.train.latest_checkpoint(checkpoints_dir))
      restore = tf1.train.import_meta_graph(meta_graph_path)
      #restore.restore(sess, tf1.train.latest_checkpoint(checkpoints_dir))
      restore.restore(sess, "checkpoints/Hazy2GT/model.ckpt-200000")
      step = int(meta_graph_path.split("-")[1].split(".")[0])
    else:
      sess.run(tf1.global_variables_initializer())
      step = 0

    coord = tf1.train.Coordinator()
    threads = tf1.train.start_queue_runners(sess=sess, coord=coord)

    try:
      fake_Y_pool = ImagePool(FLAGS.pool_size)
      fake_X_pool = ImagePool(FLAGS.pool_size)

      while not coord.should_stop():
        # get previously generated images
        fake_y_val, fake_x_val = sess.run([fake_y, fake_x])

        # train
        _, G_loss_val, D_Y_loss_val, F_loss_val, D_X_loss_val, summary = (
              sess.run(
                  [optimizers, G_loss, D_Y_loss, F_loss, D_X_loss, summary_op],
                  feed_dict={cycle_gan.fake_y: fake_Y_pool.query(fake_y_val),
                             cycle_gan.fake_x: fake_X_pool.query(fake_x_val)}
              )
        )

        train_writer.add_summary(summary, step)
        train_writer.flush()

        if step % 100 == 0:
          logging.info('-----------Step %d:-------------' % step)
          logging.info('  G_loss   : {}'.format(G_loss_val))
          logging.info('  D_Y_loss : {}'.format(D_Y_loss_val))
          logging.info('  F_loss   : {}'.format(F_loss_val))
          logging.info('  D_X_loss : {}'.format(D_X_loss_val))

        if step % 500 == 0:
          save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
          logging.info("Model saved in file: %s" % save_path)
          subprocess.call("./create_model.sh")
        step += 1

    except KeyboardInterrupt:
      logging.info('Interrupted')
      coord.request_stop()
    except Exception as e:
      coord.request_stop(e)
    finally:
      save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
      logging.info("Model saved in file: %s" % save_path)
      # When done, ask the threads to stop.
      coord.request_stop()
      coord.join(threads)

def main(unused_argv):
  train()

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  # 直接调用 main 函数，替代 tf.app.run()
  main([])
