"""

Cameron Fabbri

Evaluation by just looking at the original image and the resulting image from the network

"""

import tensorflow as tf
import cv2
import sys
import numpy as np

sys.path.insert(0, '../utils/')
sys.path.insert(0, '../inputs/')
sys.path.insert(0, '../model/')

import config
import input_
import architecture
import Image

batch_size = config.batch_size
eval_dir = config.eval_dir
checkpoint_dir = config.checkpoint_dir

def eval():
   with tf.Graph().as_default() as graph:

      images = input_.inputs("test", batch_size)

      summary_op = tf.merge_all_summaries()

      summary_writer = tf.train.SummaryWriter(eval_dir, graph)

      logits = architecture.inference(images, "test")

      variables_to_restore = tf.all_variables()
      saver = tf.train.Saver(variables_to_restore)

      with tf.Session() as sess:

         ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
         saver.restore(sess, ckpt.model_checkpoint_path)

         global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
         coord = tf.train.Coordinator()

         try:
            tf.train.start_queue_runners(sess=sess)
            threads = []

            s = 0
            while not coord.should_stop():
               for q in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                  threads.extend(q.create_threads(sess, coord=coord, daemon=True, start=True))

                  imgs, gen_imgs = sess.run([images, logits])
                  for im, gen in zip(imgs, gen_imgs):
                     im = np.uint8(im)
                     im = cv2.resize(im, (200, 200))
                     gen = np.uint8(gen)
                     gen = cv2.resize(gen, (200, 200))

                     result = np.hstack((im, gen))
                     #cv2.imshow('result', result)
                     #cv2.waitKey(0)
                     #cv2.destroyAllWindows()
                     print "Step: " + str(s)
                     cv2.imwrite('../evaluations/images/image-'+str(s)+'.png', result)
                     s += 1
                     if s == int(sys.argv[1]):
                        print "Done"
                        exit()

         except Exception as e:
            print "Error"
            raise(e)
            coord.request_stop(e)
            exit()

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)


def main(argv=None):
   eval()

if __name__ == "__main__":

   if len(sys.argv) < 2:
      print "Usage: python eval.py [num_images]"
      print "Give the number of images to test as an arg"
      exit()

   tf.app.run()
