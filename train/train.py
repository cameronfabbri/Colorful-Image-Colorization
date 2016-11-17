import tensorflow as tf
import numpy as np
import os
import sys
import numpy as np
import cv2
from optparse import OptionParser
import fnmatch

sys.path.insert(0, '../utils/')
sys.path.insert(0, '../model/')

import architecture
import time
import feed_dict as fd

def get_feed_dict(batch_size, original_images_placeholder, gray_images_placeholder, image_list, normalize):

   original_images, gray_images = fd.get_batch(batch_size, image_list, normalize)
   print original_images.shape
   exit()


   feed_dict = {
      original_images_placeholder: original_images,
      gray_images_placeholder: gray_images
   }

   return feed_dict


def train(checkpoint_dir, image_list, batch_size, normalize):
   with tf.Graph().as_default():

      global_step = tf.Variable(0, name='global_step', trainable=False)

      original_images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 144, 160, 3)) 
      gray_images_placeholder     = tf.placeholder(tf.float32, shape=(batch_size, 144, 160, 3)) 

      # image summary for tensorboard
      tf.image_summary('original_images', original_images_placeholder, max_images=100)
      tf.image_summary('gray_images', gray_images_placeholder, max_images=100)

      logits = architecture.inference(gray_images_placeholder, "train")
      loss   = architecture.loss(original_images_placeholder, logits)

      tf.scalar_summary('loss', loss)
      
      train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step=global_step)

      # summary for tensorboard graph
      summary_op = tf.merge_all_summaries()

      variables = tf.all_variables()
      init      = tf.initialize_all_variables()
      sess      = tf.Session()

      try:
         os.mkdir(checkpoint_dir)
      except:
         pass

      sess.run(init)
      print "\nRunning session\n"

      # saver for the model
      saver = tf.train.Saver(tf.all_variables())
      
      tf.train.start_queue_runners(sess=sess)

      # restore previous model if one
      ckpt = tf.train.get_checkpoint_state(checkpoint_dir+"training")
      if ckpt and ckpt.model_checkpoint_path:
         print "Restoring previous model..."
         try:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print "Model restored"
         except:
            print "Could not restore model"
            pass

      # Summary op
      graph_def = sess.graph.as_graph_def(add_shapes=True)
      summary_writer = tf.train.SummaryWriter(checkpoint_dir+"training", graph_def=graph_def)

      # Constants
      step = int(sess.run(global_step))
      #epoch_num = step/(train_size/batch_size)

      while True:
         step += 1
         feed_dict = get_feed_dict(batch_size, original_images_placeholder, gray_images_placeholder, image_list, normalize)
         _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

         if step % 10 == 0:
            print " Step: " + str(sess.run(global_step)) + " Loss: " + str(loss_value)
        
         # save tensorboard stuff
         #if step%200 == 0:
         #   summary_str = sess.run(summary_op)
         #   summary_writer.add_summary(summary_str, step)

         if step%100 == 0:
            print "Saving model"
            print
            saver.save(sess, checkpoint_dir+"training/checkpoint", global_step=global_step)
            print
         

def main(argv=None):
   parser = OptionParser(usage='usage')
   parser.add_option('-c', '--checkpoint_dir',          type='str')
   parser.add_option('-b', '--batch_size', default=100, type='int')
   parser.add_option('-d', '--data_dir', type='str')
   parser.add_option('-n', '--normalize', type='str')

   opts, args = parser.parse_args()
   opts = vars(opts)

   checkpoint_dir = opts['checkpoint_dir']
   batch_size     = opts['batch_size']
   data_dir       = opts['data_dir']
   normalize      = opts['normalize']

   if normalize == 'y':
      normalize = True
   else:
      normalize = False

   if checkpoint_dir is None:
      print "checkpoint_dir is required"
      exit()

   print
   print 'checkpoint_dir: ' + str(checkpoint_dir)
   print 'batch_size:     ' + str(batch_size)
   print 'data_dir:       ' + str(data_dir)
   print

   #answer = raw_input("All correct?\n:")
   #if answer == "n":
   #   exit()

   pattern = "*resized.png"
   image_list = list()
   for d, s, fList in os.walk(data_dir):
      for filename in fList:
         if fnmatch.fnmatch(filename, pattern):
            image_list.append(os.path.join(d,filename))

   print str(len(image_list)) + ' images...'
   train(checkpoint_dir, image_list, int(batch_size), normalize)


if __name__ == "__main__":

   if sys.argv[1] == "--help" or sys.argv[1] == "-h" or len(sys.argv) < 2:
      print
      print "-c --checkpoint_dir <str> [path to save the model]"
      print "-b --batch_size     <int> [batch size]"
      print "-d --data_dir       <str> [path to root image folder]"
      print '-n --normalize      <str> [y/n normalize training images]'
      print
      exit()


   tf.app.run()

