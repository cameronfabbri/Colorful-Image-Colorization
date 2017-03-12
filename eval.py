import cPickle as pickle
from tqdm import tqdm
import tensorflow as tf
import colorarch
from scipy import misc
import numpy as np
import argparse
import ntpath
import sys
import os
import time
import glob

sys.path.insert(0, 'ops/')
sys.path.insert(0, 'config/')

import data_ops

if __name__ == '__main__':

   CHECKPOINT_DIR = 'checkpoints/'
   IMAGES_DIR = CHECKPOINT_DIR+'images/'
   BATCH_SIZE=16

   test_images = glob.glob(sys.argv[1]+'*.*')
   num_images = len(test_images)

   Data = data_ops.loadData(sys.argv[1], BATCH_SIZE, train=False)
   # The gray 'lightness' channel in range [-1, 1]
   gray_image   = Data.inputs
   
   # The color channels in [-1, 1] range
   color_image  = Data.targets

   # architecture from
   # http://hi.cs.waseda.ac.jp/~iizuka/projects/colorization/data/colorization_sig2016.pdf
   col_img = colorarch.architecture(gray_image, train=False)

   col_img = tf.image.convert_image_dtype(col_img, dtype=tf.uint8, saturate=True)
   
   saver = tf.train.Saver(max_to_keep=1)
   
   init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
   sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
   sess.run(init)

   ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
   # restore previous model if there is one
   if ckpt and ckpt.model_checkpoint_path:
      print "Restoring previous model..."
      try:
         saver.restore(sess, ckpt.model_checkpoint_path)
         print "Model restored"
      except:
         print "Could not restore model"
         pass
 
   ########################################### training portion
   start = time.time()
   coord = tf.train.Coordinator()
   threads = tf.train.start_queue_runners(sess, coord=coord)

   prediction = np.squeeze(np.asarray(sess.run(col_img)))
   misc.imsave(IMAGES_DIR+str(i)+'.png', prediction)
   exit()
   i = 1
   for p in prediction:
      print p.shape
      exit()
      i+=1
   print 'Done. Images are in',IMAGES_DIR
