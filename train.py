import cPickle as pickle
import tensorflow as tf
import colorarch
from scipy import misc
import numpy as np
import argparse
import ntpath
import sys
import os
import time

sys.path.insert(0, 'ops/')
sys.path.insert(0, 'config/')

import data_ops

if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument('--EPOCHS',required=False,default=10,type=int,help='Number of epochs to train for')
   parser.add_argument('--DATA_DIR',       required=True,help='Directory where data is')
   parser.add_argument('--BATCH_SIZE',     required=False,type=int,default=32,help='Batch size to use')
   a = parser.parse_args()

   EPOCHS     = a.EPOCHS
   DATA_DIR   = a.DATA_DIR
   BATCH_SIZE = a.BATCH_SIZE

   CHECKPOINT_DIR = 'checkpoints/'
   IMAGES_DIR = CHECKPOINT_DIR+'images/'

   try: os.mkdir(CHECKPOINT_DIR)
   except: pass
   try: os.mkdir(IMAGES_DIR)
   except: pass
   
   # write all this info to a pickle file in the experiments directory
   exp_info = dict()
   exp_info['EPOCHS'] = EPOCHS
   exp_info['DATA_DIR']        = DATA_DIR
   exp_info['BATCH_SIZE']      = BATCH_SIZE
   exp_pkl = open(CHECKPOINT_DIR+'info.pkl', 'wb')
   data = pickle.dumps(exp_info)
   exp_pkl.write(data)
   exp_pkl.close()
   
   print
   print 'EPOCHS:          ',EPOCHS
   print 'DATA_DIR:        ',DATA_DIR
   print 'BATCH_SIZE:      ',BATCH_SIZE
   print

   # global step that is saved with a model to keep track of how many steps/epochs
   global_step = tf.Variable(0, name='global_step', trainable=False)

   # load data
   Data = data_ops.loadData(DATA_DIR, BATCH_SIZE)
   # number of training images
   num_train = Data.count
   
   # The gray 'lightness' channel in range [-1, 1]
   #L_image   = Data.inputs
   gray_image   = Data.inputs
   
   # The color channels in [-1, 1] range
   #ab_image  = Data.targets
   color_image  = Data.targets

   # architecture from
   # http://hi.cs.waseda.ac.jp/~iizuka/projects/colorization/data/colorization_sig2016.pdf
   #col_img = colorarch.netG(L_image, BATCH_SIZE)
   col_img = colorarch.architecture(gray_image)
   
   #loss = tf.reduce_mean((ab_image-col_img)**2)
   loss = tf.reduce_mean(tf.nn.l2_loss(color_image-col_img))
   train_op = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss, global_step=global_step)
   saver = tf.train.Saver(max_to_keep=1)
   
   # tensorboard summaries
   tf.summary.scalar('loss', loss)
   
   init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
   sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
   sess.run(init)

   # write out logs for tensorboard to the checkpointSdir
   summary_writer = tf.summary.FileWriter(CHECKPOINT_DIR+'/logs/', graph=tf.get_default_graph())

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
   step = sess.run(global_step)
   coord = tf.train.Coordinator()
   threads = tf.train.start_queue_runners(sess, coord=coord)
   merged_summary_op = tf.summary.merge_all()
   start = time.time()
   
   epoch_num = step/(num_train/BATCH_SIZE)
   while epoch_num < EPOCHS:
      epoch_num = step/(num_train/BATCH_SIZE)
      s = time.time()
      sess.run(train_op)
      loss_, summary = sess.run([loss, merged_summary_op])
      summary_writer.add_summary(summary, step)
      summary_writer.add_summary(summary, step)
      print 'epoch:',epoch_num,'step:',step,'loss:',loss_,'time:',time.time()-s
      step += 1
      
      if step%500 == 0:
         print 'Saving model...'
         saver.save(sess, CHECKPOINT_DIR+'checkpoint-'+str(step))
         saver.export_meta_graph(CHECKPOINT_DIR+'checkpoint-'+str(step)+'.meta')
         print 'Model saved\n'

   print 'Finished training', time.time()-start
   saver.save(sess, CHECKPOINT_DIR+'checkpoint-'+str(step))
   saver.export_meta_graph(CHECKPOINT_DIR+'checkpoint-'+str(step)+'.meta')
   exit()
