import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys

'''
   Leaky RELU
   https://arxiv.org/pdf/1502.01852.pdf
'''
def lrelu(x, leak=0.2, name='lrelu'):
   return tf.maximum(leak*x, x)

def architecture(gray_image, train=True):
   conv1 = lrelu(slim.convolution(gray_image, 32, 3, stride=1, scope='conv1', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv2 = lrelu(slim.convolution(conv1, 32, 3, stride=1, scope='conv2', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv3 = lrelu(slim.convolution(conv2, 64, 3, stride=1, scope='conv3', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv4 = lrelu(slim.convolution(conv3, 64, 3, stride=1, scope='conv4', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv5 = lrelu(slim.convolution(conv4, 128, 3, stride=1, scope='conv5', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv6 = lrelu(slim.convolution(conv5, 128, 3, stride=1, scope='conv6', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv7 = lrelu(slim.convolution(conv6, 256, 3, stride=1, scope='conv7', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv8 = lrelu(slim.convolution(conv7, 256, 3, stride=1, scope='conv8', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv9 = lrelu(slim.convolution(conv8, 128, 3, stride=1, scope='conv9', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv10 = lrelu(slim.convolution(conv9, 128, 3, stride=1, scope='conv10', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv11 = lrelu(slim.convolution(conv10, 64, 1, stride=1, scope='conv11', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv12 = lrelu(slim.convolution(conv11, 64, 1, stride=1, scope='conv12', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv13 = lrelu(slim.convolution(conv12, 32, 1, stride=1, scope='conv13', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv14 = lrelu(slim.convolution(conv13, 32, 1, stride=1, scope='conv14', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv15 = lrelu(slim.convolution(conv14, 16, 1, stride=1, scope='conv15', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv16 = lrelu(slim.convolution(conv15, 16, 1, stride=1, scope='conv16', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv17 = lrelu(slim.convolution(conv16, 8, 1, stride=1, scope='conv17', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   if train: conv17 = tf.nn.dropout(conv17, 0.8)
   conv18 = lrelu(slim.convolution(conv17, 3, 1, stride=1, scope='conv18', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   if train: conv18 = tf.nn.dropout(conv18, 0.8)
   
   tf.add_to_collection('vars', conv1)
   tf.add_to_collection('vars', conv2)
   tf.add_to_collection('vars', conv3)
   tf.add_to_collection('vars', conv4)
   tf.add_to_collection('vars', conv5)
   tf.add_to_collection('vars', conv6)
   tf.add_to_collection('vars', conv7)
   tf.add_to_collection('vars', conv8)
   tf.add_to_collection('vars', conv9)
   tf.add_to_collection('vars', conv10)
   tf.add_to_collection('vars', conv11)
   tf.add_to_collection('vars', conv12)
   tf.add_to_collection('vars', conv13)
   tf.add_to_collection('vars', conv14)
   tf.add_to_collection('vars', conv15)
   tf.add_to_collection('vars', conv16)
   tf.add_to_collection('vars', conv17)
   tf.add_to_collection('vars', conv18)
   
   return conv18

'''
'''
def netG(L_image, batch_size):
   
   conv1 = slim.convolution(L_image, 64, 3, stride=2, activation_fn=tf.identity, scope='g_conv1')
   conv1 = lrelu(conv1)

   conv2 = slim.convolution(conv1, 128, 3, stride=1, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_conv2')
   conv2 = lrelu(conv2)

   conv3 = slim.convolution(conv2, 128, 3, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_conv3')
   conv3 = lrelu(conv3)

   conv4 = slim.convolution(conv3, 256, 1, stride=1, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_conv4')
   conv4 = lrelu(conv4)
   
   conv5 = slim.convolution(conv4, 256, 3, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_conv5')
   conv5 = lrelu(conv5)

   conv6 = slim.convolution(conv5, 512, 1, stride=1, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_conv6')
   conv6 = lrelu(conv6)

   # now conv6 is used for both mid-level network and global network
   # global
   glob_conv1 = slim.convolution(conv6, 512, 3, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_glob_conv1')
   glob_conv1 = lrelu(glob_conv1)
   
   glob_conv2 = slim.convolution(glob_conv1, 512, 3, stride=1, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_glob_conv2')
   glob_conv2 = lrelu(glob_conv2)
   
   glob_conv3 = slim.convolution(glob_conv2, 512, 3, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_glob_conv3')
   glob_conv3 = lrelu(glob_conv3)
   
   glob_conv4 = slim.convolution(glob_conv3, 512, 3, stride=1, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_glob_conv4')
   glob_conv4 = lrelu(glob_conv4)

   glob_conv4 = tf.reshape(glob_conv4, [batch_size, -1])

   glob_fc1 = slim.fully_connected(glob_conv4, 1024, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_glob_fc1')
   glob_fc1 = lrelu(glob_fc1)

   glob_fc2 = slim.fully_connected(glob_fc1, 512, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_glob_fc2')
   glob_fc2 = lrelu(glob_fc2)

   glob_fc3 = slim.fully_connected(glob_fc2, 256, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_glob_fc3')
   glob_fc3 = lrelu(glob_fc3)

   # mid level
   mid_conv1 = slim.convolution(conv6, 512, 3, stride=1, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_mid_conv1')
   mid_conv1 = lrelu(mid_conv1)

   mid_conv2 = slim.convolution(mid_conv1, 256, 3, stride=1, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_mid_conv2')
   mid_conv2 = lrelu(mid_conv2)

   # FUSION LAYER
   # stack the last glob_fc with mid_conv2 so it's 32, 64, 64, 256
   glob_fc3 = tf.tile(glob_fc3, [1, 20*20])
   glob_fc3 = tf.reshape(glob_fc3, [batch_size, 20, 20, 256])
   mid_conv2 = tf.concat([mid_conv2, glob_fc3], 3)

   # colorization network
   col_conv1 = slim.convolution(mid_conv2, 128, 3, stride=1, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_col_conv1')
   col_conv1 = lrelu(col_conv1)
   # upsample - double the size
   col_conv1 = tf.image.resize_nearest_neighbor(col_conv1, [64, 64])
   
   col_conv2 = slim.convolution(col_conv1, 64, 3, stride=1, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_col_conv2')
   col_conv2 = lrelu(col_conv2)
   
   col_conv3 = slim.convolution(col_conv2, 64, 3, stride=1, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_col_conv3')
   col_conv3 = lrelu(col_conv3)
   col_conv3 = tf.image.resize_nearest_neighbor(col_conv3, [160, 160])

   col_conv4 = slim.convolution(col_conv3, 32, 3, stride=1, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_col_conv4')
   col_conv4 = lrelu(col_conv4)
   
   col_conv5 = slim.convolution(col_conv4, 3, 3, stride=1, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_col_conv5')
   col_conv5 = tf.nn.tanh(col_conv5)

   print
   print 'conv1:',conv1
   print 'conv2:',conv2
   print 'conv3:',conv3
   print 'conv4:',conv4
   print 'conv5:',conv5
   print 'conv6:',conv6
   print 'glob_conv1:',glob_conv1
   print 'glob_conv2:',glob_conv2
   print 'glob_conv3:',glob_conv3
   print 'glob_conv4:',glob_conv4
   print 'glob_fc1:',glob_fc1
   print 'glob_fc2:',glob_fc2
   print 'glob_fc3:',glob_fc3
   print 'mid_conv1:',mid_conv1
   print 'mid_conv2:',mid_conv2
   print 'col_conv1:',col_conv1
   print 'col_conv2:',col_conv2
   print 'col_conv3:',col_conv3
   print 'col_conv4:',col_conv4
   print 'col_conv5:',col_conv5
   print
   tf.add_to_collection('vars', conv1)
   tf.add_to_collection('vars', conv2)
   tf.add_to_collection('vars', conv3)
   tf.add_to_collection('vars', conv4)
   tf.add_to_collection('vars', conv5)
   tf.add_to_collection('vars', conv6)
   tf.add_to_collection('vars', glob_conv1)
   tf.add_to_collection('vars', glob_conv2)
   tf.add_to_collection('vars', glob_conv3)
   tf.add_to_collection('vars', glob_conv4)
   tf.add_to_collection('vars', glob_fc1)
   tf.add_to_collection('vars', glob_fc2)
   tf.add_to_collection('vars', glob_fc3)
   tf.add_to_collection('vars', mid_conv1)
   tf.add_to_collection('vars', mid_conv2)
   tf.add_to_collection('vars', col_conv1)
   tf.add_to_collection('vars', col_conv2)
   tf.add_to_collection('vars', col_conv3)
   tf.add_to_collection('vars', col_conv4)
   tf.add_to_collection('vars', col_conv5)
   return col_conv5
