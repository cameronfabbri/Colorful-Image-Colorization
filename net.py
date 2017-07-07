import tensorflow as tf
import sys

'''
   Leaky RELU
   https://arxiv.org/pdf/1502.01852.pdf
'''
def lrelu(x, leak=0.2, name='lrelu'):
   return tf.maximum(leak*x, x)

def architecture(gray_image, train=True):
   conv1 = lrelu(tf.layers.conv2d(gray_image, 32, 1, strides=1, name='conv1',padding='VALID'))
   print 'conv1:',conv1
   conv2 = lrelu(tf.layers.conv2d(conv1, 32, 1, strides=1, name='conv2',padding='VALID'))
   print 'conv2:',conv2
   conv3 = lrelu(tf.layers.conv2d(conv2, 64, 1, strides=1, name='conv3',padding='VALID'))
   print 'conv3:',conv3
   conv4 = lrelu(tf.layers.conv2d(conv3, 64, 1, strides=1, name='conv4',padding='VALID'))
   print 'conv4:',conv4
   conv5 = lrelu(tf.layers.conv2d(conv4, 128, 1, strides=1, name='conv5',padding='VALID'))
   print 'conv5:',conv5
   conv6 = lrelu(tf.layers.conv2d(conv5, 128, 1, strides=1, name='conv6',padding='VALID'))
   print 'conv6:',conv6
   conv7 = lrelu(tf.layers.conv2d(conv6, 256, 1, strides=1, name='conv7',padding='VALID'))
   print 'conv7:',conv7
   conv8 = lrelu(tf.layers.conv2d(conv7, 256, 1, strides=1, name='conv8',padding='VALID'))
   print 'conv8:',conv8
   conv9 = lrelu(tf.layers.conv2d(conv8, 128, 1, strides=1, name='conv9',padding='VALID'))
   print 'conv9:',conv9
   conv10 = lrelu(tf.layers.conv2d(conv9, 128, 1, strides=1, name='conv10',padding='VALID'))
   print 'conv10:',conv10
   conv11 = lrelu(tf.layers.conv2d(conv10, 64, 1, strides=1, name='conv11',padding='VALID'))
   print 'conv11:',conv11
   conv12 = lrelu(tf.layers.conv2d(conv11, 64, 1, strides=1, name='conv12',padding='VALID'))
   print 'conv12:',conv12
   conv13 = lrelu(tf.layers.conv2d(conv12, 32, 1, strides=1, name='conv13',padding='VALID'))
   print 'conv13:',conv13
   conv14 = lrelu(tf.layers.conv2d(conv13, 32, 1, strides=1, name='conv14',padding='VALID'))
   print 'conv14:',conv14
   conv15 = lrelu(tf.layers.conv2d(conv14, 16, 1, strides=1, name='conv15',padding='VALID'))
   print 'conv15:',conv15
   conv16 = lrelu(tf.layers.conv2d(conv15, 16, 1, strides=1, name='conv16',padding='VALID'))
   print 'conv16:',conv16
   conv17 = lrelu(tf.layers.conv2d(conv16, 8, 1, strides=1, name='conv17',padding='VALID'))
   print 'conv17:',conv17
   if train: conv17 = tf.nn.dropout(conv17, 0.8)
   conv18 = lrelu(tf.layers.conv2d(conv17, 3, 1, strides=1, name='conv18',padding='VALID'))
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
