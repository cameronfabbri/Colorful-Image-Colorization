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

