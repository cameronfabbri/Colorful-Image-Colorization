import tensorflow as tf
import numpy as np
import sys

FLAGS = tf.app.flags.FLAGS

num_epochs = 100

tf.app.flags.DEFINE_float('weight_decay', 0.0005,
                          """ """)
tf.app.flags.DEFINE_float('alpha', 0.1,
                          """Leaky RElu param""")

def _variable_on_cpu(name, shape, initializer):
   with tf.device('/cpu:0'):
      var = tf.get_variable(name, shape, initializer=initializer)
   return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev))
  if wd:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    weight_decay.set_shape([])
    tf.add_to_collection('losses', weight_decay)
  return var


def _conv_layer(inputs, kernel_size, stride, num_features, idx):
   with tf.variable_scope('{0}_conv'.format(idx)) as scope:
      input_channels = inputs.get_shape()[3]

      weights = _variable_with_weight_decay('weights', shape=[kernel_size, kernel_size, input_channels, num_features], stddev=0.1, wd=FLAGS.weight_decay)
      biases = _variable_on_cpu('biases', [num_features], tf.constant_initializer(0.1))

      conv = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1], padding='SAME')
      conv_biased = tf.nn.bias_add(conv, biases)

      #Leaky ReLU
      conv_rect = tf.maximum(FLAGS.alpha*conv_biased, conv_biased, name='{0}_conv'.format(idx))
      return conv_rect


def _fc_layer(inputs, hiddens, idx, flat = False, linear = False):
  with tf.variable_scope('fc{0}'.format(idx)) as scope:
    input_shape = inputs.get_shape().as_list()
    if flat:
      dim = input_shape[1]*input_shape[2]*input_shape[3]
      inputs_processed = tf.reshape(inputs, [-1,dim])
    else:
      dim = input_shape[1]
      inputs_processed = inputs

    weights = _variable_with_weight_decay('weights', shape=[dim,hiddens],stddev=0.01, wd=FLAGS.weight_decay)
    biases = _variable_on_cpu('biases', [hiddens], tf.constant_initializer(0.01))
    if linear:
      return tf.add(tf.matmul(inputs_processed,weights),biases,name=str(idx)+'_fc')

    ip = tf.add(tf.matmul(inputs_processed,weights),biases)
    return tf.maximum(FLAGS.alpha*ip,ip,name=str(idx)+'_fc')

def inference(images, name):
   print '\n \n \n'
   print images
   print
   conv1 = _conv_layer(images, 3, 1, 32, 1)
   print conv1
   conv2 = _conv_layer(conv1, 3, 1, 32, 2)
   print conv2
   conv3 = _conv_layer(conv2, 3, 1, 64, 3)
   print conv3
   conv4 = _conv_layer(conv3, 3, 1, 64, 4)
   print conv4
   conv5 = _conv_layer(conv4, 3, 1, 128, 5)
   print conv5
   conv6 = _conv_layer(conv5, 3, 1, 128, 6)
   conv7 = _conv_layer(conv6, 3, 1, 256, 7)
   conv8 = _conv_layer(conv7, 3, 1, 256, 8)
   conv9 = _conv_layer(conv8, 3, 1, 128, 9)
   conv10 = _conv_layer(conv9, 3, 1, 128, 10)
   conv11 = _conv_layer(conv10, 1, 1, 64, 11)
   conv12 = _conv_layer(conv11, 1, 1, 64, 12)
   conv13 = _conv_layer(conv12, 1, 1, 32, 13)
   conv14 = _conv_layer(conv13, 1, 1, 32, 14)
   conv15 = _conv_layer(conv14, 1, 1, 16, 15)
   conv16 = _conv_layer(conv15, 1, 1, 16, 16)
   conv17 = _conv_layer(conv16, 1, 1, 8, 17)
   conv18 = _conv_layer(conv17, 1, 1, 3, 18)
   return conv18

 
def loss (input_images, predicted_images):
   error = tf.nn.l2_loss(input_images - predicted_images)
   return error 
