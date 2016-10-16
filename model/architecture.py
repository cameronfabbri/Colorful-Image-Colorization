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

def deconv(inputs, stride, out_shape, kernel_size, num_features, idx, linear=False):
   with tf.variable_scope('fc{0}'.format(idx)) as scope:
      input_channels = inputs.get_shape()[3]

      filter_ = _variable_with_weight_decay('weights', shape=[kernel_size, kernel_size, num_features, input_channels], stddev=0.1, wd=FLAGS.weight_decay)
      strides=[1, stride, stride, 1]

      d_conv = tf.nn.conv2d_transpose(inputs, filter_, output_shape=out_shape, strides=strides, padding='SAME')

      if linear:
        return d_conv
      else:
        # activation function is lrelu as seen in super res paper
        leak = 0.2
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * d_conv + f2 * abs(d_conv)

def _phase_shift(I, r):
    bsize, a, b, c = I.get_shape().as_list()
    bsize = tf.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
    X = tf.reshape(I, (bsize, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(1, a, X)  # a, [bsize, b, r, r]
    X = tf.concat(2, [tf.squeeze(x) for x in X])  # bsize, b, a*r, r
    X = tf.split(1, b, X)  # b, [bsize, a*r, r]
    X = tf.concat(2, [tf.squeeze(x) for x in X])  # bsize, a*r, b*r
    return tf.reshape(X, (bsize, a*r, b*r, 1))

def PS(X, r, color=False):
    if color:
        Xc = tf.split(3, 3, X)
        X = tf.concat(3, [_phase_shift(x, r) for x in Xc])
    else:
        X = _phase_shift(X, r)
    return X

def inference(batch_size, images, name):

           # input, kernel size, stride, num_features, num_
   #conv1 = tf.nn.dropout(images, .8)
   print images
   out_shape = tf.pack([images.get_shape()[0], 144, 160, 64])
   d_conv1 = deconv(images, 1, out_shape, 5, 64, '1') 
   print d_conv1.get_shape() 

   out_shape = tf.pack([images.get_shape()[0], 144, 160, 64])
   d_conv2 = deconv(d_conv1, 1, out_shape, 5, 64, '2') 
   print(d_conv2.get_shape())

   out_shape = tf.pack([images.get_shape()[0], 144, 160, 3*16])
   d_conv3 = deconv(d_conv2, 1, out_shape, 5, 3*16, '3', True)
   print(d_conv3.get_shape())
   d_conv3 = PS(d_conv3, 4, color=True)
   #d_conv3 = tf.nn.sigmoid(d_conv3)
   d_conv3 = tf.nn.tanh(d_conv3)
   print(d_conv3.get_shape())
   tf.image_summary("generated", d_conv3, max_images=100) 

   return d_conv3

def loss (input_images, predicted_images):
   ### possible loss cross entropy loss ###
   #epsilon = 1e-12 
   #error = tf.reduce_mean(-(input_images * tf.log(predicted_images + epsilon) + (1.0 - input_images) * tf.log(1.0 - predicted_images + epsilon)))
   error = tf.nn.l2_loss(input_images - predicted_images)
   return error 


