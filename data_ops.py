'''

Operations used for data management

MASSIVE help from https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py

'''

from __future__ import division
from __future__ import absolute_import

from scipy import misc
from skimage import color
import collections
import tensorflow as tf
import numpy as np
import math
import time
import random
import glob
import os
import fnmatch
import cPickle as pickle

Data = collections.namedtuple('trainData', 'paths, inputs, targets, count, steps_per_epoch')


def getPaths(data_dir, ext='png'):
   pattern   = '*.'+ext
   image_paths = []
   for d, s, fList in os.walk(data_dir):
      for filename in fList:
         if fnmatch.fnmatch(filename, pattern):
            image_paths.append(os.path.join(d,filename))
   return image_paths


def loadData(data_dir, batch_size, train=True):

   if data_dir is None or not os.path.exists(data_dir):
      raise Exception('data_dir does not exist')

   if train:
      pkl_train_file = 'pokemon.pkl'

      if os.path.isfile(pkl_train_file):
         print 'Found pickle file'
         train_paths = pickle.load(open(pkl_train_file, 'rb'))
      else:
         train_paths = getPaths(data_dir)
         random.shuffle(train_paths)

         pf   = open(pkl_train_file, 'wb')
         data = pickle.dumps(train_paths)
         pf.write(data)
         pf.close()
      input_paths = train_paths
   
   else:
      #test_paths = glob.glob(data_dir+'*.*')
      #input_paths = test_paths
      input_paths = [data_dir]

   decode = tf.image.decode_image

   if len(input_paths) == 0:
      raise Exception('data_dir contains no image files')

   with tf.name_scope('load_images'):
      path_queue = tf.train.string_input_producer(input_paths, shuffle=train)
      reader = tf.WholeFileReader()
      paths, contents = reader.read(path_queue)
      raw_input_ = decode(contents)
      raw_input_ = tf.image.convert_image_dtype(raw_input_, dtype=tf.float32)

      assertion = tf.assert_equal(tf.shape(raw_input_)[2], 3, message='image does not have 3 channels')
      with tf.control_dependencies([assertion]):
         raw_input_ = tf.identity(raw_input_)

      raw_input_.set_shape([None, None, 3])

   inputs  = tf.image.rgb_to_grayscale(raw_input_)
   targets = raw_input_

   # synchronize seed for image operations so that we do the same operations to both
   # input and output images
   flip = 1
   scale_size = 180
   CROP_SIZE = 160
   seed = random.randint(0, 2**31 - 1)
   def transform(image):
      r = image
      if flip:
         r = tf.image.random_flip_left_right(r, seed=seed)
      # area produces a nice downscaling, but does nearest neighbor for upscaling
      # assume we're going to be doing downscaling here
      r = tf.image.resize_images(r, [scale_size, scale_size], method=tf.image.ResizeMethod.AREA)
      offset = tf.cast(tf.floor(tf.random_uniform([2], 0, scale_size - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
      if scale_size > CROP_SIZE:
         r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
      elif scale_size < CROP_SIZE:
         raise Exception('scale size cannot be less than crop size')
      return r

   if train:
      with tf.name_scope('input_images'):
         input_images = transform(inputs)
      with tf.name_scope('target_images'):
         target_images = transform(targets)
   else:
      input_images = tf.image.resize_images(inputs, [160, 160], method=tf.image.ResizeMethod.AREA)
      target_images = tf.image.resize_images(targets, [160, 160], method=tf.image.ResizeMethod.AREA)

   paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, input_images, target_images], batch_size=batch_size)
   steps_per_epoch = int(math.ceil(len(input_paths) / batch_size))

   return Data(
      paths=paths_batch,
      inputs=inputs_batch,
      targets=targets_batch,
      count=len(input_paths),
      steps_per_epoch=steps_per_epoch,
   )
