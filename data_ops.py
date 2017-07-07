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


def getPaths(data_dir, ext='jpg'):
   pattern   = '*.'+ext
   image_paths = []
   for d, s, fList in os.walk(data_dir):
      for filename in fList:
         if fnmatch.fnmatch(filename, pattern):
            image_paths.append(os.path.join(d,filename))
   return image_paths


def loadData(data_dir, batch_size, train=True):

   if data_dir is None or not os.path.exists(data_dir): raise Exception('data_dir does not exist')

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
      input_paths = [data_dir]

   decode = tf.image.decode_image

   if len(input_paths) == 0: raise Exception('data_dir contains no image files')
   else: print 'Found',len(input_paths),'images!'

   with tf.name_scope('load_images'):
      path_queue = tf.train.string_input_producer(input_paths, shuffle=train)
      reader = tf.WholeFileReader()
      paths, contents = reader.read(path_queue)
      raw_input_ = decode(contents)
      raw_input_ = tf.image.convert_image_dtype(raw_input_, dtype=tf.float32)

      raw_input_.set_shape([None, None, 3])

   inputs  = tf.image.rgb_to_grayscale(raw_input_)
   targets = raw_input_

   scale_size = 180
   height     = 160
   width      = 144

   seed = random.randint(0, 2**31 - 1)
   def transform(image):
      r = image
      r = tf.image.random_flip_left_right(r, seed=seed)
      r = tf.image.resize_images(r, [height, width], method=tf.image.ResizeMethod.AREA)
      #offset = tf.cast(tf.floor(tf.random_uniform([2], 0, scale_size - width + 1, seed=seed)), dtype=tf.int32)
      #r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], height, width)
      return r

   if train:
      input_images  = transform(inputs)
      target_images = transform(targets)
   else:
      input_images  = tf.image.resize_images(inputs, [160, 160], method=tf.image.ResizeMethod.AREA)
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
