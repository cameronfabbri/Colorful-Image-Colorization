import tensorflow as tf
import sys
import cv2

sys.path.insert(0, '../utils/')

def read_and_decode(filename_queue):

   reader = tf.TFRecordReader()
   _, serialized_example = reader.read(filename_queue)
   features = tf.parse_single_example(
      serialized_example,
      features={
         'gray_image': tf.FixedLenFeature([], tf.string),
         'original_image': tf.FixedLenFeature([], tf.string),
      }
   )

   gray_image = tf.decode_raw(features['gray_image'], tf.uint8)
   gray_image = tf.to_float(gray_image, name='float32')
   gray_image = tf.reshape(gray_image, [160*2, 144*2, 3])
   gray_image = gray_image/255.0
   
   original_image = tf.decode_raw(features['original_image'], tf.uint8)
   original_image = tf.to_float(original_image, name='float32')
   original_image = tf.reshape(original_image, [160*2,144*2,3])
   original_image = original_image/255.0

   return original_image, gray_image


def inputs(record_file, batch_size, type_):
   print(record_file)
   filename_queue = tf.train.string_input_producer([record_file])

   original_image, gray_image = read_and_decode(filename_queue)

   original_images, gray_images = tf.train.shuffle_batch([original_image, gray_image], 
      batch_size=batch_size, 
      num_threads=5,
      capacity=500+3*batch_size, 
      min_after_dequeue=500)
   
   return original_images, gray_images

