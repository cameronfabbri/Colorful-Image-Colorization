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
         'hd_image1': tf.FixedLenFeature([], tf.string),
         #'hd_image2': tf.FixedLenFeature([], tf.string),
         'img': tf.FixedLenFeature([], tf.string),
      }
   )

   hd_image1 = tf.decode_raw(features['hd_image1'], tf.uint8)
   hd_image1 = tf.to_float(hd_image1, name='float32')
   hd_image1 = tf.reshape(hd_image1, [576,640,3])
   hd_image1 = hd_image1/255.0
   ''' 
   hd_image2 = tf.decode_raw(features['hd_image2'], tf.uint8)
   hd_image2 = tf.to_float(hd_image2, name='float32')
   hd_image2 = tf.reshape(hd_image2, [720,800,3])
   hd_image2 = hd_image2/255.0
   '''
   img = tf.decode_raw(features['img'], tf.uint8)
   img = tf.to_float(img, name='float32')
   img = tf.reshape(img, [144,160,3])
   img = img/255.0

   return img, hd_image1


def inputs(record_file, batch_size, type_):
   print(record_file)
   filename_queue = tf.train.string_input_producer([record_file])

   img, hd_image1 = read_and_decode(filename_queue)

   imgs, hd_images1 = tf.train.shuffle_batch([img, hd_image1], 
      batch_size=batch_size, 
      num_threads=5,
      capacity=500+3*batch_size, 
      min_after_dequeue=500)
   
   return imgs, hd_images1

