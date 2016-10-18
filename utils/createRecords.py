import tensorflow as tf
from tqdm import tqdm
import sys
import cv2
import numpy as np
import os
import fnmatch

shape  = (160,144)

# helper function
def _bytes_feature(value):
   return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def run(folder, dest_dir):
   
   record = dest_dir+"images.tfrecord"

   count = 0

   print record
   if os.path.isfile(record):
      if raw_input("Record file already exists! Overwrite? (y/n)") == "n":
         exit()

   record_writer = tf.python_io.TFRecordWriter(record)

   pattern = "*.png"
   fileList = list()

   for d, s, fList in os.walk(folder):
      for filename in fList:
         if fnmatch.fnmatch(filename, pattern):
            fileList.append(os.path.join(d,filename))
   print "Getting " + str(len(fileList)) + " images..."

   for image_name in tqdm(fileList):
      
      try:
         img = cv2.imread(image_name)
         #img2 = np.zeros_like(img)
      except:
         print "corrupt image: " + str(image_name)
         continue
      
      # make a copy of the image for the desired output of the network
      try:
         original_image = img.copy()
      except:
         print "Couldn't copy image " + str(image_name)
         pass
     

      try:
         # make image gray
         img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

         # resize gray image
         img_gray = cv2.resize(img_gray, shape, interpolation=cv2.INTER_CUBIC)
         #img2 = cv2.resize(img2, shape, interpolation=cv2.INTER_CUBIC)
         #img2 = np.expand_dims(img2, axis=2)

         #img2[:,:,0] = img_gray
         #img2[:,:,1] = img_gray
         #img2[:,:,2] = img_gray
         
         # resize original image
         original_image = cv2.resize(original_image, shape, interpolation=cv2.INTER_CUBIC)

      except:
         print "Error with " + str(image_name)
         raise
         exit()
         continue

      # flatten image
      img2 = np.reshape(img_gray, [1, shape[0]*shape[1]])
      original_image = np.reshape(original_image, [1, shape[0]*shape[1]*3])

      example = tf.train.Example(features=tf.train.Features(feature={
         'original_image': _bytes_feature(original_image.tostring()),
         'gray_image': _bytes_feature(img2.tostring())}))

      try:
         record_writer.write(example.SerializeToString())
      except:
         raise
      count += 1

      if count == 200:
         break

   print "Created " + str(count) + " records"

if __name__ == "__main__":

   if len(sys.argv) < 3:
      print "Usage: python createRecords.py [source directory] [destination directory]"
      exit()

   folder = sys.argv[1]
   dest_dir  = sys.argv[2]

   run(folder, dest_dir)

