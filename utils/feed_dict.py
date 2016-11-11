'''
Cameron Fabbri

Feed dictionary for grabbing batches for training

'''
from random import shuffle
import os
import cv2
import numpy as np

def get_batch(batch_size, image_list, normalize):
   shuffle(image_list)
   image_list = image_list[:batch_size]

   original_images = []
   gray_images = []

   for image in image_list:
      filename, file_extension = os.path.splitext(image)
      gray_image = filename+'_gray.png'

      original_img = cv2.imread(image)
      original_img = original_img.astype('float')

      try:
         gray_img = cv2.imread(gray_image)
         gray_img = gray_img.astype('float')
      except:
         print 'No image ' + gray_image
         continue

      if normalize:
         original_img = original_img/255.0
         gray_img = gray_img/255.0

      original_images.append(original_img)
      gray_images.append(gray_img)

   original_images = np.asarray(original_images)
   gray_images     = np.asarray(gray_images)
   return original_images, gray_images

