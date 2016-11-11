# crop images for crystal and diamond

import cv2
import sys
import glob
from tqdm import tqdm

image_dir = sys.argv[1]
images = glob.glob(image_dir+'*.png')

print 'Cropping images'
for image in tqdm(images):
   img = cv2.imread(image)
   img = img[0:360, 120:520]
   #cv2.imshow('image',img)
   #cv2.waitKey(0)
   #cv2.destroyAllWindows()
   #exit()
   cv2.imwrite(image, img)
