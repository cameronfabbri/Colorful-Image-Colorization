"""

Cameron Fabbri

Evaluation by just looking at the original image and the resulting image from the network

"""

import tensorflow as tf
import cv2
import sys
import numpy as np

sys.path.insert(0, '../utils/')
sys.path.insert(0, '../model/')

import architecture

def eval(checkpoint_dir, image):
   with tf.Graph().as_default() as graph:

      input_image = tf.placeholder(tf.float32, shape=(1,144,160,3))
      logit = architecture.inference(input_image, "test")

      variables = tf.all_variables()
      init      = tf.initialize_all_variables()
      sess      = tf.Session()
      saver     = tf.train.Saver(variables)

      tf.train.start_queue_runners(sess=sess)

      ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

      if ckpt and ckpt.model_checkpoint_path:
         print 'Restoring model...'
         try:
            saver.restore(sess, ckpt.model_checkpoint_path)
         except:
            print 'Could not restore model'
            raise
            exit()

      graph_def = sess.graph.as_graph_def(add_shapes=True)
      
      img = cv2.imread(image)

      if  len(img.shape) != 3:
         img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
         img = np.expand_dims(img, axis=2)

      if img.shape[0] != 160:
         img = cv2.resize(img, (160,144))

      img = img.astype('float')
      img = img/255.0

      fake = np.zeros((1,144,160,3))
      fake[0,:,:,:] = img

      gen_img = sess.run([logit], feed_dict={input_image:fake})[0]
      gen_img = gen_img*255
      image_name = image.split('.png')[0]+'_output.png'

      cv2.imwrite('../images/'+image_name, gen_img[0,:,:,:])

def main(argv=None):
   eval(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
   if len(sys.argv) < 2:
      print "Usage: python eval.py /path/to/model/ image.png"
      exit()
   tf.app.run()
