# Colorizing Images

**UPDATE - Completely cleaning up code for Tensorflow 1.0 and retraining models.**

A deep learning approach to colorizing images, specifically for Pokemon.

The current model was trained on screenshots taken from Pokemon Silver, Crystal,
and Diamond, then tested on Pokemon Blue Version. Sample results below.

## Basic Training Usage
The files in the `images/train` folder are as follows:

## Evaluating on Images
I've included a trained model in the `models/` directory that you can run your own images on.
You can either run the model on one image or a folder of images. For one image, run `eval_one.py`
and pass it the model and the image as parameters. To run it on multiple images, run `eval.py`
and pass it the model and the folder to the images. `eval.py` will save your images in the 
`output` folder, where as `eval_one.py` will save them in the current directory. Examples:

## Training your own data

There are scripts included to help create your own dataset, which is desirable because
the amount of data needed to obtain good results is a good amount. The results below
were trained on about 50,000 images.

The easiest method to obtain images is to extract them from Youtube walkthrough videos of
different games. Given that you have a folder with videos 

`videos/`

`video_1.mp4`
  
`video_2.mp4`
   
`...`


use `extract_frames.sh` to extract images from each video. Just pass it the folder containing images.

Depending on if the video had a border around the game, you may need to use `crop_images.py` to crop
out the border. There are comments in the script you can uncomment to view the image before it crops
all of them to be sure the cropping is correct.

## Results

