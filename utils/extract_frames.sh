#!/bin/bash

# Script for extracting frames from multiple video files
# If you have a folder filled with videos like so...
#
# folder/
#    - video_1.mp4
#    - video_2.mp4
#    - ...

# you can simply run the script by doing `./extract_frames.sh folder/
# and it will create an individual folder for each video and place the
# images in their respective folders.

for file in "$1"/*.*; do
   destination="${file%.*}"
   echo "Extracting from $file..."
   mkdir -p "$destination"
   ffmpeg -i "$file" -r 1/1 "$destination/image_%03d.png"
done

echo ""
echo "Deleting images that are less than 40kb..."
find "$1" -name "*.png" -size -40k -delete
