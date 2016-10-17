#!/bin/bash

for file in "$1"/*.*; do
   destination="${file%.*}"
   echo "Extracting from $file..."
   mkdir -p "$destination"
   ffmpeg -i "$file" -r 1/1 "$destination/image_%03d.png"
done

echo ""
echo "Deleting images that are less than 40kb..."
find "$1" -name "*.png" -size -40k -delete
