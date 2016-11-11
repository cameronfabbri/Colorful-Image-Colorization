#!/bin/bash

# converts all jpg to png
#for f in testing/*.jpg; do
#   filename=$(basename "$f")
#   filename="${filename%.*}"
#   convert "$f" "$filename.png"
#   rm $f
#done

for f in testing/*.png; do
   convert "$f" -resize 200% resized/$f
done

for f in output/*.png; do
   convert "$f" -resize 200% resized/$f
done

