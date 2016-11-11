#!/bin/bash

for f in *.jpg; do
   filename=$(basename "$f")
   filename="${filename%.*}"
   convert "$f" "$filename.png" 
done

for f in *.png; do
   convert "$f" -resize 200% "../resized/$f"
done
