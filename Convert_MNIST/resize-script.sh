#!/bin/bash

#goal: resize any image format into 56x56 png file

if [ `ls test-images/*/*.jpg 2> /dev/null | wc -l ` -gt 0 ]; then
  echo hi
  for file in test-images/*/*.jpg; do
    convert "$file" -resize 56x56\! "${file%.*}.png"
    file "$file" #uncomment for testing
    rm "$file"
  done
fi

if [ `ls test-images/*/*.png 2> /dev/null | wc -l ` -gt 0 ]; then
  echo hi
  for file in test-images/*/*.png; do
    convert "$file" -resize 56x56\! "${file%.*}.png"
    file "$file" #uncomment for testing
  done
fi

if [ `ls training-images/*/*.jpg 2> /dev/null | wc -l ` -gt 0 ]; then
  echo hi
  for file in training-images/*/*.jpg; do
    convert "$file" -resize 56x56\! "${file%.*}.png"
    file "$file" #uncomment for testing
    rm "$file"
  done
fi

if [ `ls training-images/*/*.png 2> /dev/null | wc -l ` -gt 0 ]; then
  echo hi
  for file in training-images/*/*.png; do
    convert "$file" -resize 56x56\! "${file%.*}.png"
    file "$file" #uncomment for testing
  done
fi
