Convert MINST: converts a set of jpg and/or png images (or a mix) into mnist binary format for training 

Dependencies: imagemagick + PIL
sudo apt-get update
sudo apt-get install imagemagick php5-imagick
pip install pillow

Steps:
1. Copy-paste your jpg and/or png images into one of the class folders (0 -> benign, 1 -> malignant)
2. Change the appropriate labels in: batches.meta.txt
3. run: ./resize-script.sh to resize images to 56x56 pngs
4. run: python3 convert-mnist.py to get ubyte-3 format = same as MNIST dataset





