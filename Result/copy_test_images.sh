#!/bin/bash

# Copy test images to test_img_with_bb
# How to use: ./copy_test_images.sh < test.txt

while read line
do 
        filename=`echo "/home/venturer/google_image/0-818_resized/0-818_images/$line.jpg"`
        cp $filename /home/venturer/google_image/0-818_resized/0-818_result/test_images
done
                
