#!/bin/bash

# Copy test images to test_img_with_bb
# How to use: ./copy_test_images.sh < test.txt

while read line
do 
        filename=`echo "/home/venturer/google_image/ISIS/pos/$line.jpg"`
        mv $filename /home/venturer/google_image/ISIS/ISIS_result/test_img_with_bb/
done
                
