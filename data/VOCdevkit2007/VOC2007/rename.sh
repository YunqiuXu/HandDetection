#!/bin/bash
# Rename the *.jpg files to 000001.jpg
i=0
for img in `ls JPEGImages/*`
do 
    mv $img JPEGImages/`printf  %.6d $i`.jpg
    i=`expr $i + 1`
done
