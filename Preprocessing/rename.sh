#!/bin/bash
# Rename the *.jpg files to 000001.jpg
i=0
for img in `ls *.jpg`
do 
    mv $img `printf  %.6d $i`.jpg
    i=`expr $i + 1`
done
