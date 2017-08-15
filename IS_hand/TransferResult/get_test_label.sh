#!/bin/bash

# Get the label of test set
# How to use: ./get_test_label.sh < test.txt

while read line
do 
        filename=`echo "/home/venturer/google_image/ISIS/Annotations/$line.xml"`
        exist=`egrep "YES" $filename | wc -l`
        
        if [ $exist -gt 0 ] 
        then 
                echo "1" >> test_label.txt
        else
                echo "0" >> test_label.txt
        fi
done
                
