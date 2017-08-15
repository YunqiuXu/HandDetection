#!/bin/bash

# Get the predict result of test set
# How to use: ./get_test_pred.sh < test.txt

while read filename
do 
        exist=`egrep "$filename" YES.txt | wc -l`
        if [ $exist -gt 0 ] 
        then 
                echo "1" >> test_pred.txt
        else
                echo "0" >> test_pred.txt
        fi
done
