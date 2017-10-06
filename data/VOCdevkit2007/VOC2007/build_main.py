#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build train / val indexes
Create folder ImageSets/Main, then run this file
"""


import os

annotation_path = "./Annotations/" 
result_path = "./ImageSets/Main/"

ratio_trainval = 0.9  #(trainval/total) 
ratio_train = 0.8    #(train/trainval)

def create_train_val_trainval():
    files = os.listdir(annotation_path)
    total_cases = len(files)
    t2 = int(ratio_trainval*total_cases)
    t1 = int(ratio_train*t2)
    train_cases=files[:t1]
    val_cases=files[t1:t2]
    test_cases=files[t2:]
    
    train_txt = ""
    val_txt = ""
    trainval_txt = ""
    test_txt=""
    
    for file in train_cases:
        train_txt += file[:-4] + "\n"  #Delete ".xml"
    for file in val_cases:
        val_txt += file[:-4] + "\n"
    trainval_txt = train_txt+val_txt
    for file in test_cases:
        test_txt += file[:-4] + "\n"
        
    f = open(result_path+"train.txt","w")
    f.write(train_txt)
    f.close()
    f = open(result_path+"val.txt","w")
    f.write(val_txt)
    f.close()
    f = open(result_path+"trainval.txt","w")
    f.write(trainval_txt)
    f.close()
    f = open(result_path+"test.txt","w")
    f.write(test_txt)
    f.close()

def create_train_for_classes():               #Not being used so far
    files = os.listdir(annotation_path)
    total_cases = len(files)
    total_train = 3
    total_test = 0
    record = [[],[]]
    names = ["YES", "NO"]
    
    
    train_cases = files[:total_train]
    for case in train_cases:
        file = open(annotation_path+case)
        lines = file.readlines()
        lines = lines[1:]                  #ignore first line
        indicator = [-1,-1]
        
        for line in lines:
            line = line.strip().split(" ")
            name = line[0]
            if name == "YES":
                indicator[0] = 1
            elif name == "NO":
                indicator[1] = 1
            else:
                pass
        for i in range(2):
            record[i].append((case,indicator[i]))

    for i in range(2):
        file_path=result_path+names[i]+"_train"+".txt"
        content=""
        for k in record[i]:
            content+=k[0]+" "+str(k[1])+"\n"
        f=open(file_path,"w")
        f.write(content)
        f.close()
    #print(record)   
        
if __name__ == '__main__':
    create_train_val_trainval()
    
