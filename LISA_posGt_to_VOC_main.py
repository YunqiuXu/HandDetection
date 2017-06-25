# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 19:26:54 2017


"""
# --------------------------------------------------------
#Transform posGt to VOC2007/imagesets/main
#Used for create train.txt,val.txt,trainval.txt under main folder in VOC2007
#Written by Shaoshen Wang
# --------------------------------------------------------
#Usage:
#Put this script under train folder
#Create a new folder named "Main" in train folder
#Run this script


import os

annotation_path = "./posGt/" 
result_path = "./Main/"

ratio=0.8  #(train/total) 

def create_train_val_trainval():
    files = os.listdir(annotation_path)
    total_cases = len(files)
    total_train = int(ratio*total_cases)   
    train_cases = files[:total_train]     #Spilt into train and validation
    val_cases = files[total_train:]
    train_txt = ""
    val_txt = ""
    trainval_txt = ""
    
    for file in train_cases:
        train_txt += file[:-4] + "\n"  #Delete ".txt"
    for file in val_cases:
        val_txt += file[:-4] + "\n"
    for file in files:
        trainval_txt += file[:-4] + "\n"
    
    f = open(result_path+"train.txt","w")
    f.write(train_txt)
    f.close()
    f = open(result_path+"val.txt","w")
    f.write(val_txt)
    f.close()
    f = open(result_path+"trainval.txt","w")
    f.write(trainval_txt)
    f.close()

def create_train_for_classes():               #Not being used so far
    files = os.listdir(annotation_path)
    total_cases = len(files)
    total_train = 3
    total_test = 0
    record = [[],[],[],[]]
    names = ["leftHand_driver","rightHand_driver","leftHand_passenger","rightHand_passenger"]    
    train_cases = files[:total_train]
    
    for case in train_cases:
        file = open(annotation_path+case)
        lines = file.readlines()
        lines = lines[1:]                  #ignore first line
        indicator = [-1,-1,-1,-1]
        
        for line in lines:
            line = line.strip().split(" ")
            name = line[0]
            if name == "leftHand_driver":
                indicator[0] = 1
            elif name == "rightHand_driver":
                indicator[1] = 1
            elif name == "leftHand_passenger":
                indicator[2] = 1
            elif name == "rightHand_passenger":
                indicator[3] = 1
            else:
                pass
        for i in range(4):
            record[i].append((case,indicator[i]))

    for i in range(4):
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
    
