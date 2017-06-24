# --------------------------------------------------------
# Convert LISA dataset annotation to VOCdevkit annotation
# Written by Shaoshen Wang
# --------------------------------------------------------
#Usage:
#Put this script under train folder
#Create a new folder name "Annotations" in train folder
#Run this script

from xml.dom.minidom import Document
import xml.dom.minidom
import cv2
import os

annotation_path="./posGt/" 
image_path="./pos/"
output="./Annotations/"

def read_Lisa(name):
    file=annotation_path+name+".txt"
    f=open(file)
    lines=f.readlines()
    data=lines[1:]
    return data

def generate_xml(name,img_size):
    file=output+name+".xml"
    if os.path.exists(file):
        print("Same data name in the set!")
        pass
    else:
        #read data from LISA
        lines=read_Lisa(name)
        total_object=len(lines)
        
        
        doc=Document()
        annotation = doc.createElement('annotation')
        doc.appendChild(annotation)

        title = doc.createElement('folder')
        title_text = doc.createTextNode('VOC2007')
        title.appendChild(title_text)
        annotation.appendChild(title)

        img_name = name + '.jpg'

        title = doc.createElement('filename')
        title_text = doc.createTextNode(img_name)
        title.appendChild(title_text)
        annotation.appendChild(title)

        source = doc.createElement('source')
        annotation.appendChild(source)

        title = doc.createElement('database')
        title_text = doc.createTextNode('The VOC2007 Database')
        title.appendChild(title_text)
        source.appendChild(title)

        title = doc.createElement('annotation')
        title_text = doc.createTextNode('PASCAL VOC2007')
        title.appendChild(title_text)
        source.appendChild(title)

        size = doc.createElement('size')
        annotation.appendChild(size)

        title = doc.createElement('width')
        title_text = doc.createTextNode(str(img_size[1]))
        title.appendChild(title_text)
        size.appendChild(title)

        title = doc.createElement('height')
        title_text = doc.createTextNode(str(img_size[0]))
        title.appendChild(title_text)
        size.appendChild(title)

        title = doc.createElement('depth')
        title_text = doc.createTextNode(str(img_size[2]))
        title.appendChild(title_text)
        size.appendChild(title)

        # A loop for several objects to be detected
        #The bounding boxes are described using the top left point, a width, and a height [x y w h] in the 2D image plane.=>[xmin,ymin,xmax,ymax]
        for i in range(total_object):
            data=lines[i].strip().split(" ")
            name=data[0]
            x,y,w,h=int(data[1]),int(data[2]),int(data[3]),int(data[4])
            xmin,ymin,xmax,ymax=x,y-h,x+w,y
            
        
            object = doc.createElement('object')
            annotation.appendChild(object)
            title = doc.createElement('name')
            title_text = doc.createTextNode(name)
            title.appendChild(title_text)
            object.appendChild(title)

            bndbox = doc.createElement('bndbox')
            object.appendChild(bndbox)
            title = doc.createElement('xmin')
            title_text = doc.createTextNode(str(int(float(xmin))))
            title.appendChild(title_text)
            bndbox.appendChild(title)
            title = doc.createElement('ymin')
            title_text = doc.createTextNode(str(int(float(ymin))))
            title.appendChild(title_text)
            bndbox.appendChild(title)
            title = doc.createElement('xmax')
            title_text = doc.createTextNode(str(int(float(xmax))))
            title.appendChild(title_text)
            bndbox.appendChild(title)
            title = doc.createElement('ymax')
            title_text = doc.createTextNode(str(int(float(ymax))))
            title.appendChild(title_text)
            bndbox.appendChild(title)

        # Write the DOM object to file
        f = open(file,'w')
        f.write(doc.toprettyxml(indent = ''))
        f.close()


if __name__ == '__main__':
    files=os.listdir(annotation_path)
    for name in files:
        name=name[:-4]  #delete ".txt"
        print("Processing "+name)
        img_size = cv2.imread(image_path+name+".png").shape
        print(img_size)
        generate_xml(name,img_size)
    
    



