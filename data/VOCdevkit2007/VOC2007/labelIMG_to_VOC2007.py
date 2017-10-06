#!/usr/bin/env python
# -*- coding=utf-8 -*- 

# Transfrom XMLs made from labelIMG to VOC2007 form
# Author : Yunqiu Xu

from xml.etree.ElementTree import ElementTree,Element
import os

def read_xml(xml_name):
    '''load xml file'''
    tree = ElementTree()
    tree.parse(xml_name)  
    return tree

def write_xml(tree, output_path):
    '''write the xml file'''
    tree.write(output_path, encoding="utf-8",xml_declaration=True)

def modify_xml(xml_name, xml_path, output_path):
    tree = read_xml(xml_path + xml_name)
    # print "get tree!"
    root = tree.getroot()
    # print "get root!"
    
    # <folder>ISIS --> <folder>VOC2007
    node_folder = tree.findall("folder")[0]
    node_folder.text = "VOC2007"
    # print "change <folder>"
    
    # remove <path>
    node_path = tree.findall("path")[0]
    root.remove(node_path)
    # print "remove <path>"
    
    # <database>The VOC2007 Database</database>
    node_source = tree.findall("source")[0]
    node_database = node_source.findall("database")[0]
    node_database.text = "The VOC2007 Database"
    # print "change <database>"
    
    #<annotation>PASCAL VOC2007</annotation>
    node_annotation = Element("annotation")
    node_annotation.text = "PASCAL VOC2007"
    node_source.append(node_annotation)
    # print "add <annotation>"
    
    write_xml(tree, output_path + xml_name)
    print xml_name + " is finished!"

if __name__ == "__main__":
    xml_path = "old_annotations/"
    output_path="Annotations/"
    files = os.listdir(xml_path)
    
    # test
    print files
    
    for name in files:
        print("Processing " + name)
        modify_xml(name, xml_path, output_path)
        
        
