#!/usr/bin/env python

# resize the images to 256 * 480
# put this srcipt into image folder
# python resize.py *.jpg
import sys, cv2
imgs = sys.argv[1:]
for img_name in imgs:
    img = cv2.imread(img_name)
    img = cv2.resize(img, (480, 256))
    cv2.imwrite(img_name, img)
