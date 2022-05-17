#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob


pupil_radius_right = []
iris_radius_right = []
pupil_radius_left = []
iris_radius_left = []


for bb,img in enumerate (glob.glob("Database/*.jpg")):
    
    # Read image.
    img = cv2.imread(img, cv2.IMREAD_COLOR)

    # Convert to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #thresholding
    ret,thresh = cv2.threshold(gray,55,255,cv2.THRESH_BINARY_INV)
    #cv2.imshow('thresh', thresh)

    #choosing a 5*5 kernel full of ones for erosion
    kernel = np.ones((5,5),np.uint8)
    #closing small holes inside the foreground objects
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    #removing noise through erosion followed by dilation
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    im_floodfill = opening.copy()
    h, w = opening.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = opening | im_floodfill_inv


    #canny edge detection
    edges = cv2.Canny(im_out,100,200)
    #cv2.imshow("edges", edges)

    #finding circles using hough circle transform
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20,
                              param1=30, param2=25, minRadius=0, maxRadius=0)

    
    detected_circles = np.uint16(np.around(circles))
    
    b = 0
    for (x, y ,r) in detected_circles[0, :]:
        if r < 30:
            break
        
        cv2.circle(img, (x, y), r, (0, 255, 0), 3)
        cv2.circle(img, (x, y), 2, (0, 255, 255), 3)
        b=b+1
        cv2.circle(img, (x, y), r+70, (0, 0, 255), 3)

        if b == 1:
            pupil_radius_right.append(r)
            iris_radius_right.append(r+65)
        else:
            pupil_radius_left.append(r)
            iris_radius_left.append(r+65)
            

    #cv2.imshow('output',img)
    cv2.imwrite('output/output{}.jpg'.format(bb), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# In[4]:


import pandas as pd  

dictionary = {'pupil r1': pupil_radius_right,'pupil r2': pupil_radius_left, 'iris_r1': iris_radius_right, 'iris_r2': iris_radius_right}  
dataframe = pd.DataFrame(dictionary) 
dataframe.to_csv('radius.csv')


# In[ ]:




