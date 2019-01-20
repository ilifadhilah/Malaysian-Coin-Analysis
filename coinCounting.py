"""
coinCounting.py

YOUR WORKING FUNCTION

"""
import numpy as np
import cv2
from matplotlib import pyplot as plt


##########################
def coinCount(coinMat, i):
    # Inputs
    # coinMat: 4-D numpy array of row*col*3*numImages, 
    #          numImage denote number of images in coin set (10 in this case)
    # i: coin set image number (1, 2, ... 10)
    # Output
    # ans: Total value of the coins in the image, in float type
    #
    #########################################################################
  
    rgbImg=coinMat[:,:,:,i]
    grayImg = cv2.cvtColor(rgbImg, cv2.COLOR_RGB2GRAY)
    
    histogram,bins=np.histogram(grayImg.flatten(),256)
    index=np.where(histogram==np.max(histogram))
    retval,thr = cv2.threshold(grayImg, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    maxpoint=int(index[0])

    if retval < maxpoint:
        binimg=(255-thr)
    else:
        binimg=thr
    
    kernel = np.ones((3,3),np.uint8)
    
    binimg_dlt = cv2.dilate(binimg,kernel,iterations =5)
    
    dist_transform = cv2.distanceTransform(binimg_dlt,cv2.DIST_L2,5)
    ret, foreground = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)
    
    foreground = np.uint8(foreground)
          
    numLabels, markers, stats, centroids = cv2.connectedComponentsWithStats(foreground,8,cv2.CV_32S)
    
    markers = cv2.watershed(rgbImg,markers)
    
    total=0  
    for x in range(0,numLabels):
        area=stats[x,cv2.CC_STAT_AREA]
        r,g,b=rgbImg[centroids[x][1], centroids[x][0]]
        #print('component %d: '%x, r, g, b, area)
        if(area<4000):
            if ((98<=r<=248) and (61<=g<=240) and (30<=b<=133)):   #check for gold
                if (1056<=area<3000):
                    total+=0.5
                elif (600<=area<1056):
                    total+=0.2
                elif (449<=area<600):
                    total+=0.10   
                elif ((2<=area<62) or (65<=area<449)):
                    total+=0.05                  
            else:
                if (1750<=area<3000):               
                    total+=0.50
                elif (800<=area<1750):
                    total+=0.20     
                elif (487<=area<800):
                    total+=0.10   
                elif ((2<=area<30) or (56<=area<487)):
                    total+=0.05
        
    
    ans=total
    
    
    #########################################################################
    return round(ans, 2)
