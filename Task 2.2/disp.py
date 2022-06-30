# import cv2
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


imgL=cv.imread("left.png",0)

imgR=cv.imread("right.png",0)

#template matching is used here to find out the cordintaes of the centre of the point for the disparity
template = cv.imread('bike.png',0)
w, h = template.shape[::-1]
res = cv.matchTemplate(imgL,template,cv.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)
# print(w,h)
for pt in zip(*loc[::-1]):
    cv.rectangle(imgL, pt, (pt[0] +w , pt[1] + h), (0,0,255), 2)
    print(pt[0]+w/2,pt[1]+h/2) #location of the template
    
cv.imshow('res.png',imgL)
cv.waitKey(100)

x=570
y=494



stereo=cv.StereoBM_create(numDisparities=48,blockSize=21)
disparity=stereo.compute(imgL,imgR)




focal_length=640
distance_btw_cameras=240

print(focal_length*distance_btw_cameras/disparity[x][y])


plt.imshow(disparity,'gray')
plt.show()