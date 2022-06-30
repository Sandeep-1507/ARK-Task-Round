import cv2
import numpy as np


dictionary=cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

marker_img=np.zeros((200,200),dtype=np.uint8)
marker_img=cv2.aruco.drawMarker(dictionary,22,200,marker_img,1)
cv2.imwrite("marker22.png",marker_img)