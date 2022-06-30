import cv2
import numpy as np


cap=cv2.VideoCapture(0)

dist=np.array(( [[-3.59600442e-02 , 4.61504494e-01  ,4.34821601e-04, -1.35861355e-03
  ,-7.36531735e-01]]))

# mtx=np.array([[812.97671326  , 0.      ,   645.05630951],
#  [  0.   ,      813.93289517 ,356.96051563],
#  [  0.      ,     0.      ,     1.        ]])  

mtx=np.array( [[768.16650684  , 0.    ,     652.89033724],
 [  0.      ,   767.74399029 ,360.31414726],
 [  0.       ,    0.      ,     1.        ]])


def find_markers(frame,markerSize,totalMarkers,border_color,draw=True):

    while(True):

        # frame=cv2.resize(frame,None,fx=0.2,fy=0.2,interpolation=cv2.INTER_CUBIC)

        sucess,frame=cap.read()

        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        key=getattr(cv2.aruco,f'DICT_{markerSize}X{markerSize}_{totalMarkers}')

        aruco_dict=cv2.aruco.Dictionary_get(key)

        # aruco_dict=cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

        parameters=cv2.aruco.DetectorParameters_create()

        corners,ids,rejected=cv2.aruco.detectMarkers(gray,aruco_dict,parameters=parameters)

        if(draw):
            cv2.aruco.drawDetectedMarkers(frame,corners,ids,border_color)

        # cv2.imshow("frame",frame)

        #pose estimation starts here
        if len(corners)>0:
            for i in range(0,len(ids)):
                rvec,tvec,markerPoints=cv2.aruco.estimatePoseSingleMarkers(corners[i],0.02,mtx,dist)
                cv2.aruco.drawDetectedMarkers(frame,corners,ids)

                cv2.aruco.drawAxis(frame,mtx,dist,rvec,tvec,0.02)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

find_markers(None,6,250,(0,255,0),1)
