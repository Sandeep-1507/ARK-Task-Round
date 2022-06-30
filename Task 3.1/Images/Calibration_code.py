# Camera calibration
import cv2
import numpy as np
import glob


criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)


objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

obj_points = [] 
img_points = []  

images = glob.glob("../images/*.jpg")
i = 0
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    size = gray.shape[::-1]

    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)    

    if ret:
        obj_points.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  
        if corners2.any():
            img_points.append(corners2)
        else:
            img_points.append(corners)

        cv2.drawChessboardCorners(img, (9, 6), corners, ret)  
        i += 1
        cv2.imwrite('conimg' + str(i) + '.jpg', img)


print(len(img_points))
cv2.destroyAllWindows()


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

print("ret:", ret)
print("mtx:\n", mtx)  
print("dist:\n", dist)  
print("rvecs:\n", rvecs)  
print("tvecs:\n", tvecs) 



img = cv2.imread(images[2])
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))  
print(newcameramtx)
print("------------------use undistort function-------------------")
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
x, y, w, h = roi
dst1 = dst[y:y + h, x:x + w]
cv2.imwrite('calibresult3.jpg', dst1)
print("Method 1:dst The size of the is:", dst1.shape)