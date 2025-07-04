import numpy as np
import cv2
import glob
import time

# https://www.geeksforgeeks.org/calibratecamera-opencv-in-python/

# Define the chess board rows and columns
CHECKERBOARD = (9,6)

DIR = "/Users/wette/Documents/FHBielefeld/eigeneVorlesungen/AutonomeFahrzeuge1zu32/localization/camera_calibration/images"
DIR_UNDISTORTED = "/Users/wette/Documents/FHBielefeld/eigeneVorlesungen/AutonomeFahrzeuge1zu32/localization/camera_calibration/undistorted-images"


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
for path in glob.glob(f"{DIR}/*.png"):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(50)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

print("Error in projection : \n", ret) 
print("\nCamera matrix : \n", mtx) 
print("\nDistortion coefficients : \n", dist) 
#print("\nRotation vector : \n", rotation_vecs) 
#print("\nTranslation vector : \n", translation_vecs)

#undistort images:
for path in glob.glob(f"{DIR}/*.png"):
    filename = path.split("/")
    filename = filename[-1]

    img = cv2.imread(path)

    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx,(w, h),5)
    undistortedImage = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    cv2.imwrite(f"{DIR_UNDISTORTED}/{filename}", undistortedImage)


#cv2.imshow('Original Image', img)
#cv2.imshow('Undistorted Image', undistortedImage)
#cv2.waitKey(0)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

t_lastimage = time.time()
while True:
    ret, frame = cap.read()
    undistortedImage = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    cv2.imshow('Undistorted Image', undistortedImage)

    if cv2.waitKey(1) == ord('q'):
        break

    #dt = time.time()-t_lastimage
    #print(f"{dt}")
    #t_lastimage = time.time()
