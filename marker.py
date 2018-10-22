import numpy as np
import cv2
import sys
import time
import glob
import math



dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
#dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
#dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)

cameraMatrix = np.array([[1.53467097e+03, 0.00000000e+00, 1.29130349e+03],[0.00000000e+00, 1.53594327e+03, 9.34303421e+02],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
distCoeffs = np.array([[-0.34844261,  0.26256965, -0.00075911,  0.00117993, -0.2515726 ]])

def angles_from_rvec(rvec):
    r_mat, _j_ = cv2.Rodrigues(rvec)
    #print(r_mat)
    #a = math.atan2(r_mat[2][1], r_mat[2][2])
    #b = math.atan2(-r_mat[2][0], math.sqrt(math.pow(r_mat[2][1],2) + math.pow(r_mat[2][2],2)))
    c = math.atan2(r_mat[1][0], r_mat[0][0])
    return math.degrees(c)


if __name__ == '__main__':

  file = sys.argv[1]
  print('Processing file:', file)

  out_file = sys.argv[2]
  if out_file is None:
    out_file = 'marker_out.jpeg'

  # Read image
  frame = cv2.imread(file)

  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  parameters =  cv2.aruco.DetectorParameters_create()

  start_time = time.time()
  corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
  print("Time take: %s seconds" % (time.time() - start_time))

  print(ids)

  #dst = cv2.aruco.drawDetectedMarkers(dst, corners, ids, (0, 255, 0))

  rvecs, tvecs, _objPoints	=cv2.aruco.estimatePoseSingleMarkers(corners, 20, cameraMatrix, distCoeffs)

  for i in range(0, len(ids)):
      frame = cv2.aruco.drawAxis(frame, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 20)

  cv2.imwrite(out_file, frame)

  print(rvecs)
  print(tvecs)
  print(' ############## ')
  print(angles_from_rvec(rvecs[0]))
