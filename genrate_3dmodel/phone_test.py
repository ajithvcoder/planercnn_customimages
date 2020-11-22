import cv2 
import numpy as np
url = 'http://192.168.43.25:8080/video'
camera = cv2.VideoCapture(url)
# while(True):
#     ret, frame = cap.read()
#     #print(ret)
#     if frame is not None:
#         cv2.imshow('frame',frame)
#     q = cv2.waitKey(1)
#     if q == ord("q"):
#         break
# cv2.destroyAllWindows()

import time 

i = 0
while i<1:        
    #input("Press Enter to capture")
    start_time_1 = time.time()
    return_value, image = camera.read()
    if return_value:
        image = cv2.resize(image,(640,480))
        cv2.imwrite("phsample.png",image)
        i += 1
#del(camera)
cv2.imshow('frame',image)
q = cv2.waitKey(1)
cv2.destroyAllWindows()
