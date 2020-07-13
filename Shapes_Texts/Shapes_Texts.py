import cv2
import numpy as np

img = np.zeros((512,512,3),np.uint8)
#print(img.shape)
#img[:]=255,0,0   #The whole image
#img[200:300,100:300]=255,0,0  #img[height,width]=B,G,R

#cv2.line(img,(0,0),(300,300),(0,255,0),3)   #cv2.line(img,start_point,end_point,color,thickness)
cv2.line(img,(0,0),(img.shape[1],img.shape[0]),(0,255,0),3)
cv2.rectangle(img,(0,0),(250,350),(0,0,255),2) #thickness=cv2.FILLED means filling
cv2.circle(img,(400,50),30,(255,255,0),5)   #cv2.circle(img,center,radius,color,thickness)

cv2.putText(img,"OPENCV",(300,200),cv2.FONT_HERSHEY_COMPLEX,1,(0,150,0),1)

cv2.imshow("Image",img)

cv2.waitKey(0)