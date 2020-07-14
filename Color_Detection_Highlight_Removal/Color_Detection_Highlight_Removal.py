import cv2
import numpy as np

def empty(a):
    pass

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars",640,240)
cv2.createTrackbar("Hue Min","TrackBars",0,179,empty)
cv2.createTrackbar("Hue Max","TrackBars",19,179,empty)
cv2.createTrackbar("Sat Min","TrackBars",110,255,empty)
cv2.createTrackbar("Sat Max","TrackBars",240,255,empty)
cv2.createTrackbar("Val Min","TrackBars",153,255,empty)
cv2.createTrackbar("Val Max","TrackBars",255,255,empty)

img = cv2.imread("1.jpg")

#Resize
res = cv2.resize(img,None,fx=0.2,fy=0.2,interpolation = cv2.INTER_CUBIC)

#Erode
kernel = np.ones((5,5),np.uint8)
imgEroded = cv2.erode(res,kernel,iterations=1)

#Convert to grayscale
gray = cv2.cvtColor(imgEroded,cv2.COLOR_BGR2GRAY)

#Binarize to get highlight
ret, mask = cv2.threshold(gray,230,255,cv2.THRESH_BINARY)

#Inpaint highlight
dst = cv2.inpaint(imgEroded, mask, 3, cv2.INPAINT_TELEA) 

#Convert to HSV
HSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
hsv = cv2.cvtColor(dst,cv2.COLOR_BGR2HSV)

while True:
#    img = cv2.imread("lambo.png")
    h_min = cv2.getTrackbarPos("Hue Min","TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    print(h_min,h_max,s_min,s_max,v_min,v_max)

    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask1 = cv2.inRange(HSV,lower,upper)
    mask2 = cv2.inRange(hsv,lower,upper)

    result = cv2.bitwise_and(img,img,mask=mask1)
    imgResult = cv2.bitwise_and(dst,dst,mask=mask2)

    imgStack = stackImages(1,([imgEroded,mask,dst],[mask1,result,imgResult]))

    cv2.imshow("Stacked Images", imgStack)

    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#cv2.waitKey(0)