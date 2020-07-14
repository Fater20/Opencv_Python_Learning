import cv2
import numpy as np
import time
#空函数
def empty(a):
    pass

#图像单窗口显示函数
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

kernel = np.ones((5,5),np.uint8)

#创建HSV阈值调节滑动窗
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars",640,240)
cv2.createTrackbar("Hue Min","TrackBars",0,179,empty)
cv2.createTrackbar("Hue Max","TrackBars",0,179,empty)
cv2.createTrackbar("Sat Min","TrackBars",0,255,empty)
cv2.createTrackbar("Sat Max","TrackBars",0,255,empty)
cv2.createTrackbar("Val Min","TrackBars",0,255,empty)
cv2.createTrackbar("Val Max","TrackBars",0,255,empty)

#摄像头捕获图像
frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(2)   #摄像头捕获帧
cap.set(3, frameWidth)      #设置视频帧的宽
cap.set(4, frameHeight)     #设置视频帧的高
cap.set(10,100)             #设置图像的亮度

frame_cnt = 0
fps_str = "FPS: 0"
# img = cv2.imread('1.jpg')

# res = cv2.resize(img, None, fx=0.2, fy=0.2,interpolation=cv2.INTER_CUBIC)

# HSV = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)

while True:
    if frame_cnt == 0:
        start = time.time()
    
    frame_cnt = frame_cnt + 1
    success, frame = cap.read()   #读取帧
    #判断帧是否读取成功
    if success == False:
        print("Read Frame False!")
        break
#    res = cv2.resize(img, None, fx=0.2, fy=0.2,interpolation=cv2.INTER_CUBIC)

    imgHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  
#    img = cv2.imread("lambo.png")
    h_min = cv2.getTrackbarPos("Hue Min","TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
#    print(h_min,h_max,s_min,s_max,v_min,v_max)

    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask0 = cv2.inRange(imgHSV,lower,upper)

    col_block = cv2.bitwise_and(frame,frame,mask=mask0)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask1 = cv2.morphologyEx(mask0, cv2.MORPH_OPEN, element)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask2 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, element)

    imgCanny = cv2.Canny(mask2, 100, 150)

    maxArea = 0

    contours, hierarchy = cv2.findContours(imgCanny,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    if np.size(contours)>0:
        for i in range(0,np.size(contours)):
            area = cv2.contourArea(contours[i])
            if (area > maxArea) and (area<(frame.shape[1]-5)*(frame.shape[0]-5)):
                maxArea = area
                maxContour = contours[i]


        if maxArea > 0:
            x,y,w,h = cv2.boundingRect(maxContour) #计算点集的最外面（up-right）矩形边界
            if w - h<10 and w - h>-10:
            # 包围的矩形框
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 0, 255), 2)#draw rectangle

                centerx = int(x + w / 2)
                centery = int(y + h / 2)

#                print(centerx, centery)

                cv2.circle(frame, (centerx,centery), 2, (255, 0, 0), 2)

                # 显示效果图窗口
                cv2.imshow("POINT", frame)

            else:
                print("too low\n")

    else:
        print("No Target!")

    cv2.putText(frame,"FPS: "+fps_str,(0,30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)

    imgStack = stackImages(0.5,([frame,col_block,imgCanny],[mask0,mask1,mask2]))
    cv2.imshow("Stacked Images", imgStack)    

    cv2.waitKey(1)

    if frame_cnt == 10:
        frame_cnt = 0
        end = time.time()
        fps = int(10/(end - start))
        fps_str = str(fps)
        
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.waitKey(0)


