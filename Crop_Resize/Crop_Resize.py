import cv2

img = cv2.imread("lambo.png")
print(img.shape)#(height,width,channels)

imgResize = cv2.resize(img,(300,200)) #cv2.resize(img,(width,height))

imgCropped = img[0:200,200:500] #img[height,width]

cv2.imshow("Image",img)
cv2.imshow("Image Resize",imgResize)
cv2.imshow("Image Cropped",imgCropped)
cv2.waitKey(0)