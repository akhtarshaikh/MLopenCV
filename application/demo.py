import cv2


img = cv2.imread("ak.jpg",1)

#hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

b,g,r = cv2.split(img)
img1 = cv2.merge((b,g,r))
print(img1)
#img2 = cv2.bitwise_not(img)
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#blur = cv2.GaussianBlur(gray,(5,5),0)
#ret,thresh1 = cv2.threshold(blur,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# fx= how much amount scaling do you want to perform on x-axis
# fy = How much amount scaling do you want to perform on y-axis
# interpolation = Linear interpolation,CUBIC interpolation
#img1 = cv2.resize(img,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_LINEAR)

cv2.imshow("image",img)
#cv2.imshow("image1",hsv)
cv2.imshow("imageB",img1)
#cv2.imshow("imageG",img2)
#cv2.imshow("imageR",img3)
#cv2.imshow("image2",blur)
#cv2.imshow("image3",thresh1)

cv2.waitKey()

cv2.destroyAllWindow()
