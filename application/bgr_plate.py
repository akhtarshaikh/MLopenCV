import cv2

import numpy as np

def emptyFunction():
	pass
	return

def main():
	img1 = np.array((512,512,3),np.uint8)
	windowsname = 'changesimage'
	cv2.namedWindow(windowsname)

	cv2.createTrackbar('B',windowsname,0,255,emptyFunction)
	cv2.createTrackbar('G',windowsname,0,255,emptyFunction)
	cv2.createTrackbar('R',windowsname,0,255,emptyFunction)

	while True:
		cv2.imshow(windowsname,img1)
		
		if cv2.waitKey(1) == 27:
			break
		blue = cv2.getTrackbarPos('B',windowsname)
		green = cv2.getTrackbarPos('G',windowsname)
		red = cv2.getTrackbarPos('R',windowsname)
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
