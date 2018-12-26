import cv2

import numpy as np

def main():
	img1 = np.ones((512,512,3),np.uint8)
	cv2.line(img1,(0,100),(100,0),(255,0,0),3)
	cv2.rectangle(img1,(40,60),(80,70),(0,255,0),4)
	cv2.imshow('ak',img1)
	cv2.waitKey(0)
	cv2.destroyWindow('ak')


if __name__ == '__main__':
	main()
