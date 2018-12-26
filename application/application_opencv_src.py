import cv2
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
import os

def face_detection():
	face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml');
	eyes_casecade = cv2.CascadeClassifier('./haarcascade_eye.xml');
	#img = cv2.imread('/home/manjur/Downloads/2_two_wheeler.jpg',1);
	video = cv2.VideoCapture(0)

	while True:
		check, img = video.read()
		print("image and check")

		gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		#plt.imshow(gray_img)
		gray_eye = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


		faces = face_cascade.detectMultiScale(gray_img,scaleFactor=1.05,minNeighbors=5);
		eyes =  eyes_casecade.detectMultiScale(gray_img);
		for x,y,h,w in faces:
			print(x,y,h,w)
			#img = cv2.rectangle(gray_img,(x,y),(x+w,y+h),(0,255,255),15);
			#img = cv2.rectangle(gray_img,(x,y),(x+w,y+h),(0,255,0),3);
			img = cv2.rectangle(gray_img,(x,y),(x+w,y+h),(255,0,0),3);

		for x,y,h,w in eyes:
			img = cv2.rectangle(gray_eye,(x,y),(x+w,y+h),(0,255,0),5);

		cv2.imshow('image',img)
		cv2.imshow('grayImage',gray_img)
		#plt.imshow(img)
		#plt.show()
		#plt.show(gray_img)
	

		key=cv2.waitKey(1)

		if key == ord('q'):
			if status == 1:
				times.append(datetime.now())
			break

	for i in range(0, len(times), 2):
    		df = df.append({"Start": times[i],"End": times[i+1]}, ignore_index=True)

	df.to_csv("Times.csv")

	video.release()
	cv2.destroyAllWindows()

def face_recognition():
	subjects = ["", "Salman khan", "Amir khan"]

	#function to detect face using OpenCV
	def detect_face(img):
    		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
   		face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')

    		faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    
    		if (len(faces) == 0):
        		return None, None
    
    		(x, y, w, h) = faces[0]
    
    		return gray[y:y+w, x:x+h], faces[0]


	def prepare_training_data(data_folder_path):
	    
    		#get the directories (one directory for each subject) in data folder
    		dirs = os.listdir(data_folder_path)
    
    		faces = []
    		labels = []
    
    		for dir_name in dirs:
        
        		if not dir_name.startswith("s"):
            			continue;
            
        	#------STEP-2--------
        	#, so removing letter 's' from dir_name will give us label
        		label = int(dir_name.replace("s", ""))
        
        		subject_dir_path = data_folder_path + "/" + dir_name
        
        		subject_images_names = os.listdir(subject_dir_path)
        
        	#------STEP-3--------
        		for image_name in subject_images_names:
            
            			if image_name.startswith("."):
                			continue;
            
            			image_path = subject_dir_path + "/" + image_name

            			image = cv2.imread(image_path)
            
            			cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            			cv2.waitKey(100)
            
           		 	face, rect = detect_face(image)
            
            #------STEP-4--------
            			if face is not None:
                			faces.append(face)
                			labels.append(label)
            
    		cv2.destroyAllWindows()
    		cv2.waitKey(1)
    		cv2.destroyAllWindows()
    
    		return faces, labels


	print("Preparing data...")
	faces, labels = prepare_training_data("training-data1")
	print("Data prepared")

	#print total faces and labels
	print("Total faces: ", len(faces))
	print("Total labels: ", len(labels))


	#create our LBPH face recognizer 
	face_recognizer = cv2.face.LBPHFaceRecognizer_create()
	#train our face recognizer of our training faces
	face_recognizer.train(faces, np.array(labels))


	#given width and heigh
	def draw_rectangle(img, rect):
    		(x, y, w, h) = rect
    		cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
	#passed (x, y) coordinates. 
	def draw_text(img, text, x, y):
    		cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)


	def predict(test_img):
    		img = test_img.copy()
    		face, rect = detect_face(img)

    		label, confidence = face_recognizer.predict(face)
    		label_text = subjects[label]
    
    		draw_rectangle(img, rect)
    		draw_text(img, label_text, rect[0], rect[1]-5)
    
    		return img

	print("Predicting images...")

	#load test images
	test_img1 = cv2.imread("test-data1/test1.jpg")
	test_img2 = cv2.imread("test-data1/test2.jpg")

	#perform a prediction
	predicted_img1 = predict(test_img1)
	predicted_img2 = predict(test_img2)
	print("Prediction complete")

	#display both images
	cv2.imshow(subjects[1], cv2.resize(predicted_img1, (400, 500)))
	cv2.imshow(subjects[2], cv2.resize(predicted_img2, (400, 500)))
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.waitKey(1)
	cv2.destroyAllWindows()

def motion_detector():
	first_frame = None
	status_list = [None,None]
	times = []
	df=pd.DataFrame(columns=["Start","End"])

	video = cv2.VideoCapture(0)

	while True:
    		check, frame = video.read()
    		status = 0
    		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    		gray = cv2.GaussianBlur(gray,(21,21),0)
    		#print "hello 1"
    		if first_frame is None:
        		first_frame=gray
        		continue

    		#print "hello 2"
    		delta_frame=cv2.absdiff(first_frame,gray)
    		thresh_frame=cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    		thresh_frame=cv2.dilate(thresh_frame, None, iterations=2)

    		(_,cnts,_)=cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    		for contour in cnts:
        		if cv2.contourArea(contour) < 10000:
            			continue
        		status=1

        		(x, y, w, h)=cv2.boundingRect(contour)
        		cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 3)
    			status_list.append(status)

    		status_list=status_list[-2:]


    		if status_list[-1]==1 and status_list[-2]==0:
        		times.append(datetime.now())
    		if status_list[-1]==0 and status_list[-2]==1:
        		times.append(datetime.now())


    		cv2.imshow("Gray Frame",gray)
    		#cv2.imshow("Delta Frame",delta_frame)
    		#cv2.imshow("Threshold Frame",thresh_frame)
    		cv2.imshow("Color Frame",frame)
    		'''plt.imshow(gray)
    		plt.imshow(delta_frame)
    		plt.imshow(thresh_frame)
    		plt.imshow(frame)'''

    		key=cv2.waitKey(1)

    		if key == ord('q'):
        		if status == 1:
            			times.append(datetime.now())
       			 	break


	for i in range(0, len(times), 2):
    		df = df.append({"Start": times[i],"End": times[i+1]}, ignore_index=True)

	df.to_csv("Times.csv")

	video.release()
	cv2.destroyAllWindows()
def object_tracking_with_color():
	cap = cv2.VideoCapture(0)
    
    	if cap.isOpened():
        	ret, frame = cap.read()
    	else:
        	ret = False
    	while ret:
        	ret, frame = cap.read()
        	#convert image into hsv
		# hsv stands for
		# hue ,saturation,value
        	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        	# Blue Color
		#low = np.array([100, 50, 50]) # light blue pixel value
		#high = np.array([140, 255, 255]) #dark blue pixel value, you can change the value for proper tracking

        	# Green Color
        	#low = np.array([40, 50, 50]) # light green pixel value
        	#high = np.array([80, 255, 255]) # dark green pixel value

        	#Red Color
       		# low = np.array([140, 150, 0])
       		# high = np.array([180, 255, 255])
        	# image mask generate binary image
        	low = np.array([0, 20, 70])
        	high = np.array([20, 255, 255])
        	image_mask = cv2.inRange(hsv, low, high) # it will generate binary format of image(mask image)
        
       	 	output = cv2.bitwise_and(frame, frame, mask = image_mask)
        
        	cv2.imshow("Image mask", image_mask)
        	cv2.imshow("Original Webcam Feed", frame)
        	cv2.imshow("Color Tracking", output)
        	if cv2.waitKey(1) == 27: # exit on ESC
            		break

    	cv2.destroyAllWindows()
    	cap.release()
def image_read_write():
	cap = cv2.VideoCapture(0)
	count = 0
    	if cap.isOpened():
        	ret, frame = cap.read()
    	else:
        	ret = False
	while ret:
        	ret, frame = cap.read()
		cv2.imwrite(frame,'./test','jpg')
  		#cv2.imwrite("frame%d.jpg" % count, frame)     # save frame as JPEG file      
  		print('Read a new frame: ', frame)
  		count += 1
        	if cv2.waitKey(1) == 27: # exit on ESC
            		break

    	cv2.destroyAllWindows()
def hand_gesture():
	cap = cv2.VideoCapture(0)
	while True:
		_, frame = cap.read()
		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		blur = cv2.GaussianBlur(gray,(5,5),0)
		ret,thresh1 = cv2.threshold(blur,70,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
		#hsv = cv2.cvtColor(blurred_frame,cv2.COLOR_BGR2HSV)
        	#low = np.array([0, 20, 70])
        	#high = np.array([20, 255, 255])
        	#imagemask = cv2.inRange(hsv, low, high) # it will generate binary format of image(mask image)	
		#_,countors,_ = cv2.findContours(imagemask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
		#for countor in countors:
		#	area = cv2.contourArea(countor)
		#	if area > 1000:
		#		cv2.drawContours(frame,countors,-1,(0,255,0),3)
		#cv2.imshow("frame",frame)
		#cv2.imshow("iamge",imagemask)
		contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		for i in range(len(contours)):
            		cnt=contours[i]
            		area = cv2.contourArea(cnt)
            		if (area > max_area):
                		max_area=area
                		ci=i
  		cnt=contours[ci]
		hull = cv2.convexHull(cnt)
		drawing = np.zeros(frame.shape,np.uint8)
		cv2.drawContours(drawing,[cnt],0,(0,255,0),2)
		cv2.drawContours(drawing,[hull],0,(0,0,255),2)
		

		if cv2.waitKey(1) == 27:
			break
	cv2.destroyAllWindow()
		

def main(choice):
	while True:
		if choice == 1:
			face_detection()
			break
		elif choice == 2:
			face_recognition()
			break
		elif choice == 3:
			motion_detector()
			break
		elif choice == 4:
			object_tracking_with_color()
			break
		elif choice == 5:
			hand_gesture()
			break
		elif choice == 6:
			image_read_write()
		elif choice == 7:
			break

if __name__ == "__main__":
	print("For destrying window you need to press Esc key or q\n")
	print("Press 1 : For face detection")
	print("Press 2 : For face recognition")
	print("Press 3 : For motion detector")
	print("Press 4 : Object tracking by color(like blue green red)")
	print("Press 5 : Hand gesture")
	print("Press 6 : Read and write image")
	choice = input("Enter your choice")
	main(choice)
