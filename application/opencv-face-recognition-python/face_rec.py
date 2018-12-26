#import OpenCV module
import cv2
#import os module for reading training data directories and paths
import os
#import numpy to convert python lists to numpy arrays as 
#it is needed by OpenCV face recognizers
import numpy as np

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
