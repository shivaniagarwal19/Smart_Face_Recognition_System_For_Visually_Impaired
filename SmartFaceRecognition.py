import numpy as np
import cv2
import time
import os
from PIL import Image
class SmartFaceRecognition():
    #constructor initialization
    def __init__(self):
        print("Welcome to Smart Face Recognition")
    def FaceDetect(self):
        #Function to detect faces from the image and create dataset
        face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        id = input('enter user id')
        #id differentiates each user
        sampleN=0;
        while 1:
            #enable camera to read images
            ret, img = cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                sampleN=sampleN+1;
                cv2.imwrite("./facesData/User."+str(id)+ "." +str(sampleN)+ ".jpg", gray[y:y+h, x:x+w])
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                cv2.waitKey(100)
            #pop the window in order to provide real-time user experience
            cv2.imshow('img',img)
            #delay the screen for 1 milli second
            cv2.waitKey(1)
            if sampleN == 100:
                break
        #Free the camera
        cap.release()
        #close the preview GUI
        cv2.destroyAllWindows()
        print("Detection Part Completed")
    def TrainModel(self):
        #Function to create dataset
        recognizer = cv2.face.LBPHFaceRecognizer_create();
        path="./facesData"
        def getImagesWithID(path):
            imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
            # print image_path
            #getImagesWithID(path)
            faces = []
            IDs = []
            for imagePath in imagePaths:
          # Read the image and convert to grayscale
                facesImg = Image.open(imagePath).convert('L')
                faceNP = np.array(facesImg, 'uint8')
                # Get the label of the image
                ID= int(os.path.split(imagePath)[-1].split(".")[1])
                 # Detect the face in the image
                faces.append(faceNP)
                IDs.append(ID)
                cv2.imshow("Adding faces for traning",faceNP)
                cv2.waitKey(10)
            return np.array(IDs), faces
        Ids,faces  = getImagesWithID(path)
        recognizer.train(faces,Ids)
        recognizer.save("./faceREC/trainingdata.yml")
        cv2.destroyAllWindows()
        print("Model Successfully Trained")
    def FaceRecognize(self):
        #Function to recognize a face and categorize him or her as known or unknown based on data set
        face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
        cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        rec = cv2.face.LBPHFaceRecognizer_create();
        rec.read("./faceREC/trainingdata.yml")
        id=0
        font=cv2.FONT_HERSHEY_COMPLEX_SMALL
        while 1:
            ret, img = cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.5, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                id,conf=rec.predict(gray[y:y+h,x:x+w])
                if id==1 or id==2 or id==3 or id==4 or id==5 or id==6 or id==7 or id==8:
                    if id==1:
                        id="shashwat"
                    if id==2:
                        id="ishita"
                    if id==3:
                        id="Abhilasha"
                    if id==4:
                        id="Rohit"
                    if id==5:
                        id="Rohit"
                    if id==6:
                        id="Siddarth"
                    if id==7:
                        id="Chutiya"
                    if id==8:
                        id="Shivani"
                    if id==9:
                        id="Tridib"
                else:
                    id="Warning - the person is Unknown"
                cv2.putText(img,str(id),(x,y+h),font,1,255)
            cv2.imshow('img',img)
            print("End screen?-Y/N")
            ch=input()
            if cv2.waitKey(1) == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
obj=SmartFaceRecognition()
#User Interactive screen for function
while(1):
    print("Choose an option")
    print("Press 1: Add a new person into list of knowns")
    print("Press 2: Recognize a person as known or unknown")
    choice=int(input())
    if choice==1:
        obj.FaceDetect()
        obj.TrainModel()
    elif choice==2:
        obj.FaceRecognize()
    else:
        print("Sorry wrong input!")
        print("Would you like to choose again?\n- Y/N")
        x=input()
        if x == 'N' or 'n':
            break
        elif y == 'Y' or 'y':
            continue
        else:
            print("Shutting down system for detecting vulnerabilities")
            break
