import os
import cv2
from PIL import Image
import numpy as np

def getImageAndLabels(path):
    # save face data arrays
    facesSamples = []
    # save name data arrays
    ids = []
    # save image info
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # load recognizer
    face_detector = cv2.CascadeClassifier('/Users/caohan/Anaconda/anaconda3/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
    # list of images
    for imagePath in imagePaths:
        # Open the image, there are nine different modes of greyscale: 1,L,P,RGB,RGBA,CMYK,YCbCr,I,F
        # L is a grayscale image, each pixel becomes a value from 0-255, the darker the colour the larger the value
        PIL_img = Image.open(imagePath).convert('L')
        # Converting images to arrays, black and white and light grey. Vectorisation
        img_numpy = np.array(PIL_img, 'uint8')
        # get face Features
        faces = face_detector.detectMultiScale(img_numpy)
        # get each image ids and names
        id = int(os.path.split(imagePath)[1].split('.')[0])  # Extract
        # Preventing faceless pictures
        for x, y, w, h in faces:
            ids.append(id)
            facesSamples.append(img_numpy[y:y+h, x:x+w])
        # print face id and features
    print('id:', id)
    print('fs:', facesSamples)
    return facesSamples, ids

if __name__ == '__main__':
    # Path of an image
    path = '/Users/caohan/Desktop/Face_Recognition/save/'
    # get image Arrays and ids tags and image name
    faces, ids = getImageAndLabels(path)
    # load recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # training
    recognizer.train(faces, np.array(ids))
    # Facial information and ids are saved one by one in a file
    # save files
    recognizer.write('trainer/trainer.yml')

