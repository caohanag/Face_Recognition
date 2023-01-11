import os
import cv2
from PIL import Image
import numpy as np

def getImageAndLabels(path):
    # save face data 二维数组
    facesSamples = []
    # save name data 二维数组
    ids = []
    # save image info
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # load recognizer
    face_detector = cv2.CascadeClassifier('/Users/caohan/Anaconda/anaconda3/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
    # 遍历列表中图片
    for imagePath in imagePaths:
        # 打开图片， 灰度化有九种不同模式：1,L,P,RGB,RGBA,CMYK,YCbCr,I,F
        PIL_img = Image.open(imagePath).convert('L') # L是灰度图像，每一个像素点变为0-255数值，颜色越深值越大
        # 将图像转化为数组，黑白浅灰。向量化
        img_numpy = np.array(PIL_img, 'uint8')
        # get face Features
        faces = face_detector.detectMultiScale(img_numpy)
        # get each image ids and names
        id = int(os.path.split(imagePath)[1].split('.')[0])  # 提取
        # 预防无面容图片
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
    # 面部信息和id被一一保存在文件里
    # save files
    recognizer.write('trainer/trainer.yml')

