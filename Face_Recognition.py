import cv2
import os
import numpy as np
import urllib
import urllib.request
import hashlib

# Load training data file
recogizer = cv2.face.LBPHFaceRecognizer_create()
# Load data
recogizer.read('trainer/trainer.yml')
# Name
names = []
# warning time
warningtime = 0


# md5 Encryption
def md5(str):
    import hashlib
    m = hashlib.md5()
    m.update(str.encode("utf8"))
    return m.hexdigest()

# feedback use any SMS website Parameters
statusStr = {
'0': '短信发送成功',
    '-1': '参数不全',
    '-2': '服务器空间不支持,请确认支持curl或者fsocket,联系您的空间商解决或者更换空间',
    '30': '密码错误',
    '40': '账号不存在',
    '41': '余额不足',
    '42': '账户已过期',
    '43': 'IP地址限制',
    '50': '内容含有敏感词'
}

# warning mode use any SMS website Parameters
def warning():
    smsapi = "http://api.smsbao.com/"
    # account
    user = 'Hanbanana'
    # password
    password = md5('*********')
    # SMS Contents
    content = '[Alarm] \n reasons: ***\n time: ***'
    # SMS send number
    phone = '04*****652'

    data = urllib.urlencode({'u': user, 'p': password, 'm': phone, 'c': content})
    req = urllib2.Request(sendurl, data)
    response = urllib2.urlopen(req)
    the_page = response.read()
    print(statusStr[the_page])

# prepare recognize Image
def face_detect_demo(img):
    gray = cv2.cvColor(img, cv2.COLOR_BGR2GRAY)  # gray
    # load detector
    face_detector = cv2.CascadeClassifier('/Users/caohan/Anaconda/anaconda3/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
    # select face
    face = face_detector.detectMultiScale(gray, 1.1, 5, cv2.CASCADE_SCALE_IMAGE, (100, 100), (300, 300))
    # face = face_detector.detectMultiScale(gray)
    for x, y, w, h in face:
        cv2.rectangle(img, (x + y), (x + w, y + h), color=(0, 0, 255), thickness=2)
        cv2.circle(img, center=(x + w // 2, y + h // 2), radius=w // 2, color=(0, 255, 0), thickness=1)
        # Face Recognize
        ids, confidence = recogizer.predict(gray[y:y + h, x:x + w])  # need small value
        # pring('tag id:', ids, 'confidence:', confidence)
        if confidence > 80:
            global warningrime
            warningtime += 1
            if warningtime > 100:
                warning()
                warningtime = 0
            cv2.putText(img, 'unknow', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
        else:
            cv2.putText(img, str(names[ids - 1]), (x +10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
    cv2.imshow('result', img)
    # print('bug:', ids)

def name():
    path = '/Users/caohan/Desktop/Face_Recognition/save/'
    #names = []
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    for imagePath in imagePaths:
       name = str(os.path.split(imagePath)[1].split('.',2)[1])
       names.append(name)


cap=cv2.VideoCapture('1.mp4')
name()
while True:
    flag,frame=cap.read()
    if not flag:
        break
    face_detect_demo(frame)
    if ord(' ') == cv2.waitKey(10):
        break
cv2.destroyAllWindows()
cap.release()
#print(names)


