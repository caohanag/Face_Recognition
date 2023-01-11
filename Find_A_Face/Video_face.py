import cv2 as cv

def face_detect_demo(img):
    gary = cv.cvtColor(video, cv.COLOR_BGR2GRAY)

    face_detect = cv.CascadeClassifier('/Users/caohan/Anaconda/anaconda3/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_default.xml')  # 加载分类器，cv自带
    # 修改识别范围,限制大小（100，100），（300，300）
    face = face_detect.detectMultiScale(gary)
    for x, y, w, h in face:
        cv.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
    cv.imshow('result', img)

# 读取摄像头图像, 0为默认摄像头，其他的外接摄像头需要单独设置
cap = cv.VideoCapture(0)
# cap = cv.VideoCapture('Han.mp4')


# 循环
while True:
    flag, frame = cap.read()
    if not flag:
        break
    face_detect_demo(frame)
    if ord('q') == cv.waitKey(0):
        break
# 释放摄像头
cap.release()
# 释放内存
cv.destroyAllWindows()