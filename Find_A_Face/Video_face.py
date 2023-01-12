import cv2 as cv

def face_detect_demo(img):
    gary = cv.cvtColor(video, cv.COLOR_BGR2GRAY)

    face_detect = cv.CascadeClassifier('/Users/caohan/Anaconda/anaconda3/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_default.xml')  # 加载分类器，cv自带
    # Modify recognition range, limit size（100，100），（300，300）
    face = face_detect.detectMultiScale(gary)
    for x, y, w, h in face:
        cv.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
    cv.imshow('result', img)

# Read camera image, 0 is the default camera, other external cameras need to be set.
cap = cv.VideoCapture(0)
# cap = cv.VideoCapture('Han.mp4')


# loop
while True:
    flag, frame = cap.read()
    if not flag:
        break
    face_detect_demo(frame)
    if ord('q') == cv.waitKey(0):
        break
# release camera
cap.release()
# Release memory
cv.destroyAllWindows()