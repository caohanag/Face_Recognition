import cv2 as cv


def face_detect_demo():
    gary = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    face_detect = cv.CascadeClassifier('/Users/caohan/Anaconda/anaconda3/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_default.xml')  # 加载分类器，cv自带
    # Modify recognition range, limit size（100，100），（300，300）
    face = face_detect.detectMultiScale(gary)
    for x, y, w, h in face:
        cv.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
    cv.imshow('result', img)


# read image
img = cv.imread('Zhou.png')
# Create detection functions
face_detect_demo()

# wait
while True:
    if ord('q') == cv.waitKey(0):
        break

cv.destroyAllWindows()
