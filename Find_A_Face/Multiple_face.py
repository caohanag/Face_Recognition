import cv2 as cv


def face_detect_demo():
    gary = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    face_detect = cv.CascadeClassifier('/Users/caohan/Anaconda/anaconda3/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_default.xml')  # 加载分类器，cv自带
    # 修改识别范围,限制大小（100，100），（300，300）
    face = face_detect.detectMultiScale(gary)
    for x, y, w, h in face:
        cv.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
    cv.imshow('result', img)


# 读取图像
img = cv.imread('Zhou.png')
# 建立检测函数
face_detect_demo()

# 等待
while True:
    if ord('q') == cv.waitKey(0):
        break

cv.destroyAllWindows()
