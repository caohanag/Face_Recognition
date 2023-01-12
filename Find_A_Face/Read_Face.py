# import cv model
import cv2 as cv

# read picture
# cv.imread; cv.imshow: cv-self-functions
img = cv.imread('Zhou.jpg')

# Coordinates
x, y, w, h = 100, 100, 100, 100
# Drawing rectangles
cv.rectangle(img, (x, y, x + w, y + h), color=(0, 0, 255), thickness=3)

# Drawing circles
cv.circle(img, center=(x + w, y + h), radius=100, color=(255, 0, 0), thickness=5)

# display
cv.imshow('re_img', img)

# Converting an image to gray scale
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # (image, color)
# display gray image
cv.imshow('gray', gray_img)
# save gray image
cv.imwrite('Zhou.jpg', gray_img)

# display pictures
cv.imshow('read_img', gray_img)

# modify size of image
resize_img = cv.resize(img, dsize=(200, 200))
# display original
cv.imshow('img', img)
# display modified
cv.imshow('resize_img', resize_img)
# print original image
print('original', img.shape)
# print modified image
print('modified', resize_img.shape)


# Detection functions
def face_detect_demo():
    gary = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_detect = cv.CascadeClassifier('/Users/caohan/Anaconda/anaconda3/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')  # 加载分类器，cv自带
    face = face_detect.datectMultiScale(gary, 1.01, 5, 0, (100, 100), (300, 300))  # 修改识别范围,限制大小（100，100），（300，300）
    for x, y, w, h in face:
        cv.rectangle(img, (x, y), (x + w, x + h), color=(0, 0, 255), thickness=2)
    cv.imshow('result', img)

    img = cv.imread('Zhou.jpg')
    # Create detection functions
    face_detect_demo()


# delay, show image
# 0 is for no limited waiting. 1 is for 1 second.
while True:
    if ord('q') == cv.waitKey(0):
        break

# Release memory
cv.destroyAllWindows()
