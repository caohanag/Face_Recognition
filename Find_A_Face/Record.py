import cv2

cap = cv2.VideoCapture(0)

flag = 1
num = 1

while (cap.isOpened()):  # 检测是否在开启状态
    ret_flag, Vshow = cap.read()  # 得到每帧图像
    cv2.imshow("Capture_Test", Vshow)  # 显示图像
    k = cv2.waitKey(1) & 0xFF  # 按键判断
    if k == ord('s'):  # press s to save
        # str(num)是编号1，2，3...
        cv2.imwrite("/Users/caohan/Desktop/Face_Recognition/save/" + str(num) + ".name" + ".jpg", Vshow)
        print("success to save" + str(num) + ".jpg")
        print("-----------")
        num += 1
    elif k == ord('b'):  # exit
        break

cap.release()
cv2.destroyAllWindows()
