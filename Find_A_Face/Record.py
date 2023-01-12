import cv2

cap = cv2.VideoCapture(0)

flag = 1
num = 1

while (cap.isOpened()):  # Detects if it is on
    ret_flag, Vshow = cap.read()  # Get each frame
    cv2.imshow("Capture_Test", Vshow)  # display
    k = cv2.waitKey(1) & 0xFF  # press key
    if k == ord('s'):  # press s to save
        # str(num) 1，2，3...
        cv2.imwrite("/Users/caohan/Desktop/Face_Recognition/save/" + str(num) + ".name" + ".jpg", Vshow)
        print("success to save" + str(num) + ".jpg")
        print("-----------")
        num += 1
    elif k == ord('b'):  # exit
        break

cap.release()
cv2.destroyAllWindows()
