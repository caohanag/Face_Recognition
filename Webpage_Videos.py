import cv2

class CaptureVideo(object):
    def net_video(self):
        # get online videos
    cam = cv2.VideoCapture("rtmp://192.168.0.10/live/test")
    while cam.isOpened():
        sucess, frame = cam.read()
        cv2.imshow("Network", frame)
        cv2.waitKey(1)
if __name__ == "__main__":
    capture_video = CaptureVideo()
    capture_video.net_video()