# -*- coding: utf-8 -*- 
import cv2

class Camera():
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if(self.cap is None):
            raise Exception("Fail to find camera device.")
        if(not self.cap.isOpened()):
            raise Exception("Fail to open camera device.")

    def __del__(self):
        if(self.cap.isOpened()):
            self.cap.release()

    def get_frame(self):
        success, frame = self.cap.read()
        if not success:
            raise Exception("Read frame error.")
        
        return frame