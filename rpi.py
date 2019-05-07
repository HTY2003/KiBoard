from picamera.array import PiRGBArray
from picamera import PiCamera
import time

class PiCameraWrapper():
    def __init__(self):
        self.camera = PiCamera()
        self.resolution = (640, 480)
        self.framerate = 30
        self.rawCapture = PiRGBArray(self.camera, size=self.resolution)
        time.sleep(0.1)

    def image(format):
        self.camera.capture(self.rawCapture, format=format, resize=(128,64))
        return self.rawCapture.array

    def video(format):
        return self.camera.capture_continuous(self.rawCapture, format=format, use_video_port=True)
