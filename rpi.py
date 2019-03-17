from picamera.array import PiRGBArray
from picamera import PiCamera
import time

class PiCameraWrapper():
    def __init__(self, resolution, framerate):
        self.camera = PiCamera()
        self.resolution = resolution
        self.framerate = framerate
        self.rawCapture = PiRGBArray(self.camera, size=self.resolution)
        time.sleep(0.1)
    
    def image(format):
        self.camera.capture(self.rawCapture, format=format)
        return self.rawCapture.array

    def video(framerate, format):
        return self.camera.capture_continuous(self.rawCapture, format=format, use_video_port=True)

