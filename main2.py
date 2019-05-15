import picamera
import time
import threading
from image_helpers import graytorgb
from background_subtract import BackgroundSubtract
from colour_segment import ColourSegment

done = False
lock = threading.Lock()
pool = []

class ImageProcessor(threading.Thread):
    def __init__(self, stream):
        super(ImageProcessor, self).__init__()
        self.stream = stream
        self.event = threading.Event()
        self.terminated = False
        self.start()

    def run(self):
        global done
        while not self.terminated:
            if self.event.wait(1):
                try:
                    self.stream.seek(0)
                finally:
                    self.stream.seek(0)
                    self.stream.truncate()
                    self.event.clear()
                    with lock:
                        pool.append(self)

def streams():
    while not done:
        with lock:
            if pool:
                processor = pool.pop()
            else:
                processor = None
        if processor:
            yield processor.stream
            processor.event.set()
        else:
            time.sleep(0.1)

with picamera.PiCamera() as camera:
    pool = [ImageProcessor() for i in range(4)]
    camera.resolution = (640, 480)
    camera.framerate = 30
    time.sleep(2)
    camera.shutter_speed = camera.exposure_speed
    camera.exposure_mode = 'off'
    g = camera.awb_gains
    camera.awb_mode = 'off'
    camera.awb_gains = g
    camera.hflip = True
    camera.vflip = True
    pool = [ImageProcessor(picamera.array.PiRGBArray(camera)) for i in range(4)]
    bgsub = BackgroundSubtract()
    segment = ColourSegment()
    for i in range(30):
        camera.capture(streams(), format='bgr')
        image = stream.array
        bgsub.calibrate(image)

while pool:
    with lock:
        processor = pool.pop()
    processor.terminated = True
    processor.join()
