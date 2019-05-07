import picamera
import io
import time
from image_segment import BackgroundSubtract, HandSegment

stream = io.BytesIO()

with picamera.PiCamera() as camera:
    camera.resolution = (640, 480)
    camera.framerate = 30
    # Wait for the automatic gain control to settle
    time.sleep(2)
    # Now fix the values
    camera.shutter_speed = camera.exposure_speed
    camera.exposure_mode = 'off'
    g = camera.awb_gains
    camera.awb_mode = 'off'
    camera.awb_gains = g
    camera.capture(stream, format='jpeg')
    with picamera.array.PiRGBArray(camera) as stream:
        bgsub = BackgroundSubtract()
        for i in range(30):
            camera.capture(stream, format='bgr')
            image = stream.array
            bgsub.calibrate(image)
        while True:
            camera.capture(stream, format='bgr')
            image = stream.array
            mask = bgsub.foreground(image)
            frame = graytorgb(mask, image)
            cv2.imshow("Subtracted", frame)
            cv2.waitKey(0)
