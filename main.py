import picamera
import io
import time
from image_helpers import graytorgb
from background_subtract import BackgroundSubtract
from hand_segment import HandSegment
from shadow_analysis import ShadowAnalysis

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
        segment = HandSegment()
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
            mask2 = segment.extract(frame)
            mask2 = segment.denoise(mask2)
            mask2 = segment.contour(mask2)
            frame2 = graytorgb(mask2, image)
            cv2.imshow("Detected", frame2)
            cv2.waitKey(0)
