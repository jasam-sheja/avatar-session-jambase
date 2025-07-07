
from typing import Tuple
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QThread, Signal, Qt, QObject
import cv2
import numpy as np

class CameraThread(QThread):
    change_pixmap_signal = Signal(QPixmap, np.ndarray)

    def __init__(self, index, scale:Tuple[int, int]=None, /, parent: QObject | None = None) -> None:
        super().__init__(parent=parent)
        self.index = index
        self.scale = scale


        self.cap = cv2.VideoCapture(self.index) # 0 for default camera
        if not self.cap.isOpened():
            print(f"Error: Could not open camera, {self.index}.")
            return
        if not self.cap.set(cv2.CAP_PROP_FPS, 30):
            print(f"Warning: Could not set FPS for camera, {self.index}.")
        if not self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')):
            print(f"Warning: Could not set FOURCC for camera, {self.index}.")
        if not self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920):
            print(f"Warning: Could not set frame width for camera, {self.index}.")
        if not self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080):
            print(f"Warning: Could not set frame height for camera, {self.index}.")

    def run(self):
        if not self.cap.isOpened():
            print(f"Error: Could not open camera, {self.index}.")
            return

        while True:
            ret, frame = self.cap.read()
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                if self.scale is not None:
                    p = convert_to_qt_format.scaled(*self.scale, Qt.AspectRatioMode.KeepAspectRatio)
                else:
                    p = convert_to_qt_format
                self.change_pixmap_signal.emit(QPixmap.fromImage(p), rgb_image)
            else:
                break
        self.cap.release()

    def stop(self):
        if self.cap is not None:
            self.cap.release()
        self.quit()
        self.wait()

def test_cameras():
    import cv2, glob, os
    candidates = sorted(glob.glob('/dev/v4l/by-id/*'))[::-1]
    for path in candidates:
        cap = cv2.VideoCapture(path, cv2.CAP_V4L2)   # explicit backend
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print(f'{os.path.basename(path):55s}', 'OK' if cap.isOpened() else 'FAIL')
        cap.release()