from PySide6.QtCore import Qt, QThreadPool, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
)

import numpy as np
import cv2
import torch

from av_jambase.demov2 import Demo
from ..utils.threads import Worker


class StreamViewer(QVBoxLayout):
    # Signal to indicate validity change of the stream viewer
    validityChanged = Signal(bool)
    # Signal to update the manipulated view
    updateManipView = Signal(QPixmap)
    newManipFrame = Signal(np.ndarray)

    def __init__(self, cam_label: str, manip_label: str):
        super().__init__()
        self.cam_view = QLabel(cam_label)
        self.cam_view.setFrameShape(QFrame.Box)
        self.cam_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cam_view.setStyleSheet("background-color: black;")
        self.manip_view = QLabel(manip_label)
        self.manip_view.setFrameShape(QFrame.Box)
        self.manip_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.manip_view.setStyleSheet("background-color: black;")
        self.addWidget(self.cam_view)
        self.addWidget(self.manip_view)

        # Connect signals and slots
        self.updateManipView.connect(self.manip_view.setPixmap)

        self.demo = None
        self._show_demo = False
        self._demo_running = False
        self.cam_cap: cv2.VideoCapture = None
        # Limit to one thread for camera processing
        self.cuda_pool = QThreadPool(maxThreadCount=3)

    def is_valid(self) -> bool:
        """Check if the user ID is valid."""
        return (
            self.demo is not None
            and self._demo_running
        )

    def set_cam_cap(self, cam_cap: cv2.VideoCapture):
        """Set the camera capture object."""
        self.cam_cap = cam_cap

    def activate_demo(self, show: bool = True):
        """Activate or deactivate the demo."""
        self._show_demo = show

    def deactivate_demo(self):
        """Deactivate the demo."""
        self._show_demo = False
        if self.demo is not None:
            self.demo.reset()
        self.cuda_pool.clear()
        print(self.cuda_pool.activeThreadCount(), self.cuda_pool.maxThreadCount())
        self.manip_view.clear()
        print('deactivate_demo')

    def set_demo(self, demo: Demo):
        isvalid = self.is_valid()
        self.demo = demo
        if isvalid != self.is_valid():
            self.validityChanged.emit(self.is_valid())

    def connect_stream(self, pixmap: QPixmap, rgb_image: np.ndarray):
        self.cam_view.setPixmap(
            pixmap.scaled(
                self.cam_view.size().shrunkBy(self.cam_view.contentsMargins()),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        if self.demo is None or not self.demo.is_ready():
            return
        # delete the previous CUDA tasks to keep it real-time
        self.cuda_pool.clear()
        self.cuda_pool.start(Worker(self.update_manip_view, rgb_image))

    def update_manip_view(self, rgb_image: np.ndarray):
        # Update the manipulated view with the current frame
        if rgb_image.shape > (640, 640, 3):
            rgb_image = cv2.resize(
                rgb_image,
                (640, int(640 * rgb_image.shape[0] / rgb_image.shape[1])),
                interpolation=cv2.INTER_LANCZOS4,
            )
        with torch.cuda.stream(torch.cuda.Stream()):
            try:
                rgb_manip = self.demo.apply(rgb_image)
            except TypeError as e:
                return  # The demo is closed or not ready
        if rgb_manip is None:
            return  # The demo is closed or not ready

        isvalid = self.is_valid()
        self._demo_running = True
        if isvalid != self.is_valid():
            self.validityChanged.emit(self.is_valid())
        if not self._show_demo:
            return
        # Convert the manipulated RGB image to QImage
        rgb_manip = np.ascontiguousarray(rgb_manip)
        self.newManipFrame.emit(rgb_manip)
        h, w, ch = rgb_manip.shape
        bytes_per_line = ch * w
        img = QImage(rgb_manip.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        # self.manip_view.setPixmap(QPixmap.fromImage(img))
        pixmap = QPixmap.fromImage(img)
        self.updateManipView.emit(
            pixmap.scaled(
                self.manip_view.size() * (3 / 4),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def release(self):
        """Release the camera."""

    def wait(self):
        self.cuda_pool.waitForDone()

    def stop(self):
        self.wait()
        self.release()
