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

from av_jambase.demo import Demo
from ..utils.threads import Worker


class StreamViewer(QVBoxLayout):
    validityChanged = Signal(bool)

    def __init__(self, cam_label: str, manip_label: str):
        super().__init__()
        self.cam_view = QLabel(cam_label)
        self.cam_view.setFrameShape(QFrame.Box)
        self.cam_view.setAlignment(Qt.AlignCenter)
        self.cam_view.setStyleSheet("background-color: black;")
        self.manip_view = QLabel(manip_label)
        self.manip_view.setFrameShape(QFrame.Box)
        self.manip_view.setAlignment(Qt.AlignCenter)
        self.manip_view.setStyleSheet("background-color: black;")
        self.addWidget(self.cam_view)
        self.addWidget(self.manip_view)

        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("ID:"))
        self.lineedit = QLineEdit()
        self.lineedit.setPlaceholderText("Enter user ID")
        control_layout.addWidget(self.lineedit)
        control_layout.addStretch()
        self.accept_btn = QPushButton("Set")
        self.accept_btn.setEnabled(False)
        # accept_btn.setToolTip("Set the user ID for the stream viewer.")
        control_layout.addWidget(self.accept_btn)
        self.addLayout(control_layout)

        # Connect signals and slots
        self.lineedit.textChanged.connect(self.on_id_changed)
        self.accept_btn.clicked.connect(self.on_set_clicked)

        self.demo = None
        self._show_demo = False
        self.cam_cap: cv2.VideoCapture = None
        self.cam_writer: cv2.VideoWriter = None
        self.manip_writer: cv2.VideoWriter = None
        # Limit to one thread for camera processing
        self.cam_write_pool = QThreadPool(maxThreadCount=1)
        self.manip_write_pool = QThreadPool(maxThreadCount=1)
        self.cuda_pool = QThreadPool(maxThreadCount=3)

    def on_id_changed(self, text: str):
        if self.lineedit.text().strip():
            self.accept_btn.setEnabled(True)
        else:
            self.accept_btn.setEnabled(False)

    def on_set_clicked(self):
        """Handle the click event for the 'Set' button."""
        user_id = self.lineedit.text().strip()
        if not user_id:
            raise ValueError("User ID cannot be empty.")
        isvalid = self.is_valid()
        self.lineedit.setEnabled(False)
        if isvalid != self.is_valid():
            self.validityChanged.emit(True)
        self.accept_btn.setEnabled(False)
        self.accept_btn.hide()

    def is_valid(self) -> bool:
        """Check if the user ID is valid."""
        return not self.lineedit.isEnabled() and self.demo is not None

    def set_cam_cap(self, cam_cap: cv2.VideoCapture):
        """Set the camera capture object."""
        self.cam_cap = cam_cap

    @property
    def user_id(self) -> str:
        """Get the user ID for the stream viewer."""
        return self.lineedit.text().strip()

    def activate_demo(self, show: bool = True):
        """Activate or deactivate the demo."""
        self._show_demo = show

    def set_demo(self, demo: Demo):
        isvalid = self.is_valid()
        self.demo = demo
        if isvalid != self.is_valid():
            self.validityChanged.emit(self.is_valid())

    def set_cam_writer(self, cam_writer):
        self.cam_writer = cam_writer

    def set_manip_writer(self, manip_writer):
        self.manip_writer = manip_writer

    def connect_stream(self, pixmap: QPixmap, rgb_image: np.ndarray):
        self.cam_view.setPixmap(
            pixmap.scaled(
                self.cam_view.size().shrunkBy(self.cam_view.contentsMargins()),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        if self.cam_writer is not None:
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            self.cam_write_pool.start(Worker(self.cam_writer.write, bgr_image))
        if self.demo is None:
            return
        # delete the previous CUDA tasks to keep it real-time
        self.cuda_pool.clear()
        self.cuda_pool.start(Worker(self.update_manip_view, rgb_image))

    def update_manip_view(self, rgb_image: np.ndarray):
        # Update the manipulated view with the current frame
        if rgb_image.shape > (480, 640, 3):
            rgb_image = cv2.resize(
                rgb_image,
                (640, int(480 * rgb_image.shape[0] / rgb_image.shape[1])),
                interpolation=cv2.INTER_LANCZOS4,
            )
        with torch.cuda.stream(torch.cuda.Stream()):
            rgb_manip = self.demo.apply(rgb_image)
        if not self._show_demo:
            return
        # Convert the manipulated RGB image to QImage
        rgb_manip = np.ascontiguousarray(rgb_manip)
        h, w, ch = rgb_manip.shape
        bytes_per_line = ch * w
        img = QImage(rgb_manip.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        # self.manip_view.setPixmap(QPixmap.fromImage(img))
        pixmap = QPixmap.fromImage(img)
        self.manip_view.setPixmap(
            pixmap.scaled(
                self.manip_view.size() * (3 / 4),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        if self.manip_writer is not None:
            bgr_image = cv2.cvtColor(rgb_manip, cv2.COLOR_RGB2BGR)
            self.manip_write_pool.start(Worker(self.manip_writer.write, bgr_image))

    def release(self):
        """Release the camera and manipulation writers."""
        if self.cam_writer is not None:
            self.cam_writer.release()
            self.cam_writer = None
        if self.manip_writer is not None:
            self.manip_writer.release()
            self.manip_writer = None

    def wait(self):
        self.cam_write_pool.waitForDone()
        self.manip_write_pool.waitForDone()
        self.cuda_pool.waitForDone()

    def stop(self):
        self.wait()
        self.release()
