import sys
from pathlib import Path
from typing import Any, Dict

import yaml
from PySide6.QtCore import QThreadPool, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QWidget,
)
from PySide6.QtGui import QIntValidator

import cv2

from av_jambase.demo import Demo, parse_args

from .camera import CameraThread
from .utils.threads import Worker
from .components.stream_viewer import StreamViewer
from .components.session_manager import AvatarSessionManager
from .components.session_timer import StopwatchWidget
from .components.loading_widget import LoadingWidget
from .utils import files as files_utils

__dir__ = Path(__file__).resolve().parent


class MainWindow(QWidget):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.session_manager = AvatarSessionManager(cfg)

        self.setWindowTitle(cfg.get("window_title", "Stream Comparison"))
        main_layout = QHBoxLayout()

        def setup(num: int, user: str):
            assert user in {"left", "right"}, "User must be either 'left' or 'right'."
            assert 1 <= num <= 2, "User number must be either 1 or 2."

            cam_layout = StreamViewer(f"user{num}", f"reenactment{num}")
            cam_box = QGroupBox(f"参加者{num}")
            cam_box.setLayout(cam_layout)
            main_layout.addWidget(cam_box, stretch=1)
            # Feed the camera stream
            cam = CameraThread(cfg[f"{user}_user"]["user_cam"])
            cam.change_pixmap_signal.connect(cam_layout.connect_stream)
            cam_layout.set_cam_cap(cam.cap)
            # run the camera thread
            cam.start()
            return cam_layout, cam

        self.left_stream, left_cam = setup(1, "left")
        self.right_stream, right_cam = setup(2, "right")

        last_column_layout = QVBoxLayout()

        self.session_timer = StopwatchWidget()
        self.avatar_timer = StopwatchWidget()
        last_column_layout.addStretch()
        last_column_layout.addWidget(self.session_timer)
        last_column_layout.addWidget(self.avatar_timer)
        last_column_layout.addStretch()



        self.loading_widget = LoadingWidget(gif_path=cfg.get("loading_gif_path", None))
        last_column_layout.addWidget(self.loading_widget)

        experiment_controls = QGroupBox("Experiment Controls")
        experiment_controls_layout = QHBoxLayout()
        experiment_controls_layout.addWidget(QLabel("Experiment"))
        self.exp_textedit = QLineEdit()
        self.exp_textedit.setPlaceholderText("experiment here.")
        self.exp_textedit.setValidator(QIntValidator())
        experiment_controls_layout.addWidget(self.exp_textedit)
        self.start_btn = QPushButton("Start")
        self.start_btn.setEnabled(False)
        self.next_btn = QPushButton("Next")
        self.next_btn.setEnabled(False)
        self.next_btn.hide()  # Hide the next button initially
        self.finish_btn = QPushButton("Finish")
        self.finish_btn.setEnabled(False)
        self.finish_btn.hide()  # Hide the next button initially

        experiment_controls_layout.addWidget(self.start_btn)
        experiment_controls_layout.addWidget(self.next_btn)
        experiment_controls_layout.addWidget(self.finish_btn)
        experiment_controls.setLayout(experiment_controls_layout)
        last_column_layout.addWidget(experiment_controls, stretch=0)

        main_layout.addLayout(last_column_layout, stretch=0)

        # Connect signals and slots
        self.start_btn.clicked.connect(self.on_click_start)
        self.next_btn.clicked.connect(self.on_next_click)
        self.finish_btn.clicked.connect(self.on_finish_click)
        self.exp_textedit.textChanged.connect(self.activate_start_btn)
        self.left_stream.validityChanged.connect(self.activate_start_btn)
        self.right_stream.validityChanged.connect(self.activate_start_btn)

        self.setLayout(main_layout)
        self.resize(1200, 600)
        self.threads = [left_cam, right_cam]

        self.threadpool = QThreadPool()
        # setup left demo
        worker = Worker(
            Demo,
            parse_args(
                f"--source_image {self.session_manager.get_left_avatar()} --preprocess_driving".split()
            ),
        )
        worker.signals.result.connect(self.left_stream.set_demo)
        worker.signals.result.connect(self.activate_start_btn)
        worker.signals.error.connect(print)
        self.threadpool.start(worker)
        # setup right demo
        worker = Worker(
            Demo,
            parse_args(
                f"--source_image {self.session_manager.get_right_avatar()} --preprocess_driving".split()
            ),
        )
        worker.signals.result.connect(self.right_stream.set_demo)
        worker.signals.result.connect(self.activate_start_btn)
        worker.signals.error.connect(print)
        self.threadpool.start(worker)

    def activate_start_btn(self, *args, **kwargs):
        """Activate the next button if both left and right choices are selected."""
        if (
            self.exp_textedit.text().strip() != ""
            and self.left_stream.is_valid()
            and self.right_stream.is_valid()
        ):
            self.start_btn.setEnabled(True)
            self.loading_widget.stop()
            self.loading_widget.hide()
        else:
            self.start_btn.setEnabled(False)

    def on_click_start(self):
        """Start the experiment."""
        self.session_timer.start()
        self.avatar_timer.start()
        self.exp_textedit.setEnabled(False)
        self.start_btn.hide()
        self.next_btn.show()
        QTimer.singleShot(10_000, lambda: self.next_btn.setEnabled(True))
        self.left_stream.activate_demo()
        self.right_stream.activate_demo()

        def setup(stream: StreamViewer, user: str):
            assert user in {"left", "right"}, "User must be either 'left' or 'right'."

            # setup the writers
            fps = stream.cam_cap.get(cv2.CAP_PROP_FPS)
            width = int(stream.cam_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(stream.cam_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer_file = files_utils.get_stream_path(
                self.exp_textedit.text(),
                stream.user_id,
                self.session_manager.save_path,
            )
            writer_file.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                writer_file,
                fourcc=cv2.VideoWriter.fourcc(*cfg.get("save_fourcc", "mp4v")),
                fps=fps,
                frameSize=(width, height),
            )
            stream.set_cam_writer(writer)
            writer = cv2.VideoWriter(
                files_utils.get_reenactment_path(
                    self.exp_textedit.text(),
                    stream.user_id,
                    self.session_manager.session_num,
                    self.session_manager.save_path,
                ).as_posix(),
                fourcc=cv2.VideoWriter.fourcc(*cfg.get("save_fourcc", "mp4v")),
                fps=fps,
                frameSize=cfg[f"{user}_user"].get("reenactment_img_size", (256, 256)),
            )
            if writer is None or not writer.isOpened():
                raise ValueError(
                    "Failed to create video writer for camera.",
                )
            stream.set_manip_writer(writer)

        setup(self.left_stream, "left")
        setup(self.right_stream, "right")
        files_utils.link_other_subject(
            self.exp_textedit.text(),
            self.left_stream.user_id,
            self.right_stream.user_id,
            self.session_manager.save_path,
        )

    def on_next_click(self):
        self.session_manager.next_session()
        self.avatar_timer.reset().start()

        self.next_btn.setEnabled(False)
        QTimer.singleShot(10_000, lambda: self.next_btn.setEnabled(True))

        def update(stream: StreamViewer, user: str, new_source: str):
            assert user in {"left", "right"}, "User must be either 'left' or 'right'."

            # setup the writers
            fps = stream.cam_cap.get(cv2.CAP_PROP_FPS)
            writer = cv2.VideoWriter(
                files_utils.get_reenactment_path(
                    self.exp_textedit.text(),
                    stream.user_id,
                    self.session_manager.session_num,
                    self.session_manager.save_path,
                ).as_posix(),
                fourcc=cv2.VideoWriter.fourcc(*cfg.get("save_fourcc", "mp4v")),
                fps=fps,
                frameSize=cfg[f"{user}_user"].get("reenactment_img_size", (256, 256)),
            )
            if writer is None or not writer.isOpened():
                raise ValueError(
                    "Failed to create video writer for camera.",
                )
            switch = stream.manip_writer
            worker = Worker(
                stream.demo.prep_source,
                new_source,
            )
            self.threadpool.start(worker)
            stream.set_manip_writer(writer)
            switch.release()

        update(self.left_stream, "left", self.session_manager.get_left_avatar())
        update(self.right_stream, "right", self.session_manager.get_right_avatar())

        if self.session_manager.available_sessions() <= 0:
            # the experiment is over
            self.next_btn.hide()
            self.finish_btn.show()
            QTimer.singleShot(10_000, lambda: self.finish_btn.setEnabled(True))

    def on_finish_click(self):
        """Finish the experiment."""
        self.hide()
        for thread in self.threads:
            thread.stop()
        self.left_stream.stop()
        self.right_stream.stop()
        self.close()

    def closeEvent(self, event):
        event.accept()
        QApplication.quit()


if __name__ == "__main__":
    import torch

    with torch.inference_mode():
        cfg = yaml.safe_load((__dir__ / "config.yaml").read_text(encoding="utf-8"))
        app = QApplication(sys.argv)
        window = MainWindow(cfg)
        window.show()
        sys.exit(app.exec())
