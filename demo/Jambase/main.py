import sys
from pathlib import Path
from typing import Any, Dict

import yaml
from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import (
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QWidget,
)
import cv2

from av_jambase.demo import Demo, parse_args

from .camera import CameraThread
from .utils.threads import Worker
from .components.questionnaire import Questionnaire
from .components.stream_viewer import StreamViewer
from .components.session_manager import AvatarSessionManager

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
            # setup the writers
            fps = cam.cap.get(cv2.CAP_PROP_FPS)
            width = int(cam.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cam.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(
                self.session_manager.save_path.joinpath(f"user{num}.mp4").as_posix(),
                fourcc=cv2.VideoWriter.fourcc(*cfg.get("save_fourcc", "mp4v")),
                fps=fps,
                frameSize=(width, height),
            )
            cam_layout.cam_cap = cam.cap
            cam_layout.set_cam_writer(writer)
            writer = cv2.VideoWriter(
                self.session_manager.save_path.joinpath(
                    f"reenactment{num}-session{self.session_manager.session_num}.mp4"
                ).as_posix(),
                fourcc=cv2.VideoWriter.fourcc(*cfg.get("save_fourcc", "mp4v")),
                fps=fps,
                frameSize=cfg[f"{user}_user"].get("reenactment_img_size", (256, 256)),
            )
            if writer is None or not writer.isOpened():
                raise ValueError(
                    "Failed to create video writer for camera.",
                    self.session_manager.save_path.joinpath(
                        f"reenactment{num}-session{self.session_manager.session_num}.mp4"
                    ).as_posix(),
                    cv2.VideoWriter.fourcc(*cfg.get("save_fourcc", "mp4v")),
                    fps,
                    cfg[f"{user}_user"].get("reenactment_img_size", (256, 256)),
                )
            cam_layout.set_manip_writer(writer)
            # run the camera thread
            cam.start()
            return cam_layout, cam

        self.left_stream, left_cam = setup(1, "left")
        self.right_stream, right_cam = setup(2, "right")

        # Section 3: Questionnaire
        questionnaire_cfg = cfg.get(
            "questionnaire",
            {
                "question": "How good is the avatar?",
                "choices": ["Great", "Good", "Okay", "Bad"],
            },
        )
        # session folder
        questionnaire_cfg["save_path"] = self.session_manager.save_path.as_posix()
        questionnaire_layout = Questionnaire(
            questionnaire_cfg["question"],
            questionnaire_cfg["choices"],
            cfg=questionnaire_cfg,
        )
        questionnaire_layout.next_btn.setEnabled(False)
        questionnaire_box = QGroupBox("Questionnaire")
        questionnaire_box.setLayout(questionnaire_layout)
        main_layout.addWidget(questionnaire_box, stretch=0)
        questionnaire_layout.next_btn.clicked.connect(self.start_next)
        questionnaire_layout.left_group.buttonClicked.connect(self.activate_next_btn)
        questionnaire_layout.right_group.buttonClicked.connect(self.activate_next_btn)
        self.questionnaire = questionnaire_layout

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
        worker.signals.error.connect(print)
        self.threadpool.start(worker)

    def activate_next_btn(self, *args, **kwargs):
        """Activate the next button if both left and right choices are selected."""
        if (
            self.questionnaire.is_valid()
            and self.right_stream.demo is not None
            and self.left_stream.demo is not None
        ):
            self.questionnaire.next_btn.setEnabled(True)
        else:
            self.questionnaire.next_btn.setEnabled(False)

    def start_next(self):
        if self.questionnaire.next_btn.text() == "Finish":
            # Save the questionnaire responses and close the application
            self.questionnaire.record()
            self.close()
            return
        self.session_manager.next_session()

        # Record the current questionnaire responses and reset the question
        self.questionnaire.record()
        self.questionnaire.reset_question()
        self.questionnaire.next_btn.setEnabled(False)

        # close the video writers
        save_path = self.cfg["save_path"]

        def update(num: int, user: str, stream: StreamViewer):
            assert user in {"left", "right"}, "User must be either 'left' or 'right'."
            assert 1 <= num <= 2, "User number must be either 1 or 2."
            ## Reenactment video writer
            fps = stream.cam_cap.get(cv2.CAP_PROP_FPS)
            writer = cv2.VideoWriter(
                self.session_manager.save_path.joinpath(
                    f"reenactment{num}-session{self.session_manager.session_num}.mp4"
                ).as_posix(),
                fourcc=cv2.VideoWriter.fourcc(*self.cfg.get("save_fourcc", "mp4v")),
                fps=fps,
                frameSize=self.cfg[f"{user}_user"].get(
                    "reenactment_img_size", (256, 256)
                ),
            )
            if writer is None or not writer.isOpened():
                raise ValueError(
                    "Failed to create video writer for left camera.",
                    self.session_manager.save_path.joinpath(
                        f"reenactment{num}-session{self.session_manager.session_num}.mp4"
                    ).as_posix(),
                    cv2.VideoWriter.fourcc(*self.cfg.get("save_fourcc", "mp4v")),
                    fps,
                    self.cfg[f"{user}_user"].get("reenactment_img_size", (256, 256)),
                )
            switch = stream.manip_writer
            stream.set_manip_writer(writer)
            switch.release()

        update(1, "left", self.left_stream)
        update(2, "right", self.right_stream)

        # Start the next reenactment
        self.left_stream.demo.prep_source(self.session_manager.get_left_avatar())
        self.right_stream.demo.prep_source(self.session_manager.get_right_avatar())

        if self.session_manager.available_sessions() <= 0:
            # the experiment is over
            self.questionnaire.next_btn.setText("Finish")
            return

    def closeEvent(self, event):
        self.hide()
        for thread in self.threads:
            thread.stop()
        event.accept()
        # Release the video writers
        if self.left_stream.cam_writer is not None:
            self.left_stream.cam_writer.release()
        if self.left_stream.manip_writer is not None:
            self.left_stream.manip_writer.release()
        if self.right_stream.cam_writer is not None:
            self.right_stream.cam_writer.release()
        if self.right_stream.manip_writer is not None:
            self.right_stream.manip_writer.release()


if __name__ == "__main__":
    import torch

    with torch.inference_mode():
        cfg = yaml.safe_load((__dir__ / "config.yaml").read_text(encoding="utf-8"))
        app = QApplication(sys.argv)
        window = MainWindow(cfg)
        window.show()
        sys.exit(app.exec())
