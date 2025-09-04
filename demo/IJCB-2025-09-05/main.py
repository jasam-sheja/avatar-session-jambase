import sys
from pathlib import Path
from typing import Any, Dict
import logging

import yaml
from PySide6.QtCore import QThreadPool, QTimer
from PySide6.QtGui import QAction, QIntValidator
from PySide6.QtWidgets import (
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from av_jambase.demov2 import Demo, parse_args

from .camera import CameraThread
from .components.loading_widget import LoadingWidget
from .components.session_manager import AvatarSessionManager
from .components.stream_viewer import StreamViewer
from .utils.threads import Worker
from .utils.database import CFGDatabase
from .login import show_login
from .components.auth_widget import AuthWidget

__dir__ = Path(__file__).resolve().parent

class MainWindow(QMainWindow):
    def __init__(self, cfg_path: Path):
        super().__init__()
        self.cfg_path = cfg_path
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        self.cfg = cfg
        self.session_manager = AvatarSessionManager(cfg)
        self.database_manager = CFGDatabase(cfg)

        self.setWindowTitle(cfg.get("window_title", "Stream Comparison"))
        main_layout = QHBoxLayout()

        def setup():
            cam_layout = StreamViewer(f"user", f"reenactment")
            cam_box = QGroupBox(f"参加者")
            cam_box.setLayout(cam_layout)
            main_layout.addWidget(cam_box, stretch=1)
            # Feed the camera stream
            cam = CameraThread(cfg["input"]["cam"], **cfg["input"].get("_kwargs_", {}))
            cam.change_pixmap_signal.connect(cam_layout.connect_stream)
            cam_layout.set_cam_cap(cam.cap)
            # run the camera thread
            cam.start()
            return cam_layout, cam

        self.stream, cam = setup()
        # self.auth = AuthWidget(cfg["auth"]["config_path"])
        # self.stream.newManipFrame.connect(self.auth.add)
        # main_layout.addLayout(self.auth)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        self.resize(1200, 600)

        # Create a menu bar
        menu_bar = self.menuBar()
        # Create a User menu
        user_menu = menu_bar.addMenu("User") 

        # Create actions
        login_action = QAction("Login", self)
        login_action.triggered.connect(self.show_login)
        user_menu.addAction(login_action)
        self._login_action = login_action

        logout_action = QAction("Logout", self)
        logout_action.triggered.connect(self.logout)
        user_menu.addAction(logout_action)
        logout_action.setVisible(False)
        self._logout_action = logout_action

        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(QApplication.quit)
        user_menu.addAction(quit_action)

        self.threads = [cam]

        self.threadpool = QThreadPool()
        # setup left demo
        worker = Worker(
            Demo,
            parse_args(f"--preprocess_driving --visualize all".split()),
        )
        worker.signals.result.connect(self.stream.set_demo)
        worker.signals.error.connect(print)
        self.threadpool.start(worker)

    def reload_cfg(self):
        self.cfg.update(yaml.safe_load(self.cfg_path.read_text(encoding="utf-8")))

    def show_login(self):
        if (user := show_login(self.database_manager, self)) is None:
            return
        self.reload_cfg()
        self._logout_action.setVisible(True)
        self._login_action.setVisible(False)
        # TODO: Start avatar session
        self.user = user
        self.stream.demo.prep_source(self.database_manager.get_avatar(user))
        self.stream.activate_demo()

    def logout(self):
        self._login_action.setVisible(True)
        self._logout_action.setVisible(False)
        # TODO: Stop avatar session
        self.stream.deactivate_demo()
        # self.auth.reset()

    def on_click_start(self):
        """Start the experiment."""
        self.stream.activate_demo()

    def on_finish_click(self):
        """Finish the experiment."""
        self.hide()
        for thread in self.threads:
            thread.stop()
        self.stream.stop()
        self.close()

    def closeEvent(self, event):
        event.accept()
        QApplication.quit()  # TODO: check if this is necessary


if __name__ == "__main__":
    import torch

    torch.set_float32_matmul_precision("high")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s :%(message)s")

    with torch.inference_mode():
        
        app = QApplication(sys.argv)
        window = MainWindow((__dir__ / "config.yaml"))
        window.show()
        # show_login()
        sys.exit(app.exec())
