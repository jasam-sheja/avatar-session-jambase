from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QMovie

class LoadingWidget(QWidget):
    def __init__(self, parent=None, gif_path=None):
        super().__init__(parent)
        self.label = QLabel()
        self.movie = None
        if gif_path:
            self.movie = QMovie(gif_path)
            self.label.setMovie(self.movie)
        else:
            # fallback: spinner animation using Unicode
            self.frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            self.frame_idx = 0
            self.timer = QTimer(self)
            self.timer.setInterval(100)
            self.timer.timeout.connect(self.update_text)
            self.label.setText(self.frames[self.frame_idx])

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.start()

    def start(self):
        if self.movie:
            self.movie.start()
        else:
            self.timer.start()

    def stop(self):
        if self.movie:
            self.movie.stop()
        else:
            self.timer.stop()

    def update_text(self):
        self.frame_idx = (self.frame_idx + 1) % len(self.frames)
        self.label.setText(self.frames[self.frame_idx] + " Loading...")
