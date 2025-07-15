from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget


class StopwatchWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.elapsed_time = 0  # milliseconds
        self.running = False

        self.timer = QTimer(self)
        self.timer.setInterval(10)  # update every 10 ms
        self.timer.timeout.connect(self.update_time)

        self.label = QLabel(self.format_time(self.elapsed_time))
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("font-size: 48px;")

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

    def format_time(self, ms):
        seconds = ms // 1000
        minutes = seconds // 60
        return f"{minutes%60:02}:{seconds%60:02}"

    def update_time(self):
        if self.running:
            self.elapsed_time += 10
            self.label.setText(self.format_time(self.elapsed_time))

    def start(self):
        if not self.running:
            self.running = True
            self.timer.start()
        return self

    def stop(self):
        if self.running:
            self.running = False
            self.timer.stop()
        return self

    def reset(self):
        self.stop()
        self.elapsed_time = 0
        self.label.setText(self.format_time(self.elapsed_time))
        return self
