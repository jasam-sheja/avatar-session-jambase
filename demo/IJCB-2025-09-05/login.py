import sys
from PySide6.QtCore import Qt, QSettings, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QCheckBox,
    QMessageBox,
    QFormLayout,
    QSpacerItem,
    QSizePolicy,
)

# --- Fake user store (replace with real auth) ---
USERS = {
    "ammar": "ammar",
    "allam": "allam",
}


class LoginDialog(QDialog):
    authenticated = Signal(str)  # emits the username on success

    def __init__(self, database, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Login")
        self.setModal(True)
        self.setMinimumWidth(320)
        self.database = database

        self.settings = QSettings("ExampleCo", "LoginDemo")

        # Widgets
        self.user_edit = QLineEdit()
        self.user_edit.setPlaceholderText("Username")
        self.user_edit.setClearButtonEnabled(True)

        self.pass_edit = QLineEdit()
        self.pass_edit.setPlaceholderText("Password")
        self.pass_edit.setEchoMode(QLineEdit.Password)
        self.pass_edit.setClearButtonEnabled(True)

        self.show_pw = QCheckBox("Show password")
        self.show_pw.toggled.connect(
            lambda checked: self.pass_edit.setEchoMode(
                QLineEdit.Normal if checked else QLineEdit.Password
            )
        )

        self.remember = QCheckBox("Remember username")

        self.login_btn = QPushButton("Log in")
        self.login_btn.setDefault(True)
        self.cancel_btn = QPushButton("Cancel")

        # Layout
        form = QFormLayout()
        form.addRow("Username:", self.user_edit)
        form.addRow("Password:", self.pass_edit)

        btns = QHBoxLayout()
        btns.addStretch(1)
        btns.addWidget(self.cancel_btn)
        btns.addWidget(self.login_btn)

        root = QVBoxLayout()
        root.addLayout(form)
        root.addWidget(self.show_pw)
        root.addWidget(self.remember)
        root.addItem(QSpacerItem(0, 8, QSizePolicy.Minimum, QSizePolicy.Expanding))
        root.addLayout(btns)
        self.setLayout(root)

        # Signals
        self.login_btn.clicked.connect(self.try_login)
        self.cancel_btn.clicked.connect(self.reject)
        self.pass_edit.returnPressed.connect(self.try_login)
        self.user_edit.returnPressed.connect(self.try_login)

        # Prefill last username
        last_user = self.settings.value("last_username", "", str)
        if last_user:
            self.user_edit.setText(last_user)
            self.remember.setChecked(True)

    def try_login(self):
        username = self.user_edit.text().strip()
        password = self.pass_edit.text()

        if not username or not password:
            QMessageBox.warning(
                self, "Missing info", "Please enter both username and password."
            )
            return

        if not self.database.check_pass(username, password):
            QMessageBox.critical(self, "Login failed", "Invalid username or password.")
            return

        # Remember username optionally
        if self.remember.isChecked():
            self.settings.setValue("last_username", username)
        else:
            self.settings.remove("last_username")

        self.authenticated.emit(username)
        self.accept()


class MainWindow(QMainWindow):
    logout_requested = Signal()

    def __init__(self, username: str, parent=None):
        super().__init__(parent)
        self.username = username
        self.setWindowTitle("Main App")
        self.resize(640, 400)

        # Central widget
        label = QLabel(f"Welcome, {self.username}!", alignment=Qt.AlignCenter)
        label.setStyleSheet("font-size: 20px;")
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(label)
        self.setCentralWidget(container)

        # Menu / actions
        logout_action = QAction("Log out", self)
        logout_action.triggered.connect(self.logout_requested.emit)

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)

        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        file_menu.addAction(logout_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

        # Small status bar hint
        self.statusBar().showMessage("Logged in as: " + self.username)


def show_login(database, parent=None) -> str | None:
    """Show the login dialog; return the username on success, else None."""
    dlg = LoginDialog(database, parent)
    result = dlg.exec()
    return dlg.user_edit.text().strip() if result == QDialog.Accepted else None
