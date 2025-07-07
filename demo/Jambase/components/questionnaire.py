import json
from pathlib import Path
from typing import Any, Dict

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QVBoxLayout,
)


class Questionnaire(QVBoxLayout):
    def __init__(self, question: str, choices: list, cfg: Dict[str, Any] = {}):
        super().__init__()
        self.left_group = QButtonGroup()
        self.right_group = QButtonGroup()
        font_size = 20

        # setup the layout for choices
        choices_layout = QHBoxLayout()
        left_col = QVBoxLayout()
        left_col.setAlignment(Qt.AlignmentFlag.AlignRight)
        left_col.addStretch()
        center_col = QVBoxLayout()
        center_col.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center_col.addStretch()
        right_col = QVBoxLayout()
        right_col.setAlignment(Qt.AlignmentFlag.AlignLeft)
        right_col.addStretch()
        for ch in choices:
            left_btn = QRadioButton()
            right_btn = QRadioButton()
            self.left_group.addButton(left_btn)
            self.right_group.addButton(right_btn)
            label = QLabel(ch)
            left_btn.setText(ch)
            right_btn.setText(ch)
            left_btn.setStyleSheet(
                """
    QRadioButton { color: transparent;      /* text invisible */
                   padding-left: 0px; }     /* optional: close gap */
"""
            )
            left_btn.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
            right_btn.setStyleSheet(
                """
    QRadioButton { color: transparent;      /* text invisible */
                   padding-left: 0px; }     /* optional: close gap */
"""
            )
            label.setStyleSheet(f"font-size: {font_size-4}px;")
            label.setAlignment(Qt.AlignCenter)
            left_col.addWidget(left_btn)
            left_col.addStretch()
            center_col.addWidget(label)
            center_col.addStretch()
            right_col.addWidget(right_btn)
            right_col.addStretch()
        choices_layout.addLayout(left_col)
        choices_layout.addLayout(center_col)
        choices_layout.addLayout(right_col)

        label = QLabel(question)
        label.setStyleSheet(f"font-size: {font_size}px; font-weight: bold;")
        label.setAlignment(Qt.AlignCenter)
        self.addWidget(label)
        self.addLayout(choices_layout)

        self.cfg = cfg

        self._l_recorded = []
        self._r_recorded = []

    def is_valid(self):
        """Check if both left and right choices are selected."""
        left_choice = self.left_group.checkedButton()
        right_choice = self.right_group.checkedButton()
        return left_choice is not None and right_choice is not None

    def record(self):
        """Record the selected choices."""
        if self.is_valid():
            left_choice = self.left_group.checkedButton()
            right_choice = self.right_group.checkedButton()
            left_choice_text = left_choice.text()
            right_choice_text = right_choice.text()
            self._l_recorded.append(left_choice_text)
            self._r_recorded.append(right_choice_text)
        else:
            raise ValueError("Both left and right choices must be selected.")

    def save_left(self, file_path: Path):
        """Save the selected choices."""

        file_path.write_text(
            json.dumps(self._l_recorded, indent=4, ensure_ascii=False), encoding="utf-8"
        )

    def save_right(self, file_path: Path):
        """Save the selected choices."""
        file_path.write_text(
            json.dumps(self._r_recorded, indent=4, ensure_ascii=False), encoding="utf-8"
        )

    def reset_question(self):
        """Reset the state of the questionnaire."""
        self.left_group.setExclusive(False)
        self.right_group.setExclusive(False)
        for btn in self.left_group.buttons():
            btn.setChecked(False)
        for btn in self.right_group.buttons():
            btn.setChecked(False)
        self.left_group.setExclusive(True)
        self.right_group.setExclusive(True)
