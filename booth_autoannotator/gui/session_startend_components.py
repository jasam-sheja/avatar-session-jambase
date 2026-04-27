#!/usr/bin/env python3
"""Shared GUI components for session start/end annotation tools."""

from __future__ import annotations

import copy
import os
import threading
import time
from typing import Dict, List, Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, QTime, Signal, Slot
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QTimeEdit,
    QVBoxLayout,
    QWidget,
)

from ..door_angle import DoorState
from ..face_detect import FaceDetection
from ..models import AnnotationValue, Session
from ..speech_vad import SpeechSegment

def fmt_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h:
        return f"{h:d}:{m:02d}:{s:06.3f}"
    return f"{m:02d}:{s:06.3f}"


def qtime_to_sec(qt: QTime) -> float:
    return qt.hour() * 3600 + qt.minute() * 60 + qt.second() + qt.msec() / 1000.0


def sec_to_qtime(t: float) -> QTime:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t % 1) * 1000))
    return QTime(h, m, s, ms)


# ──────────────────────────────────────────────────────────────────────────────
# VideoThread
# ──────────────────────────────────────────────────────────────────────────────

class VideoThread(QThread):
    """Decodes video frames in a background thread, emitting QImages."""

    frame_ready = Signal(QImage, float)   # (frame, timestamp_seconds)

    def __init__(self, video_path: str, parent=None):
        super().__init__(parent)
        self._video_path = video_path
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 30.0
        self._total_frames: int = 0
        self._speed: float = 1.0
        self._paused: bool = True
        self._pause_event = threading.Event()
        self._seek_to: Optional[float] = None
        self._lock = threading.Lock()
        self._running: bool = True

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self._video_path)
        if not self._cap.isOpened():
            return False
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return True

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def duration(self) -> float:
        return self._total_frames / self._fps if self._fps > 0 else 0.0

    @property
    def total_frames(self) -> int:
        return self._total_frames

    # ---- slots ----

    @Slot()
    def play(self):
        self._paused = False
        self._pause_event.set()

    @Slot()
    def pause(self):
        self._paused = True
        self._pause_event.clear()

    @Slot(float)
    def seek(self, t: float):
        with self._lock:
            self._seek_to = t
        # unblock the loop briefly to service the seek
        self._pause_event.set()

    @Slot(float)
    def set_speed(self, s: float):
        self._speed = max(0.05, s)

    def stop(self):
        self._running = False
        self._paused = False
        self._pause_event.set()
        self.wait(3000)

    # ---- thread loop ----

    def run(self):
        cap = self._cap
        if cap is None or not cap.isOpened():
            return

        frame_interval = 1.0 / self._fps
        next_due = time.perf_counter()

        while self._running:
            # service pending seek
            with self._lock:
                seek_t = self._seek_to
                self._seek_to = None

            if seek_t is not None:
                frame_idx = int(round(seek_t * self._fps))
                frame_idx = max(0, min(frame_idx, self._total_frames - 1))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                next_due = time.perf_counter()

            if self._paused:
                # emit one frame so the display updates on seek-while-paused
                if seek_t is not None:
                    ret, frame = cap.read()
                    if ret:
                        pos = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
                        self.frame_ready.emit(self._to_qimage(frame), pos / self._fps)
                self._pause_event.wait()
                if self._paused:
                    self._pause_event.clear()
                next_due = time.perf_counter()
                continue

            ret, frame = cap.read()
            if not ret:
                self.pause()
                continue

            pos = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
            self.frame_ready.emit(self._to_qimage(frame), pos / self._fps)

            next_due += frame_interval / max(0.05, self._speed)
            sleep_s = next_due - time.perf_counter()
            if sleep_s > 0.001:
                time.sleep(sleep_s)

        cap.release()

    @staticmethod
    def _to_qimage(frame: np.ndarray) -> QImage:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        return QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()


# ──────────────────────────────────────────────────────────────────────────────
# VideoLabel
# ──────────────────────────────────────────────────────────────────────────────

class VideoLabel(QLabel):
    """Displays a scaled video frame with face-bbox and timestamp overlay."""

    clicked = Signal()

    def __init__(self, cam_name: str = "Camera", parent=None):
        super().__init__(parent)
        self._cam_name = cam_name
        self._pixmap: Optional[QPixmap] = None
        self._timestamp: float = 0.0
        self._face_detections: List[FaceDetection] = []
        self._orig_w: int = 1
        self._orig_h: int = 1
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background: #1a1a1a;")

    @Slot(QImage, float)
    def update_frame(self, img: QImage, t: float):
        self._timestamp = t
        self._orig_w = img.width()
        self._orig_h = img.height()
        self._pixmap = QPixmap.fromImage(img)
        self.update()

    def set_face_detections(self, detections: List[FaceDetection]):
        self._face_detections = detections

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._pixmap:
            return

        painter = QPainter(self)
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x_off = (self.width() - scaled.width()) // 2
        y_off = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x_off, y_off, scaled)

        sx = scaled.width() / max(1, self._orig_w)
        sy = scaled.height() / max(1, self._orig_h)

        # Face bbox closest to current timestamp
        nearby = [
            d for d in self._face_detections
            if abs(d.timestamp - self._timestamp) < 0.5 and d.bbox is not None
        ]
        if nearby:
            best = min(nearby, key=lambda d: abs(d.timestamp - self._timestamp))
            x1, y1, x2, y2 = best.bbox
            rx = x_off + int(x1 * sx)
            ry = y_off + int(y1 * sy)
            rw = int((x2 - x1) * sx)
            rh = int((y2 - y1) * sy)
            painter.setPen(QPen(QColor(0, 220, 80), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(rx, ry, rw, rh)
            if best.det_score is not None:
                painter.setPen(QColor(0, 220, 80))
                painter.setFont(QFont("Monospace", 9))
                painter.drawText(rx, max(y_off + 12, ry - 4), f"{best.det_score:.2f}")

        # HUD
        painter.setPen(QColor(255, 255, 255, 210))
        painter.setFont(QFont("Monospace", 10, QFont.Weight.Bold))
        painter.drawText(x_off + 6, y_off + 18, f"{self._cam_name}  {fmt_time(self._timestamp)}")
        painter.end()

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)


# ──────────────────────────────────────────────────────────────────────────────
# TimelineWidget
# ──────────────────────────────────────────────────────────────────────────────

_ROW_H = 18 # height of each data row (door/speech/face)
_ROW_GAP = 4 # vertical gap between rows
_HEADER_H = 20 # height of top header row (labels)
_LABEL_W = 82 # width of left label column
_PLAYHEAD_W = 2 # width of the playhead line

_STATUS_COLORS: Dict[str, QColor] = {
    "complete":     QColor(60,  180,  80, 120),
    "needs_review": QColor(220, 160,  40, 120),
    "incomplete":   QColor(200,  60,  60, 120),
}


class TimelineWidget(QWidget):
    """Six-row timeline with draggable playhead and session spans."""

    playhead_moved  = Signal(float)   # seconds
    session_selected = Signal(str)    # session_id
    session_modified = Signal(str, str, float)  # (session_id, field_name, value)

    _ROW_LABELS = [
        "Cam1 Door", "Cam1 Speech", "Cam1 Face",
        "Cam2 Door", "Cam2 Speech", "Cam2 Face",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._duration: float = 1.0 # total timeline duration in seconds (used for scaling)
        self._playhead: float = 0.0 # current playhead position in seconds
        self._zoom: float = 1.0 # user-controlled zoom level (1.0 = fit to width)
        self._h_scroll: float = 0.0 # horizontal scroll offset in seconds (for zooming/panning)
        self._dragging = False
        # Panning state for Shift+drag
        self._panning: bool = False
        self._pan_start_x: Optional[float] = None
        self._pan_start_h_scroll: float = 0.0

        self._door_events:    List[List[DoorState]]     = [[], []]
        self._speech_segs:    List[List[SpeechSegment]] = [[], []]
        self._face_dets:      List[List[FaceDetection]] = [[], []]
        self._door_states:    List[List[DoorState]]     = [[], []]
        self._cam1_sessions:  List[Session]             = []
        self._cam2_sessions:  List[Session]             = []
        self._offsets:        List[float]               = [0.0, 0.0, 0.0, 0.0] # [cam1_offset, cam2_offset, audio1_offset, audio2_offset]
        self._selected_id:    Optional[str]             = None

        self.setMinimumHeight(self._total_h())
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setMouseTracking(True)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu)

    # ---- public setters ----

    def set_duration(self, duration: float):
        self._duration = max(1.0, duration)
        self.update()

    def set_data(
        self,
        door_events: List[List[DoorState]],
        speech_segs: List[List[SpeechSegment]],
        face_dets: List[List[FaceDetection]],
        door_states: List[List[DoorState]],
    ):
        self._door_events = door_events
        self._speech_segs = speech_segs
        self._face_dets   = face_dets
        self._door_states = door_states
        self.update()

    def set_sessions(
        self,
        cam1_sessions: List[Session],
        cam2_sessions: Optional[List[Session]] = None,
    ):
        self._cam1_sessions = cam1_sessions
        self._cam2_sessions = cam2_sessions if cam2_sessions is not None else []
        self.update()

    def set_offsets(self, cam1_offset: float, cam2_offset: float = 0.0, audio1_offset: Optional[float] = None, audio2_offset: Optional[float] = None):
        """Set per-camera wall-clock offsets (seconds) used to shift session bars on the timeline."""
        self._offsets = [cam1_offset, cam2_offset, audio1_offset if audio1_offset is not None else 0.0, audio2_offset if audio2_offset is not None else 0.0]
        self.update()

    def set_selected_session(self, sid: Optional[str]):
        self._selected_id = sid
        self.update()

    @Slot(float)
    def set_playhead(self, t: float):
        self._playhead = t
        self.update()

    # ---- coordinate helpers ----

    def _total_h(self) -> int:
        return _HEADER_H + len(self._ROW_LABELS) * (_ROW_H + _ROW_GAP)

    def _pps(self) -> float:
        """Pixels per second, based on current zoom and widget width."""
        avail = max(1, self.width() - _LABEL_W)
        return (avail / max(1.0, self._duration)) * self._zoom

    def _t_to_x(self, t: float) -> int:
        return _LABEL_W + int((t - self._h_scroll) * self._pps())

    def _x_to_t(self, x: float) -> float:
        return self._h_scroll + (x - _LABEL_W) / self._pps()

    def _row_y(self, row: int) -> int:
        return _HEADER_H + row * (_ROW_H + _ROW_GAP)

    # ---- paint ----

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        w, h = self.width(), self.height()

        painter.fillRect(0, 0, w, h, QColor(30, 30, 30))

        # Row backgrounds + labels
        label_font = QFont("Monospace", 8)
        painter.setFont(label_font)
        for row, label in enumerate(self._ROW_LABELS):
            ry = self._row_y(row)
            bg = QColor(42, 42, 42) if row % 2 == 0 else QColor(36, 36, 36)
            painter.fillRect(0, ry, w, _ROW_H, bg)
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(2, ry, _LABEL_W - 2, _ROW_H,
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, label)
            # separator between cam1 and cam2 groups
            if row == 2:
                painter.setPen(QPen(QColor(70, 70, 70), 1))
                sep_y = ry + _ROW_H + _ROW_GAP // 2
                painter.drawLine(0, sep_y, w, sep_y)

        # Session spans — cam1 on rows 0-2, cam2 on rows 3-5
        for cam_sessions, base_row, cam_offset in (
            (self._cam1_sessions, 0, self._offsets[0]),
            (self._cam2_sessions, 3, self._offsets[1]),
        ):
            for session in cam_sessions:
                enter_t = (session.times.enter_time.t + cam_offset) if session.times.enter_time else None
                exit_t  = (session.times.exit_time.t  + cam_offset) if session.times.exit_time  else None
                if enter_t is None:
                    continue
                x1 = self._t_to_x(enter_t)
                x2 = self._t_to_x(exit_t) if exit_t else self._t_to_x(self._duration)
                color = QColor(_STATUS_COLORS.get(session.validation.status,
                                                  _STATUS_COLORS["incomplete"]))
                if session.session_id == self._selected_id:
                    color.setAlpha(200)
                for row in range(base_row, base_row + 3):
                    painter.fillRect(x1, self._row_y(row), max(1, x2 - x1), _ROW_H, color)
                if x2 - x1 > 20:
                    painter.setPen(QColor(255, 255, 255, 200))
                    painter.setFont(QFont("Monospace", 7))
                    painter.drawText(x1 + 2, self._row_y(base_row), x2 - x1 - 2, _ROW_H,
                                     Qt.AlignmentFlag.AlignVCenter, session.session_id)

        # Door events (rows 0 & 3) — apply per-camera offsets (wall-clock)
        for cam, base_row in enumerate([0, 3]):
            cam_offset = self._offsets[cam] if cam < len(self._offsets) else 0.0
            for ev in self._door_events[cam]:
                t = ev.timestamp + cam_offset
                x = self._t_to_x(t)
                if ev.state == "open":
                    color = QColor(255, 80, 80) 
                elif ev.state == "closed":
                    color = QColor(80, 120, 255)
                elif ev.state == "partial":
                    color = QColor(255, 160, 40)
                else:
                    color = QColor(250, 250, 250)
                ry = self._row_y(base_row)
                painter.fillRect(x - 1, ry, 3, _ROW_H, color)

        # Speech segments (rows 1 & 4) — apply per-camera offsets (wall-clock)
        for cam, base_row in enumerate([1, 4]):
            cam_offset = self._offsets[cam] if cam < len(self._offsets) else 0.0
            audio_offset = self._offsets[2 + cam] if len(self._offsets) > 2 else 0.0
            for seg in self._speech_segs[cam]:
                x1 = self._t_to_x(seg.start + cam_offset - audio_offset)
                x2 = self._t_to_x(seg.end + cam_offset - audio_offset)
                ry = self._row_y(base_row)
                painter.fillRect(x1, ry + 2, max(1, x2 - x1), _ROW_H - 4, QColor(60, 200, 80))

        # Face detections (rows 2 & 5) – merge into contiguous blocks (apply offsets)
        for cam, base_row in enumerate([2, 5]):
            cam_offset = self._offsets[cam] if cam < len(self._offsets) else 0.0
            block_start: Optional[float] = None
            prev_t: Optional[float] = None
            for det in self._face_dets[cam]:
                if det.detected:
                    if block_start is None:
                        block_start = det.timestamp + cam_offset
                    prev_t = det.timestamp + cam_offset
                else:
                    if block_start is not None and prev_t is not None:
                        ry = self._row_y(base_row)
                        painter.fillRect(self._t_to_x(block_start), ry + 2,
                                         max(1, self._t_to_x(prev_t) - self._t_to_x(block_start)),
                                         _ROW_H - 4, QColor(80, 160, 220))
                        block_start = None
            if block_start is not None and prev_t is not None:
                ry = self._row_y(base_row)
                painter.fillRect(self._t_to_x(block_start), ry + 2,
                                 max(1, self._t_to_x(prev_t) - self._t_to_x(block_start)),
                                 _ROW_H - 4, QColor(80, 160, 220))

        # Time ruler
        painter.fillRect(0, 0, w, _HEADER_H, QColor(22, 22, 22))
        painter.setPen(QColor(110, 110, 110))
        painter.setFont(QFont("Monospace", 7))
        step = self._nice_step()
        t = 0.0
        while t <= self._duration + step:
            x = self._t_to_x(t)
            if _LABEL_W <= x <= w:
                painter.drawLine(x, 0, x, _HEADER_H // 2)
                painter.drawText(x + 2, 0, 70, _HEADER_H,
                                 Qt.AlignmentFlag.AlignVCenter, fmt_time(t))
            t += step

        # Playhead
        px = self._t_to_x(self._playhead)
        painter.setPen(QPen(QColor(255, 50, 50), _PLAYHEAD_W))
        painter.drawLine(px, 0, px, h)
        painter.end()

    def _nice_step(self) -> float:
        pps = self._pps()
        for step in [0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600]:
            if pps * step >= 60:
                return float(step)
        return 600.0

    # ---- mouse ----

    def _session_at(self, x: float) -> Optional[Session]:
        t = self._x_to_t(x)
        for cam_sessions, cam_offset in (
            (self._cam1_sessions, self._offsets[0]),
            (self._cam2_sessions, self._offsets[1]),
        ):
            for s in cam_sessions:
                enter_t = s.times.enter_time.t if s.times.enter_time else None
                exit_t  = s.times.exit_time.t  if s.times.exit_time  else None
                if enter_t is None:
                    continue
                # compare timeline position against offset-shifted local times
                if (enter_t + cam_offset) <= t <= ((exit_t + cam_offset) if exit_t else self._duration):
                    return s
        return None

    def _offset_for_session(self, session: Session) -> float:
        """Return the cam offset for the given session object."""
        if session in self._cam1_sessions:
            return self._offsets[0]
        if session in self._cam2_sessions:
            return self._offsets[1]
        return 0.0

    def mousePressEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            if event.button() == Qt.MouseButton.LeftButton:
                # start panning (hold Shift + drag with left mouse)
                self._dragging = True
                self._panning = True
                self._pan_start_x = event.position().x()
                self._pan_start_h_scroll = self._h_scroll
                try:
                    self.setCursor(Qt.CursorShape.ClosedHandCursor)
                except Exception:
                    pass
        elif event.button() == Qt.MouseButton.LeftButton:
            t = max(0.0, min(self._x_to_t(event.position().x()), self._duration))
            self._dragging = True
            self._playhead = t
            self.update()
            self.playhead_moved.emit(t)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging and (event.modifiers() & Qt.KeyboardModifier.ShiftModifier) and (event.buttons() & Qt.MouseButton.LeftButton):
            # If panning was initiated (Shift+drag), adjust horizontal scroll
            if self._panning:
                try:
                    cur_x = event.position().x()
                    start_x = self._pan_start_x if self._pan_start_x is not None else cur_x
                    dx = cur_x - start_x
                    pps = max(1e-6, self._pps())
                    # dragging right (dx>0) should decrease h_scroll so content moves right
                    delta_seconds = -dx / pps
                    new_h = self._pan_start_h_scroll + delta_seconds
                    avail = max(1, self.width() - _LABEL_W)
                    visible_secs = avail / self._pps() if self._pps() > 0 else self._duration
                    max_h = max(0.0, self._duration - visible_secs)
                    self._h_scroll = max(0.0, min(max_h, new_h))
                    self.update()
                except Exception:
                    pass
        elif self._dragging and (event.buttons() & Qt.MouseButton.LeftButton):
            t = max(0.0, min(self._x_to_t(event.position().x()), self._duration))
            self._playhead = t
            self.update()
            self.playhead_moved.emit(t)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        # If panning was active, clear state and restore cursor
        if self._panning:
            self._panning = False
            self._pan_start_x = None
            self._pan_start_h_scroll = 0.0
            try:
                self.unsetCursor()
            except Exception:
                pass
        self._dragging = False
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self._zoom = max(0.5, min(50.0, self._zoom * factor))
            self.update()
        else:
            super().wheelEvent(event)

    def _on_context_menu(self, pos):
        session = self._session_at(pos.x())
        t = self._x_to_t(pos.x())
        menu = QMenu(self)
        if session:
            a_je = menu.addAction(f"Jump to enter_time  ({session.session_id})")
            a_jx = menu.addAction(f"Jump to exit_time   ({session.session_id})")
            menu.addSeparator()
            a_se = menu.addAction("Set enter_time here")
            a_sx = menu.addAction("Set exit_time here")
            menu.addSeparator()
            a_sel = menu.addAction(f"Select {session.session_id}")
            action = menu.exec(self.mapToGlobal(pos))
            cam_offset = self._offset_for_session(session)
            if action == a_je and session.times.enter_time:
                # emit wall-clock time so the playhead lands on the right frame
                self.playhead_moved.emit(session.times.enter_time.t + cam_offset)
            elif action == a_jx and session.times.exit_time:
                self.playhead_moved.emit(session.times.exit_time.t + cam_offset)
            elif action == a_se:
                # t is wall-clock; convert to local video time before storing
                local_t = t - cam_offset
                session.times.enter_time = AnnotationValue(t=local_t, source="manual",
                                                           confidence=1.0, method="gui")
                self.session_selected.emit(session.session_id)
                self.session_modified.emit(session.session_id, "enter_time", local_t)
                self.update()
            elif action == a_sx:
                local_t = t - cam_offset
                session.times.exit_time = AnnotationValue(t=local_t, source="manual",
                                                          confidence=1.0, method="gui")
                self.session_selected.emit(session.session_id)
                self.session_modified.emit(session.session_id, "exit_time", local_t)
                self.update()
            elif action == a_sel:
                self.session_selected.emit(session.session_id)
        else:
            a_move = menu.addAction(f"Seek to {fmt_time(t)}")
            if menu.exec(self.mapToGlobal(pos)) == a_move:
                self.playhead_moved.emit(t)


# ──────────────────────────────────────────────────────────────────────────────
# SessionEditorWidget
# ──────────────────────────────────────────────────────────────────────────────

class SessionEditorWidget(QWidget):
    """Form for viewing and editing a single Session's annotation fields."""

    session_saved      = Signal()
    annotation_changed = Signal(str, float)   # (field_name, new_time)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._session: Optional[Session] = None
        self._playhead: float = 0.0
        self._offset: float = 0.0  # wall-clock offset for active cam (subtract to get local time)
        self._undo_stack: List[Session] = []
        self._redo_stack: List[Session] = []
        self._building = False
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(2, 2, 2, 2)
        outer.setSpacing(4)

        self._group = QGroupBox("No session selected")
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # enter_time
        enter_row = QHBoxLayout()
        self._enter_edit = QTimeEdit()
        self._enter_edit.setDisplayFormat("HH:mm:ss.zzz")
        self._enter_edit.timeChanged.connect(self._on_enter_changed)
        self._btn_set_enter = QPushButton("Set ▸")
        self._btn_set_enter.setFixedWidth(52)
        self._btn_set_enter.clicked.connect(self._on_set_enter)
        enter_row.addWidget(self._enter_edit)
        enter_row.addWidget(self._btn_set_enter)
        form.addRow("enter_time:", enter_row)

        # exit_time
        exit_row = QHBoxLayout()
        self._exit_edit = QTimeEdit()
        self._exit_edit.setDisplayFormat("HH:mm:ss.zzz")
        self._exit_edit.timeChanged.connect(self._on_exit_changed)
        self._btn_set_exit = QPushButton("Set ▸")
        self._btn_set_exit.setFixedWidth(52)
        self._btn_set_exit.clicked.connect(self._on_set_exit)
        exit_row.addWidget(self._exit_edit)
        exit_row.addWidget(self._btn_set_exit)
        form.addRow("exit_time:", exit_row)

        # door events (read-only)
        self._lbl_door_closed = QLabel("—")
        self._lbl_door_opened = QLabel("—")
        form.addRow("door_closed:", self._lbl_door_closed)
        form.addRow("door_opened:", self._lbl_door_opened)

        # subject_id
        self._subject_edit = QLineEdit()
        self._subject_edit.setPlaceholderText("[MANUAL_REQUIRED]")
        self._subject_edit.textChanged.connect(self._on_subject_changed)
        form.addRow("subject_id:", self._subject_edit)

        # validation status
        self._lbl_validation = QLabel("—")
        form.addRow("validation:", self._lbl_validation)

        # notes
        self._notes_edit = QPlainTextEdit()
        self._notes_edit.setMaximumHeight(56)
        self._notes_edit.setPlaceholderText("Freeform notes…")
        self._notes_edit.textChanged.connect(self._on_notes_changed)
        form.addRow("notes:", self._notes_edit)

        self._group.setLayout(form)
        outer.addWidget(self._group)

        # Buttons
        btn_row = QHBoxLayout()
        self._btn_save    = QPushButton("Save")
        self._btn_revert  = QPushButton("Revert")
        self._btn_undo    = QPushButton("Undo")
        self._btn_redo    = QPushButton("Redo")
        for b in [self._btn_save, self._btn_revert, self._btn_undo, self._btn_redo]:
            b.setEnabled(False)
            btn_row.addWidget(b)
        self._btn_save.clicked.connect(self._on_save)
        self._btn_revert.clicked.connect(self._on_revert)
        self._btn_undo.clicked.connect(self._on_undo)
        self._btn_redo.clicked.connect(self._on_redo)
        outer.addLayout(btn_row)
        outer.addStretch()

    # ---- public ----

    def load_session(self, session: Optional[Session]):
        self._session = session
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._populate()

    def set_playhead(self, t: float):
        self._playhead = t

    def set_offset(self, offset: float):
        """Set the wall-clock offset for the active camera. Subtracted when writing local times."""
        self._offset = offset

    # ---- populate ----

    def _populate(self):
        s = self._session
        self._building = True
        try:
            if s is None:
                self._group.setTitle("No session selected")
                self._enter_edit.setTime(QTime(0, 0, 0, 0))
                self._exit_edit.setTime(QTime(0, 0, 0, 0))
                self._lbl_door_closed.setText("—")
                self._lbl_door_opened.setText("—")
                self._subject_edit.setText("")
                self._notes_edit.setPlainText("")
                self._lbl_validation.setText("—")
                for b in [self._btn_save, self._btn_revert]:
                    b.setEnabled(False)
            else:
                self._group.setTitle(f"Session: {s.session_id}")
                self._enter_edit.setTime(sec_to_qtime(s.times.enter_time.t) if s.times.enter_time else QTime(0, 0, 0, 0))
                self._exit_edit.setTime(sec_to_qtime(s.times.exit_time.t)   if s.times.exit_time  else QTime(0, 0, 0, 0))
                dc = s.events.door_closed
                do = s.events.door_opened
                self._lbl_door_closed.setText(f"{fmt_time(dc.t)} ({dc.source})" if dc else "—")
                self._lbl_door_opened.setText(f"{fmt_time(do.t)} ({do.source})" if do else "—")
                self._subject_edit.setText(s.subject.subject_id)
                self._notes_edit.setPlainText(s.freeform_notes)
                self._lbl_validation.setText(s.validation.status)
                self._btn_save.setEnabled(True)
                self._btn_revert.setEnabled(True)
        finally:
            self._building = False
        self._refresh_undo_redo()

    def _refresh_undo_redo(self):
        self._btn_undo.setEnabled(bool(self._undo_stack))
        self._btn_redo.setEnabled(bool(self._redo_stack))

    def _push_undo(self):
        if self._session:
            self._undo_stack.append(copy.deepcopy(self._session))
            self._redo_stack.clear()
            self._refresh_undo_redo()

    # ---- callbacks ----

    def _on_enter_changed(self, qt: QTime):
        if self._building or self._session is None:
            return
        t = qtime_to_sec(qt)
        self._push_undo()
        self._session.times.enter_time = AnnotationValue(t=t, source="manual", confidence=1.0, method="gui")
        self.annotation_changed.emit("enter_time", t)

    def _on_exit_changed(self, qt: QTime):
        if self._building or self._session is None:
            return
        t = qtime_to_sec(qt)
        self._push_undo()
        self._session.times.exit_time = AnnotationValue(t=t, source="manual", confidence=1.0, method="gui")
        self.annotation_changed.emit("exit_time", t)

    def _on_set_enter(self):
        if self._session is None:
            return
        self._push_undo()
        local_t = self._playhead - self._offset
        self._session.times.enter_time = AnnotationValue(t=local_t, source="manual", confidence=1.0, method="gui")
        self._building = True
        self._enter_edit.setTime(sec_to_qtime(local_t))
        self._building = False
        self.annotation_changed.emit("enter_time", local_t)

    def _on_set_exit(self):
        if self._session is None:
            return
        self._push_undo()
        local_t = self._playhead - self._offset
        self._session.times.exit_time = AnnotationValue(t=local_t, source="manual", confidence=1.0, method="gui")
        self._building = True
        self._exit_edit.setTime(sec_to_qtime(local_t))
        self._building = False
        self.annotation_changed.emit("exit_time", local_t)

    def _on_subject_changed(self, text: str):
        if self._building or self._session is None:
            return
        self._push_undo()
        self._session.subject.subject_id = text
        self.annotation_changed.emit("subject_id", 0.0)

    def _on_notes_changed(self):
        if self._building or self._session is None:
            return
        self._push_undo()
        self._session.freeform_notes = self._notes_edit.toPlainText()
        self.annotation_changed.emit("notes", 0.0)

    def _on_save(self):
        self.session_saved.emit()

    def _on_revert(self):
        if self._session and self._undo_stack:
            rev = self._undo_stack[0]
            self._undo_stack.clear()
            self._redo_stack.clear()
            self._session.times = copy.deepcopy(rev.times)
            self._session.subject = copy.deepcopy(rev.subject)
            self._session.freeform_notes = rev.freeform_notes
            self._populate()

    def _on_undo(self):
        if self._session and self._undo_stack:
            self._redo_stack.append(copy.deepcopy(self._session))
            prev = self._undo_stack.pop()
            self._session.times = copy.deepcopy(prev.times)
            self._session.subject = copy.deepcopy(prev.subject)
            self._session.freeform_notes = prev.freeform_notes
            self._populate()
            self.annotation_changed.emit("undo", 0.0)

    def _on_redo(self):
        if self._session and self._redo_stack:
            self._undo_stack.append(copy.deepcopy(self._session))
            nxt = self._redo_stack.pop()
            self._session.times = copy.deepcopy(nxt.times)
            self._session.subject = copy.deepcopy(nxt.subject)
            self._session.freeform_notes = nxt.freeform_notes
            self._populate()
            self.annotation_changed.emit("redo", 0.0)

# --------
# AudioThread
# ---------
class AudioThread(QThread):
    """Decodes audio samples in a background thread, emitting numpy arrays."""

    samples_ready = Signal(np.ndarray, float)   # (samples, timestamp_seconds)

    def __init__(self, video_path: str, parent=None):
        super().__init__(parent)
        self._video_path = video_path
        self._running: bool = True
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 30.0
        self._total_frames: int = 0

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self._video_path)
        if not self._cap.isOpened():
            return False
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return True

    def stop(self):
        self._running = False
        self.wait(3000)

    def run(self):
        cap = self._cap
        if cap is None or not cap.isOpened():
            return

        # Note: OpenCV doesn't provide direct access to raw audio samples.
        # In a real implementation, you'd likely use a library like PyAV or ffmpeg-python
        # to decode audio frames. Here we'll just emit silence for demonstration.

        frame_interval = 1.0 / self._fps
        t = 0.0

        while self._running and t < (self._total_frames / self._fps):
            # Emit silence (or decoded audio samples in a real implementation)
            samples = np.zeros(44100, dtype=np.float32)  # 1 second of silence at 44.1 kHz
            self.samples_ready.emit(samples, t)

            time.sleep(frame_interval)
            t += frame_interval

# ---------- Audio Manager (QtMultimedia) ----------


class AudioManager:
    """Audio manager implemented using PySide6.QtMultimedia (QMediaPlayer + QAudioOutput).
    Designed to be created from the Qt GUI thread. Public API matches prior
    implementations: play(start_time), pause(), resume(), stop(), seek(pos),
    get_position(), set_volume(), cleanup().
    """

    def __init__(self, audio_path: Optional[str] = None, parent=None):
        self.audio_path = audio_path
        self._available = False
        self._player = None
        self._audio_output = None
        self._duration = 0.0
        self._position = 0.0
        self._volume = 1.0

        if not audio_path or not os.path.exists(audio_path):
            return

        try:
            from PySide6.QtMultimedia import (
                QAudioBufferOutput,
                QAudioDevice,
                QAudioOutput,
                QMediaDevices,
                QMediaFormat,
                QMediaMetaData,
                QMediaPlayer,
            )
            from PySide6.QtCore import QUrl
        except Exception as e:
            print(f"[audio] QtMultimedia not available: {e}")
            return

        try:
            # Create player and audio output, attach to default audio device
            self._player = QMediaPlayer(parent)
            self._audio_output = QAudioOutput()

            # If available, prefer setting explicit output device
            try:
                dev = QMediaDevices.defaultAudioOutput()
                if dev is not None:
                    self._audio_output.setDevice(dev)
            except Exception:
                pass

            self._player.setAudioOutput(self._audio_output)

            # Load source
            try:
                self._player.setSource(QUrl.fromLocalFile(str(audio_path)))
            except Exception:
                # Older Qt may use setSource differently; try setSource with QUrl
                try:
                    self._player.setSource(QUrl.fromLocalFile(str(audio_path)))
                except Exception:
                    print(f"[audio] failed to set source for QMediaPlayer: {audio_path}")
                    pass

            # Connect signals to track position/duration
            try:
                self._player.durationChanged.connect(self._on_duration_changed)
                self._player.positionChanged.connect(self._on_position_changed)
            except Exception:
                pass

            # default volume
            try:
                self._audio_output.setVolume(self._volume)
            except Exception:
                pass

            self._available = True
        except Exception as e:
            print(f"[audio] failed to initialize QMediaPlayer: {e}")
            self._player = None
            self._audio_output = None
            self._available = False

    def _on_duration_changed(self, d: int):
        try:
            self._duration = float(d) / 1000.0
        except Exception:
            self._duration = 0.0

    def _on_position_changed(self, p: int):
        try:
            self._position = float(p) / 1000.0
        except Exception:
            self._position = 0.0

    def play(self, start_time: float = 0.0):
        if not self._available:
            return
        try:
            ms = int(max(0.0, start_time) * 1000)
            # set position then play
            try:
                self._player.setPosition(ms)
            except Exception:
                # some Qt versions may require setPosition on sourceChanged; ignore
                pass
            self._player.play()
        except Exception:
            pass

    def pause(self):
        if not self._available:
            return
        try:
            self._player.pause()
        except Exception:
            pass

    def resume(self):
        if not self._available:
            return
        try:
            self._player.play()
        except Exception:
            pass

    def stop(self):
        if not self._available:
            return
        try:
            self._player.stop()
        except Exception:
            pass

    def seek(self, position: float):
        if not self._available:
            return
        try:
            ms = int(max(0.0, position) * 1000)
            self._player.setPosition(ms)
        except Exception:
            pass

    def get_position(self) -> float:
        if not self._available:
            return 0.0
        try:
            return float(self._position)
        except Exception:
            try:
                return float(self._player.position()) / 1000.0
            except Exception:
                return 0.0

    def set_volume(self, vol: float):
        """Set volume (0-100 or 0.0-1.0)."""
        if not self._available or self._audio_output is None:
            return
        try:
            if vol > 1:
                v = max(0.0, min(1.0, float(vol) / 100.0))
            else:
                v = max(0.0, min(1.0, float(vol)))
            self._volume = v
            self._audio_output.setVolume(v)
        except Exception:
            pass

    def cleanup(self):
        if not self._available:
            return
        try:
            self._player.stop()
            # best-effort cleanup
            try:
                self._player.setSource(None)
            except Exception:
                pass
            try:
                self._player.deleteLater()
            except Exception:
                pass
            try:
                self._audio_output.deleteLater()
            except Exception:
                pass
        except Exception:
            pass
        finally:
            self._player = None
            self._audio_output = None
            self._available = False



# ──────────────────────────────────────────────────────────────────────────────
# MainWindow
# ──────────────────────────────────────────────────────────────────────────────

STATUS_ICONS = {"complete": "✓", "needs_review": "⚠", "incomplete": "✗"}
SPEED_VALUES = [0.25, 0.5, 1.0, 2.0, 4.0]
