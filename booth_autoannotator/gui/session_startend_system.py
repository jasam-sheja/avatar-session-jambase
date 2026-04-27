#!/usr/bin/env python3
"""System GUI for visualizing and correcting session start/end times.

This window operates on recording *roots* (date -> recording folders), scans all
recordings at startup, and loads heavy data only for the selected recording.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..door_angle import DoorState
from ..face_detect import FaceDetection
from ..models import AnnotationDocument, AnnotationValue, Session
from ..speech_vad import SpeechSegment
from .session_startend_components import (
    SPEED_VALUES,
    STATUS_ICONS,
    SessionEditorWidget,
    TimelineWidget,
    VideoLabel,
    VideoThread,
    fmt_time,
    AudioManager,
)

logger = logging.getLogger(__name__)

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_REC_RE = re.compile(r"^recording-\d+$")


@dataclass(frozen=True)
class RecordingKey:
    date: str
    recording_id: str

    @property
    def label(self) -> str:
        return f"{self.date}/{self.recording_id}"


@dataclass(frozen=True)
class RecordingPaths:
    key: RecordingKey
    recording_dir: Path
    output_dir: Path
    cache_dir: Path
    video_paths: Tuple[Path, Path]
    audio_paths: Tuple[Path, Path]
    output_json_paths: Tuple[Path, Path]
    cache_door_paths: Tuple[Path, Path]
    cache_face_paths: Tuple[Path, Path]
    cache_speech_paths: Tuple[Path, Path]
    media_complete: bool


@dataclass
class RecordingLoadResult:
    key: RecordingKey
    paths: RecordingPaths
    doc1: AnnotationDocument
    doc2: AnnotationDocument
    door_events: List[List[DoorState]]
    speech_segs: List[List[SpeechSegment]]
    face_dets: List[List[FaceDetection]]
    door_states: List[List[DoorState]]


def _build_paths(
    recording_root: Path,
    output_root: Path,
    cache_root: Path,
    key: RecordingKey,
) -> RecordingPaths:
    recording_dir = recording_root / key.date / key.recording_id
    output_dir = output_root / key.date / key.recording_id
    cache_dir = cache_root / key.date / key.recording_id

    video_paths = (
        recording_dir / "stream-001.mp4",
        recording_dir / "stream-002.mp4",
    )
    audio_paths = (
        recording_dir / "audio-001.wav",
        recording_dir / "audio-002.wav",
    )
    output_json_paths = (
        output_dir / "001.json",
        output_dir / "002.json",
    )
    cache_door_paths = (
        cache_dir / "stream-001_door_states.pkl",
        cache_dir / "stream-002_door_states.pkl",
    )
    cache_face_paths = (
        cache_dir / "stream-001_face_detections.pkl",
        cache_dir / "stream-002_face_detections.pkl",
    )
    cache_speech_paths = (
        cache_dir / "audio-001_speech_segments.pkl",
        cache_dir / "audio-002_speech_segments.pkl",
    )

    media_complete = all(p.exists() for p in video_paths + audio_paths)

    return RecordingPaths(
        key=key,
        recording_dir=recording_dir,
        output_dir=output_dir,
        cache_dir=cache_dir,
        video_paths=video_paths,
        audio_paths=audio_paths,
        output_json_paths=output_json_paths,
        cache_door_paths=cache_door_paths,
        cache_face_paths=cache_face_paths,
        cache_speech_paths=cache_speech_paths,
        media_complete=media_complete,
    )


def scan_recordings(
    recording_root: Path,
    output_root: Path,
    cache_root: Path,
) -> Tuple[Dict[str, List[RecordingKey]], Dict[RecordingKey, RecordingPaths], Dict[str, int]]:
    index: Dict[str, List[RecordingKey]] = {}
    by_key: Dict[RecordingKey, RecordingPaths] = {}
    invalid_media = 0

    if not recording_root.exists():
        return index, by_key, {"dates": 0, "recordings": 0, "invalid_media": 0}

    for date_dir in sorted(recording_root.iterdir(), key=lambda p: p.name):
        if not date_dir.is_dir() or not _DATE_RE.match(date_dir.name):
            continue

        keys: List[RecordingKey] = []
        for rec_dir in sorted(date_dir.iterdir(), key=lambda p: p.name):
            if not rec_dir.is_dir() or not _REC_RE.match(rec_dir.name):
                continue

            key = RecordingKey(date=date_dir.name, recording_id=rec_dir.name)
            paths = _build_paths(recording_root, output_root, cache_root, key)
            if not paths.media_complete:
                invalid_media += 1
            keys.append(key)
            by_key[key] = paths

        if keys:
            index[date_dir.name] = keys

    stats = {
        "dates": len(index),
        "recordings": sum(len(v) for v in index.values()),
        "invalid_media": invalid_media,
    }
    return index, by_key, stats


class RecordingLoadThread(QThread):
    loaded = Signal(object)
    failed = Signal(str)
    progress = Signal(int, str)

    def __init__(self, paths: RecordingPaths, parent=None):
        super().__init__(parent)
        self._paths = paths

    @staticmethod
    def _try_load_cache(path: Path):
        try:
            if path.exists():
                with open(path, "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning("Cache load failed for %s: %s", path, e)
        return None

    @staticmethod
    def _load_doc(path: Path) -> AnnotationDocument:
        if path.exists():
            return AnnotationDocument.load(str(path))
        return AnnotationDocument()

    def run(self):
        try:
            paths = self._paths
            # emit starting progress
            try:
                self.progress.emit(0, "Starting…")
            except Exception:
                pass
            if self.isInterruptionRequested():
                try:
                    self.failed.emit("cancelled")
                except Exception:
                    pass
                return
            doc1 = self._load_doc(paths.output_json_paths[0])
            try:
                self.progress.emit(10, "Loaded metadata (cam1)")
            except Exception:
                pass
            if self.isInterruptionRequested():
                try:
                    self.failed.emit("cancelled")
                except Exception:
                    pass
                return
            doc2 = self._load_doc(paths.output_json_paths[1])
            try:
                self.progress.emit(20, "Loaded metadata (cam2)")
            except Exception:
                pass
            if self.isInterruptionRequested():
                try:
                    self.failed.emit("cancelled")
                except Exception:
                    pass
                return

            door_events: List[List[DoorState]] = [[], []]
            speech_segs: List[List[SpeechSegment]] = [[], []]
            face_dets: List[List[FaceDetection]] = [[], []]
            door_states: List[List[DoorState]] = [[], []]

            for cam in (0, 1):
                door = self._try_load_cache(paths.cache_door_paths[cam])
                face = self._try_load_cache(paths.cache_face_paths[cam])
                if isinstance(door, list):
                    door_events[cam] = door
                    door_states[cam] = door
                if isinstance(face, list):
                    face_dets[cam] = face
                try:
                    pct = 30 + cam * 10
                    self.progress.emit(pct, f"Loaded door/face cache (cam{cam+1})")
                except Exception:
                    pass
                if self.isInterruptionRequested():
                    try:
                        self.failed.emit("cancelled")
                    except Exception:
                        pass
                    return

            for cam in (0, 1):
                speech = self._try_load_cache(paths.cache_speech_paths[cam])
                if isinstance(speech, tuple) and len(speech) >= 1:
                    segments = speech[0]
                    if isinstance(segments, list):
                        speech_segs[cam] = segments
                elif isinstance(speech, list):
                    speech_segs[cam] = speech
                try:
                    pct = 60 + cam * 20
                    self.progress.emit(pct, f"Loaded speech cache (cam{cam+1})")
                except Exception:
                    pass
                if self.isInterruptionRequested():
                    try:
                        self.failed.emit("cancelled")
                    except Exception:
                        pass
                    return

            try:
                self.progress.emit(95, "Finalizing…")
            except Exception:
                pass
            if self.isInterruptionRequested():
                try:
                    self.failed.emit("cancelled")
                except Exception:
                    pass
                return

            result = RecordingLoadResult(
                key=paths.key,
                paths=paths,
                doc1=doc1,
                doc2=doc2,
                door_events=door_events,
                speech_segs=speech_segs,
                face_dets=face_dets,
                door_states=door_states,
            )
            try:
                self.progress.emit(100, "Done")
            except Exception:
                pass
            self.loaded.emit(result)
        except Exception as e:
            logger.error("Failed to load recording context: %s", e, exc_info=True)
            self.failed.emit(str(e))


class MainWindow(QMainWindow):
    """Main application window for multi-recording annotation."""

    def __init__(
        self,
        recording_dir: Path,
        output_dir: Path,
        cache_dir: Path,
        auto_save: bool = False,
    ):
        super().__init__()
        self._recording_root = recording_dir
        self._output_root = output_dir
        self._cache_root = cache_dir
        self._auto_save = auto_save

        self._index: Dict[str, List[RecordingKey]] = {}
        self._paths_by_key: Dict[RecordingKey, RecordingPaths] = {}
        self._tree_items: Dict[RecordingKey, QTreeWidgetItem] = {}

        self._current_key: Optional[RecordingKey] = None
        self._current_paths: Optional[RecordingPaths] = None

        self._playhead: float = 0.0
        self._playing: bool = False
        self._speed: float = 1.0
        self._wall_clock_last: float = 0.0

        self._doc1: Optional[AnnotationDocument] = None
        self._doc2: Optional[AnnotationDocument] = None

        self._door_events: List[List[DoorState]] = [[], []]
        self._speech_segs: List[List[SpeechSegment]] = [[], []]
        self._face_dets: List[List[FaceDetection]] = [[], []]
        self._door_states: List[List[DoorState]] = [[], []]

        self._threads: List[Optional[VideoThread]] = [None, None]
        # Support two audio managers (one per camera)
        self._audios: List[Optional[AudioManager]] = [None, None]
        self._loader: Optional[RecordingLoadThread] = None
        self._loading: bool = False
        self._suppress_tree_signal: bool = False
        self._suppress_list_signal: bool = False

        self._active_cam: int = 0
        self._cam_offsets: List[float] = [0.0, 0.0]
        self._audio_offsets: List[float] = [0.0, 0.0]
        self._selected_idx: List[int] = [-1, -1]
        self._changed_session_ids: List[Set[str]] = [set(), set()]
        self._deleted_session_ids: List[Set[str]] = [set(), set()]
        self._dirty: bool = False

        self.setWindowTitle("Booth Session Annotator (System)")
        self._build_ui()
        self._reload_recording_index(select_first=True)

    # ─── UI construction ──────────────────────────────────────────────────

    def _build_ui(self):
        self.setMinimumSize(1400, 780)

        mb = self.menuBar()
        file_menu = mb.addMenu("&File")
        act_save = file_menu.addAction("&Save current recording")
        act_save.setShortcut("S")
        act_save.triggered.connect(self._save_all)

        act_reload = file_menu.addAction("&Reload recording index")
        act_reload.setShortcut("Ctrl+R")
        act_reload.triggered.connect(lambda: self._reload_recording_index(select_first=False))

        file_menu.addSeparator()
        file_menu.addAction("&Quit").triggered.connect(self.close)

        edit_menu = mb.addMenu("&Edit")
        act_undo = edit_menu.addAction("&Undo")
        act_undo.setShortcut("Ctrl+Z")
        act_undo.triggered.connect(lambda: self._session_editor._on_undo())
        act_redo = edit_menu.addAction("&Redo")
        act_redo.setShortcut("Ctrl+Y")
        act_redo.triggered.connect(lambda: self._session_editor._on_redo())

        nav_menu = mb.addMenu("&Navigate")
        act_prev_rec = nav_menu.addAction("Previous recording")
        act_prev_rec.setShortcut("Ctrl+PgUp")
        act_prev_rec.triggered.connect(self._prev_recording)
        act_next_rec = nav_menu.addAction("Next recording")
        act_next_rec.setShortcut("Ctrl+PgDown")
        act_next_rec.triggered.connect(self._next_recording)

        central = QWidget()
        self.setCentralWidget(central)
        root_lay = QVBoxLayout(central)
        root_lay.setContentsMargins(4, 4, 4, 4)
        root_lay.setSpacing(4)

        main_split = QSplitter(Qt.Orientation.Horizontal)

        # ── Left: recording tree ───────────────────────────────────────────
        left = QWidget()
        left.setMinimumWidth(250)
        left.setMaximumWidth(360)
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(2, 2, 2, 2)
        left_lay.setSpacing(4)

        hdr = QHBoxLayout()
        hdr.addWidget(QLabel("Recordings"))
        hdr.addStretch()
        self._btn_reload = QPushButton("Reload")
        self._btn_reload.setFixedWidth(70)
        self._btn_reload.clicked.connect(lambda: self._reload_recording_index(select_first=False))
        hdr.addWidget(self._btn_reload)
        left_lay.addLayout(hdr)

        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setAlternatingRowColors(True)
        self._tree.currentItemChanged.connect(self._on_tree_item_changed)
        left_lay.addWidget(self._tree)

        self._tree_stats = QLabel("0 dates / 0 recordings")
        self._tree_stats.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        left_lay.addWidget(self._tree_stats)

        main_split.addWidget(left)

        # ── Right: annotation editor area ──────────────────────────────────
        right_split = QSplitter(Qt.Orientation.Horizontal)

        # media column (videos + timeline + transport)
        media = QWidget()
        media_lay = QVBoxLayout(media)
        media_lay.setContentsMargins(0, 0, 0, 0)
        media_lay.setSpacing(4)

        vid_row = QHBoxLayout()
        self._video_label1 = VideoLabel("Cam 1")
        self._video_label2 = VideoLabel("Cam 2")
        vid_row.addWidget(self._video_label1)
        vid_row.addWidget(self._video_label2)
        media_lay.addLayout(vid_row, stretch=3)

        self._timeline = TimelineWidget()
        tl_scroll = QScrollArea()
        tl_scroll.setWidgetResizable(True)
        tl_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        tl_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        tl_scroll.setWidget(self._timeline)
        tl_scroll.setFixedHeight(self._timeline._total_h() + 22)
        media_lay.addWidget(tl_scroll)

        transport = QWidget()
        tl = QHBoxLayout(transport)
        tl.setContentsMargins(4, 2, 4, 2)
        tl.setSpacing(4)

        def tb(label: str, w: int = 36) -> QPushButton:
            b = QPushButton(label)
            b.setFixedWidth(w)
            return b

        self._btn_to_start = tb("|◄")
        self._btn_to_start.setToolTip("Jump to session start")
        self._btn_step_back = tb("◄◄")
        self._btn_step_back.setToolTip("Step back 5 seconds")
        self._btn_playpause = tb("▶", 44)
        self._btn_playpause.setToolTip("Play/Pause (Space)")
        self._btn_step_fwd = tb("▶▶")
        self._btn_step_fwd.setToolTip("Step forward 5 seconds")
        self._btn_to_end = tb("▶|")
        self._btn_to_end.setToolTip("Jump to session end")
        self._btn_prev_sess = tb("◀ Sess", 64)
        self._btn_next_sess = tb("Sess ▶", 64)

        self._speed_combo = QComboBox()
        for lbl in ["0.25×", "0.5×", "1×", "2×", "4×"]:
            self._speed_combo.addItem(lbl)
        self._speed_combo.setCurrentIndex(2)
        self._speed_combo.currentIndexChanged.connect(self._on_speed_changed)

        self._volume_slider = QSlider(Qt.Orientation.Horizontal)
        self._volume_slider.setRange(0, 100)
        self._volume_slider.setValue(80)
        self._volume_slider.setFixedWidth(80)
        self._volume_slider.valueChanged.connect(self._on_volume_changed)

        self._time_label = QLabel("00:00.000 / 00:00.000")
        self._time_label.setFont(QFont("Monospace", 10))

        for w in [
            self._btn_to_start,
            self._btn_step_back,
            self._btn_playpause,
            self._btn_step_fwd,
            self._btn_to_end,
        ]:
            tl.addWidget(w)
        tl.addSpacing(6)
        tl.addWidget(self._btn_prev_sess)
        tl.addWidget(self._btn_next_sess)
        tl.addSpacing(6)
        tl.addWidget(QLabel("Speed:"))
        tl.addWidget(self._speed_combo)
        tl.addSpacing(6)
        tl.addWidget(QLabel("🔊"))
        tl.addWidget(self._volume_slider)
        tl.addSpacing(6)
        tl.addWidget(self._time_label)
        tl.addStretch()
        media_lay.addWidget(transport)

        right_split.addWidget(media)

        # session column
        sess_col = QWidget()
        sess_col.setMinimumWidth(260)
        sess_col.setMaximumWidth(360)
        sess_lay = QVBoxLayout(sess_col)
        sess_lay.setContentsMargins(4, 4, 4, 4)
        sess_lay.setSpacing(4)

        # ── Cam 1 Sessions ──
        cam1_hdr = QHBoxLayout()
        self._lbl_cam1_sessions = QLabel("Cam 1 Sessions")
        bold_font = QFont()
        bold_font.setBold(True)
        self._lbl_cam1_sessions.setFont(bold_font)  # cam1 active by default
        cam1_hdr.addWidget(self._lbl_cam1_sessions)
        cam1_hdr.addStretch()
        self._btn_add_sess_cam1 = QPushButton("+ Add")
        self._btn_add_sess_cam1.setFixedWidth(52)
        self._btn_add_sess_cam1.setToolTip("Add new Cam 1 session at playhead (A)")
        self._btn_del_sess_cam1 = QPushButton("− Del")
        self._btn_del_sess_cam1.setFixedWidth(52)
        self._btn_del_sess_cam1.setToolTip("Remove selected Cam 1 session (Delete)")
        cam1_hdr.addWidget(self._btn_add_sess_cam1)
        cam1_hdr.addWidget(self._btn_del_sess_cam1)
        sess_lay.addLayout(cam1_hdr)

        self._session_list_cam1 = QListWidget()
        self._session_list_cam1.setAlternatingRowColors(True)
        sess_lay.addWidget(self._session_list_cam1, stretch=1)

        # ── Cam 2 Sessions ──
        cam2_hdr = QHBoxLayout()
        self._lbl_cam2_sessions = QLabel("Cam 2 Sessions")
        cam2_hdr.addWidget(self._lbl_cam2_sessions)
        cam2_hdr.addStretch()
        self._btn_add_sess_cam2 = QPushButton("+ Add")
        self._btn_add_sess_cam2.setFixedWidth(52)
        self._btn_add_sess_cam2.setToolTip("Add new Cam 2 session at playhead")
        self._btn_del_sess_cam2 = QPushButton("− Del")
        self._btn_del_sess_cam2.setFixedWidth(52)
        self._btn_del_sess_cam2.setToolTip("Remove selected Cam 2 session")
        cam2_hdr.addWidget(self._btn_add_sess_cam2)
        cam2_hdr.addWidget(self._btn_del_sess_cam2)
        sess_lay.addLayout(cam2_hdr)

        self._session_list_cam2 = QListWidget()
        self._session_list_cam2.setAlternatingRowColors(True)
        sess_lay.addWidget(self._session_list_cam2, stretch=1)

        self._session_editor = SessionEditorWidget()
        sess_lay.addWidget(self._session_editor)
        right_split.addWidget(sess_col)

        right_split.setStretchFactor(0, 3)
        right_split.setStretchFactor(1, 1)
        main_split.addWidget(right_split)

        main_split.setStretchFactor(0, 1)
        main_split.setStretchFactor(1, 4)
        root_lay.addWidget(main_split)

        self.statusBar().showMessage("Ready")

        self._ui_timer = QTimer(self)
        self._ui_timer.setInterval(33)
        self._ui_timer.timeout.connect(self._on_ui_tick)
        self._ui_timer.start()

        # signals
        self._btn_to_start.clicked.connect(self._jump_to_session_start)
        self._btn_step_back.clicked.connect(lambda: self._step_frames(-1))
        self._btn_playpause.clicked.connect(self._toggle_play)
        self._btn_step_fwd.clicked.connect(lambda: self._step_frames(1))
        self._btn_to_end.clicked.connect(self._jump_to_session_end)
        self._btn_prev_sess.clicked.connect(self._prev_session)
        self._btn_next_sess.clicked.connect(self._next_session)

        self._timeline.playhead_moved.connect(self._on_timeline_seek)
        self._timeline.session_selected.connect(self._on_session_selected_by_id)
        self._timeline.session_modified.connect(self._on_timeline_session_modified)

        self._btn_add_sess_cam1.clicked.connect(lambda: self._do_add_session(0))
        self._btn_del_sess_cam1.clicked.connect(lambda: self._do_remove_session(0))
        self._btn_add_sess_cam2.clicked.connect(lambda: self._do_add_session(1))
        self._btn_del_sess_cam2.clicked.connect(lambda: self._do_remove_session(1))
        self._session_list_cam1.currentRowChanged.connect(
            lambda row: self._on_list_row_changed(0, row)
        )
        self._session_list_cam1.itemDoubleClicked.connect(self._on_list_double_click)
        self._session_list_cam2.currentRowChanged.connect(
            lambda row: self._on_list_row_changed(1, row)
        )
        self._session_list_cam2.itemDoubleClicked.connect(self._on_list_double_click)

        self._session_editor.session_saved.connect(self._save_all)
        self._session_editor.annotation_changed.connect(self._on_annotation_changed)

    # ─── Recording index / tree ───────────────────────────────────────────

    def _reload_recording_index(self, select_first: bool):
        prev_key = self._current_key

        self._index, self._paths_by_key, stats = scan_recordings(
            self._recording_root,
            self._output_root,
            self._cache_root,
        )

        logger.info(
            "Recording scan: %d dates, %d recordings, %d with missing media",
            stats["dates"],
            stats["recordings"],
            stats["invalid_media"],
        )
        self._tree_stats.setText(
            f"{stats['dates']} dates / {stats['recordings']} recordings"
            f" ({stats['invalid_media']} missing media)"
        )
        self._populate_tree()

        if not self._paths_by_key:
            self._set_loading_state(False)
            self.statusBar().showMessage("No recordings found")
            self._current_key = None
            self._current_paths = None
            self._unload_runtime(reset_docs=True)
            self._update_title()
            return

        if prev_key and prev_key in self._paths_by_key:
            self._select_tree_key(prev_key, silent=True)
        elif select_first:
            first_key = self._first_selectable_key()
            if first_key:
                self._select_tree_key(first_key, silent=False)

    def _populate_tree(self):
        self._suppress_tree_signal = True
        self._tree.clear()
        self._tree_items.clear()

        for date in sorted(self._index.keys()):
            parent = QTreeWidgetItem([date])
            parent.setFlags(Qt.ItemFlag.ItemIsEnabled)

            for key in sorted(self._index[date], key=lambda k: k.recording_id):
                child = QTreeWidgetItem([key.recording_id])
                child.setData(0, Qt.ItemDataRole.UserRole, key)
                self._tree_items[key] = child

                paths = self._paths_by_key[key]
                if not paths.media_complete:
                    child.setText(0, f"{key.recording_id}  (missing media)")
                    child.setDisabled(True)

                parent.addChild(child)

            self._tree.addTopLevelItem(parent)
            parent.setExpanded(True)

        self._suppress_tree_signal = False

    def _select_tree_key(self, key: Optional[RecordingKey], silent: bool):
        if key is None:
            return
        item = self._tree_items.get(key)
        if item is None:
            return

        if silent:
            self._suppress_tree_signal = True
        self._tree.setCurrentItem(item)
        if silent:
            self._suppress_tree_signal = False

    def _first_selectable_key(self) -> Optional[RecordingKey]:
        for date in sorted(self._index.keys()):
            for key in sorted(self._index[date], key=lambda k: k.recording_id):
                paths = self._paths_by_key.get(key)
                if paths and paths.media_complete:
                    return key
        return None

    def _ordered_keys(self) -> List[RecordingKey]:
        keys: List[RecordingKey] = []
        for date in sorted(self._index.keys()):
            keys.extend(sorted(self._index[date], key=lambda k: k.recording_id))
        return [k for k in keys if self._paths_by_key.get(k) and self._paths_by_key[k].media_complete]

    def _on_tree_item_changed(self, item: Optional[QTreeWidgetItem], _prev: Optional[QTreeWidgetItem]):
        if self._suppress_tree_signal or item is None or self._loading:
            return

        key = item.data(0, Qt.ItemDataRole.UserRole)
        if not isinstance(key, RecordingKey):
            return

        if key == self._current_key:
            return

        if not self._maybe_allow_recording_switch():
            self._select_tree_key(self._current_key, silent=True)
            return

        self._load_recording_async(key)

    def _maybe_allow_recording_switch(self) -> bool:
        if not self._dirty:
            return True

        action = self._prompt_unsaved_changes(
            "You have unsaved changes. Save before switching recordings?"
        )
        if action == "cancel":
            return False
        if action == "save":
            return self._save_all()
        return True

    def _prompt_unsaved_changes(self, text: str) -> str:
        box = QMessageBox(self)
        box.setWindowTitle("Unsaved changes")
        box.setText(text)
        save_btn = box.addButton("Save", QMessageBox.ButtonRole.AcceptRole)
        discard_btn = box.addButton("Discard", QMessageBox.ButtonRole.DestructiveRole)
        cancel_btn = box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(save_btn)
        box.exec()

        clicked = box.clickedButton()
        if clicked == save_btn:
            return "save"
        if clicked == discard_btn:
            return "discard"
        if clicked == cancel_btn:
            return "cancel"
        return "cancel"

    def _prev_recording(self):
        self._select_recording_delta(-1)

    def _next_recording(self):
        self._select_recording_delta(1)

    def _select_recording_delta(self, delta: int):
        keys = self._ordered_keys()
        if not keys:
            return

        if self._current_key in keys:
            idx = keys.index(self._current_key)
        else:
            idx = 0
        nidx = max(0, min(len(keys) - 1, idx + delta))
        self._select_tree_key(keys[nidx], silent=False)

    # ─── Recording load / unload ──────────────────────────────────────────

    def _set_loading_state(self, loading: bool):
        self._loading = loading
        self._tree.setEnabled(not loading)
        self._btn_reload.setEnabled(not loading)

    def _load_recording_async(self, key: RecordingKey):
        paths = self._paths_by_key.get(key)
        if not paths:
            self.statusBar().showMessage(f"Recording not found in index: {key.label}")
            return
        if not paths.media_complete:
            QMessageBox.warning(self, "Missing media", f"Recording {key.label} is missing required media files.")
            return

        self._set_loading_state(True)
        self.statusBar().showMessage(f"Loading {key.label} ...")
        self._pause()
        self._unload_runtime(reset_docs=True)

        loader = RecordingLoadThread(paths, self)
        self._loader = loader
        loader.loaded.connect(self._on_recording_loaded)
        loader.failed.connect(self._on_recording_failed)
        loader.finished.connect(self._on_loader_finished)

        # Progress dialog to show load progress and allow cancellation
        try:
            pd = QProgressDialog(f"Loading {key.label}…", "Cancel", 0, 100, self)
            pd.setWindowTitle("Loading recording")
            pd.setWindowModality(Qt.WindowModal)
            pd.setMinimumDuration(0)
            pd.setValue(0)

            def _on_progress(pct: int, msg: str):
                try:
                    pd.setValue(int(pct))
                    pd.setLabelText(str(msg))
                except Exception:
                    pass

            loader.progress.connect(_on_progress)
            pd.canceled.connect(lambda: loader.requestInterruption())
            loader.finished.connect(pd.close)
            pd.show()
        except Exception:
            # If progress dialog can't be created, continue silently
            pass

        loader.start()

    def _on_recording_loaded(self, payload: object):
        if not isinstance(payload, RecordingLoadResult):
            self._on_recording_failed("Unexpected loader payload")
            return

        self._current_key = payload.key
        self._current_paths = payload.paths
        self._doc1 = payload.doc1
        self._doc2 = payload.doc2

        self._door_events = payload.door_events
        self._speech_segs = payload.speech_segs
        self._face_dets = payload.face_dets
        self._door_states = payload.door_states

        self._timeline.set_data(
            self._door_events,
            self._speech_segs,
            self._face_dets,
            self._door_states,
        )
        self._video_label1.set_face_detections(self._face_dets[0])
        self._video_label2.set_face_detections(self._face_dets[1])

        self._cam_offsets[0] = self._video_offset(self._doc1) if self._doc1 and self._doc1.video else 0.0
        self._cam_offsets[1] = self._video_offset(self._doc2) if self._doc2 and self._doc2.video else 0.0
        self._audio_offsets[0] = self._audio_offset(self._doc1) if self._doc1 and self._doc1.audio else 0.0
        self._audio_offsets[1] = self._audio_offset(self._doc2) if self._doc2 and self._doc2.audio else 0.0

        self._timeline.set_sessions(self._sessions(0), self._sessions(1))
        self._timeline.set_offsets(self._cam_offsets[0], self._cam_offsets[1], self._audio_offsets[0], self._audio_offsets[1])
        self._timeline.set_selected_session(None)
        self._populate_session_list()
        self._session_editor.load_session(None)
        self._active_cam = 0
        self._selected_idx = [-1, -1]
        self._update_cam_headers()

        self._changed_session_ids = [set(), set()]
        self._deleted_session_ids = [set(), set()]
        self._dirty = False

        self._start_runtime_for_current_paths()
        self._update_title()
        self.statusBar().showMessage(
            f"Loaded {payload.key.label} — "
            f"Cam1: {len(self._sessions(0))} session(s), "
            f"Cam2: {len(self._sessions(1))} session(s)",
            4000,
        )

    def _on_recording_failed(self, message: str):
        if message and str(message).lower().startswith("cancel"):
            # User cancelled the load — show a brief status message instead of a modal warning
            self.statusBar().showMessage("Load cancelled", 2000)
            logger.info("Recording load cancelled by user")
            return
        self.statusBar().showMessage(f"Failed to load recording: {message}")
        logger.error("Failed to load selected recording: %s", message)
        QMessageBox.warning(self, "Load failed", message)

    def _on_loader_finished(self):
        self._set_loading_state(False)
        self._loader = None

    def _start_runtime_for_current_paths(self):
        self._stop_runtime_threads()

        if not self._current_paths:
            return

        duration = 0.0
        if self._doc1 and self._doc1.video:
            duration = max(duration, float(self._doc1.video.duration_sec))

        for cam in (0, 1):
            path = self._current_paths.video_paths[cam]
            if not path.exists():
                continue
            thread = VideoThread(str(path))
            if not thread.open():
                logger.warning("Could not open video file: %s", path)
                continue
            self._threads[cam] = thread
            duration = max(duration, thread.duration)
            label = self._video_label1 if cam == 0 else self._video_label2
            thread.frame_ready.connect(label.update_frame)
            thread.start()

        self._timeline.set_duration(max(1.0, duration))
        self._seek(0.0)

        # Initialize per-camera audio managers (both cams if available).
        self._audios = [None, None]
        for cam in (0, 1):
            audio_path = self._current_paths.audio_paths[cam]
            if audio_path.exists():
                try:
                    aud = AudioManager(str(audio_path))
                    # initialize volume from UI if available
                    try:
                        vol = self._volume_slider.value() / 100.0
                        aud.set_volume(vol)
                    except Exception:
                        pass
                    self._audios[cam] = aud
                except Exception as e:
                    logger.warning("Audio init failed for %s: %s", audio_path, e)

    def _stop_runtime_threads(self):
        for i, thread in enumerate(self._threads):
            if thread:
                thread.stop()
            self._threads[i] = None

    def _unload_runtime(self, reset_docs: bool):
        self._pause()
        self._stop_runtime_threads()

        # Cleanup both audio managers
        for i, aud in enumerate(getattr(self, "_audios", [None, None])):
            if aud:
                try:
                    aud.cleanup()
                except Exception:
                    pass
            try:
                self._audios[i] = None
            except Exception:
                pass

        self._playhead = 0.0
        self._update_time_label()

        self._video_label1.set_face_detections([])
        self._video_label2.set_face_detections([])

        self._door_events = [[], []]
        self._speech_segs = [[], []]
        self._face_dets = [[], []]
        self._door_states = [[], []]
        self._timeline.set_data(self._door_events, self._speech_segs, self._face_dets, self._door_states)
        self._timeline.set_sessions([])
        self._timeline.set_offsets(0.0, 0.0)
        self._timeline.set_selected_session(None)
        self._timeline.set_duration(1.0)
        self._timeline.set_playhead(0.0)
        self._cam_offsets = [0.0, 0.0]

        self._session_list_cam1.blockSignals(True)
        self._session_list_cam2.blockSignals(True)
        self._session_list_cam1.clear()
        self._session_list_cam2.clear()
        self._session_list_cam1.blockSignals(False)
        self._session_list_cam2.blockSignals(False)
        self._session_editor.load_session(None)
        self._selected_idx = [-1, -1]
        self._changed_session_ids = [set(), set()]
        self._deleted_session_ids = [set(), set()]

        if reset_docs:
            self._doc1 = None
            self._doc2 = None

    # ─── Playback ──────────────────────────────────────────────────────────

    @property
    def _active_doc(self) -> Optional[AnnotationDocument]:
        return self._doc1 if self._active_cam == 0 else self._doc2

    @property
    def _active_offset(self) -> float:
        return self._cam_offsets[self._active_cam]

    @property
    def _main_doc(self) -> Optional[AnnotationDocument]:
        if self._doc1.video.time_stamp <= self._doc2.video.time_stamp:
            return self._doc2
        else:
            return self._doc1
        
    def _video_offset(self, doc: AnnotationDocument) -> float:
        """Returns the video timestamp offset (in seconds) of the given document relative to the main document."""
        return (doc.video.time_stamp - self._main_doc.video.time_stamp) / 1000.0
    
    def _audio_offset(self, doc: AnnotationDocument) -> float:
        """Returns the audio timestamp offset (in seconds) of the given document relative to the main document."""
        if getattr(doc, "audio", None) is not None:
            return float(doc.audio.offset_sec) / 1000.0
        return 0.0


    def _toggle_play(self):
        if self._playing:
            self._pause()
        else:
            self._play()

    def _play(self):
        self._playing = True
        self._wall_clock_last = time.perf_counter()
        self._btn_playpause.setText("⏸")
        for t in self._threads:
            if t:
                t.play()
        # Start both audios in sync with the wall-clock playhead.
        for cam, aud in enumerate(getattr(self, "_audios", [None, None])):
            if aud:
                try:
                    doc = self._doc1 if cam == 0 else self._doc2
                    ao = 0.0
                    if doc and getattr(doc, "audio", None) is not None:
                        ao = float(doc.audio.offset_sec) / 1000.0
                    cam_offset = self._cam_offsets[cam] if cam < len(self._cam_offsets) else 0.0
                    # audio file position = wall-clock playhead - cam_video_offset + audio_offset
                    audio_pos = self._playhead - cam_offset + ao
                    if audio_pos < 0.0:
                        audio_pos = 0.0
                    aud.play(audio_pos)
                except Exception:
                    pass

    def _pause(self):
        self._playing = False
        self._btn_playpause.setText("▶")
        for t in self._threads:
            if t:
                t.pause()
        for aud in getattr(self, "_audios", [None, None]):
            if aud:
                try:
                    aud.pause()
                except Exception:
                    pass

    def _seek(self, t: float):
        dur = self._timeline._duration
        t = max(0.0, min(t, dur))
        self._playhead = t

        if self._threads[0]:
            self._threads[0].seek(t - self._video_offset(self._doc1))
        if self._threads[1]:
            self._threads[1].seek(t - self._video_offset(self._doc2))
        # Seek both audio players (audio offset is stored in ms; convert to s)
        for cam, aud in enumerate(getattr(self, "_audios", [None, None])):
            if aud:
                try:
                    doc = self._doc1 if cam == 0 else self._doc2
                    ao = 0.0
                    if doc and getattr(doc, "audio", None) is not None:
                        ao = float(doc.audio.offset_sec) / 1000.0
                    cam_offset = self._cam_offsets[cam] if cam < len(self._cam_offsets) else 0.0
                    audio_pos = t - cam_offset + ao
                    if audio_pos < 0.0:
                        audio_pos = 0.0
                    aud.seek(audio_pos)
                except Exception:
                    pass

        self._timeline.set_playhead(t)
        self._session_editor.set_playhead(t)
        self._update_time_label()

    def _step_frames(self, n: int, large: bool = False):
        fps = self._threads[0].fps if self._threads[0] else 30.0
        step = (10 if large else 1) / fps
        self._seek(self._playhead + (step if n > 0 else -step))

    def _on_ui_tick(self):
        if not self._playing:
            return
        now = time.perf_counter()
        dt = (now - self._wall_clock_last) * self._speed
        self._wall_clock_last = now
        self._playhead = min(self._playhead + dt, self._timeline._duration)
        self._timeline.set_playhead(self._playhead)
        self._session_editor.set_playhead(self._playhead)
        self._update_time_label()
        if self._playhead >= self._timeline._duration:
            self._pause()

    def _update_time_label(self):
        self._time_label.setText(
            f"{fmt_time(self._playhead)} / {fmt_time(self._timeline._duration)}"
        )

    def _on_speed_changed(self, idx: int):
        self._speed = SPEED_VALUES[idx]
        for t in self._threads:
            if t:
                t.set_speed(self._speed)

    def _on_volume_changed(self, _val: int):
        v = float(_val)
        # set_volume accepts 0..1 or 0..100; normalize to 0..1
        if v > 1.0:
            v = max(0.0, min(1.0, v / 100.0))
        else:
            v = max(0.0, min(1.0, v))
        for aud in getattr(self, "_audios", [None, None]):
            if aud:
                try:
                    aud.set_volume(v)
                except Exception:
                    pass

    def _on_timeline_seek(self, t: float):
        self._seek(t)

    # ─── Sessions and editor sync ──────────────────────────────────────────

    def _sessions(self, cam: int) -> List[Session]:
        doc = self._doc1 if cam == 0 else self._doc2
        return doc.sessions if doc else []

    def _index_for_session_id(self, sid: str, cam: int) -> int:
        for i, s in enumerate(self._sessions(cam)):
            if s.session_id == sid:
                return i
        return -1

    def _populate_session_list(self):
        self._populate_session_list_for(0)
        self._populate_session_list_for(1)

    def _populate_session_list_for(self, cam: int):
        lst = self._session_list_cam1 if cam == 0 else self._session_list_cam2
        lst.blockSignals(True)
        lst.clear()
        for s in self._sessions(cam):
            enter_str = fmt_time(s.times.enter_time.t) if s.times.enter_time else "??"
            exit_str = fmt_time(s.times.exit_time.t) if s.times.exit_time else "??"
            icon = STATUS_ICONS.get(s.validation.status, "·")
            lst.addItem(QListWidgetItem(f"{icon} {s.session_id}  {enter_str} → {exit_str}"))
        lst.blockSignals(False)

    def _current_session(self) -> Optional[Session]:
        ss = self._sessions(self._active_cam)
        idx = self._selected_idx[self._active_cam]
        if 0 <= idx < len(ss):
            return ss[idx]
        return None

    def _update_cam_headers(self):
        bold = QFont()
        bold.setBold(True)
        normal = QFont()
        if self._active_cam == 0:
            self._lbl_cam1_sessions.setFont(bold)
            self._lbl_cam2_sessions.setFont(normal)
        else:
            self._lbl_cam1_sessions.setFont(normal)
            self._lbl_cam2_sessions.setFont(bold)

    def _mark_session_changed(self, sid: str):
        if not sid:
            return
        self._deleted_session_ids[self._active_cam].discard(sid)
        self._changed_session_ids[self._active_cam].add(sid)
        self._dirty = True
        self._update_title()

    def _mark_session_deleted(self, sid: str):
        if not sid:
            return
        self._changed_session_ids[self._active_cam].discard(sid)
        self._deleted_session_ids[self._active_cam].add(sid)
        self._dirty = True
        self._update_title()

    def _next_session_id(self) -> str:
        used = (
            {s.session_id for s in self._sessions(0)}
            | {s.session_id for s in self._sessions(1)}
        )
        i = 1
        while True:
            sid = f"S{i:03d}"
            if sid not in used:
                return sid
            i += 1

    def _do_add_session(self, cam: int):
        self._active_cam = cam
        self._add_session()

    def _do_remove_session(self, cam: int):
        self._active_cam = cam
        self._remove_session()

    def _add_session(self):
        doc = self._active_doc
        if doc is None:
            return

        from ..models import SessionTimes

        sid = self._next_session_id()
        enter_val = AnnotationValue(t=self._playhead - self._active_offset, source="manual", confidence=1.0, method="gui")
        new_s = Session(session_id=sid, times=SessionTimes(enter_time=enter_val))

        doc.sessions.append(new_s)
        key_fn = lambda s: s.times.enter_time.t if s.times.enter_time else float("inf")
        doc.sessions.sort(key=key_fn)

        self._timeline.set_sessions(self._sessions(0), self._sessions(1))
        self._populate_session_list_for(self._active_cam)

        lst = self._session_list_cam1 if self._active_cam == 0 else self._session_list_cam2
        idx = self._index_for_session_id(sid, self._active_cam)
        if idx >= 0:
            lst.setCurrentRow(idx)

        self._mark_session_changed(sid)
        self.statusBar().showMessage(
            f"Added session {sid} (Cam {self._active_cam + 1})", 2000
        )
        if self._auto_save:
            self._save_all()

    def _remove_session(self):
        doc = self._active_doc
        if doc is None:
            return
        idx = self._selected_idx[self._active_cam]
        ss = self._sessions(self._active_cam)
        if not (0 <= idx < len(ss)):
            return
        s = ss[idx]

        reply = QMessageBox.question(
            self,
            "Remove session",
            f"Remove session {s.session_id}? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        sid = s.session_id
        doc.sessions = [x for x in doc.sessions if x.session_id != sid]

        self._session_editor.load_session(None)
        self._selected_idx[self._active_cam] = -1
        self._timeline.set_sessions(self._sessions(0), self._sessions(1))
        self._timeline.set_selected_session(None)
        self._populate_session_list_for(self._active_cam)

        self._mark_session_deleted(sid)
        self.statusBar().showMessage(f"Removed session {sid}", 2000)
        if self._auto_save:
            self._save_all()

    def _jump_to_session_start(self):
        s = self._current_session()
        if s and s.times.enter_time:
            self._seek(s.times.enter_time.t + self._active_offset)

    def _jump_to_session_end(self):
        s = self._current_session()
        if s and s.times.exit_time:
            self._seek(s.times.exit_time.t + self._active_offset)

    def _prev_session(self):
        lst = self._session_list_cam1 if self._active_cam == 0 else self._session_list_cam2
        n = lst.count()
        if n:
            lst.setCurrentRow(max(0, self._selected_idx[self._active_cam] - 1))

    def _next_session(self):
        lst = self._session_list_cam1 if self._active_cam == 0 else self._session_list_cam2
        n = lst.count()
        if n:
            lst.setCurrentRow(min(n - 1, self._selected_idx[self._active_cam] + 1))

    def _on_list_row_changed(self, cam: int, row: int):
        if self._suppress_list_signal:
            return
        self._active_cam = cam
        self._selected_idx[cam] = row
        # Clear selection in the other list
        self._suppress_list_signal = True
        other = self._session_list_cam2 if cam == 0 else self._session_list_cam1
        other.clearSelection()
        other.setCurrentRow(-1)
        self._selected_idx[1 - cam] = -1
        self._suppress_list_signal = False
        self._update_cam_headers()
        ss = self._sessions(cam)
        session = ss[row] if 0 <= row < len(ss) else None
        self._session_editor.load_session(session)
        self._session_editor.set_offset(self._cam_offsets[cam])
        self._timeline.set_selected_session(session.session_id if session else None)
        if session and session.times.enter_time:
            self._seek(session.times.enter_time.t + self._cam_offsets[cam])

    def _on_list_double_click(self, _item):
        self._jump_to_session_start()
        self._play()

    def _on_session_selected_by_id(self, session_id: str):
        for cam in (0, 1):
            idx = self._index_for_session_id(session_id, cam)
            if idx >= 0:
                self._active_cam = cam
                lst = self._session_list_cam1 if cam == 0 else self._session_list_cam2
                lst.setCurrentRow(idx)
                return

    def _on_timeline_session_modified(self, session_id: str, _field_name: str, _value: float):
        for cam in (0, 1):
            if self._index_for_session_id(session_id, cam) >= 0:
                self._active_cam = cam
                break
        self._mark_session_changed(session_id)
        idx = self._index_for_session_id(session_id, self._active_cam)
        self._refresh_list_item(idx)
        self._timeline.set_sessions(self._sessions(0), self._sessions(1))
        if self._auto_save:
            self._save_all()

    def _on_annotation_changed(self, _field: str, _t: float):
        s = self._current_session()
        if not s:
            return
        self._mark_session_changed(s.session_id)
        self._refresh_list_item(self._selected_idx[self._active_cam])
        self._timeline.set_sessions(self._sessions(0), self._sessions(1))
        if self._auto_save:
            self._save_all()

    def _refresh_list_item(self, idx: int):
        ss = self._sessions(self._active_cam)
        if not (0 <= idx < len(ss)):
            return
        s = ss[idx]
        icon = STATUS_ICONS.get(s.validation.status, "·")
        enter_str = fmt_time(s.times.enter_time.t) if s.times.enter_time else "??"
        exit_str = fmt_time(s.times.exit_time.t) if s.times.exit_time else "??"
        lst = self._session_list_cam1 if self._active_cam == 0 else self._session_list_cam2
        item = lst.item(idx)
        if item:
            item.setText(f"{icon} {s.session_id}  {enter_str} → {exit_str}")

    # ─── Save (session patch mode) ─────────────────────────────────────────

    @staticmethod
    def _patch_sessions_json(
        json_path: Path,
        doc: AnnotationDocument,
        changed_ids: Set[str],
        deleted_ids: Set[str],
    ):
        doc_map = {s.session_id: s.to_dict() for s in doc.sessions}

        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid JSON object in {json_path}")
        else:
            payload = doc.to_dict()

        sessions = payload.get("sessions", [])
        if not isinstance(sessions, list):
            sessions = []

        # Normalize to dict entries with session_id only.
        normalized: List[dict] = []
        for sess in sessions:
            if isinstance(sess, dict) and "session_id" in sess:
                normalized.append(sess)
        sessions = normalized

        if deleted_ids:
            sessions = [s for s in sessions if s.get("session_id") not in deleted_ids]

        by_id = {
            str(sess.get("session_id")): i
            for i, sess in enumerate(sessions)
            if isinstance(sess, dict) and "session_id" in sess
        }

        for sid in changed_ids:
            if sid in deleted_ids:
                continue
            session_payload = doc_map.get(sid)
            if session_payload is None:
                continue
            idx = by_id.get(sid)
            if idx is None:
                sessions.append(session_payload)
                by_id[sid] = len(sessions) - 1
            else:
                sessions[idx] = session_payload

        payload["sessions"] = sessions
        if "schema_version" not in payload:
            payload["schema_version"] = doc.schema_version
        if "tool" not in payload:
            payload["tool"] = doc.tool.to_dict()
        if "video" not in payload and doc.video:
            payload["video"] = doc.video.to_dict()
        if "audio" not in payload and doc.audio:
            payload["audio"] = doc.audio.to_dict()
        if "notes" not in payload:
            payload["notes"] = doc.notes

        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _save_all(self) -> bool:
        if not self._current_paths or not self._doc1 or not self._doc2:
            return False

        changed0 = set(self._changed_session_ids[0])
        deleted0 = set(self._deleted_session_ids[0])
        changed1 = set(self._changed_session_ids[1])
        deleted1 = set(self._deleted_session_ids[1])
        if not changed0 and not deleted0 and not changed1 and not deleted1:
            self.statusBar().showMessage("No pending changes", 1500)
            return True

        try:
            if changed0 or deleted0:
                self._patch_sessions_json(
                    self._current_paths.output_json_paths[0],
                    self._doc1,
                    changed0,
                    deleted0,
                )
            if changed1 or deleted1:
                self._patch_sessions_json(
                    self._current_paths.output_json_paths[1],
                    self._doc2,
                    changed1,
                    deleted1,
                )
            self._changed_session_ids = [set(), set()]
            self._deleted_session_ids = [set(), set()]
            self._dirty = False
            self._update_title()
            self.statusBar().showMessage("Saved ✓", 3000)
            return True
        except Exception as e:
            logger.error("Save failed: %s", e, exc_info=True)
            self.statusBar().showMessage(f"Save failed: {e}")
            QMessageBox.critical(self, "Save failed", str(e))
            return False

    # ─── Title / shortcuts / lifecycle ────────────────────────────────────

    def _update_title(self):
        suffix = ""
        if self._current_key:
            suffix = f" - {self._current_key.label}"
        dirty = " *" if self._dirty else ""
        self.setWindowTitle(f"Booth Session Annotator (System){suffix}{dirty}")

    def keyPressEvent(self, event):
        key = event.key()
        mods = event.modifiers()
        large = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)

        if key == Qt.Key.Key_Space:
            self._toggle_play()
        elif key == Qt.Key.Key_Left:
            self._prev_session() if ctrl else self._step_frames(-1, large=large)
        elif key == Qt.Key.Key_Right:
            self._next_session() if ctrl else self._step_frames(1, large=large)
        elif key == Qt.Key.Key_BracketLeft:
            self._session_editor._on_set_enter()
        elif key == Qt.Key.Key_BracketRight:
            self._session_editor._on_set_exit()
        elif key == Qt.Key.Key_S and not ctrl:
            self._save_all()
        elif key == Qt.Key.Key_N:
            self._next_session()
            self._play()
        elif key == Qt.Key.Key_P:
            self._prev_session()
        elif key == Qt.Key.Key_Z and ctrl:
            self._session_editor._on_undo()
        elif key == Qt.Key.Key_Y and ctrl:
            self._session_editor._on_redo()
        elif key == Qt.Key.Key_A and not ctrl:
            self._add_session()
        elif key == Qt.Key.Key_Delete:
            self._remove_session()
        elif key == Qt.Key.Key_Escape:
            self._session_list_cam1.clearSelection()
            self._session_list_cam1.setCurrentRow(-1)
            self._session_list_cam2.clearSelection()
            self._session_list_cam2.setCurrentRow(-1)
            self._selected_idx = [-1, -1]
            self._session_editor.load_session(None)
        elif key == Qt.Key.Key_PageUp and ctrl:
            self._prev_recording()
        elif key == Qt.Key.Key_PageDown and ctrl:
            self._next_recording()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        if self._dirty:
            action = self._prompt_unsaved_changes("You have unsaved changes. Save before closing?")
            if action == "cancel":
                event.ignore()
                return
            if action == "save" and not self._save_all():
                event.ignore()
                return

        self._ui_timer.stop()
        self._pause()
        self._unload_runtime(reset_docs=True)

        if self._loader and self._loader.isRunning():
            self._loader.wait(3000)
        self._loader = None

        super().closeEvent(event)


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Booth Session Annotator system GUI (multi-recording)",
    )
    parser.add_argument("recording", help="Path to root recording directory")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output root directory",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Cache root directory for .pkl files",
    )
    parser.add_argument(
        "--auto-save",
        action="store_true",
        help="Save annotations after every edit",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    setup_logging(args.verbose)

    app = QApplication(sys.argv)
    app.setApplicationName("BoothSessionAnnotatorSystem")

    window = MainWindow(
        recording_dir=Path(args.recording),
        output_dir=args.output,
        cache_dir=args.cache_dir,
        auto_save=args.auto_save,
    )
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()