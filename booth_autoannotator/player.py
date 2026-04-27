#!/usr/bin/env python3
"""
PyCV Player — a sophisticated OpenCV video player with audio sync support.

Features
- Play/Pause, frame-step, time seeks
- Mouse scrub timeline (click/drag)
- Zoom (mouse wheel) + Pan (right-drag), reset zoom (middle click)
- In/Out markers + loop segment
- Bookmarks (add/next/prev) + export JSON
- Audio playback with sync and offset control
- Snapshot PNG
- Fullscreen toggle
- HUD + Help overlay

Install:
  pip install opencv-python numpy

Audio playback requires one of:
  pip install pygame  # Recommended for best compatibility
  pip install pyaudio librosa
  pip install sounddevice soundfile

Run:
  python pycv_player.py /path/to/video.mp4
  python pycv_player.py /path/to/video.mp4 --audio /path/to/audio.wav
  python pycv_player.py /path/to/video.mp4 --audio /path/to/audio.wav --audio-offset -0.5
  python pycv_player.py 0               # webcam index
"""

import argparse
import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ---------- Utilities ----------


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def fmt_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{s:06.3f}"
    return f"{m:02d}:{s:06.3f}"


def safe_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except Exception:
        return None


def get_window_size(win: str, fallback: Tuple[int, int]) -> Tuple[int, int]:
    # OpenCV 4.x supports getWindowImageRect; fall back if unavailable.
    try:
        x, y, w, h = cv2.getWindowImageRect(win)
        if w > 0 and h > 0:
            return (w, h)
    except Exception:
        pass
    return fallback


# ---------- Audio Manager ----------


class AudioManager:
    """
    Manages audio playback with synchronization to video.
    Supports multiple backends: pygame, pyaudio, sounddevice.
    """

    def __init__(self, audio_path: Optional[str] = None):
        self.audio_path = audio_path
        self.backend = None
        self.is_playing = False
        self.current_position = 0.0
        self.duration = 0.0
        self.thread = None
        self._stop_event = threading.Event()

        if audio_path and os.path.exists(audio_path):
            self._init_backend()

    def _init_backend(self):
        """Try to initialize audio backend in order of preference."""
        backends = [
            ("sounddevice", self._init_sounddevice),
            ("pyaudio", self._init_pyaudio),
            ("pygame", self._init_pygame),
        ]

        for name, init_func in backends:
            try:
                init_func()
                self.backend = name
                print(f"[audio] initialized with {name} backend")
                return
            except Exception as e:
                print(f"[audio] {name} failed: {e}")
                continue

        print("[audio] no suitable audio backend found")

    def _init_pygame(self):
        """Initialize pygame mixer backend."""
        import pygame

        pygame.mixer.init()
        pygame.mixer.music.load(self.audio_path)
        self.duration = pygame.mixer.Sound(self.audio_path).get_length()

    def _init_pyaudio(self):
        """Initialize pyaudio backend."""
        import librosa

        self.audio_data, self.sample_rate = librosa.load(self.audio_path, sr=None)
        self.duration = len(self.audio_data) / self.sample_rate

    def _init_sounddevice(self):
        """Initialize sounddevice backend."""
        import soundfile as sf

        self.audio_data, self.sample_rate = sf.read(self.audio_path)
        self.duration = len(self.audio_data) / self.sample_rate

    def play(self, start_time: float = 0.0):
        """Start audio playback from specified time."""
        if not self.backend:
            return

        self.current_position = start_time
        self.is_playing = True
        self._stop_event.clear()

        if self.backend == "pygame":
            self._play_pygame(start_time)
        elif self.backend == "pyaudio":
            self._play_thread()
        elif self.backend == "sounddevice":
            self._play_sounddevice(start_time)

    def _play_pygame(self, start_time: float):
        """Play audio using pygame at specified time."""
        import pygame

        pygame.mixer.music.play(loops=0, start=start_time)

    def _play_sounddevice(self, start_time: float):
        """Play audio using sounddevice in background thread."""
        import sounddevice as sd

        self.thread = threading.Thread(
            target=self._play_sounddevice_thread, args=(start_time,), daemon=True
        )
        self.thread.start()

    def _play_sounddevice_thread(self, start_time: float):
        """Background thread for sounddevice playback."""
        import sounddevice as sd

        start_sample = int(start_time * self.sample_rate)
        chunk = self.audio_data[start_sample:]

        if len(chunk) > 0:
            sd.play(chunk, samplerate=self.sample_rate)
            sd.wait()

    def _play_thread(self):
        """Background thread for pyaudio playback."""
        import pyaudio

        self.thread = threading.Thread(target=self._play_pyaudio_thread, daemon=True)
        self.thread.start()

    def _play_pyaudio_thread(self):
        """Background thread for pyaudio playback."""
        import pyaudio

        chunk_size = 1024
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=self.audio_data.shape[1]
            if len(self.audio_data.shape) > 1
            else 1,
            rate=self.sample_rate,
            output=True,
        )

        start_sample = int(self.current_position * self.sample_rate)
        for i in range(start_sample, len(self.audio_data), chunk_size):
            if self._stop_event.is_set():
                break

            chunk = self.audio_data[i : i + chunk_size]
            if len(chunk) > 0:
                stream.write(chunk.astype(np.float32).tobytes())
            self.current_position = i / self.sample_rate

        stream.stop_stream()
        stream.close()
        p.terminate()

    def pause(self):
        """Pause audio playback."""
        if not self.backend:
            return

        self.is_playing = False
        if self.backend == "pygame":
            import pygame

            pygame.mixer.music.pause()
        else:
            self._stop_event.set()

    def resume(self):
        """Resume audio playback."""
        if not self.backend or not self.is_playing:
            return

        self.is_playing = True
        if self.backend == "pygame":
            import pygame

            pygame.mixer.music.unpause()

    def stop(self):
        """Stop audio playback."""
        if not self.backend:
            return

        self.is_playing = False
        self._stop_event.set()

        if self.backend == "pygame":
            import pygame

            pygame.mixer.music.stop()

        if self.thread:
            self.thread.join(timeout=1.0)

    def seek(self, position: float):
        """Seek audio to specified time."""
        if not self.backend:
            return

        self.current_position = clamp(position, 0.0, self.duration)
        if self.is_playing:
            self.stop()
            self.play(self.current_position)

    def get_position(self) -> float:
        """Get current playback position."""
        if not self.backend or not self.is_playing:
            return self.current_position

        if self.backend == "pygame":
            import pygame

            if pygame.mixer.music.get_busy():
                return pygame.mixer.music.get_pos() / 1000.0
        return self.current_position

    def cleanup(self):
        """Clean up audio resources."""
        if self.backend:
            self.stop()
            if self.backend == "pygame":
                import pygame

                pygame.mixer.quit()



# ---------- PreCompute CAP ----------


class ListVideoCap:
    """
    A simple VideoCapture-like class that serves frames from a pre-loaded list.
    Used for testing or as a placeholder.
    """

    def __init__(self, frames: List[np.ndarray], fps: float = 30.0):
        self.frames = frames
        self.fps = fps
        self.total_frames = len(frames)
        self.pos = 0  # next frame index

    def isOpened(self) -> bool:
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.pos >= self.total_frames:
            return (False, None)
        fr = self.frames[self.pos]
        self.pos += 1
        return (True, fr)

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FPS:
            return self.fps
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return float(self.total_frames)
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            return float(self.pos)
        return 0.0

    def set(self, prop_id: int, value: float) -> bool:
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            v = int(clamp(int(value), 0, max(0, self.total_frames - 1)))
            self.pos = v
            return True
        return False
    
    def release(self):
        pass


# ---------- Player State ----------


@dataclass
class PlayerState:
    paused: bool = False
    show_hud: bool = True
    show_help: bool = False
    fullscreen: bool = False

    speed: float = 1.0  # playback speed multiplier
    repeat_all: bool = False
    audio_enabled: bool = True
    audio_offset: float = 0.0  # audio offset in seconds (positive = audio ahead)

    # Zoom / pan in source-frame coordinates
    zoom: float = 1.0  # 1..6
    pan_cx: float = 0.5  # normalized center (0..1)
    pan_cy: float = 0.5

    # Image adjustments
    brightness: int = 0  # beta in convertScaleAbs
    contrast: float = 1.0  # alpha in convertScaleAbs

    # Segment loop (in/out in frames)
    in_frame: Optional[int] = None
    out_frame: Optional[int] = None
    loop_segment: bool = False

    bookmarks: List[int] = field(default_factory=list)

    # Mouse interaction
    scrubbing: bool = False
    right_drag: bool = False
    last_mouse: Tuple[int, int] = (0, 0)
    hover_bar_x: Optional[int] = None


# ---------- Main Player ----------


class PyCVPlayer:
    def __init__(
        self,
        src: str | cv2.VideoCapture,
        start_frame: float = 0.0,
        audio_path: Optional[str] = None,
        audio_offset: float = 0.0,
        window: str = "PyCV Player",
    ):
        self.window = window
        self.src = src

        self.state = PlayerState(audio_offset=audio_offset)

        # Timeline bar appearance
        self.bar_h = 44
        self.pad = 10

        # Audio manager
        self.audio = AudioManager(audio_path)

        # Capture
        self.cap = self._open_capture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open source: {src}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not self.fps or self.fps <= 1e-3:
            self.fps = 30.0

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.is_live = self.total_frames <= 0  # webcam/stream

        # Frame state
        self.cur_frame_idx = 0
        self.cur_frame = None  # last decoded BGR frame
        self.last_render_size = (1280, 720)

        # Seek to start time if possible
        if start_frame > 0 and not self.is_live:
            self.seek_frame(start_frame, show_frame=True)
        else:
            self._read_one(show_if_fail=True)

        # Window setup
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
        cv2.setMouseCallback(self.window, self._on_mouse)

        # Timing
        self.next_due = time.perf_counter()

        # Export path for sidecar JSON / snapshots
        self.sidecar_dir, self.sidecar_base = self._sidecar_paths(src)

    def _open_capture(self, src: str | cv2.VideoCapture) -> cv2.VideoCapture:
        # If user passes a digit like "0", treat as webcam index.
        idx = safe_int(src)
        if idx is not None and src.strip() == str(idx):
            return cv2.VideoCapture(idx)
        if isinstance(src, str):
            return cv2.VideoCapture(src)
        return src

    def _sidecar_paths(self, src: str | cv2.VideoCapture) -> Tuple[str, str]:
        # For webcam, use current directory
        if safe_int(src) is not None and src.strip().isdigit():
            return (os.getcwd(), f"webcam_{src}")
        elif isinstance(src, str):
            p = os.path.abspath(src)
            d = os.path.dirname(p)
            base = os.path.splitext(os.path.basename(p))[0]
            return (d, base)
        else:
            return (os.getcwd(), self.src_name)

    # ---- Capture / seeking ----

    def duration_s(self) -> float:
        if self.is_live:
            return 0.0
        return self.total_frames / self.fps if self.total_frames > 0 else 0.0

    def frame_time_s(self, frame_idx: int) -> float:
        return frame_idx / self.fps

    def _read_one(self, show_if_fail=False) -> bool:
        ret, fr = self.cap.read()
        if not ret:
            if show_if_fail and self.cur_frame is not None:
                return False
            return False

        self.cur_frame = fr
        # CAP_PROP_POS_FRAMES returns "next frame index", so subtract 1 for displayed frame.
        pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
        if not self.is_live:
            self.cur_frame_idx = max(0, pos - 1)
        else:
            self.cur_frame_idx += 1
        return True

    def seek_frame(self, frame_idx: int, show_frame=True):
        if self.is_live:
            return
        frame_idx = int(clamp(frame_idx, 0, max(0, self.total_frames - 1)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # Sync audio to new frame position with offset applied
        if self.audio.backend and not self.state.paused:
            t = self.frame_time_s(frame_idx)
            # Apply audio offset: positive offset plays audio earlier
            audio_t = t + self.state.audio_offset
            self.audio.seek(audio_t)
        
        if show_frame:
            self._read_one(show_if_fail=True)

    def seek_time(self, t: float, show_frame=True):
        if self.is_live:
            return
        t = clamp(float(t), 0.0, self.duration_s())
        self.seek_frame(int(round(t * self.fps)), show_frame=show_frame)

    # ---- Mouse handling ----

    def _bar_rect(self, win_w: int, win_h: int) -> Tuple[int, int, int, int]:
        x0 = 0
        y0 = win_h - self.bar_h
        return (x0, y0, win_w, self.bar_h)

    def _in_bar(self, x: int, y: int, win_w: int, win_h: int) -> bool:
        _, y0, w, h = self._bar_rect(win_w, win_h)
        return (0 <= x < w) and (y0 <= y < y0 + h)

    def _bar_x_to_frame(self, x: int, win_w: int) -> int:
        if self.is_live or self.total_frames <= 1:
            return 0
        t = clamp(x / max(1, win_w - 1), 0.0, 1.0)
        return int(round(t * (self.total_frames - 1)))

    def _on_mouse(self, event, x, y, flags, userdata=None):
        win_w, win_h = get_window_size(self.window, self.last_render_size)
        self.last_render_size = (win_w, win_h)

        self.state.hover_bar_x = x if self._in_bar(x, y, win_w, win_h) else None

        if event == cv2.EVENT_LBUTTONDOWN:
            if self._in_bar(x, y, win_w, win_h):
                self.state.scrubbing = True
                if not self.is_live:
                    self.seek_frame(self._bar_x_to_frame(x, win_w), show_frame=True)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.state.scrubbing and (not self.is_live):
                self.seek_frame(self._bar_x_to_frame(x, win_w), show_frame=True)
            if self.state.right_drag:
                self._pan_by_mouse(x, y, win_w, win_h)
        elif event == cv2.EVENT_LBUTTONUP:
            self.state.scrubbing = False

        # Right-drag for pan when zoomed
        if event == cv2.EVENT_RBUTTONDOWN:
            self.state.right_drag = True
            self.state.last_mouse = (x, y)
        elif event == cv2.EVENT_RBUTTONUP:
            self.state.right_drag = False

        # Middle click resets zoom/pan
        if event == cv2.EVENT_MBUTTONDOWN:
            self.state.zoom = 1.0
            self.state.pan_cx, self.state.pan_cy = 0.5, 0.5

        # Mouse wheel zoom (Windows / some builds)
        if event == cv2.EVENT_MOUSEWHEEL:
            # flags carries wheel delta in high 16 bits for some OpenCV builds
            delta = (flags >> 16) & 0xFFFF
            if delta & 0x8000:  # negative
                delta = -((~delta & 0xFFFF) + 1)
            self._zoom_at_cursor(x, y, delta, win_w, win_h)

    def _zoom_at_cursor(
        self, mx: int, my: int, wheel_delta: int, win_w: int, win_h: int
    ):
        if self.cur_frame is None:
            return
        if wheel_delta == 0:
            return

        # Compute cursor position in source image coordinates under current view
        src_h, src_w = self.cur_frame.shape[:2]
        view = self._compute_view_rect(src_w, src_h)

        # Map mouse to displayed video region (excluding bar) and then to view rect.
        video_h = max(1, win_h - self.bar_h)
        # Fit-to-window rect for the video image
        scale = min(win_w / src_w, video_h / src_h)
        disp_w = int(round(src_w * scale))
        disp_h = int(round(src_h * scale))
        ox = (win_w - disp_w) // 2
        oy = (video_h - disp_h) // 2

        # Clamp mouse to video display area
        mx2 = clamp(mx, ox, ox + disp_w - 1)
        my2 = clamp(my, oy, oy + disp_h - 1)

        u = (mx2 - ox) / max(1, disp_w - 1)
        v = (my2 - oy) / max(1, disp_h - 1)

        vx, vy, vw, vh = view
        ix = vx + u * (vw - 1)
        iy = vy + v * (vh - 1)

        # Zoom change
        step = 1.12
        if wheel_delta > 0:
            new_zoom = self.state.zoom * step
        else:
            new_zoom = self.state.zoom / step
        new_zoom = float(clamp(new_zoom, 1.0, 6.0))

        # Update zoom, then adjust pan so cursor stays roughly anchored
        self.state.zoom = new_zoom

        # New view size
        new_vw = src_w / self.state.zoom
        new_vh = src_h / self.state.zoom

        # Set center so that (ix, iy) stays at same u,v location
        new_cx = (ix - (u - 0.5) * new_vw) / src_w
        new_cy = (iy - (v - 0.5) * new_vh) / src_h
        self.state.pan_cx = float(clamp(new_cx, 0.0, 1.0))
        self.state.pan_cy = float(clamp(new_cy, 0.0, 1.0))

    def _pan_by_mouse(self, x: int, y: int, win_w: int, win_h: int):
        if self.cur_frame is None or self.state.zoom <= 1.001:
            self.state.last_mouse = (x, y)
            return

        lx, ly = self.state.last_mouse
        dx = x - lx
        dy = y - ly
        self.state.last_mouse = (x, y)

        # Pan sensitivity based on zoom and window size
        src_h, src_w = self.cur_frame.shape[:2]
        view_w = src_w / self.state.zoom
        view_h = src_h / self.state.zoom

        # Convert pixels in window to normalized pan shift
        # Approx: how much of view does a window pixel represent?
        video_h = max(1, win_h - self.bar_h)
        scale = min(win_w / src_w, video_h / src_h)
        disp_w = max(1, int(round(src_w * scale)))
        disp_h = max(1, int(round(src_h * scale)))

        shift_x = (dx / disp_w) * (view_w / src_w)
        shift_y = (dy / disp_h) * (view_h / src_h)

        # Right-drag: move the image with the mouse (invert pan)
        self.state.pan_cx = float(clamp(self.state.pan_cx - shift_x, 0.0, 1.0))
        self.state.pan_cy = float(clamp(self.state.pan_cy - shift_y, 0.0, 1.0))

    # ---- Rendering ----

    def _compute_view_rect(self, src_w: int, src_h: int) -> Tuple[int, int, int, int]:
        # Returns (x, y, w, h) in source-image coordinates for zoom/pan crop.
        z = max(1.0, float(self.state.zoom))
        if z <= 1.001:
            return (0, 0, src_w, src_h)

        vw = src_w / z
        vh = src_h / z

        cx = self.state.pan_cx * src_w
        cy = self.state.pan_cy * src_h

        x0 = int(round(cx - vw / 2))
        y0 = int(round(cy - vh / 2))

        x0 = int(clamp(x0, 0, max(0, src_w - int(round(vw)))))
        y0 = int(clamp(y0, 0, max(0, src_h - int(round(vh)))))

        vw_i = int(clamp(int(round(vw)), 1, src_w))
        vh_i = int(clamp(int(round(vh)), 1, src_h))
        return (x0, y0, vw_i, vh_i)

    def _apply_adjustments(self, frame: np.ndarray) -> np.ndarray:
        # brightness/contrast
        fr = cv2.convertScaleAbs(
            frame, alpha=self.state.contrast, beta=self.state.brightness
        )
        return fr

    def _compose_canvas(self) -> np.ndarray:
        if self.cur_frame is None:
            return np.zeros((720, 1280, 3), dtype=np.uint8)

        src = self._apply_adjustments(self.cur_frame)
        src_h, src_w = src.shape[:2]

        win_w, win_h = get_window_size(self.window, self.last_render_size)
        self.last_render_size = (win_w, win_h)

        # Allocate canvas
        canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)

        video_h = max(1, win_h - self.bar_h)

        # Crop for zoom/pan
        vx, vy, vw, vh = self._compute_view_rect(src_w, src_h)
        view = src[vy : vy + vh, vx : vx + vw]

        # Fit view into available area while keeping aspect ratio
        scale = min(win_w / vw, video_h / vh)
        disp_w = max(1, int(round(vw * scale)))
        disp_h = max(1, int(round(vh * scale)))

        resized = cv2.resize(view, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

        ox = (win_w - disp_w) // 2
        oy = (video_h - disp_h) // 2
        canvas[oy : oy + disp_h, ox : ox + disp_w] = resized

        # Draw timeline bar
        self._draw_bar(canvas, win_w, win_h)

        # HUD
        if self.state.show_hud:
            self._draw_hud(canvas)

        # Help overlay
        if self.state.show_help:
            self._draw_help(canvas)

        return canvas

    def _draw_bar(self, canvas: np.ndarray, win_w: int, win_h: int):
        x0, y0, w, h = self._bar_rect(win_w, win_h)
        cv2.rectangle(
            canvas, (x0, y0), (x0 + w - 1, y0 + h - 1), (30, 30, 30), thickness=-1
        )

        if self.is_live or self.total_frames <= 1:
            # Live: simple indicator
            cv2.putText(
                canvas,
                "LIVE (no seeking)",
                (10, y0 + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (220, 220, 220),
                2,
                cv2.LINE_AA,
            )
            return

        # Progress fraction
        frac = self.cur_frame_idx / max(1, self.total_frames - 1)
        px = int(round(frac * (w - 1)))

        # Base line
        y_line = y0 + h // 2
        cv2.line(canvas, (0, y_line), (w - 1, y_line), (80, 80, 80), 2)

        # Played portion
        cv2.line(canvas, (0, y_line), (px, y_line), (210, 210, 210), 4)

        # Handle
        cv2.circle(canvas, (px, y_line), 8, (245, 245, 245), -1, cv2.LINE_AA)

        # In/Out markers
        if self.state.in_frame is not None:
            ix = int(
                round((self.state.in_frame / max(1, self.total_frames - 1)) * (w - 1))
            )
            cv2.line(canvas, (ix, y0 + 6), (ix, y0 + h - 7), (80, 220, 80), 2)
            cv2.putText(
                canvas,
                "I",
                (ix + 4, y0 + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (80, 220, 80),
                2,
                cv2.LINE_AA,
            )
        if self.state.out_frame is not None:
            ox = int(
                round((self.state.out_frame / max(1, self.total_frames - 1)) * (w - 1))
            )
            cv2.line(canvas, (ox, y0 + 6), (ox, y0 + h - 7), (80, 160, 255), 2)
            cv2.putText(
                canvas,
                "O",
                (ox + 4, y0 + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (80, 160, 255),
                2,
                cv2.LINE_AA,
            )

        # Hover time preview
        if self.state.hover_bar_x is not None:
            hx = int(clamp(self.state.hover_bar_x, 0, w - 1))
            hframe = self._bar_x_to_frame(hx, w)
            ht = self.frame_time_s(hframe)
            txt = fmt_time(ht)
            cv2.line(canvas, (hx, y0 + 6), (hx, y0 + h - 7), (140, 140, 140), 1)
            tw = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
            tx = int(clamp(hx - tw // 2, 4, w - tw - 4))
            cv2.putText(
                canvas,
                txt,
                (tx, y0 + h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (230, 230, 230),
                2,
                cv2.LINE_AA,
            )

        # Current time text (right side)
        cur_t = self.frame_time_s(self.cur_frame_idx)
        dur_t = self.duration_s()
        right_txt = f"{fmt_time(cur_t)} / {fmt_time(dur_t)}"
        (tw, th), _ = cv2.getTextSize(right_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(
            canvas,
            right_txt,
            (w - tw - 10, y0 + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )

    @property
    def src_name(self) -> str:
        return self.src if isinstance(self.src, str) else "video_capture"

    def _draw_hud(self, canvas: np.ndarray):
        lines = []
        src_name = self.src_name
        lines.append(
            f"{os.path.basename(src_name) if os.path.exists(src_name) else src_name}"
        )
        if not self.is_live:
            lines.append(
                f"Frame {self.cur_frame_idx+1}/{self.total_frames}  ({self.fps:.3f} fps)"
            )
        else:
            lines.append(f"Frame {self.cur_frame_idx}  ({self.fps:.3f} fps est.)")
        
        audio_status = (
            f"[audio: {self.audio.backend or 'off'}]"
            if self.audio.backend
            else ""
        )
        offset_str = (
            f"(offset: {self.state.audio_offset:+.3f}s)"
            if self.audio.backend and self.state.audio_offset != 0.0
            else ""
        )
        lines.append(
            f"{'PAUSED' if self.state.paused else 'PLAYING'}   speed {self.state.speed:.2f}x   zoom {self.state.zoom:.2f}x  {audio_status} {offset_str}"
        )
        if (
            self.state.loop_segment
            and self.state.in_frame is not None
            and self.state.out_frame is not None
        ):
            a = fmt_time(self.frame_time_s(self.state.in_frame))
            b = fmt_time(self.frame_time_s(self.state.out_frame))
            lines.append(f"LOOP segment: {a} → {b}")
        if self.state.repeat_all:
            lines.append("REPEAT: ON")
        if self.state.bookmarks:
            lines.append(
                f"Bookmarks: {len(self.state.bookmarks)} (b add, n/prev, m/next)"
            )

        x, y = 12, 26
        for i, ln in enumerate(lines):
            cv2.putText(
                canvas,
                ln,
                (x, y + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (240, 240, 240),
                2,
                cv2.LINE_AA,
            )

    def _draw_help(self, canvas: np.ndarray):
        help_lines = [
            "Controls:",
            "  Space: Play/Pause",
            "  , / . : Frame step -/+",
            "  Left/Right arrows: Seek -/+ 5s     Up/Down arrows: Seek -/+ 30s",
            "  - / = : Speed down/up             0: reset speed to 1x",
            "  1..5 : Speed presets (0.25x, 0.5x, 1x, 2x, 4x)",
            "  i: set IN     o: set OUT     p: toggle loop IN->OUT     c: clear IN/OUT",
            "  r: toggle repeat-all        u: toggle audio (if available)",
            "  [ / ] : Audio offset -/+ 0.1s      \\ : Reset audio offset to 0s",
            "  b: add bookmark     n: prev bookmark     m: next bookmark",
            "  s: snapshot PNG     e: export markers/bookmarks JSON",
            "  a/z: brightness -/+     d/x: contrast -/+     v: reset adjustments",
            "  h: toggle help      t: toggle HUD       f: fullscreen",
            "",
            "Mouse:",
            "  Left click/drag on timeline: scrub",
            "  Mouse wheel: zoom in/out",
            "  Right-drag: pan (when zoomed)",
            "  Middle click: reset zoom/pan",
            "",
            "Quit: q or ESC",
        ]
        # Semi-transparent box
        overlay = canvas.copy()
        x0, y0 = 24, 40
        line_h = 22
        box_w = 980
        box_h = line_h * (len(help_lines) + 1)
        cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)

        y = y0 + 30
        for ln in help_lines:
            cv2.putText(
                canvas,
                ln,
                (x0 + 18, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )
            y += line_h

    # ---- Actions ----

    def toggle_fullscreen(self):
        self.state.fullscreen = not self.state.fullscreen
        prop = cv2.WINDOW_FULLSCREEN if self.state.fullscreen else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(self.window, cv2.WND_PROP_FULLSCREEN, prop)

    def snapshot(self):
        if self.cur_frame is None:
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        if not self.is_live:
            t = self.frame_time_s(self.cur_frame_idx)
            name = f"{self.sidecar_base}_f{self.cur_frame_idx:08d}_{t:0.3f}s_{ts}.png"
        else:
            name = f"{self.sidecar_base}_frame{self.cur_frame_idx:08d}_{ts}.png"
        path = os.path.join(self.sidecar_dir, name)
        cv2.imwrite(path, self.cur_frame)
        print(f"[snapshot] {path}")

    def export_json(self):
        data = {
            "source": self.src_name,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "in_frame": self.state.in_frame,
            "out_frame": self.state.out_frame,
            "loop_segment": self.state.loop_segment,
            "repeat_all": self.state.repeat_all,
            "bookmarks": sorted(set(int(x) for x in self.state.bookmarks)),
            "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        name = f"{self.sidecar_base}_markers.json"
        path = os.path.join(self.sidecar_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[export] {path}")

    def add_bookmark(self):
        if self.is_live:
            return
        self.state.bookmarks.append(int(self.cur_frame_idx))
        self.state.bookmarks = sorted(set(self.state.bookmarks))

    def goto_bookmark(self, direction: int):
        if self.is_live or not self.state.bookmarks:
            return
        bms = sorted(self.state.bookmarks)
        cur = self.cur_frame_idx
        if direction > 0:
            # next
            for b in bms:
                if b > cur:
                    self.seek_frame(b, show_frame=True)
                    return
            self.seek_frame(bms[0], show_frame=True)  # wrap
        else:
            # prev
            for b in reversed(bms):
                if b < cur:
                    self.seek_frame(b, show_frame=True)
                    return
            self.seek_frame(bms[-1], show_frame=True)  # wrap

    def clear_inout(self):
        self.state.in_frame = None
        self.state.out_frame = None
        self.state.loop_segment = False

    def set_in(self):
        if self.is_live:
            return
        self.state.in_frame = int(self.cur_frame_idx)
        # keep ordering if out exists
        if (
            self.state.out_frame is not None
            and self.state.out_frame < self.state.in_frame
        ):
            self.state.out_frame = self.state.in_frame

    def set_out(self):
        if self.is_live:
            return
        self.state.out_frame = int(self.cur_frame_idx)
        if (
            self.state.in_frame is not None
            and self.state.out_frame < self.state.in_frame
        ):
            self.state.in_frame = self.state.out_frame

    def toggle_loop_segment(self):
        if self.is_live:
            return
        if self.state.in_frame is None or self.state.out_frame is None:
            self.state.loop_segment = False
            return
        if self.state.out_frame <= self.state.in_frame:
            self.state.loop_segment = False
            return
        self.state.loop_segment = not self.state.loop_segment

    def seek_seconds(self, delta_s: float):
        if self.is_live:
            return
        self.seek_time(self.frame_time_s(self.cur_frame_idx) + delta_s, show_frame=True)

    def frame_step(self, delta_frames: int):
        if self.is_live:
            return
        self.seek_frame(self.cur_frame_idx + delta_frames, show_frame=True)

    def adjust_speed(self, mul: float):
        self.state.speed = float(clamp(self.state.speed * mul, 0.05, 16.0))

    def set_speed(self, v: float):
        self.state.speed = float(clamp(v, 0.05, 16.0))

    def toggle_audio(self):
        """Toggle audio playback on/off."""
        if not self.audio.backend:
            return
        self.state.audio_enabled = not self.state.audio_enabled
        if self.state.audio_enabled and not self.state.paused:
            t = self.frame_time_s(self.cur_frame_idx)
            audio_t = t + self.state.audio_offset
            self.audio.play(audio_t)
        else:
            self.audio.pause()

    def adjust_audio_offset(self, delta: float):
        """Adjust audio offset by delta seconds (can be positive or negative)."""
        if not self.audio.backend:
            return
        self.state.audio_offset = float(self.state.audio_offset + delta)
        # Re-sync audio with new offset if playing
        if not self.state.paused and self.audio.is_playing:
            t = self.frame_time_s(self.cur_frame_idx)
            audio_t = t + self.state.audio_offset
            self.audio.seek(audio_t)

    def reset_audio_offset(self):
        """Reset audio offset to zero."""
        if not self.audio.backend:
            return
        self.state.audio_offset = 0.0
        # Re-sync audio if playing
        if not self.state.paused and self.audio.is_playing:
            t = self.frame_time_s(self.cur_frame_idx)
            self.audio.seek(t)

    def _sync_audio_with_playback(self):
        """Synchronize audio with video during playback."""
        if not self.audio.backend or not self.state.audio_enabled:
            return
        
        if self.state.paused:
            if self.audio.is_playing:
                self.audio.pause()
        else:
            if not self.audio.is_playing:
                t = self.frame_time_s(self.cur_frame_idx)
                # Apply audio offset: positive offset plays audio earlier
                audio_t = t + self.state.audio_offset
                self.audio.play(audio_t)

    # ---- Keyboard handling ----

    def _handle_key(self, k: int) -> bool:
        """
        Returns True if should quit.
        Uses waitKeyEx codes for arrows (varies by platform).
        """
        # ASCII
        if k in (27, ord("q")):  # ESC or q
            return True

        if k == ord(" "):
            self.state.paused = not self.state.paused
            # reset timing to avoid jump
            self.next_due = time.perf_counter()
            return False

        if k == ord("h"):
            self.state.show_help = not self.state.show_help
            return False

        if k == ord("t"):
            self.state.show_hud = not self.state.show_hud
            return False

        if k == ord("f"):
            self.toggle_fullscreen()
            return False

        if k == ord("r"):
            self.state.repeat_all = not self.state.repeat_all
            return False

        if k == ord("u"):
            self.toggle_audio()
            return False

        # Audio offset controls
        if k == ord("[") or k == ord("{"):
            self.adjust_audio_offset(-0.1)
            return False
        if k == ord("]") or k == ord("}"):
            self.adjust_audio_offset(+0.1)
            return False
        if k == ord("\\") or k == ord("|"):
            self.reset_audio_offset()
            return False

        if k == ord("s"):
            self.snapshot()
            return False

        if k == ord("e"):
            self.export_json()
            return False

        if k == ord("b"):
            self.add_bookmark()
            return False

        if k == ord("m"):  # next bookmark
            self.goto_bookmark(+1)
            return False

        if k == ord("n"):  # prev bookmark
            self.goto_bookmark(-1)
            return False

        if k == ord("i"):
            self.set_in()
            return False

        if k == ord("o"):
            self.set_out()
            return False

        if k == ord("p"):
            self.toggle_loop_segment()
            return False

        if k == ord("c"):
            self.clear_inout()
            return False

        # Frame stepping
        if k == ord(","):
            self.state.paused = True
            self.frame_step(-1)
            return False
        if k == ord("."):
            self.state.paused = True
            self.frame_step(+1)
            return False

        # Speed controls
        if k == ord("-") or k == ord("_"):
            self.adjust_speed(1 / 1.1)
            return False
        if k == ord("=") or k == ord("+"):
            self.adjust_speed(1.1)
            return False
        if k == ord("0"):
            self.set_speed(1.0)
            return False
        if k == ord("1"):
            self.set_speed(0.25)
            return False
        if k == ord("2"):
            self.set_speed(0.5)
            return False
        if k == ord("3"):
            self.set_speed(1.0)
            return False
        if k == ord("4"):
            self.set_speed(2.0)
            return False
        if k == ord("5"):
            self.set_speed(4.0)
            return False

        # Brightness/contrast quick tweaks
        if k == ord("a"):  # brightness down
            self.state.brightness = int(clamp(self.state.brightness - 5, -100, 100))
            return False
        if k == ord("z"):  # brightness up
            self.state.brightness = int(clamp(self.state.brightness + 5, -100, 100))
            return False
        if k == ord("d"):  # contrast down
            self.state.contrast = float(clamp(self.state.contrast / 1.05, 0.2, 4.0))
            return False
        if k == ord("x"):  # contrast up
            self.state.contrast = float(clamp(self.state.contrast * 1.05, 0.2, 4.0))
            return False
        if k == ord("v"):  # reset adjustments
            self.state.brightness = 0
            self.state.contrast = 1.0
            return False

        # Arrow keys (platform-dependent codes)
        # Common OpenCV waitKeyEx codes:
        #  - Windows: left 2424832, up 2490368, right 2555904, down 2621440
        #  - Some builds: left 81, up 82, right 83, down 84
        LEFT = {81, 2424832 & 0xFF}
        RIGHT = {83, 2555904 & 0xFF}
        UP = {82, 2490368 & 0xFF}
        DOWN = {84, 2621440 & 0xFF}
        if (k & 0xFF) in LEFT:
            self.seek_seconds(-5.0)
            return False
        if (k & 0xFF) in RIGHT:
            self.seek_seconds(+5.0)
            return False
        if (k & 0xFF) in UP:
            self.seek_seconds(-30.0)
            return False
        if (k & 0xFF) in DOWN:
            self.seek_seconds(+30.0)
            return False

        return False

    # ---- Playback loop ----

    def _handle_end_of_video(self):
        if self.is_live:
            return
        if (
            self.state.loop_segment
            and self.state.in_frame is not None
            and self.state.out_frame is not None
        ):
            self.seek_frame(self.state.in_frame, show_frame=True)
            return
        if self.state.repeat_all:
            self.seek_frame(0, show_frame=True)
            return
        # Stop at end
        self.state.paused = True
        # stay on last frame

    def step_playback_if_due(self):
        if self.state.paused or self.state.scrubbing:
            self._sync_audio_with_playback()
            return

        # Segment loop: if we pass out_frame, jump back
        if (
            (not self.is_live)
            and self.state.loop_segment
            and self.state.in_frame is not None
            and self.state.out_frame is not None
        ):
            if self.cur_frame_idx >= self.state.out_frame:
                self.seek_frame(self.state.in_frame, show_frame=True)
                return

        now = time.perf_counter()
        dt = (1.0 / self.fps) / max(0.05, self.state.speed)
        if now < self.next_due:
            return
        # Schedule next due; if we're lagging, catch up gently
        self.next_due = now + dt

        ok = self._read_one(show_if_fail=True)
        if not ok:
            self._handle_end_of_video()
        
        # Ensure audio is in sync during playback
        self._sync_audio_with_playback()

    def run(self, destroy_on_exit=True):
        # Prime timing
        self.next_due = time.perf_counter()

        try:
            while True:
                self.step_playback_if_due()

                canvas = self._compose_canvas()
                cv2.imshow(self.window, canvas)

                k = cv2.waitKeyEx(1)
                if k != -1:
                    if self._handle_key(k):
                        break
        finally:
            self.audio.cleanup()
            self.cap.release()
            if destroy_on_exit:
                cv2.destroyAllWindows()


# ---------- Entry point ----------


def main():
    ap = argparse.ArgumentParser(
        description="Sophisticated OpenCV video player with audio sync support (mouse + shortcuts)."
    )
    ap.add_argument("source", help="Video path or webcam index (e.g., 0)")
    ap.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to audio file to sync with video",
    )
    ap.add_argument(
        "--audio-offset",
        type=float,
        default=0.0,
        help="Initial audio offset in seconds (positive = audio ahead, negative = audio behind)",
    )
    ap.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Start time in seconds (file sources only).",
    )
    args = ap.parse_args()

    player = PyCVPlayer(
        args.source,
        start_frame=args.start,
        audio_path=args.audio,
        audio_offset=args.audio_offset,
    )
    print("Controls: space play/pause, arrows seek, i/o in/out, p toggle loop, r repeat, u toggle audio, +/- speed, [/] audio offset, b bookmark, h help")
    player.run()


if __name__ == "__main__":
    main()
