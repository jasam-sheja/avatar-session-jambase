# Session Start/End System Implementation Plan

## How To Use This Plan
- Work from top to bottom.
- For each Decision block, delete all unselected options before starting dependent tasks.
- Keep exactly one option per Decision block.
- Mark completed tasks by changing `[ ]` to `[x]`.

## Objective
Implement `booth_autoannotator.gui.session_startend_system` as a multi-recording annotation GUI that:
- Browses a root Avatar directory by date and recording.
- Loads one recording context at a time (videos, audio, cache, annotation docs).
- Lets annotators edit session start/end metadata.
- Saves JSON outputs under the mirrored output root structure.


## Programmer Notes
- Problem: The current annotation (Add/Del) workflow is not quite right. Given the two videos, each video should have its own session list and editing capabilities. There should be two `_session_list` for each camera, and selecting a session in either list should update the shared editor and timeline of both cameras. Start by understanding the current code, then update the plan in @file:plan.md to reflect the necessary changes to support dual session lists and editing.
- Problem: Only cam1 can be annotated in current GUI, but we have two cameras per recording. How to allow annotating both cam1 and cam2 sessions in the same interface? There should be two `_session_list` for each camera, and selecting a session in either list should update the shared editor and timeline of both cameras. Edits should be saved to the corresponding `001.json` or `002.json` based on which session list is active.
- Feature: While loading a recording a gui element should show the progress of loading the videos, audio, and cache overlayed on the window. This is especially important for the videos which can take a long time to load.

## Definition Of Done
- Directory tree is functional and supports switching recordings.
- Video playback, timeline, session list, and editor all update when recording changes.
- Editing enter/exit times works and persists to `001.json` and `002.json` in output root.
- Missing caches and missing files are handled without crashing.
- Unsaved-change behavior is implemented according to selected decision.
- App cleanly stops worker threads/audio when switching recordings and on close.

## Decision D1: Reuse Strategy For Existing GUI Code
- [x] Extract shared components to a new module (for example `gui/session_startend_components.py`) and import from both GUIs.

## Decision D2: Directory Tree Implementation
- [x] Build a custom `QTreeWidget` from scanned recording index (date -> recording).

## Decision D3: Recording Loading Policy
- [x] Scan all recording folders at startup, then load heavy assets only for selected recording.

## Decision D4: Unsaved Changes On Recording Switch
- [x] Option A (recommended): Prompt user with Save / Discard / Cancel dialog.

## Decision D5: Save Granularity
- [x] Save only changed session data by patching JSON.

## Decision D6: Background Loading Mechanism
- [x] Use `QThread` worker for loading docs/caches and then bind results in main thread.

## Phase 0 - Freeze Interfaces And Data Contracts
- [x] Confirm the folder contract for avatar/cache/output roots.
- [x] Define `RecordingKey` and `RecordingPaths` structures (date, recording_id, file paths).
- [x] Define app state contract: current recording key, loaded docs, dirty flag, selected session index.
- [x] Define conversion and helper contracts (time formatting, path mapping output/cache from recording key).

Acceptance criteria:
- One place in code documents path mapping rules.
- Invalid recording folders are skipped with log warnings.

## Phase 1 - Build Recording Index Layer
- [x] Implement scanner for root recording directory: discover `YYYY-MM-DD/recording-XXX`.
- [x] Validate required media files per recording (`stream-001.mp4`, `stream-002.mp4`, `audio-001.wav`, `audio-002.wav`).
- [x] Build index object used by tree UI and load operations.
- [x] Add basic metrics logging (total dates, total recordings, invalid entries).

Acceptance criteria:
- Index produces stable sorted order (date ascending, recording ascending).
- App can start even when some recordings are incomplete.

## Phase 2 - Build Main Window Shell
- [x] Create left pane container for directory tree.
- [x] Create right pane container with video area, timeline area, session list, and editor.
- [x] Add status bar messages for startup, loading, loaded, and error states.
- [x] Wire top-level actions: save, reload index, jump next/previous recording (if selected).

Acceptance criteria:
- Shell renders even before any recording is selected.
- No hard crash when roots are empty.

## Phase 3 - Integrate Core Annotation Components
Depends on Decision D1.

- [x] Reuse or extract `VideoThread`.
- [x] Reuse or extract `VideoLabel`.
- [x] Reuse or extract `TimelineWidget`.
- [x] Reuse or extract `SessionEditorWidget`.
- [x] Keep API compatibility with current signal/slot usage where possible.

Acceptance criteria:
- Components compile/import cleanly in both GUIs (if shared module approach selected).
- Existing single-recording GUI behavior remains unchanged.

## Phase 4 - Recording Context Loader And Switch Lifecycle
Depends on Decisions D3 and D6.

- [x] Implement `load_recording_context(recording_key)` that resolves paths and loads:
  - annotation docs (`001.json`, `002.json`) from output root,
  - cache data from cache root,
  - video and audio backends.
- [x] Implement safe unload of previous context:
  - stop video threads,
  - release captures,
  - cleanup audio manager,
  - clear widgets/state.
- [x] Implement guarded switch flow with unsaved-change policy (Decision D4).

Acceptance criteria:
- Switching between recordings multiple times does not leak threads or stale frames.
- Timeline/session list always reflect the active recording only.

## Phase 5 - Directory Tree Behavior
Depends on Decision D2.

- [x] Populate tree with date parent nodes and recording child nodes.
- [x] Store `RecordingKey` in each selectable recording item.
- [x] On selection, trigger recording switch flow.
- [x] Visually indicate current recording and loading state.

Acceptance criteria:
- Tree selection reliably loads matching recording.
- Invalid/missing recording entries are disabled or visually marked.

## Phase 6 - Session Editing And Save Flow
Depends on Decision D5.

- [x] Populate session list from active `doc1.sessions`.
- [x] Keep editor and timeline synced with selected session.
- [x] Apply enter/exit edits as manual `AnnotationValue` with confidence and method metadata.
- [x] Mark dirty state on edits and refresh session list status text/icon.
- [x] Save active recording JSON to output root mirrored path.
- [x] If auto-save is enabled, save after each edit.

Acceptance criteria:
- Edits persist after app restart.
- Save failures are surfaced in status bar and logs with actionable message.

## Phase 7 - Robustness, UX, And Keyboard Shortcuts
- [x] Ensure keyboard shortcuts work with focus in tree/list/editor widgets.
- [x] Add user feedback for missing cache files (non-fatal).
- [x] Add load progress indicator (status text at minimum).
- [x] Ensure close event blocks or prompts on unsaved changes per Decision D4.

Acceptance criteria:
- No unhandled exceptions during normal navigation/edit/save flows.
- User can recover from missing data files without restarting app.

## Phase 9 - Dual Session Lists Per Camera

**Problem being solved:** The current GUI only exposes `doc1.sessions` via a single `_session_list`. Cam2 sessions (`doc2`) cannot be viewed or edited, and all edits — including Add/Delete — are routed only to `doc1`. This phase replaces the single session list with two per-camera lists and routes editing and saving to the correct document based on which list is active.

### State changes

- Add `_active_cam: int = 0` to `MainWindow.__init__`. Tracks which camera's list last had focus (0 = cam1, 1 = cam2).
- Add `_active_doc` property: returns `self._doc1` if `_active_cam == 0` else `self._doc2`.
- Replace `_selected_idx: int` with `_selected_idx: List[int] = [-1, -1]` (one per camera).
- Replace `_changed_session_ids: Set[str]` and `_deleted_session_ids: Set[str]` with per-camera lists:
  `_changed_session_ids: List[Set[str]] = [set(), set()]`
  `_deleted_session_ids: List[Set[str]] = [set(), set()]`

### UI changes

- [x] In the session column, replace the single `_session_list` with two stacked list widgets:
  - `_session_list_cam1: QListWidget` labelled "Cam 1 Sessions" with its own "+ Add / − Del" buttons.
  - `_session_list_cam2: QListWidget` labelled "Cam 2 Sessions" with its own "+ Add / − Del" buttons.
  - The shared `SessionEditorWidget` remains below (or beside) both lists.
- [x] Visually highlight the active camera's list header (bold label or border).

### Logic changes

- [x] `_sessions(cam: int) -> List[Session]`: helper that returns `doc1.sessions` or `doc2.sessions` by cam index. Replace all uses of `_sessions()` throughout the file.
- [x] `_populate_session_list()`: call a new `_populate_session_list_for(cam)` for each cam that fills `_session_list_cam1` or `_session_list_cam2`.
- [x] On selection in `_session_list_cam1`: set `_active_cam = 0`, update `_selected_idx[0]`, load session into shared editor, update timeline selection, seek to session start.
- [x] On selection in `_session_list_cam2`: set `_active_cam = 1`, update `_selected_idx[1]`, load session into shared editor, update timeline selection, seek to session start.
- [x] Cross-list deselection: when cam1 list gets a selection, clear cam2 list selection, and vice versa. Use `_suppress_list_signal` flag or `blockSignals()` to avoid re-entrancy.
- [x] `_current_session()`: return `_active_doc.sessions[_selected_idx[_active_cam]]` if in range, else `None`.
- [x] `_index_for_session_id(sid, cam)`: search in `_sessions(cam)`.
- [x] `_add_session()`: add a new `Session` only to `_active_doc.sessions` at the current playhead. Do NOT mirror to the other doc automatically. Refresh only the active cam's list.
- [x] `_remove_session()`: remove from `_active_doc.sessions` only. Refresh only the active cam's list.
- [x] `_mark_session_changed(sid)`: route to `_changed_session_ids[_active_cam]`, discard from `_deleted_session_ids[_active_cam]`.
- [x] `_mark_session_deleted(sid)`: route to `_deleted_session_ids[_active_cam]`, discard from `_changed_session_ids[_active_cam]`.
- [x] `_refresh_list_item(idx)`: refresh the item in the active cam's list widget.
- [x] `_next_session_id()`: check IDs in both `doc1.sessions` and `doc2.sessions` to avoid collisions.
- [x] `_prev_session` / `_next_session`: navigate within the active cam's list.
- [x] `_jump_to_session_start` / `_jump_to_session_end`: use `_current_session()` (already routed via active cam after the above changes).

### Timeline changes

- [x] `_on_recording_loaded`: pass sessions from **both** docs to the timeline. The timeline already supports rendering sessions from a flat list; extend `TimelineWidget` to accept two lists and render cam1 sessions on rows 0-2 and cam2 sessions on rows 3-5.
- [x] `_on_timeline_session_modified(session_id, field_name, value)`: determine which doc owns `session_id` (check both), set `_active_cam` accordingly, then call `_mark_session_changed(session_id)`.
- [x] `_on_session_selected_by_id(session_id)`: search both docs, set `_active_cam` to the owning cam, update the correct list row.

### Save changes

- [x] `_save_all()`: save `doc1` using `_changed_session_ids[0]` / `_deleted_session_ids[0]` to `output_json_paths[0]`, and `doc2` using `_changed_session_ids[1]` / `_deleted_session_ids[1]` to `output_json_paths[1]`. Clear each set independently.
- [x] `_dirty` flag remains global (True if either cam has changes).

### Unload / reset changes

- [x] `_unload_runtime`: clear both session lists, reset `_selected_idx = [-1, -1]`, clear both changed/deleted sets.

### Acceptance criteria
- Cam1 list and Cam2 list each show sessions from their respective docs independently.
- Selecting a session in either list loads it into the shared editor and highlights it on the timeline.
- Adding/removing a session in cam1 does not affect cam2's list, and vice versa.
- Save writes each doc's patch to the correct JSON file (`001.json` for cam1, `002.json` for cam2).
- Dirty flag and unsaved-change prompt work correctly for edits to either camera.
- No regressions: switching recordings, undo/redo, keyboard shortcuts all continue working.

## Phase 10 - Sync Video Playhead With Timeline Annotations

**Problem being solved:** Session enter/exit times are stored in *local video time* (seconds from the start of that stream), but the timeline operates in *wall-clock time* (a shared global reference starting at `t = 0` for the earliest stream). When the two streams have a non-zero relative offset (`doc.video.time_stamp` differs between cameras), sessions from the later stream appear shifted left on the timeline and the playhead no longer lands on the correct frame when jumping to a session.

The offset helper `_video_offset(doc) → float` already exists in `MainWindow` and returns `(doc.video.time_stamp - main_doc.video.time_stamp) / 1000.0` in seconds.

### Reference frames (must stay consistent after this phase)

| Value | Stored in | Reference frame |
|---|---|---|
| `session.times.enter_time.t` | JSON / doc | **Local video time** (unchanged — do not rewrite JSONs) |
| `session.times.exit_time.t` | JSON / doc | **Local video time** |
| `TimelineWidget._playhead` | in-memory | **Wall-clock time** |
| `_seek(t)` argument | in-memory | **Wall-clock time** |
| Manual edits written back to session | JSON / doc | **Local video time** (convert before storing) |

### State / API changes

- [x] Add `_cam_offsets: List[float] = [0.0, 0.0]` to `MainWindow.__init__` — cam1 and cam2 offsets in seconds relative to the earliest stream.
- [x] Populate `_cam_offsets` in `_on_recording_loaded` after docs are loaded:
  ```python
  self._cam_offsets[0] = self._video_offset(self._doc1) if self._doc1 else 0.0
  self._cam_offsets[1] = self._video_offset(self._doc2) if self._doc2 else 0.0
  ```
- [x] Reset `_cam_offsets = [0.0, 0.0]` in `_unload_runtime`.
- [x] Add a read-only `_active_offset` property: `return self._cam_offsets[self._active_cam]`.

### TimelineWidget changes

- [x] Add `_offsets: List[float] = [0.0, 0.0]` field to `TimelineWidget.__init__`.
- [x] Add `set_offsets(cam1_offset: float, cam2_offset: float)` method that stores both offsets and calls `self.update()`.
- [x] In `paintEvent`, apply the per-cam offset when converting session time to x-coordinate:
  ```python
  for cam_sessions, base_row, offset in (
      (self._cam1_sessions, 0, self._offsets[0]),
      (self._cam2_sessions, 3, self._offsets[1]),
  ):
      for session in cam_sessions:
          enter_t = session.times.enter_time.t + offset if session.times.enter_time else None
          exit_t  = session.times.exit_time.t  + offset if session.times.exit_time  else None
          ...
  ```
- [x] In `_session_at(x)`, compute the candidate time `t = self._x_to_t(x)`, then for each session subtract the cam offset before comparing with enter/exit:
  ```python
  local_t = t - offset
  if enter_t <= local_t <= (exit_t or duration):
      return session
  ```
  Alternatively, compare `t` directly against the offset-shifted enter/exit (simpler).
- [x] In the context-menu and drag handlers that write back to `session.times`, subtract the cam offset before storing so the value stays in local video time:
  ```python
  local_t = self._x_to_t(x) - cam_offset_for_this_session
  session.times.enter_time = AnnotationValue(t=local_t, source="manual", ...)
  ```
- [x] Update `set_sessions` call in `_on_recording_loaded` to also call `self._timeline.set_offsets(self._cam_offsets[0], self._cam_offsets[1])`.

### MainWindow seek/jump changes

- [x] `_jump_to_session_start`: seek to `session.times.enter_time.t + self._active_offset` (wall-clock).
- [x] `_jump_to_session_end`: seek to `session.times.exit_time.t + self._active_offset` (wall-clock).
- [x] `_add_session`: when the user adds a session at the current playhead `_playhead`, store the enter time as `_playhead - self._active_offset` (local time). Confirm this is already the case or fix.
- [x] `_on_timeline_session_modified`: the `value` emitted from the timeline is already in wall-clock time after the timeline fix above. Before writing it to the session, convert to local time:
  ```python
  local_t = value - self._cam_offsets[owner_cam]
  ```

### SessionEditorWidget changes

- [x] The editor displays `session.times.enter_time.t` and `exit_time.t` which are in local time. Decide on display convention:
  - Keep displaying local time (no change to editor widget itself). The wall-clock offset is only relevant for the shared timeline and seek operations.
- [x] Ensure that when the user edits times in the `SessionEditorWidget` the emitted signal and the value written back are still in local time (currently correct).

### Verification checklist

- [x] Record with non-zero offset (`doc1.video.time_stamp != doc2.video.time_stamp`): session bars on timeline align with the playhead when playing.
- [x] Clicking a session bar on the timeline seeks the correct video frame.
- [x] "Add session at playhead" inserts a session whose enter_time is in local video coordinates (verify by checking the saved JSON).
- [x] Dragging a session bar on the timeline updates the stored local time correctly.
- [x] Recording with zero offset (both streams have same timestamp): no regression.
- [x] `_unload_runtime` → reload: offsets reset and recomputed correctly.

### Acceptance criteria
- Timeline session bars visually align with the video playhead for both cameras at all times.
- Session times in the saved JSON remain in local video time (no format change).
- Zero-offset recordings continue to work identically to pre-phase behaviour.

## Phase 8 - Verification And Test Coverage
- [ ] Add unit tests for path mapping and index scanning logic.
- [ ] Add tests for mirrored output/cache path resolution.
- [ ] Add smoke test checklist for manual QA:
  - start app with real dataset root,
  - switch among at least 5 recordings,
  - edit and save at least 3 sessions,
  - restart app and verify persistence.
- [ ] Add regression check for single-recording GUI to ensure no breakage.

Acceptance criteria:
- Core utility tests pass.
- Manual smoke test checklist is reproducible and documented.

## Implementation Order (Recommended)
- [x] Resolve Decisions D1-D6 first.
- [x] Complete Phases 0-2.
- [x] Complete Phase 3.
- [x] Complete Phases 4-6.
- [x] Finish Phase 7.
- [ ] Complete Phase 9 (dual session lists).
- [x] Complete Phase 10 (timeline offset sync).
- [ ] Finish Phase 8 (tests).

## Notes For Programmer
- If Decision D1 is Option B, prioritize extracting shared code before adding new features.
- Prefer small, reviewable commits by phase.
- Keep logging concise but specific (include recording key in messages).
- Avoid changing annotation schema unless explicitly required.