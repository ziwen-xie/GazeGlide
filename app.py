"""Local webcam eye scrolling for Windows. Camera frames are never saved."""
import ctypes
from ctypes import wintypes
import os
import sys
from pathlib import Path
import time
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk
from control import ScrollControl
from gaze import GazeMap, TARGETS, head_features, head_adjustment

ROOT = Path(__file__).resolve().parent
# Some Windows Python installations omit Tcl/Tk's bundled search path in venvs.
for variable, folder, marker in [('TCL_LIBRARY', 'tcl8.6', 'init.tcl'), ('TK_LIBRARY', 'tk8.6', 'tk.tcl')]:
    bundled = Path(sys.base_prefix) / 'tcl' / folder
    if (bundled / marker).is_file():
        os.environ.setdefault(variable, str(bundled))
user32 = ctypes.WinDLL('user32', use_last_error=True)
try:
    user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
except AttributeError:
    user32.SetProcessDPIAware()
kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
user32.GetForegroundWindow.restype = wintypes.HWND
kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
BROWSERS = {'chrome.exe', 'msedge.exe', 'firefox.exe', 'brave.exe', 'opera.exe', 'vivaldi.exe'}

def monitors():
    displays = []
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HANDLE, wintypes.HDC, ctypes.POINTER(wintypes.RECT), wintypes.LPARAM)
    def collect(handle, dc, rect, data):
        r = rect.contents
        displays.append((r.left, r.top, r.right-r.left, r.bottom-r.top))
        return True
    callback = callback_type(collect)
    user32.EnumDisplayMonitors(None, None, callback, 0)
    return sorted(displays, key=lambda r: (r[0], r[1]))

def position_window(window, rect):
    window.update_idletasks()
    user32.GetParent.argtypes = [wintypes.HWND]
    user32.GetParent.restype = wintypes.HWND
    hwnd = user32.GetParent(window.winfo_id()) or window.winfo_id()
    user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.UINT]
    user32.SetWindowPos(hwnd, wintypes.HWND(-1), *map(int, rect), 0x10)
    return hwnd

def browser_window():
    hwnd = user32.GetForegroundWindow()
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    handle = kernel32.OpenProcess(0x1000, False, pid.value)
    if not handle:
        return None
    try:
        size = wintypes.DWORD(4096)
        name = ctypes.create_unicode_buffer(size.value)
        if kernel32.QueryFullProcessImageNameW(handle, 0, name, ctypes.byref(size)):
            if os.path.basename(name.value).lower() in BROWSERS:
                return hwnd
    finally:
        kernel32.CloseHandle(handle)
    return None

def gaze_value(landmarks):
    # Iris position projected along each eye's vertical axis; independent of image size.
    values = []
    for iris, top, bottom, left, right in [(468, 159, 145, 33, 133), (473, 386, 374, 362, 263)]:
        def xy(i):
            return np.array([landmarks[i].x, landmarks[i].y])
        axis = xy(bottom) - xy(top)
        width = np.linalg.norm(xy(right) - xy(left))
        opening = np.linalg.norm(axis)
        if width <= 0 or opening / width < 0.13:
            return None  # Closed eyes: stop immediately.
        # Eyelids move with gaze, so measuring against them can cancel the signal.
        # Use the stable eye corners and a perpendicular axis instead.
        horizontal = (xy(right) - xy(left)) / width
        vertical = np.array([-horizontal[1], horizontal[0]])
        if vertical[1] < 0:
            vertical = -vertical
        midpoint = (xy(left) + xy(right)) / 2
        if horizontal[0] < 0:
            horizontal = -horizontal
        offset = xy(iris) - midpoint
        values.append([float(np.dot(offset, horizontal) / width), float(np.dot(offset, vertical) / width)])
    return np.mean(values, axis=0)

class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('GazeGlide')
        self.root.geometry('680x850')
        self.root.attributes('-topmost', True)
        self.root.after(1500, lambda: self.root.attributes('-topmost', False))
        self.cap = None
        self.detector = None
        self.control = None
        self.gaze_map = None
        self.head_reference = None
        self.latest_head = None
        self.latest_eyes = None
        self.eye_bias = np.zeros(2)
        self.calibrated_screen = None
        self.displays = monitors()
        self.overlay = None
        self.enabled = False
        self.filtered = None
        self.cal = None
        self.keys = {}
        self.preview = ttk.Label(self.root)
        self.status = tk.StringVar(value='Connect your webcam, then start the camera.')
        ttk.Label(self.root, text='GazeGlide', font=('Segoe UI', 24)).pack(pady=12)
        ttk.Label(self.root, text='Look up to scroll up. Look down to scroll down.\nLook at the middle to stop.', justify='center').pack()
        row = ttk.Frame(self.root)
        row.pack(pady=12)
        ttk.Label(row, text='Camera index:').pack(side='left')
        self.camera = tk.StringVar(value='0')
        ttk.Spinbox(row, from_=0, to=9, width=4, textvariable=self.camera).pack(side='left', padx=8)
        ttk.Button(row, text='Start camera', command=self.start).pack(side='left')
        screen_row = ttk.Frame(self.root)
        screen_row.pack(pady=4)
        ttk.Label(screen_row, text='Calibration screen:').pack(side='left', padx=6)
        self.screen_select = ttk.Combobox(screen_row, state='readonly', width=38,
            values=[f'Screen {i+1}: {w} x {h} at ({x}, {y})' for i, (x,y,w,h) in enumerate(self.displays)])
        self.screen_select.pack(side='left')
        self.screen_select.current(0)
        self.screen_select.bind('<<ComboboxSelected>>', self.screen_changed)
        self.show_region = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.root, text='Show transparent gaze square (after calibration)', variable=self.show_region,
            command=self.refresh_overlay).pack(pady=4)
        self.use_head = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.root, text='Include head movement (experimental)', variable=self.use_head,
            command=self.reset_motion).pack()
        head_row = ttk.Frame(self.root)
        head_row.pack(pady=3)
        ttk.Label(head_row, text='Head sensitivity:').pack(side='left')
        self.head_strength = tk.DoubleVar(value=1.)
        ttk.Scale(head_row, from_=0, to=2, variable=self.head_strength, length=180).pack(side='left', padx=8)
        ttk.Button(head_row, text='Recenter at current head position', command=self.recenter_head).pack(side='left')
        self.head_status = tk.StringVar(value='Head tracking: waiting for camera')
        ttk.Label(self.root, textvariable=self.head_status).pack()
        ttk.Label(self.root, text='To recenter: look at screen center, then click Recenter.').pack()
        ttk.Button(self.root, text='Calibrate (F8)', command=self.calibrate).pack(pady=4)
        ttk.Button(self.root, text='Enable / pause (F9)', command=self.toggle).pack(pady=4)
        ttk.Label(self.root, textvariable=self.status, wraplength=520, justify='center').pack(pady=12)
        ttk.Label(self.root, text='F10: stop and quit globally • F9: pause globally\nScrolling runs only when a supported browser is active.').pack()
        self.preview.pack(pady=8)
        ttk.Label(self.root, text='Keep both eyes visible. Move your eyes toward each dot during calibration.').pack()
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.root.after(30, self.tick)

    def reset_motion(self):
        self.filtered = None
        self.enabled = False
        if self.control:
            self.control.reset()
        self.status.set('Head setting changed. Scrolling paused; F9 resumes.')

    def recenter_head(self):
        if self.gaze_map is None or self.latest_head is None or self.latest_eyes is None:
            self.status.set('Calibrate and make sure both eyes are visible first.')
            return
        self.head_reference = self.latest_head.copy()
        self.eye_bias = self.gaze_map.mean - self.latest_eyes
        self.reset_motion()
        self.status.set('Current gaze set to screen center. Scrolling paused. Look at screen center when recentering; F9 resumes.')

    def screen_changed(self, event=None):
        self.cancel_cal()
        self.enabled = False
        self.control = self.gaze_map = None
        self.hide_overlay()
        self.status.set('Screen changed. Calibrate this screen before tracking.')

    def hide_overlay(self):
        if self.overlay:
            self.overlay.withdraw()

    def refresh_overlay(self):
        if not self.show_region.get():
            self.hide_overlay()

    def draw_region(self, point):
        if not self.show_region.get() or self.cal or self.calibrated_screen is None:
            self.hide_overlay()
            return
        if self.overlay is None:
            self.overlay = tk.Toplevel(self.root)
            self.overlay.withdraw()
            self.overlay.overrideredirect(True)
            self.overlay.configure(bg='#ff00ff')
            self.overlay.attributes('-transparentcolor', '#ff00ff')
            canvas = tk.Canvas(self.overlay, width=180, height=180, bg='#ff00ff', highlightthickness=0)
            canvas.pack(fill='both', expand=True)
            canvas.create_rectangle(4, 4, 176, 176, outline='#00d4aa', width=4)
            hwnd = position_window(self.overlay, (0, 0, 180, 180))
            user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
            user32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
            # Layered + transparent + no activation + tool window: clicks pass through.
            style = user32.GetWindowLongW(hwnd, -20)
            user32.SetWindowLongW(hwnd, -20, style | 0x80000 | 0x20 | 0x08000000 | 0x80)
        x, y, width, height = self.calibrated_screen
        left = x + int(np.clip(point[0]*width-90, 0, max(0,width-180)))
        top = y + int(np.clip(point[1]*height-90, 0, max(0,height-180)))
        self.overlay.deiconify()
        position_window(self.overlay, (left, top, 180, 180))

    def start(self):
        if self.cap is not None:
            return
        cap = None
        try:
            cap = cv2.VideoCapture(int(self.camera.get()), cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ok, _ = cap.read()
            if not ok:
                raise RuntimeError('Camera unavailable. Close other camera apps or try index 1 or 2.')
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(ROOT / 'models/face_landmarker.task')),
                running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=1,
                output_facial_transformation_matrixes=True,
                min_face_detection_confidence=0.6, min_tracking_confidence=0.6)
            self.detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)
            self.cap = cap
            self.status.set('Camera ready. Click Calibrate. Sit facing the camera in good light.')
        except Exception as exc:
            if cap is not None:
                cap.release()
            messagebox.showerror('Camera', str(exc))

    def toggle(self):
        if self.cal:
            return
        if self.control is None:
            self.status.set('Start the camera and calibrate first.')
            return
        self.enabled = not self.enabled
        self.control.reset()
        self.status.set('Enabled. Switch to your browser. F9 pauses.' if self.enabled else 'Paused. F9 enables scrolling.')

    def calibrate(self):
        if self.cap is None or self.cal:
            self.status.set('Start the camera first.')
            return
        self.enabled = False
        self.control = None
        self.gaze_map = None
        self.filtered = None
        self.hide_overlay()
        screen = self.displays[self.screen_select.current()]
        window = tk.Toplevel(self.root)
        window.overrideredirect(True)
        window.attributes('-topmost', True)
        position_window(window, screen)
        window.focus_force()
        canvas = tk.Canvas(window, bg='#101827', highlightthickness=0)
        canvas.pack(fill='both', expand=True)
        self.cal = dict(window=window, canvas=canvas, screen=screen, stage=0, start=time.monotonic(), samples=[], values=[], heads=[])
        window.bind('<Escape>', lambda e: self.cancel_cal())
        window.protocol('WM_DELETE_WINDOW', self.cancel_cal)

    def cancel_cal(self):
        if self.cal:
            self.cal['window'].destroy()
            self.cal = None
            self.status.set('Calibration cancelled. Scrolling paused.')

    def calibration_frame(self, value, now, head=None):
        cal = self.cal
        elapsed = now - cal['start']
        canvas = cal['canvas']
        width, height = canvas.winfo_width(), canvas.winfo_height()
        target = TARGETS[cal['stage']]
        x, y = target[0]*width, target[1]*height
        canvas.delete('all')
        remaining = max(0., 5-elapsed)
        canvas.create_oval(x-38, y-38, x+38, y+38, fill='#5eead4', outline='')
        canvas.create_arc(x-44, y-44, x+44, y+44, start=90, extent=-360*remaining/5,
            style='arc', outline='white', width=4)
        canvas.create_text(x, y, text=f'{remaining:.1f}', fill='#101827', font=('Segoe UI', 20, 'bold'), tags='dot_timer')
        canvas.create_text(width/2, height*0.32, fill='white', font=('Segoe UI', 22),
            text=f"Look at the dot ({cal['stage']+1}/9). Keep your head still.\n{'Settling' if elapsed <= 2 else 'Measuring'} • Esc cancels")
        if elapsed > 2 and value is not None and head is not None:
            cal['samples'].append(value)
            cal['heads'].append(head)
        if elapsed >= 5:
            if len(cal['samples']) < 15:
                self.cancel_cal()
                self.status.set('Could not see open eyes clearly. Improve lighting and calibrate again.')
                return
            median = np.median(cal['samples'], axis=0)
            cal['values'].append(median)
            cal['samples'] = []
            cal['stage'] += 1
            cal['start'] = now
            if cal['stage'] == 9:
                values = cal['values']
                screen = cal['screen']
                reference = np.median(cal['heads'], axis=0)
                self.cancel_cal()
                try:
                    self.gaze_map = GazeMap(values)
                    self.head_reference = reference
                    self.eye_bias = np.zeros(2)
                    self.calibrated_screen = screen
                    self.control = ScrollControl(.15, .5, .85)
                    self.filtered = None
                    self.status.set('Calibrated. The square shows estimated gaze. F9 enables browser scrolling; tracking works while paused.')
                except ValueError as exc:
                    self.status.set(str(exc))

    def tick(self):
        for key, action in [(0x77, self.calibrate), (0x78, self.toggle), (0x79, self.close)]:
            pressed = bool(user32.GetAsyncKeyState(key) & 0x8000)
            if pressed and not self.keys.get(key, False):
                action()
                if key == 0x79:
                    return
            self.keys[key] = pressed
        now = time.monotonic()
        value = None
        head = None
        try:
            if self.cap is not None:
                ok, frame = self.cap.read()
                if ok:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = self.detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(now*1000))
                    if result.face_landmarks:
                        marks = result.face_landmarks[0]
                        value = gaze_value(marks)
                        if result.facial_transformation_matrixes:
                            center = [(marks[33].x+marks[263].x)/2, (marks[33].y+marks[263].y)/2]
                            width = np.hypot(marks[33].x-marks[263].x, marks[33].y-marks[263].y)
                            head = head_features(result.facial_transformation_matrixes[0], center, width)
                        for index in (468, 473):
                            mark = result.face_landmarks[0][index]
                            cv2.circle(rgb, (int(mark.x*rgb.shape[1]), int(mark.y*rgb.shape[0])), 3, (50, 255, 100), -1)
                    image = Image.fromarray(cv2.flip(rgb, 1))
                    image.thumbnail((360, 270))
                    self.preview_image = ImageTk.PhotoImage(image)
                    self.preview.configure(image=self.preview_image)
                self.latest_head, self.latest_eyes = head, value
                self.head_status.set('Head unavailable' if head is None else
                    f'Head yaw: {np.degrees(head[0]):+.0f}°   pitch: {np.degrees(head[1]):+.0f}°   sensitivity: {self.head_strength.get():.1f}')
                if self.cal:
                    self.calibration_frame(value, now, head)
                else:
                    if value is not None:
                        value = value + self.eye_bias
                        if self.use_head.get() and self.head_reference is not None:
                            correction = head_adjustment(head, self.head_reference, self.head_strength.get()) if head is not None else None
                            value = None if correction is None else value + correction
                    if value is None:
                        self.filtered = None
                    else:
                        self.filtered = value if self.filtered is None else 0.25*value + 0.75*self.filtered
                    hwnd = browser_window() if self.enabled else None
                    point = self.gaze_map.predict(self.filtered) if self.gaze_map and self.filtered is not None else None
                    if point is not None:
                        self.draw_region(point)
                    else:
                        self.hide_overlay()
                    browser_on_screen = False
                    if hwnd and self.calibrated_screen:
                        rect = wintypes.RECT()
                        user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
                        user32.GetWindowRect(hwnd, ctypes.byref(rect))
                        sx, sy, sw, sh = self.calibrated_screen
                        browser_on_screen = sx <= (rect.left+rect.right)/2 < sx+sw and sy <= (rect.top+rect.bottom)/2 < sy+sh
                    if self.control:
                        direction = self.control.step(None if point is None else point[1], now, self.enabled and browser_on_screen)
                        if direction:
                            # WM_MOUSEWHEEL, one notch; sent only to the active browser.
                            sx, sy, sw, sh = self.calibrated_screen
                            gx, gy = int(sx + point[0]*sw), int(sy + point[1]*sh)
                            coords = (gx & 0xffff) | ((gy & 0xffff) << 16)
                            user32.PostMessageW(hwnd, 0x020A, ((direction*120) & 0xffff) << 16, coords)
                    if self.enabled:
                        state = 'Eyes not visible — stopped' if value is None else ('Browser outside calibrated screen or inactive — stopped' if not browser_on_screen else 'Tracking — ' + {0:'hold', 1:'up', -1:'down'}[self.control.direction])
                        self.status.set(state + '. F9 pauses; F10 quits.')
        except Exception as exc:
            self.enabled = False
            self.hide_overlay()
            if self.control:
                self.control.reset()
            self.cancel_cal()
            self.status.set(f'Paused after error: {exc}')
        self.root.after(30, self.tick)

    def close(self):
        self.enabled = False
        if self.cap is not None:
            self.cap.release()
        if self.detector is not None:
            self.detector.close()
        self.root.destroy()

if __name__ == '__main__':
    App().root.mainloop()
