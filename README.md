# GazeGlide

**Browse at a glance.**

Windows webcam prototype for hands-free browser scrolling. Frames are processed locally in memory, never recorded or uploaded. Model weights come from Google's MediaPipe distribution.

## Setup

Requires Windows 10/11, Python 3.13, and a webcam recognized by Windows.

Clone or download this repository, then double-click `setup.bat` to install dependencies and download the face landmark model. Once setup finishes, double-click `start.bat`.

Alternatively, run these commands from the project folder in PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_model.py
.\.venv\Scripts\python.exe app.py
```

The model and virtual environment are downloaded during setup and are excluded from Git. Setup needs internet access; camera processing runs locally.

## Use

1. Double-click `start.bat`.
2. Click **Start camera**. If it cannot open your USB webcam, close other camera apps and try camera index 1 or 2.
3. Select the monitor in **Calibration screen**, then click **Calibrate**. Look directly at each of the nine dots for five seconds while keeping your head still. Calibration lasts 45 seconds and appears on your selected monitor.
   After calibration, a square with a transparent center follows your estimated gaze region on that display, even while scrolling is paused. Use **Show transparent gaze square** to hide it. Clicks pass through the square. Changing screens requires recalibration. The webcam does not automatically identify which monitor you are looking at.
   Each dot contains a countdown and a progress ring. The first two seconds allow your gaze to settle; the final three collect measurements.
4. Open a long browser page. Place your mouse pointer over its scrolling content, then press **F9**.
5. Look toward the bottom to scroll down, toward the top to scroll up. Looking at the center stops. Scrolling starts after about 0.7 seconds of sustained gaze.
6. **F9** pauses/resumes, **F8** recalibrates, **F10** exits. These keys work while the browser is active. Esc cancels calibration.

Supported process names: Chrome, Edge, Firefox, Brave, Opera, Vivaldi. Scrolling stops when another app becomes active, the active browser's center lies outside the calibrated monitor, or open eyes are no longer detected. It starts paused and requires calibration each launch. Browser wheel-event handling and nested scroll areas can vary; real browser testing is needed.

Mount the camera near the selected screen center, face a light source, and keep your seating position steady. Recalibrate if you move. A nine-point affine mapping converts iris offsets to approximate screen coordinates; the 180-pixel square indicates an estimated region, not measured accuracy. Calibration rejects weak or inconsistent samples using held-out target prediction. An ordinary webcam estimates iris position; this prototype cannot promise precise gaze tracking. Glasses, reflections, low resolution, and head movement may affect results. No driver-specific integration is needed if Windows recognizes the webcam.

## Head movement

**Include head movement** adds an experimental adjustment based on the face transformation matrix (yaw/pitch) and face position relative to calibration. **Head sensitivity** controls its strength; zero removes the adjustment. The square responds to combined eye and head measurements. This is a local approximation with heuristic gains, not a calibrated 3D gaze ray. Head motion while your eyes hold a fixed screen target may still cause drift. Calibration should still be done with a steady head.

To change seating position, look at the selected screen's center and click **Recenter at current head position**. This resets the head reference and centers the current eye measurement, and pauses scrolling. Large head turns or distance changes stop tracking until you return or recenter. For more reliable head movement handling, collect extra calibration samples at known targets with varied head poses, or calibrate camera/screen geometry and intersect a combined eye/head direction ray with the screen.

## Development

Python 3.13 was used for setup. Dependencies are pinned in `requirements.txt`.

Run `download_model.py` to download the model from Google's MediaPipe distribution to `models/face_landmarker.task` if missing.

Run `.venv\Scripts\python.exe -m unittest -v` for dwell, direction, tracking-loss, and calibration checks.

MediaPipe API reference: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python
