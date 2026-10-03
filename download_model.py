"""Download the MediaPipe model from Google's official distribution."""
from pathlib import Path
from urllib.request import urlretrieve
import zipfile

MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task'
MODEL_PATH = Path(__file__).resolve().parent / 'models' / 'face_landmarker.task'

def main():
    if MODEL_PATH.is_file() and zipfile.is_zipfile(MODEL_PATH):
        print('Face landmark model is ready.')
        return
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = MODEL_PATH.with_suffix('.download')
    try:
        print('Downloading face landmark model from Google...')
        urlretrieve(MODEL_URL, temporary)
        if not zipfile.is_zipfile(temporary):
            raise RuntimeError('Downloaded file is not a valid MediaPipe model bundle.')
        temporary.replace(MODEL_PATH)
        print('Face landmark model is ready.')
    finally:
        temporary.unlink(missing_ok=True)

if __name__ == '__main__':
    main()
