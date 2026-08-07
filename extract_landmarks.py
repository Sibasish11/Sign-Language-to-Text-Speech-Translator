import os
import urllib.request
import cv2
import numpy as np
import pandas as pd
from mediapipe.tasks.python.vision import HandLandmarker
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat

DATA_DIR = './data/raw'
MODEL_DIR = './models'
HAND_MODEL_PATH = os.path.join(MODEL_DIR, 'hand_landmarker.task')
HAND_MODEL_URL = 'https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task'

os.makedirs('./data/processed', exist_ok=True)


def download_hand_model(model_path: str) -> str:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if not os.path.exists(model_path):
        print('Downloading MediaPipe hand landmarker model...')
        urllib.request.urlretrieve(HAND_MODEL_URL, model_path)
        print(f'Downloaded hand landmarker model to {model_path}')
    return model_path


def main() -> None:
    model_path = download_hand_model(HAND_MODEL_PATH)
    data = []
    labels = []

    with HandLandmarker.create_from_model_path(model_path) as hand_landmarker:
        for dir_ in sorted(os.listdir(DATA_DIR)):
            dir_path = os.path.join(DATA_DIR, dir_)
            if not os.path.isdir(dir_path):
                continue

            for img_path in sorted(os.listdir(dir_path)):
                img_file = os.path.join(dir_path, img_path)
                img = cv2.imread(img_file)
                if img is None:
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)
                mp_image = Image(ImageFormat.SRGB, img_rgb)
                results = hand_landmarker.detect(mp_image)

                if not results.hand_landmarks:
                    continue

                hand_landmarks = results.hand_landmarks[0]
                x_ = [lm.x for lm in hand_landmarks]
                y_ = [lm.y for lm in hand_landmarks]
                data_aux = []

                min_x = min(x_)
                min_y = min(y_)
                for lm in hand_landmarks:
                    data_aux.append(lm.x - min_x)
                    data_aux.append(lm.y - min_y)

                if len(data_aux) == 42:
                    data.append(data_aux)
                    labels.append(dir_)

    if not data:
        raise RuntimeError('No hand landmarks were extracted. Check that data/raw contains valid hand images.')

    df = pd.DataFrame(data)
    df['label'] = labels
    df.to_csv('./data/processed/landmark_data.csv', index=False)
    print('Landmarks extracted and saved to data/processed/landmark_data.csv')


if __name__ == '__main__':
    main()
