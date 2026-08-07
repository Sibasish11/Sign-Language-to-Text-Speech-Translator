import os
import urllib.request
import pickle
import cv2
import numpy as np
from mediapipe.tasks.python.vision import HandLandmarker
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat

MODEL_URL = 'https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task'
HAND_MODEL_PATH = './models/hand_landmarker.task'


def download_hand_model(model_path: str) -> str:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if not os.path.exists(model_path):
        print('Downloading MediaPipe hand landmarker model...')
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print(f'Downloaded hand landmarker model to {model_path}')
    return model_path


def main() -> None:
    hand_model_path = download_hand_model(HAND_MODEL_PATH)
    model_dict = pickle.load(open('./models/model.p', 'rb'))
    model = model_dict['model']

    cap = cv2.VideoCapture(0)
    current_word = ''
    predicted_character = ''

    with HandLandmarker.create_from_model_path(hand_model_path) as hand_landmarker:
        while True:
            data_aux = []
            x_ = []
            y_ = []

            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)
            mp_image = Image(ImageFormat.SRGB, frame_rgb)
            results = hand_landmarker.detect(mp_image)

            if results.hand_landmarks:
                hand_landmarks = results.hand_landmarks[0]
                for lm in hand_landmarks:
                    x_.append(lm.x)
                    y_.append(lm.y)
                    cx = int(lm.x * frame.shape[1])
                    cy = int(lm.y * frame.shape[0])
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

                if x_ and y_:
                    min_x = min(x_)
                    min_y = min(y_)
                    for lm in hand_landmarks:
                        data_aux.append(lm.x - min_x)
                        data_aux.append(lm.y - min_y)

                    if len(data_aux) == 42:
                        prediction = model.predict([np.asarray(data_aux)])
                        predicted_character = prediction[0]
                        cv2.putText(frame, predicted_character, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

            cv2.putText(frame, f'Word: {current_word}', (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Sign Language Translator', frame)

            key = cv2.waitKey(1)
            if key == ord('a'):
                current_word += predicted_character
            if key == ord('s'):
                current_word += ' '
            if key == ord('c'):
                current_word = ''
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
