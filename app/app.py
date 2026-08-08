import os
import urllib.request
import streamlit as st
import cv2
import numpy as np
import pickle
from text_to_speech import speak_text
from mediapipe.tasks.python.vision import HandLandmarker
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat

MODEL_URL = 'https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task'
HAND_MODEL_PATH = './models/hand_landmarker.task'


def download_hand_model(model_path: str) -> str:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(MODEL_URL, model_path)
    return model_path


st.title('ASL to Text & Speech Translator')

model_dict = pickle.load(open('./models/model.p', 'rb'))
model = model_dict['model']

if 'sentence' not in st.session_state:
    st.session_state.sentence = ''

run = st.checkbox('Capture frame')
FRAME_WINDOW = st.image([])

if run:
    hand_model_path = download_hand_model(HAND_MODEL_PATH)
    with HandLandmarker.create_from_model_path(hand_model_path) as hand_landmarker:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)
            mp_image = Image(ImageFormat.SRGB, rgb_frame)
            results = hand_landmarker.detect(mp_image)
            predicted_character = ''

            if results.hand_landmarks:
                hand_landmarks = results.hand_landmarks[0]
                x_ = [lm.x for lm in hand_landmarks]
                y_ = [lm.y for lm in hand_landmarks]
                data_aux = []
                min_x = min(x_)
                min_y = min(y_)
                for lm in hand_landmarks:
                    data_aux.append(lm.x - min_x)
                    data_aux.append(lm.y - min_y)
                    cx = int(lm.x * frame.shape[1])
                    cy = int(lm.y * frame.shape[0])
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

                if len(data_aux) == 42:
                    predicted_character = model.predict([np.asarray(data_aux)])[0]

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels='RGB')
            st.write('Prediction:', predicted_character)

if st.button('Speak Word'):
    speak_text(st.session_state.sentence)
