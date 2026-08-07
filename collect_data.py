import cv2
import os

DATA_DIR = './data/raw'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Classes to collect: A-Z (You can start with A, B, C for testing)
classes = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
dataset_size = 100

cap = cv2.VideoCapture(0)

for j in classes:
    if not os.path.exists(os.path.join(DATA_DIR, j)):
        os.makedirs(os.path.join(DATA_DIR, j))

    print(f'Collecting data for class {j}. Press "Q" when ready.')
    
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, f'Ready? Press "Q" to capture: {j}', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, j, f'{counter}.jpg'), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
