import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import time
from scipy.ndimage import center_of_mass, shift

# Load trained model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

model = NeuralNet(784, 100, 10)
model.load_state_dict(torch.load("mnist_model.pth", map_location=torch.device('cpu')))
model.eval()

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

canvas = np.zeros((480, 640), dtype=np.uint8)
is_writing = False
prev_x, prev_y = None, None

DRAW_FRAME_SIZE = 280
frame_top_left = (180, 100)
frame_bottom_right = (180 + DRAW_FRAME_SIZE, 100 + DRAW_FRAME_SIZE)

def center_digit(image):
    cy, cx = center_of_mass(image)
    if np.isnan(cx) or np.isnan(cy):
        return image
    shiftx = int(np.round(image.shape[1]/2.0 - cx))
    shifty = int(np.round(image.shape[0]/2.0 - cy))
    return shift(image, shift=(shifty, shiftx), mode='constant', cval=0)

def preprocess_digit(img):
    x1, y1 = frame_top_left
    x2, y2 = frame_bottom_right
    roi = img[y1:y2, x1:x2]

    # Preprocess the ROI
    roi = cv2.GaussianBlur(roi, (5, 5), 0)
    _, roi = cv2.threshold(roi, 50, 255, cv2.THRESH_BINARY)
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = center_digit(roi)

    roi = roi / 255.0
    roi = torch.tensor(roi, dtype=torch.float32).view(-1, 784)
    return roi

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)

instructions = [
    "Controls:",
    "Press 'w' - Toggle writing mode",
    "Press 'r' - Recognize and show prediction",
    "Press 'c' - Clear screen",
    "Press 'Esc' - Exit"
]

last_prediction_text = ""
prediction_timer = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    overlay = frame.copy()
    cv2.rectangle(overlay, frame_top_left, frame_bottom_right, (255, 255, 255), 2)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(overlay, handLms, mp_hands.HAND_CONNECTIONS)
            index_finger = handLms.landmark[8]
            cx, cy = int(index_finger.x * w), int(index_finger.y * h)

            if frame_top_left[0] < cx < frame_bottom_right[0] and frame_top_left[1] < cy < frame_bottom_right[1]:
                if is_writing:
                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x, prev_y), (cx, cy), 255, 10)
                    prev_x, prev_y = cx, cy
                    cv2.circle(overlay, (cx, cy), 5, (0, 255, 0), -1)
                else:
                    prev_x, prev_y = None, None

    color_canvas = np.zeros_like(overlay)
    color_canvas[canvas > 0] = (0, 0, 255)
    overlay = cv2.addWeighted(overlay, 1.0, color_canvas, 0.8, 0)

    for i, line in enumerate(instructions):
        cv2.putText(overlay, line, (10, 20 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    if last_prediction_text and time.time() < prediction_timer:
        (tw, th), _ = cv2.getTextSize(last_prediction_text, cv2.FONT_HERSHEY_DUPLEX, 2, 3)
        fx = frame_top_left[0] + (DRAW_FRAME_SIZE - tw) // 2
        fy = frame_top_left[1] + (DRAW_FRAME_SIZE + th) // 2
        cv2.putText(overlay, last_prediction_text, (fx, fy),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

    key = cv2.waitKey(1)
    if key == ord('w'):
        is_writing = not is_writing
    elif key == ord('c'):
        canvas[:] = 0
        last_prediction_text = ""
    elif key == ord('r'):
        input_img = canvas.copy()
        input_img[input_img > 0] = 255
        roi = input_img[frame_top_left[1]:frame_bottom_right[1], frame_top_left[0]:frame_bottom_right[0]]
        if np.count_nonzero(roi) > 300:
            digit = preprocess_digit(input_img)
            output = model(digit)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs).item()
            confidence = probs[0][pred].item()

            last_prediction_text = f"{pred} ({confidence*100:.1f}%)"
            prediction_timer = time.time() + 3
            canvas[:] = 0
        else:
            print("Draw something in the box to predict.")

    cv2.imshow("Digit Recognizer", overlay)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
