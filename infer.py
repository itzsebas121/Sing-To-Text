"""
Inferencia en tiempo real usando la cámara y el modelo entrenado.

Uso:
  python infer.py --model model.joblib

Presiona 'q' para salir.
"""
import argparse
import collections
import time
import joblib

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def extract_landmarks(hand_landmarks):
    if hand_landmarks is None:
        return [0.0] * 63
    out = []
    for lm in hand_landmarks.landmark:
        out.extend([lm.x, lm.y, lm.z])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='model.joblib', help='Ruta al modelo guardado')
    parser.add_argument('--camera', type=int, default=0)
    args = parser.parse_args()

    data = joblib.load(args.model)
    pipeline = data['pipeline']
    le = data['le']

    cap = cv2.VideoCapture(args.camera)
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6) as hands:
        recent = collections.deque(maxlen=12)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Mirror frame so camera behaves like a mirror for the user
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            h, w, _ = frame.shape
            label = None
            # prepare combined two-hand vector
            landmarks_all = [[0.0] * 63, [0.0] * 63]
            if results.multi_hand_landmarks:
                def avg_x(landmarks):
                    return sum([lm.x for lm in landmarks.landmark]) / len(landmarks.landmark)

                hands_sorted = sorted(results.multi_hand_landmarks, key=avg_x)
                for i, hand_landmarks in enumerate(hands_sorted[:2]):
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    landmarks_all[i] = extract_landmarks(hand_landmarks)

                combined = landmarks_all[0] + landmarks_all[1]
                X = np.array(combined).reshape(1, -1)
                try:
                    pred = pipeline.predict(X)[0]
                    label = le.inverse_transform([pred])[0]
                    recent.append(label)
                except Exception:
                    label = None
            else:
                recent.append(None)

            # Majority vote over recent
            most_common = None
            if len(recent) > 0:
                votes = [r for r in recent if r is not None]
                if votes:
                    most_common = collections.Counter(votes).most_common(1)[0][0]

            if most_common:
                text = f'Prediction: {most_common}'
            elif label:
                text = f'Prediction: {label}'
            else:
                text = 'Prediction: -'

            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow('Sign-to-Text (press q to quit)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
