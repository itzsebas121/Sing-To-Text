"""
Inferencia por secuencia (frases). Graba una secuencia con 'r' y la clasifica con un modelo LSTM.

Uso:
  python infer_seq.py --model seq_model.h5 --le le_seq.joblib

Presiona 'r' para empezar/detener la grabación de la frase; presiona 'q' para salir.
"""
import argparse
import collections
import time
import joblib
import numpy as np
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def extract_landmarks(hand_landmarks):
    if hand_landmarks is None:
        return [0.0] * 63
    out = []
    for lm in hand_landmarks.landmark:
        out.extend([lm.x, lm.y, lm.z])
    return out


def pad_sequence(seq, max_len):
    n_features = seq.shape[1]
    out = np.zeros((max_len, n_features), dtype=np.float32)
    L = min(len(seq), max_len)
    out[:L, :] = seq[:L]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='seq_model.h5')
    parser.add_argument('--le', default='le_seq.joblib')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--max-len', type=int, default=60, help='Max frames expected per sequence')
    args = parser.parse_args()

    model = None
    try:
        model = __import__('tensorflow.keras').models.load_model(args.model)
    except Exception as e:
        import tensorflow as tf
        model = tf.keras.models.load_model(args.model)

    le = joblib.load(args.le)

    cap = cv2.VideoCapture(args.camera)
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6) as hands:
        recording = False
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            landmarks_all = [[0.0]*63, [0.0]*63]
            if results.multi_hand_landmarks:
                def avg_x(landmarks):
                    return sum([lm.x for lm in landmarks.landmark]) / len(landmarks.landmark)

                hands_sorted = sorted(results.multi_hand_landmarks, key=avg_x)
                for i, hand_landmarks in enumerate(hands_sorted[:2]):
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    landmarks_all[i] = extract_landmarks(hand_landmarks)

            comb = np.array(landmarks_all[0] + landmarks_all[1], dtype=np.float32)

            if recording:
                frames.append(comb)
                cv2.putText(frame, f'Recording frames: {len(frames)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            else:
                cv2.putText(frame, 'Press r to record sequence', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

            cv2.imshow('Infer sequence (r=toggle, q=quit)', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                recording = not recording
                if not recording and frames:
                    seq = np.array(frames, dtype=np.float32)
                    seq_padded = pad_sequence(seq, args.max_len)
                    X = np.expand_dims(seq_padded, axis=0)
                    preds = model.predict(X)
                    p = np.argmax(preds, axis=1)[0]
                    label = le.inverse_transform([p])[0]
                    print(f'Predicted sequence label: {label} (prob={preds[0,p]:.3f})')
                    frames = []

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
