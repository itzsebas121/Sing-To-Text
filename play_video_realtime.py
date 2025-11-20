"""
Reproduce un video y realiza inferencia en tiempo real por ventanas deslizantes,
mostrando subtítulos durante la reproducción.

Uso:
  python play_video_realtime.py --input video.mp4 --model seq_model_rf.joblib --le le_seq_rf.joblib

Opciones: `--window`, `--stride`, `--threshold`, `--stability` (igual que infer_realtime).
"""
import argparse
import cv2
import joblib
import numpy as np
import mediapipe as mp
from collections import deque, Counter
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def extract_landmarks(hand_landmarks):
    if hand_landmarks is None:
        return [0.0] * 63
    out = []
    for lm in hand_landmarks.landmark:
        out.extend([lm.x, lm.y, lm.z])
    return out


def sequence_to_feature(seq):
    arr = np.array(seq)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    first = arr[0]
    last = arr[-1]
    diff = last - first
    return np.concatenate([mean, std, diff])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--model', default='seq_model_rf.joblib')
    parser.add_argument('--le', default='le_seq_rf.joblib')
    parser.add_argument('--window', type=int, default=40)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--stability', type=int, default=3)
    args = parser.parse_args()

    clf = joblib.load(args.model)
    le = None
    if args.le and os.path.exists(args.le):
        le = joblib.load(args.le)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print('No se pudo abrir el video:', args.input)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    window = deque(maxlen=args.window)
    last_predictions = deque(maxlen=args.stability)
    frame_idx = 0

    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # mirror to match real-time behavior
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

            combined = landmarks_all[0] + landmarks_all[1]
            window.append(combined)

            show_label = ''
            # when window full and at stride, predict
            if len(window) == args.window and frame_idx % args.stride == 0:
                feat = sequence_to_feature(list(window)).reshape(1, -1)
                try:
                    probs = clf.predict_proba(feat)[0]
                    p_idx = int(np.argmax(probs))
                    p_prob = float(probs[p_idx])
                    if le is not None:
                        label = le.inverse_transform([p_idx])[0]
                    else:
                        try:
                            label = clf.classes_[p_idx]
                        except Exception:
                            label = str(p_idx)
                    last_predictions.append((label, p_prob))
                except Exception:
                    pass

            if len(last_predictions) >= args.stability:
                labels = [p for p,pr in last_predictions]
                most_common, count = Counter(labels).most_common(1)[0]
                avg_prob = sum([pr for (l,pr) in last_predictions if l==most_common]) / max(1, sum(1 for (l,pr) in last_predictions if l==most_common))
                if count >= args.stability and avg_prob >= args.threshold:
                    show_label = most_common.replace('_',' ')

            h, w, _ = frame.shape
            if show_label:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, h-60), (w, h), (0,0,0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                cv2.putText(frame, show_label, (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            else:
                cv2.putText(frame, 'Playing...', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,0), 2)

            cv2.imshow('Play Video Real-time Subtitles (q to quit)', frame)
            if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
                break
            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
