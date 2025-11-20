"""
Procesa un video, detecta manos frame a frame, usa un clasificador de secuencias
(`seq_model_rf.joblib`) para predecir frases en ventanas deslizantes y genera
un video subtitulado con las predicciones.

Uso:
  python process_video.py --input video.mp4 --output out_subtitled.mp4 --model seq_model_rf.joblib --le le_seq_rf.joblib

El algoritmo (simple): extrae landmarks por frame, mantiene una ventana de `--window` frames,
convierte la ventana a características (media,std,diff) y predice; si la probabilidad
es suficientemente alta y la predicción se mantiene durante `--min-windows` ventanas
consecutivas, se genera un subtítulo en el video.
"""
import argparse
import joblib
import cv2
import mediapipe as mp
import numpy as np
import os
from collections import deque, Counter

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
    seq = np.array(seq)
    mean = seq.mean(axis=0)
    std = seq.std(axis=0)
    first = seq[0]
    last = seq[-1]
    diff = last - first
    return np.concatenate([mean, std, diff])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default='out_subtitled.mp4')
    parser.add_argument('--model', default='seq_model_rf.joblib')
    parser.add_argument('--le', default='le_seq_rf.joblib')
    parser.add_argument('--window', type=int, default=40, help='Frames per window')
    parser.add_argument('--stride', type=int, default=8, help='Stride between windows')
    parser.add_argument('--min-windows', type=int, default=3, help='Minimum consecutive windows to accept a subtitle')
    parser.add_argument('--threshold', type=float, default=0.4, help='Confidence threshold')
    args = parser.parse_args()

    clf = joblib.load(args.model)
    le = joblib.load(args.le)

    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6) as hands:
        frames = []
        frame_idx = 0
        window_preds = []  # (frame_idx, pred_label, prob)
        # We'll store subtitles as list of (start_sec, end_sec, label)
        subtitles = []
        # To speed up, read all frames landmarks
        landmarks_list = []
        while True:
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
                    landmarks_all[i] = extract_landmarks(hand_landmarks)
            combined = landmarks_all[0] + landmarks_all[1]
            landmarks_list.append(combined)
            frames.append(frame)
            frame_idx += 1

        n = len(landmarks_list)
        # Sliding windows
        i = 0
        preds_windows = []
        while i + args.window <= n:
            window = landmarks_list[i:i+args.window]
            feat = sequence_to_feature(window).reshape(1, -1)
            probs = clf.predict_proba(feat)[0]
            p_idx = np.argmax(probs)
            p_label = le.inverse_transform([p_idx])[0]
            p_prob = probs[p_idx]
            preds_windows.append((i, p_label, p_prob))
            i += args.stride

        # Collapse consecutive windows with same label passing threshold
        accepted = []
        for idx, label, prob in preds_windows:
            accepted.append((idx, label, prob))

        # Group into segments: convert window indices to frame ranges
        segments = []
        if accepted:
            cur_label = accepted[0][1]
            cur_start = accepted[0][0]
            cur_end = cur_start + args.window
            cur_count = 1 if accepted[0][2] >= args.threshold else 0
            for j in range(1, len(accepted)):
                idx, label, prob = accepted[j]
                start = idx
                end = idx + args.window
                if label == cur_label:
                    cur_end = end
                    if prob >= args.threshold:
                        cur_count += 1
                else:
                    if cur_count >= args.min_windows:
                        segments.append((cur_start, cur_end, cur_label))
                    cur_label = label
                    cur_start = start
                    cur_end = end
                    cur_count = 1 if prob >= args.threshold else 0
            if cur_count >= args.min_windows:
                segments.append((cur_start, cur_end, cur_label))

        # Now write frames and overlay subtitles according to segments
        seg_idx = 0
        for fi, frame in enumerate(frames):
            # find active subtitle
            text = ''
            while seg_idx < len(segments) and (fi > segments[seg_idx][1]):
                seg_idx += 1
            if seg_idx < len(segments):
                s, e, label = segments[seg_idx]
                if fi >= s and fi <= e:
                    text = label.replace('_', ' ')

            if text:
                # draw semi-transparent rectangle
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, h-60), (w, h), (0,0,0), -1)
                alpha = 0.6
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                cv2.putText(frame, text, (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

            out.write(frame)

    cap.release()
    out.release()
    print('Processed video saved to', args.output)


if __name__ == '__main__':
    main()
