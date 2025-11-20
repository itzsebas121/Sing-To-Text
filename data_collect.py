"""
Recolección de datos para reconocimiento de señas con MediaPipe Hands.

Uso:
  python data_collect.py --label A --samples 200

Presiona 'q' para salir antes de completar.
"""
import argparse
import csv
import os
import time
from typing import List

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def extract_landmarks(hand_landmarks) -> List[float]:
    """Flatten landmarks to a list [x0,y0,z0,...] normalized relative to image."""
    if hand_landmarks is None:
        return [0.0] * 63
    out = []
    for lm in hand_landmarks.landmark:
        out.extend([lm.x, lm.y, lm.z])
    return out


def ensure_header(path: str):
    if not os.path.exists(path):
        header = ['label'] + [f'l{i}_{c}' for i in range(21) for c in ('x', 'y', 'z')]
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)


def main():
    parser = argparse.ArgumentParser(description='Recolecta landmarks de MediaPipe y guarda en CSV')
    parser.add_argument('--label', required=True, help='Etiqueta para las muestras (ej: A, B, hola)')
    parser.add_argument('--output', default='dataset.csv', help='Archivo CSV de salida')
    parser.add_argument('--sequence', action='store_true', help='Grabar secuencias (presiona r para toggle)')
    parser.add_argument('--samples', type=int, default=200, help='Número de muestras a recoger')
    parser.add_argument('--camera', type=int, default=0, help='Índice de la cámara')
    args = parser.parse_args()

    ensure_header(args.output)

    cap = cv2.VideoCapture(args.camera)
    # support up to 2 hands; use mirror display and processing so camera feels natural
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6) as hands:
        collected = 0
        last_save_time = 0
        seq_recording = False
        seq_frames = []
        os.makedirs('sequences', exist_ok=True)
        index_path = os.path.join('sequences', 'index.csv')
        # Ensure index exists
        if not os.path.exists(index_path):
            with open(index_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['path', 'label'])

        print(f"Iniciando recolección para etiqueta '{args.label}' - objetivo: {args.samples} muestras")
        while cap.isOpened() and collected < args.samples:
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror mode: flip horizontally so camera behaves like a mirror
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            h, w, _ = frame.shape
            # Prepare two-hand feature vector: default zeros
            landmarks_all = [[0.0] * 63, [0.0] * 63]
            if results.multi_hand_landmarks:
                # Sort hands by x to have deterministic order (left-to-right in image)
                def avg_x(landmarks):
                    return sum([lm.x for lm in landmarks.landmark]) / len(landmarks.landmark)

                hands_sorted = sorted(results.multi_hand_landmarks, key=avg_x)
                for i, hand_landmarks in enumerate(hands_sorted[:2]):
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    landmarks_all[i] = extract_landmarks(hand_landmarks)

            # Flatten both-hand features (hand0 then hand1)
            combined = landmarks_all[0] + landmarks_all[1]

            # Sequence recording mode: toggle with 'r'
            if args.sequence:
                cv2.putText(frame, f"Recording: {'ON' if seq_recording else 'OFF'} (r to toggle)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                if seq_recording:
                    seq_frames.append(combined)
                    cv2.putText(frame, f'Recording frames: {len(seq_frames)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            else:
                now = time.time()
                # Save per-frame samples at a rate to avoid duplicates
                if now - last_save_time > 0.12 and any(v != 0.0 for v in combined):
                    with open(args.output, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([args.label] + combined)
                    collected += 1
                    last_save_time = now
                    print(f"Guardada muestra {collected}/{args.samples}")

            cv2.putText(frame, f'Label: {args.label}  Collected: {collected}/{args.samples}', (10, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Data collection - Mirror mode (press q to quit)', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if args.sequence and key == ord('r'):
                # toggle recording
                seq_recording = not seq_recording
                if not seq_recording and seq_frames:
                    # save sequence to .npy
                    fname = f"sequences/{args.label}_{int(time.time())}.npy"
                    import numpy as _np
                    _np.save(fname, _np.array(seq_frames, dtype=_np.float32))
                    with open(index_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([fname, args.label])
                    print(f"Saved sequence {fname} (frames={len(seq_frames)})")
                    seq_frames = []

    cap.release()
    cv2.destroyAllWindows()
    print('Recolección finalizada.')


if __name__ == '__main__':
    main()
