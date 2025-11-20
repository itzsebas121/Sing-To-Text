"""
Inferencia en tiempo real (ventana deslizante) desde la cámara.

Funcionamiento:
- Captura frames de la cámara en modo espejo.
- Extrae landmarks de hasta 2 manos por frame (mano0 + mano1 -> 126 features).
- Mantiene una ventana deslizante de `--window` frames y, cada `--stride` frames,
  calcula features (media/std/diff) y predice con el clasificador de secuencias.
- Suaviza predicciones con una cola corta y muestra subtítulos en pantalla cuando
  la predicción es estable y supera `--threshold`.

Uso:
  python infer_realtime.py --model seq_model_rf.joblib --le le_seq_rf.joblib

Opciones principales:
  --window N        Frames por ventana (por defecto 40)
  --stride S        Saltos entre ventanas (por defecto 8)
  --threshold t     Umbral de confianza (por defecto 0.5)
  --stability k     Ventanas consecutivas iguales necesarias (por defecto 3)
  --camera i        Índice de la cámara (por defecto 0)
"""
import argparse
import time
from collections import deque, Counter
import os

import joblib
import cv2
import numpy as np
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


def sequence_to_feature(seq):
    arr = np.array(seq)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    first = arr[0]
    last = arr[-1]
    diff = last - first
    return np.concatenate([mean, std, diff])


def try_load_model(model_path):
    # Prefer joblib models (RandomForest). If a Keras model is present, try to load it.
    if model_path.endswith('.joblib') or model_path.endswith('.pkl'):
        data = joblib.load(model_path)
        # If pipeline style (like earlier), return pipeline and label encoder
        if isinstance(data, dict) and 'pipeline' in data and 'le' in data:
            return ('sk_pipeline', data['pipeline'], data['le'])
        # else assume it's a classifier and require a separate le
        return ('sk_clf', data, None)
    else:
        # try keras
        try:
            from tensorflow.keras.models import load_model
            model = load_model(model_path)
            return ('keras', model, None)
        except Exception:
            raise RuntimeError('Modelo no soportado o no se pudo cargar: ' + model_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='seq_model_rf.joblib', help='Modelo de secuencias (joblib o Keras)')
    parser.add_argument('--le', default='le_seq_rf.joblib', help='LabelEncoder (joblib) si aplica')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--window', type=int, default=40)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--stability', type=int, default=3)
    args = parser.parse_args()

    model_type = None
    clf = None
    le = None
    # try to load label encoder if provided
    if args.le and os.path.exists(args.le):
        try:
            le = joblib.load(args.le)
        except Exception:
            le = None

    # load model
    mt = None
    try:
        mt = try_load_model(args.model)
    except Exception as e:
        print('Error cargando el modelo:', e)
        return

    model_type = mt[0]
    if model_type == 'sk_pipeline':
        clf = mt[1]
        if le is None:
            le = mt[2]
    elif model_type == 'sk_clf':
        clf = mt[1]
    elif model_type == 'keras':
        clf = mt[1]

    cap = cv2.VideoCapture(args.camera)
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6) as hands:
        window = deque(maxlen=args.window)
        frame_idx = 0
        last_predictions = deque(maxlen=args.stability)
        show_label = ''
        show_since = 0
        # to compute predictions each stride
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            landmarks_all = [[0.0] * 63, [0.0] * 63]
            if results.multi_hand_landmarks:
                def avg_x(landmarks):
                    return sum([lm.x for lm in landmarks.landmark]) / len(landmarks.landmark)
                hands_sorted = sorted(results.multi_hand_landmarks, key=avg_x)
                for i, hand_landmarks in enumerate(hands_sorted[:2]):
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    landmarks_all[i] = extract_landmarks(hand_landmarks)

            combined = landmarks_all[0] + landmarks_all[1]
            window.append(combined)

            # Predict at strides
            label_pred = None
            prob_pred = 0.0
            if len(window) == args.window and frame_idx % args.stride == 0:
                feat = sequence_to_feature(list(window)).reshape(1, -1)
                try:
                    if model_type in ('sk_pipeline', 'sk_clf'):
                        probs = clf.predict_proba(feat)[0]
                        p_idx = int(np.argmax(probs))
                        prob_pred = float(probs[p_idx])
                        if le is not None:
                            label_pred = le.inverse_transform([p_idx])[0]
                        else:
                            # if classifier has classes_ attribute
                            try:
                                label_pred = clf.classes_[p_idx]
                            except Exception:
                                label_pred = str(p_idx)
                    elif model_type == 'keras':
                        probs = clf.predict(feat)
                        p_idx = int(np.argmax(probs, axis=1)[0])
                        prob_pred = float(probs[0, p_idx])
                        if le is not None:
                            label_pred = le.inverse_transform([p_idx])[0]
                        else:
                            label_pred = str(p_idx)
                except Exception as e:
                    # prediction failed — ignore
                    label_pred = None
                    prob_pred = 0.0

                if label_pred is not None:
                    last_predictions.append((label_pred, prob_pred))

            # decide what to show: check last_predictions stability
            show_label = ''
            if len(last_predictions) >= args.stability:
                labels = [p for p,pr in last_predictions]
                probs = [pr for p,pr in last_predictions]
                most_common, count = Counter(labels).most_common(1)[0]
                avg_prob = sum([pr for (l,pr) in last_predictions if l==most_common]) / max(1, sum(1 for (l,pr) in last_predictions if l==most_common))
                if count >= args.stability and avg_prob >= args.threshold:
                    show_label = most_common.replace('_', ' ')

            # display overlay
            h, w, _ = frame.shape
            if show_label:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, h-60), (w, h), (0,0,0), -1)
                alpha = 0.6
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                cv2.putText(frame, show_label, (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            else:
                cv2.putText(frame, 'Listening...', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,0), 2)

            cv2.imshow('Real-time Sign-to-Text (press q to quit)', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
