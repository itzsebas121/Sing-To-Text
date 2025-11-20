"""
Entrenamiento de modelo secuencial (LSTM) a partir de secuencias guardadas en `sequences/`.

El dataset de secuencias debe contener `sequences/index.csv` con columnas `path,label`.

Uso:
  python train_seq.py --index sequences/index.csv --model seq_model.h5

Guarda `seq_model.h5` (Keras) y `le.joblib` con el LabelEncoder.
"""
import argparse
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.callbacks import EarlyStopping


def load_sequences(index_csv, max_len=None):
    df = pd.read_csv(index_csv)
    X = []
    y = []
    lengths = []
    for _, row in df.iterrows():
        path = row['path']
        label = row['label']
        if not os.path.exists(path):
            continue
        seq = np.load(path)
        X.append(seq.astype(np.float32))
        y.append(label)
        lengths.append(len(seq))

    if not X:
        raise ValueError('No sequences found in index or files missing')

    if max_len is None:
        max_len = max(lengths)

    # Pad sequences with zeros to max_len
    n_samples = len(X)
    n_features = X[0].shape[1]
    X_pad = np.zeros((n_samples, max_len, n_features), dtype=np.float32)
    for i, seq in enumerate(X):
        L = min(len(seq), max_len)
        X_pad[i, :L, :] = seq[:L]

    return X_pad, np.array(y), max_len


def build_model(input_shape, n_classes):
    model = Sequential([
        Masking(mask_value=0.0, input_shape=input_shape),
        LSTM(128, return_sequences=False),
        Dense(64, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', default='sequences/index.csv', help='CSV index with path,label')
    parser.add_argument('--model', default='seq_model.h5', help='Output Keras model file')
    parser.add_argument('--le', default='le_seq.joblib', help='LabelEncoder output')
    parser.add_argument('--max-len', type=int, default=None, help='Max sequence length (pad/truncate)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    X, y_str, max_len = load_sequences(args.index, max_len=args.max_len)
    print(f'Loaded {len(X)} sequences, max_len={max_len}, features={X.shape[2]}')

    le = LabelEncoder()
    y = le.fit_transform(y_str)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = build_model(input_shape=(max_len, X.shape[2]), n_classes=len(le.classes_))
    es = EarlyStopping(patience=5, restore_best_weights=True)

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=args.epochs, batch_size=args.batch_size, callbacks=[es])

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test accuracy: {acc:.4f}')

    model.save(args.model)
    joblib.dump(le, args.le)
    print(f'Saved model: {args.model} and LabelEncoder: {args.le}')


if __name__ == '__main__':
    main()
