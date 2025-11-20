"""
Entrenamiento alternativo sin TensorFlow: convierte secuencias en features fijas
(media, std, delta) y entrena un RandomForest para clasificar frases.

Salida: `seq_model_rf.joblib` y `le_seq_rf.joblib`.
"""
import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def sequence_to_feature(seq):
    # seq: (T, F)
    mean = seq.mean(axis=0)
    std = seq.std(axis=0)
    first = seq[0]
    last = seq[-1]
    diff = last - first
    feat = np.concatenate([mean, std, diff])
    return feat


def load_index(index_csv):
    df = pd.read_csv(index_csv)
    X = []
    y = []
    for _, row in df.iterrows():
        path = row['path']
        label = row['label']
        if not os.path.exists(path):
            continue
        seq = np.load(path)
        X.append(sequence_to_feature(seq))
        y.append(label)
    return np.array(X), np.array(y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', default='sequences/index.csv')
    parser.add_argument('--model', default='seq_model_rf.joblib')
    parser.add_argument('--le', default='le_seq_rf.joblib')
    args = parser.parse_args()

    X, y = load_index(args.index)
    print('Loaded', X.shape)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, preds))
    print(classification_report(y_test, preds, target_names=le.classes_))

    joblib.dump(clf, args.model)
    joblib.dump(le, args.le)
    print('Saved model and label encoder')


if __name__ == '__main__':
    main()
