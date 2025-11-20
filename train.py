"""
Entrenamiento del modelo a partir de `dataset.csv`.

Uso:
  python train.py --input dataset.csv --model model.joblib

El script entrena un Pipeline (StandardScaler + RandomForest) y guarda el pipeline y el LabelEncoder.
"""
import argparse
import os
import joblib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='dataset.csv', help='CSV con columnas: label,l0_x,...')
    parser.add_argument('--model', default='model.joblib', help='Ruta para guardar el modelo')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"No se encontró {args.input}. Ejecuta primero `data_collect.py` para crear el dataset.")
        return

    df = pd.read_csv(args.input)
    if 'label' not in df.columns:
        print('CSV no contiene columna `label`.')
        return

    X = df.drop('label', axis=1).astype(float).values
    y = df['label'].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=42))
    ])

    print('Entrenando modelo...')
    pipeline.fit(X, y_enc)

    # Evaluación simple
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)
    preds = pipeline.predict(X_test)
    print('Accuracy (test):', accuracy_score(y_test, preds))
    print('Reporte de clasificación:')
    print(classification_report(y_test, preds, target_names=le.classes_))

    print(f'Guardando modelo en {args.model} ...')
    joblib.dump({'pipeline': pipeline, 'le': le}, args.model)
    print('Modelo guardado.')


if __name__ == '__main__':
    main()
