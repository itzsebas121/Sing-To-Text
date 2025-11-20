"""
Generador de dataset sintético para pruebas.

Genera un `dataset.csv` con la misma estructura que `data_collect.py` (columna `label` + 21*3 landmarks).
Cada clase tiene un centroid en el espacio 63-D y se muestrean valores con ruido gaussiano.

Uso:
  python generate_synthetic_dataset.py --classes A,B,C,HELLO --samples-per-class 200 --output dataset.csv
"""
import argparse
import csv
import numpy as np
import os


def make_header():
    return ['label'] + [f'l{i}_{c}' for i in range(21) for c in ('x', 'y', 'z')]


def generate(classes, samples_per_class, scale, seed=42):
    rng = np.random.RandomState(seed)
    n_features = 21 * 3
    centroids = {cls: rng.rand(n_features) for cls in classes}
    rows = []
    for cls in classes:
        centroid = centroids[cls]
        for _ in range(samples_per_class):
            sample = centroid + rng.normal(loc=0.0, scale=scale, size=n_features)
            # Clip to [0,1] to mimic normalized landmarks
            sample = np.clip(sample, 0.0, 1.0)
            rows.append([cls] + sample.tolist())
    rng.shuffle(rows)
    return rows


def _asl_letters():
    return [chr(ord('A') + i) for i in range(26)]


def _digits():
    return [str(i) for i in range(10)]


def _common_words():
    return ['HELLO', 'THANKS', 'YES', 'NO', 'PLEASE', 'GOOD', 'MORNING', 'SORRY']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', default='', help='Lista de etiquetas separadas por comas')
    parser.add_argument('--preset', default='', choices=['asl', 'asl_digits', 'asl_plus'],
                        help='Presets: "asl"=A-Z, "asl_digits"=A-Z+0-9, "asl_plus"=A-Z+0-9+words')
    parser.add_argument('--samples-per-class', type=int, default=300, help='Número de muestras por clase')
    parser.add_argument('--scale', type=float, default=0.03, help='Ruido gaussiano (std)')
    parser.add_argument('--output', default='dataset.csv')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    classes = []
    if args.preset:
        if args.preset == 'asl':
            classes = _asl_letters()
        elif args.preset == 'asl_digits':
            classes = _asl_letters() + _digits()
        elif args.preset == 'asl_plus':
            classes = _asl_letters() + _digits() + _common_words()

    if args.classes:
        extra = [c.strip() for c in args.classes.split(',') if c.strip()]
        classes = classes + extra if classes else extra

    # If still empty, fallback to small default
    if not classes:
        classes = ['A', 'B', 'C', 'HELLO']

    rows = generate(classes, args.samples_per_class, args.scale, seed=args.seed)

    # Ensure output directory
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(make_header())
        for r in rows:
            writer.writerow(r)

    print(f'Generated {len(rows)} samples ({len(classes)} classes) -> {args.output}')


if __name__ == '__main__':
    main()
