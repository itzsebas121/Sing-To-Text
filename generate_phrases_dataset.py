"""
Generador de secuencias sintéticas para frases en español.

Genera archivos `.npy` en `sequences/` y un `sequences/index.csv` con columnas `path,label`.

Cada frase se modela como una secuencia de frames con centroides y ruido; la duración varía aleatoriamente.

Uso:
  python generate_phrases_dataset.py --output sequences/index.csv --samples-per-phrase 200
"""
import argparse
import os
import csv
import numpy as np


DEFAULT_PHRASES = [
    'HOLA', 'BUENOS_DIAS', 'BUENAS_NOCHES', 'GRACIAS', 'POR_FAVOR', 'DISCULPA',
    'LO_SIENTO', 'SI', 'NO', 'TE_QUIERO', 'HASTA_LUEGO', 'PASA_ME_AQUEL',
    'DONDE_ESTA_EL_BANO', 'TENGO_HAMBRE', 'TENGO_SED', 'AYUDA', 'NO_ENTIENDO',
    'REPETIR_POR_FAVOR', 'QUE_HORA_ES', 'BUEN_TRABAJO', 'ESTOY_CANSADO', 'COMO_ESTAS',
    'MUCHO_GUSTO', 'FELIZ_CUMPLEANOS', 'QUIERO_IR', 'ESPERA', 'VENGO_AHI', 'ESPERA_UNA_SEG',
    'MAS_DESPACIO', 'MUY_BIEN', 'NO_SE', 'ME_LLAMO', 'DE_NADA', 'CLARO', 'PERDON', 'PUEDES_AJUDAR'
]


def make_dirs(out_index_path):
    seq_dir = os.path.dirname(os.path.abspath(out_index_path))
    os.makedirs(seq_dir, exist_ok=True)
    return seq_dir


def generate_one_sequence(phrase, n_frames, n_features=126, seed=None):
    rng = np.random.RandomState(seed)
    # Simulate a smooth motion by interpolating between a few random control points
    n_control = max(2, n_frames // 20)
    control = rng.rand(n_control, n_features)
    # Interpolate
    t = np.linspace(0, 1, n_frames)
    seq = np.zeros((n_frames, n_features), dtype=np.float32)
    for i in range(n_features):
        seq[:, i] = np.interp(t, np.linspace(0, 1, n_control), control[:, i])
    # Add per-frame noise
    seq += rng.normal(scale=0.02, size=seq.shape)
    seq = np.clip(seq, 0.0, 1.0)
    return seq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='sequences/index.csv')
    parser.add_argument('--phrases', default='', help='Comma-separated extra phrases')
    parser.add_argument('--samples-per-phrase', type=int, default=200)
    parser.add_argument('--min-frames', type=int, default=20)
    parser.add_argument('--max-frames', type=int, default=80)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    phrases = list(DEFAULT_PHRASES)
    if args.phrases:
        extras = [p.strip().replace(' ', '_').upper() for p in args.phrases.split(',') if p.strip()]
        phrases += extras

    seq_dir = make_dirs(args.output)

    rng = np.random.RandomState(args.seed)
    records = []
    total = 0
    for phrase in phrases:
        for i in range(args.samples_per_phrase):
            n_frames = rng.randint(args.min_frames, args.max_frames + 1)
            seq = generate_one_sequence(phrase, n_frames, n_features=126, seed=rng.randint(2**31))
            fname = os.path.join(seq_dir, f"{phrase}_{i}_{int(rng.randint(1e9))}.npy")
            np.save(fname, seq)
            records.append((fname, phrase))
            total += 1
        print(f'Generated {args.samples_per_phrase} sequences for phrase: {phrase}')

    # write index
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'label'])
        for r in records:
            writer.writerow(r)

    print(f'Total sequences generated: {total} -> index: {args.output}')


if __name__ == '__main__':
    main()
