# Sing-To-Text

Proyecto simple en Python para convertir lenguaje de señas (mano) a texto usando MediaPipe y un clasificador tradicional.

Características:
- Recolección de datos desde la cámara usando MediaPipe Hands (`data_collect.py`).
- Entrenamiento de un clasificador (RandomForest) en `train.py`.
- Inferencia en tiempo real con la cámara en `infer.py`.

Requisitos
---------

Instala las dependencias (usa PowerShell en Windows):

```powershell
python -m pip install -r requirements.txt
```

Recolectar datos
----------------

Opciones:

- Recolectar desde la cámara (recomendado para datos reales):

```powershell
python data_collect.py --label A --samples 200
```

- Generar un dataset sintético grande (útil para pruebas rápidas). El repositorio incluye presets para generar muchas señas (letras A-Z, dígitos 0-9 y palabras comunes):

```powershell
# Genera dataset con preset asl_plus (A-Z, 0-9, palabras) con 300 muestras por clase
python generate_synthetic_dataset.py --preset asl_plus --samples-per-class 300 --output dataset.csv
```

Notas: los scripts crean/añaden a `dataset.csv`. El formato es `label` seguido por 63 valores (21 landmarks x 3).

Entrenar el modelo
------------------

Cuando tengas suficientes muestras para varias etiquetas:

```powershell
python train.py --input dataset.csv --model model.joblib
```

Esto guardará `model.joblib` con el pipeline y el codificador de etiquetas.

Inferencia en tiempo real
------------------------

Ejecuta la inferencia con la cámara:

```powershell
python infer.py --model model.joblib
```

Presiona `q` para salir.

Inferencia de frases / secuencias
--------------------------------

Para grabar frases (secuencia de frames) y entrenar un modelo LSTM:

1. Graba secuencias usando `data_collect.py` con el flag `--sequence`. Con la interfaz en espejo:

```powershell
# Inicia el script y presiona 'r' para empezar/detener la grabación de la secuencia
python data_collect.py --label "PASAME_AQUEL" --sequence --samples 50
```

Las secuencias se guardarán en la carpeta `sequences/` como archivos `.npy` y `sequences/index.csv` tendrá las rutas y etiquetas.

2. Entrena un modelo secuencial (LSTM) con `train_seq.py`:

```powershell
python train_seq.py --index sequences/index.csv --model seq_model.h5 --le le_seq.joblib --max-len 80
```

3. Infiere frases en tiempo real con `infer_seq.py` (presiona `r` para grabar la frase y la clasifica):

```powershell
python infer_seq.py --model seq_model.h5 --le le_seq.joblib --max-len 80
```

Notas importantes
-----------------
- El modo cámara está en espejo por defecto para que sea más fácil alinear tu mano con lo que ves en pantalla.
- La captación soporta hasta dos manos; los features se guardan como mano0 + mano1 (cada mano 21 landmarks x 3 = 63; total 126 valores por frame).
- Para frases largas, aumenta `--max-len` al entrenar y al inferir; las secuencias se rellenan/truncarán a ese largo.
- Para mejor robustez con datos reales, recolecta muchas variaciones (ángulos, distancias, ropa de fondo) y considera normalizar por la caja delimitadora de la mano (puedo añadir esa opción si quieres).

Inferencia continua (tiempo real)
--------------------------------

Si prefieres inferencia continua sin tener que presionar `r` para grabar, usa `infer_realtime.py`. Este script mantiene una ventana deslizante de frames y predice automáticamente cuando la ventana está completa. Muestra subtítulos en pantalla en tiempo real.

Ejemplo:

```powershell
# Usa el modelo RandomForest entrenado sobre secuencias (o un modelo Keras si tienes uno)
python infer_realtime.py --model seq_model_rf.joblib --le le_seq_rf.joblib --window 40 --stride 8 --threshold 0.5
```

Opciones importantes:
- `--window`: número de frames por ventana.
- `--stride`: cada cuántos frames se evalúa una nueva ventana.
- `--threshold`: umbral de confianza mínimo para mostrar subtítulo.
- `--stability`: cuántas ventanas consecutivas deben coincidir para aceptar la predicción.

Consejo: empieza con `--threshold 0.4 --stability 2` para ajustar rápidamente, luego sube el umbral para reducir falsos positivos.

Consejos y mejoras
------------------
- Recolecta varias variaciones por etiqueta (ángulos, distancia, mano izquierda/derecha si aplica).
- Aumenta `samples` por etiqueta para mejor generalización.
- Cambia el clasificador por una red neuronal si quieres mejor rendimiento.
- Añade preprocesado (normalización relativa al bounding box de la mano) para robustez.

Limitaciones
------------

Este repositorio ofrece una canalización completa de ejemplo (captura → entrenamiento → inferencia). No incluye un modelo pre-entrenado para todas las señas del mundo; necesitarás recolectar y entrenar con tus propias etiquetas.

Licencia
--------

Código de ejemplo para uso didáctico.
# Sing-To-Text
