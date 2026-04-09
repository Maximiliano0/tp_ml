# 🏃 TP Machine Learning — Human Action Recognition

Trabajo práctico de Machine Learning: clasificación de imágenes de actividades humanas utilizando el dataset [Human Action Recognition (HAR)](https://www.kaggle.com/datasets/meetnagadia/human-action-recognition-har-dataset) de Kaggle.

## 📁 Estructura del proyecto

```
tp_ml/
├── data_prep/                 # Exploración y preparación de datos
│   ├── data_explore.py        # Script de exploración del dataset
│   ├── dataset_description.md # Documentación detallada del dataset
│   ├── dataset_description.html
│   └── output/                # Gráficos generados
├── data_trans/                # Adecuación de imágenes
│   ├── data_adecuate.py       # Resize + CLAHE + RGB + Oversampling
│   ├── data_transform_description.md
│   ├── data_transform_description.html
│   └── output/                # Ejemplos visuales del pipeline
├── data_train/                # Entrenamiento de la CNN
│   ├── train_cnn.py           # ResNet50 Transfer Learning V2, RGB 320×240
│   ├── train_cnn_colab.ipynb  # Notebook Colab (GPU T4, AMP)
│   ├── train_results.md
│   ├── train_results.html
│   └── output/                # Modelo (.pth), métricas (JSON), gráficos
├── datos_har/                 # Dataset (no versionado)
│   ├── dataset/               # 12,600 imágenes originales etiquetadas
│   ├── dataset_tr/            # 12,570 imágenes procesadas (RGB, CLAHE)
│   ├── new_data/              # 5,410 imágenes sin etiqueta (originales)
│   ├── dataset.csv            # Etiquetas: filename → label
│   └── new_data.csv           # Lista de filenames sin etiqueta
├── requirements.txt
└── .gitignore
```

## 📊 Dataset

| Propiedad | Valor |
|-----------|-------|
| **Clases** | 15 actividades humanas |
| **Imágenes etiquetadas** | 12,600 (840/clase) |
| **Imágenes procesadas** | 12,570 (838/clase, oversampling) |
| **Formato original** | JPEG, RGB |
| **Formato procesado** | RGB, 256×192 px |
| **Preprocesamiento** | Resize → CLAHE (LAB) → RGB |

Las 15 clases: `calling`, `clapping`, `cycling`, `dancing`, `drinking`, `eating`, `fighting`, `hugging`, `laughing`, `listening_to_music`, `running`, `sitting`, `sleeping`, `texting`, `using_laptop`.

## 🧠 Modelo (ResNet50 — Transfer Learning V2, 2 fases)

| Propiedad | Valor |
|-----------|-------|
| **Arquitectura** | ResNet50 (pretrained ImageNet1K_V2) + head custom |
| **Entrada** | 3×320×240 (RGB, normalización ImageNet) |
| **Parámetros** | ~24.6M (backbone congelable + head entrenable) |
| **Head** | Linear(2048→256) → ReLU → Dropout(0.3) → Linear(256→15) |
| **Entrenamiento** | 2 fases: head (10 ep) → progressive unfreezing (100 ep máx) |
| **Progressive unfreezing** | F2 épocas 1-10: layer3+layer4+fc; época 11+: todo |
| **Optimizador** | AdamW (head lr=1e-3/1e-4, backbone lr=5e-5, wd=1e-3) |
| **Scheduler** | SequentialLR (LinearLR warmup 5ep + CosineAnnealingLR) |
| **Early stopping** | Paciencia 10 épocas (por test loss, fase 2) |
| **Label smoothing** | 0.05 |
| **Gradient clipping** | max_norm=1.0 |
| **MixUp/CutMix** | Desactivado (AUGMIX_PROB=0.0) |
| **Augmentation** | HFlip, Rotation(20°), ColorJitter, RandAugment(mag=6), Erasing(0.10) |

## 🚀 Cómo ejecutar

```powershell
# Crear entorno virtual
python -m venv tp_ml_env
.\tp_ml_env\Scripts\Activate.ps1

# Instalar dependencias
pip install -r requirements.txt

# 1. Exploración del dataset
python data_prep/data_explore.py

# 2. Adecuación de imágenes (CLAHE + RGB + oversampling)
python data_trans/data_adecuate.py

# 3. Entrenamiento de la CNN
python data_train/train_cnn.py
```

> La primera ejecución de `data_explore.py` descarga el dataset desde Kaggle (~328 MB). Se requiere `~/.kaggle/kaggle.json` con credenciales válidas.

## 📄 Documentación

- **Dataset**: [`data_prep/dataset_description.md`](data_prep/dataset_description.md)
- **Transformación**: [`data_trans/data_transform_description.md`](data_trans/data_transform_description.md)
- **Entrenamiento**: [`data_train/train_results.md`](data_train/train_results.md)
- **Notebook Colab**: [`data_train/train_cnn_colab.ipynb`](data_train/train_cnn_colab.ipynb)
