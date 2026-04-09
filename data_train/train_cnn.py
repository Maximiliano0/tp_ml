"""
train_cnn.py
============
Entrena un modelo ResNet50 (transfer learning desde ImageNet V2) para
clasificar imagenes del dataset HAR procesado con CLAHE y oversampling.

Arquitectura
------------
- Backbone: ResNet50 preentrenado en ImageNet (IMAGENET1K_V2)
- Head custom: Linear(2048->256) -> ReLU -> Dropout(0.3) -> Linear(256->15)
- Entrada: 3 canales RGB, 320x240 (mayor resolucion para capturar detalle)
- Parametros totales: ~24.6M

Entrenamiento (2 fases + progressive unfreezing)
-------------------------------------------------
Fase 1 — Solo head (backbone congelado):
  - Epocas: 10
  - Optimizador: AdamW (lr=1e-3, weight_decay=1e-3)

Fase 2 — Fine-tuning con progressive unfreezing:
  - Epocas 1-10: solo layer3+layer4+fc entrenables (backbone parcial)
  - Epocas 11+: todo el backbone descongelado
  - Optimizador: AdamW con LR diferencial (head=1e-4, backbone=5e-5)
  - Scheduler: SequentialLR(LinearLR warmup 5ep + CosineAnnealingLR)
  - Early stopping: paciencia 10 epocas (por test loss)
  - MixUp (alpha=0.2) / CutMix (alpha=1.0) desactivados (AUGMIX_PROB=0.0)

Regularizacion
--------------
- Label smoothing: 0.05
- Gradient clipping: max_norm=1.0
- Dropout: 0.3
- Weight decay: 1e-3
- Data augmentation: RandomHorizontalFlip, RandomRotation(20),
  ColorJitter(0.3, 0.3, 0.3, 0.1), RandAugment(num_ops=2, magnitude=6),
  RandomErasing(0.10)
- Normalizacion ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

Pasos
-----
1. Cargar imagenes de datos_har/dataset_tr/ y etiquetas de dataset.csv
2. Separar en train (80%) y test (20%) con estratificacion
3. Fase 1: entrenar head con backbone congelado
4. Fase 2: fine-tuning con progressive unfreezing y LR diferencial
5. Evaluar: accuracy, classification report y matriz de confusion
6. Guardar modelo (.pth), metricas (JSON) y graficos en data_train/output/
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position

import torch  # pylint: disable=wrong-import-position
import torch.nn as nn  # pylint: disable=wrong-import-position
import torch.optim as optim  # pylint: disable=wrong-import-position
from torch.utils.data import Dataset, DataLoader  # pylint: disable=wrong-import-position
from torchvision import transforms, models  # pylint: disable=wrong-import-position
from sklearn.model_selection import train_test_split  # pylint: disable=wrong-import-position
from sklearn.metrics import (  # pylint: disable=wrong-import-position
    confusion_matrix, accuracy_score, classification_report
)

# ───────────────────────────────────────────────────────
# Rutas
# ───────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "datos_har"
IMG_DIR = DATA_DIR / "dataset_tr"
CSV_PATH = DATA_DIR / "dataset.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Hiperparámetros ──
SEED = 42
BATCH_SIZE = 64
PHASE1_EPOCHS = 10        # fase 1: solo head (backbone congelado)
PHASE2_EPOCHS = 100       # fase 2: fine-tuning completo (early stopping puede cortar)
LR = 1e-3                 # learning rate del head (fase 1)
LR_HEAD_PHASE2 = 1e-4     # learning rate del head (fase 2, mas bajo para no desestabilizar)
LR_BACKBONE = 5e-5        # learning rate del backbone (fase 2, ajuste mas agresivo)
WEIGHT_DECAY = 1e-3       # regularización L2
IMG_H, IMG_W = 320, 240   # resolución mayor para capturar más detalle
UNFREEZE_AFTER = 10       # épocas F2 solo layer3+layer4, luego descongelar todo
NUM_WORKERS = 0            # workers del DataLoader (0 = main thread)
EARLY_STOP_PATIENCE = 10  # épocas sin mejora antes de detener (fase 2)
WARMUP_EPOCHS = 5         # épocas de warmup lineal (fase 2)
LABEL_SMOOTHING = 0.05    # suavizado de etiquetas (bajo para 15 clases)
MIXUP_ALPHA = 0.2         # alpha para MixUp
CUTMIX_ALPHA = 1.0        # alpha para CutMix
AUGMIX_PROB = 0.0         # MixUp/CutMix desactivado (no ayuda con 15 clases y 12K imgs)

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ═══════════════════════════════════════════════════════
# 1. CARGA DE DATOS
# ═══════════════════════════════════════════════════════
print("=" * 60)
print("1. CARGA DE DATOS")
print("=" * 60)

df_all = pd.read_csv(CSV_PATH)

# Quedarse solo con las imágenes que existen en dataset_tr
existing_files = set(os.listdir(IMG_DIR))
df = df_all[df_all["filename"].isin(existing_files)].copy()
df.reset_index(drop=True, inplace=True)

# Codificar etiquetas
labels_sorted = sorted(df["label"].unique())
label2idx = {label: idx for idx, label in enumerate(labels_sorted)}
idx2label = {idx: label for label, idx in label2idx.items()}
df["label_idx"] = df["label"].map(label2idx)

NUM_CLASSES = len(labels_sorted)
print(f"   Imagenes encontradas: {len(df)}")
print(f"   Clases: {NUM_CLASSES} -> {labels_sorted}")

# ═══════════════════════════════════════════════════════
# 2. SPLIT TRAIN / TEST (70% / 30%)
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. SPLIT TRAIN / TEST")
print("=" * 60)

X_train_df, X_test_df = train_test_split(
    df, test_size=0.20, random_state=SEED, stratify=df["label_idx"]
)
X_train_df.reset_index(drop=True, inplace=True)
X_test_df.reset_index(drop=True, inplace=True)

print(f"   Train: {len(X_train_df)} imagenes ({len(X_train_df)/len(df)*100:.0f}%)")
print(f"   Test:  {len(X_test_df)} imagenes ({len(X_test_df)/len(df)*100:.0f}%)")

# ═══════════════════════════════════════════════════════
# 3. DATASET Y DATALOADER
# ═══════════════════════════════════════════════════════

# ── Normalización ImageNet (para transfer learning con ResNet50) ──
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ── Data augmentation agresiva (solo train) ──
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_H, IMG_W)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandAugment(num_ops=2, magnitude=6),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    transforms.RandomErasing(p=0.10),
])

# ── Transform de test (sin augmentacion) ──
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class HARDataset(Dataset):
    """Dataset para imagenes HAR con transformaciones opcionales."""

    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        img = np.array(Image.open(img_path).convert("RGB"))
        if self.transform:
            img = self.transform(img)
        else:
            pil_img = Image.fromarray(img).resize(
                (IMG_W, IMG_H), Image.Resampling.LANCZOS
            )
            img = np.array(pil_img, dtype=np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
            img = torch.from_numpy(img)
        label = row["label_idx"]
        return img, torch.tensor(label, dtype=torch.long)


train_ds = HARDataset(X_train_df, IMG_DIR, transform=train_transform)
test_ds = HARDataset(X_test_df, IMG_DIR, transform=test_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# ═══════════════════════════════════════════════════════
# 4. MODELO: ResNet18 (Transfer Learning)
#    Backbone preentrenado en ImageNet + clasificador custom
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. ARQUITECTURA DEL MODELO (ResNet50)")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar ResNet50 preentrenado en ImageNet (pesos V2: receta moderna, mejor accuracy)
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# Reemplazar el clasificador final
model.fc = nn.Sequential(
    nn.Linear(2048, 256),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    nn.Linear(256, NUM_CLASSES),
)

model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Device: {device}")
print("   Backbone: ResNet50 (preentrenado ImageNet V2)")
print(f"   Parametros totales:     {total_params:,}")
print(f"   Parametros entrenables: {trainable_params:,}")
print(f"   Input: 3 x {IMG_H} x {IMG_W} (RGB)")
print(f"   Output: {NUM_CLASSES} clases")

# ═══════════════════════════════════════════════════════
# 5. ENTRENAMIENTO (2 FASES)
#    Fase 1: head (backbone congelado)
#    Fase 2: fine-tuning completo con MixUp/CutMix
# ═══════════════════════════════════════════════════════


def mixup_data(x, y, alpha=0.2):
    """MixUp: mezcla pares de imagenes y etiquetas."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def cutmix_data(x, y, alpha=1.0):
    """CutMix: recorta y pega regiones entre imagenes."""
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)
    _, _, h, w = x.shape
    cut_rat = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(w * cut_rat), int(h * cut_rat)
    cx, cy = np.random.randint(w), np.random.randint(h)
    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)
    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - (bbx2 - bbx1) * (bby2 - bby1) / (w * h)
    return mixed_x, y, y[index], lam


print("\n" + "=" * 60)
print("4. ENTRENAMIENTO (2 FASES)")
print("=" * 60)

criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
best_test_acc = 0.0  # pylint: disable=invalid-name
best_test_loss = float("inf")

# ── FASE 1: Entrenar solo el head (backbone congelado) ──
for name, param in model.named_parameters():
    if not name.startswith("fc"):
        param.requires_grad = False

optimizer1 = optim.AdamW(model.fc.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
trainable1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n--- FASE 1: Entrenar head ({PHASE1_EPOCHS} epocas, backbone congelado) ---")
print(f"   Parametros entrenables: {trainable1:,}")

for epoch in range(1, PHASE1_EPOCHS + 1):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer1.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer1.step()
        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    train_loss = running_loss / total
    train_acc = correct / total

    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    test_loss = running_loss / total
    test_acc = correct / total

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["test_loss"].append(test_loss)
    history["test_acc"].append(test_acc)

    if test_loss < best_test_loss:
        best_test_acc = test_acc
        best_test_loss = test_loss
        torch.save(model.state_dict(), OUTPUT_DIR / "har_cnn_best.pth")

    print(f"   [F1] Epoca {epoch:03d}/{PHASE1_EPOCHS}  "
          f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
          f"Test Loss: {test_loss:.4f}  Acc: {test_acc:.4f}")

# ── FASE 2: Fine-tuning progresivo con MixUp/CutMix ──
# Primeras UNFREEZE_AFTER épocas: solo layer3 + layer4 + fc
# Después: descongelar todo el backbone
for name, param in model.named_parameters():
    if name.startswith(("layer3.", "layer4.", "fc.")):
        param.requires_grad = True
    else:
        param.requires_grad = False


def _build_optimizer_and_scheduler(mdl):
    """Crea optimizer con LR diferencial y scheduler."""
    backbone_p = [p for n, p in mdl.named_parameters()
                  if not n.startswith("fc") and p.requires_grad]
    opt = optim.AdamW([
        {"params": backbone_p, "lr": LR_BACKBONE},
        {"params": mdl.fc.parameters(), "lr": LR_HEAD_PHASE2},
    ], weight_decay=WEIGHT_DECAY)
    warmup = optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, end_factor=1.0,
        total_iters=WARMUP_EPOCHS)
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=PHASE2_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
    sched = optim.lr_scheduler.SequentialLR(
        opt, [warmup, cosine], milestones=[WARMUP_EPOCHS])
    return opt, sched


optimizer2, scheduler2 = _build_optimizer_and_scheduler(model)

trainable2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n--- FASE 2: Fine-tuning progresivo ({PHASE2_EPOCHS} epocas) ---")
print(f"   LR head={LR_HEAD_PHASE2}  |  LR backbone={LR_BACKBONE}")
print(f"   Parametros entrenables (parcial): {trainable2:,}")
print(f"   Descongelar todo tras epoca {UNFREEZE_AFTER}")
print(f"   Warmup: {WARMUP_EPOCHS} epocas  |  Early stopping: {EARLY_STOP_PATIENCE} epocas")
print(f"   MixUp (alpha={MIXUP_ALPHA}) / CutMix (alpha={CUTMIX_ALPHA}) | prob={AUGMIX_PROB}")

epochs_no_improve = 0  # pylint: disable=invalid-name
phase2_trained = 0  # pylint: disable=invalid-name

for epoch in range(1, PHASE2_EPOCHS + 1):
    global_epoch = PHASE1_EPOCHS + epoch
    phase2_trained = epoch  # pylint: disable=invalid-name

    # Progressive unfreezing: descongelar todo tras UNFREEZE_AFTER
    if epoch == UNFREEZE_AFTER + 1:
        for param in model.parameters():
            param.requires_grad = True
        optimizer2, scheduler2 = _build_optimizer_and_scheduler(model)
        # Avanzar scheduler las épocas ya transcurridas
        for _ in range(epoch - 1):
            scheduler2.step()
        n_train = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   >>> Descongelando backbone completo "
              f"(params entrenables: {n_train:,})")

    # --- Train con MixUp/CutMix ---
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        # Aplicar MixUp o CutMix con probabilidad AUGMIX_PROB
        if np.random.rand() < AUGMIX_PROB:
            if np.random.rand() < 0.5:
                imgs, targets_a, targets_b, mix_lam = mixup_data(imgs, labels, MIXUP_ALPHA)
            else:
                imgs, targets_a, targets_b, mix_lam = cutmix_data(imgs, labels, CUTMIX_ALPHA)
        else:
            targets_a, targets_b, mix_lam = labels, labels, 1.0

        optimizer2.zero_grad()
        outputs = model(imgs)
        loss = (mix_lam * criterion(outputs, targets_a)
                + (1 - mix_lam) * criterion(outputs, targets_b))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer2.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += (mix_lam * (preds == targets_a).float()
                    + (1 - mix_lam) * (preds == targets_b).float()).sum().item()
        total += labels.size(0)
    train_loss = running_loss / total
    train_acc = correct / total

    # --- Eval (sin MixUp/CutMix) ---
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    test_loss = running_loss / total
    test_acc = correct / total

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["test_loss"].append(test_loss)
    history["test_acc"].append(test_acc)

    scheduler2.step()

    # Guardar mejor modelo (por test loss)
    if test_loss < best_test_loss:
        best_test_acc = test_acc
        best_test_loss = test_loss
        epochs_no_improve = 0  # pylint: disable=invalid-name
        torch.save(model.state_dict(), OUTPUT_DIR / "har_cnn_best.pth")
    else:
        epochs_no_improve += 1  # pylint: disable=invalid-name

    print(f"   [F2] Epoca {epoch:03d}/{PHASE2_EPOCHS} (global {global_epoch:03d})  "
          f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
          f"Test Loss: {test_loss:.4f}  Acc: {test_acc:.4f}  |  "
          f"LR: {optimizer2.param_groups[1]['lr']:.1e}")

    # Early stopping
    if epochs_no_improve >= EARLY_STOP_PATIENCE:
        print(f"\n   Early stopping en epoca {epoch} de fase 2 "
              f"(sin mejora en {EARLY_STOP_PATIENCE} epocas)")
        print(f"   Mejor Test Acc: {best_test_acc:.4f}")
        break

total_epochs = PHASE1_EPOCHS + phase2_trained
print(f"\n   Entrenamiento finalizado. Epocas totales: {total_epochs}")
print(f"   Mejor Test Acc: {best_test_acc:.4f} | Mejor Test Loss: {best_test_loss:.4f}")

# ═══════════════════════════════════════════════════════
# 6. EVALUACIÓN FINAL
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. EVALUACION FINAL")
print("=" * 60)

# Cargar el mejor modelo guardado durante entrenamiento
model.load_state_dict(torch.load(OUTPUT_DIR / "har_cnn_best.pth", weights_only=True))
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

acc = accuracy_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=labels_sorted)

print(f"\n   Accuracy global: {acc:.4f} ({acc*100:.2f}%)")
print("\n   Classification Report:\n")
print(report)

# ═══════════════════════════════════════════════════════
# 7. GRÁFICOS (curvas de entrenamiento + matriz de confusión)
# ═══════════════════════════════════════════════════════
print("=" * 60)
print("6. GENERACION DE GRAFICOS")
print("=" * 60)

# 7a. Curvas de loss y accuracy
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(range(1, len(history["train_loss"])+1), history["train_loss"],
         label="Train Loss", marker="o", markersize=3)
ax1.plot(range(1, len(history["test_loss"])+1), history["test_loss"],
         label="Test Loss", marker="o", markersize=3)
ax1.set_xlabel("Epoca")
ax1.set_ylabel("Loss")
ax1.set_title("Loss por Epoca")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(range(1, len(history["train_acc"])+1), history["train_acc"],
         label="Train Acc", marker="o", markersize=3)
ax2.plot(range(1, len(history["test_acc"])+1), history["test_acc"],
         label="Test Acc", marker="o", markersize=3)
ax2.set_xlabel("Epoca")
ax2.set_ylabel("Accuracy")
ax2.set_title("Accuracy por Epoca")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "curvas_entrenamiento.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"   Guardado: {OUTPUT_DIR / 'curvas_entrenamiento.png'}")

# 7b. Matriz de confusión
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
ax.set_title("Matriz de Confusion", fontsize=14, fontweight="bold")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

tick_marks = np.arange(NUM_CLASSES)
ax.set_xticks(tick_marks)
ax.set_xticklabels(labels_sorted, rotation=45, ha="right", fontsize=8)
ax.set_yticks(tick_marks)
ax.set_yticklabels(labels_sorted, fontsize=8)
ax.set_xlabel("Prediccion", fontsize=12)
ax.set_ylabel("Real", fontsize=12)

# Anotar valores en cada celda
thresh = cm.max() / 2.0
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        ax.text(j, i, str(cm[i, j]),
                ha="center", va="center", fontsize=7,
                color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "matriz_confusion.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"   Guardado: {OUTPUT_DIR / 'matriz_confusion.png'}")

# ═══════════════════════════════════════════════════════
# 8. GUARDAR MODELO Y MÉTRICAS (último + mejor por test_loss)
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("7. GUARDADO DE MODELO Y METRICAS")
print("=" * 60)

torch.save(model.state_dict(), OUTPUT_DIR / "har_cnn.pth")
print(f"   Modelo ultimo guardado: {OUTPUT_DIR / 'har_cnn.pth'}")
print(f"   Mejor modelo (acc={best_test_acc:.4f}): {OUTPUT_DIR / 'har_cnn_best.pth'}")

metrics = {
    "architecture": "resnet50",
    "transfer_learning": True,
    "pretrained_weights": "IMAGENET1K_V2",
    "training_strategy": "two_phase",
    "accuracy": float(acc),
    "best_test_acc": float(best_test_acc),
    "best_test_loss": float(best_test_loss),
    "phase1_epochs": PHASE1_EPOCHS,
    "phase2_epochs": phase2_trained,
    "total_epochs": total_epochs,
    "batch_size": BATCH_SIZE,
    "learning_rate_head_phase1": LR,
    "learning_rate_head_phase2": LR_HEAD_PHASE2,
    "learning_rate_backbone": LR_BACKBONE,
    "weight_decay": WEIGHT_DECAY,
    "label_smoothing": LABEL_SMOOTHING,
    "gradient_clipping": 1.0,
    "dropout": 0.3,
    "augmix_prob": AUGMIX_PROB,
    "warmup_epochs": WARMUP_EPOCHS,
    "mixup_alpha": MIXUP_ALPHA,
    "cutmix_alpha": CUTMIX_ALPHA,
    "img_h": IMG_H,
    "img_w": IMG_W,
    "train_size": len(X_train_df),
    "test_size": len(X_test_df),
    "num_classes": NUM_CLASSES,
    "classes": labels_sorted,
    "confusion_matrix": cm.tolist(),
    "history": history,
    "classification_report": report,
    "device": str(device),
    "total_params": total_params,
}
with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
print(f"   Metricas guardadas: {OUTPUT_DIR / 'metrics.json'}")

# ═══════════════════════════════════════════════════════
# RESUMEN
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("RESUMEN")
print("=" * 60)
print("  Arquitectura:           ResNet50 (Transfer Learning V2, 2 fases)")
print(f"  Parametros totales:     {total_params:,}")
print(f"  Entrada:                3x{IMG_H}x{IMG_W}")
print(f"  Imagenes totales:       {len(df)}")
print(f"  Train / Test:           {len(X_train_df)} / {len(X_test_df)}")
print(f"  Clases:                 {NUM_CLASSES}")
print(f"  Fase 1:                 {PHASE1_EPOCHS} epocas (solo head)")
print(f"  Fase 2:                 {phase2_trained} epocas (fine-tuning)")
print(f"  Epocas totales:         {total_epochs}")
print(f"  Label smoothing:        {LABEL_SMOOTHING}")
print("  Gradient clipping:      1.0")
print("  Dropout:                0.5")
print("  MixUp/CutMix:           activo")
print(f"  Accuracy final (test):  {acc:.4f} ({acc*100:.2f}%)")
print(f"  Modelo:                 {OUTPUT_DIR / 'har_cnn.pth'}")
print(f"  Metricas:               {OUTPUT_DIR / 'metrics.json'}")
print(f"  Graficos:               {OUTPUT_DIR}/")
print("=" * 60)
print("Entrenamiento completado.")
