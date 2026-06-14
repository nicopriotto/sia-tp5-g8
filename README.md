# TP5 — Deep Learning: Autoencoders

## Estado actual

| Ejercicio | Estado |
|---|---|
| **1a** — Autoencoder básico (font.h, latente 2D) | ✅ Completo |
| **1b** — Denoising Autoencoder | 🔲 Pendiente |
| **2** — VAE | 🔲 Pendiente |

---

## 1a — Qué hay disponible para reutilizar en 1b

### Infraestructura lista (no tocar)

`autoencoder_core/` es la librería compartida entre 1a y 1b. Todo lo que sigue ya funciona y
está probado:

| Módulo | Qué hace |
|---|---|
| `model.py` — `Autoencoder` | MLP encoder/decoder con latente 2D. Forward, backward, encode, decode. Dropout incluido. |
| `training.py` — `train_autoencoder` | Loop de entrenamiento con early stopping, L2, gradient clipping, **soporte de denoising ya cableado** (ver abajo). |
| `losses.py` | BCE y MSE con sus deltas de backprop. |
| `activations.py` | tanh, sigmoid, relu, leaky_relu con sus derivadas. |
| `optimizers.py` | Adam y SGD-momentum. |
| `noise.py` — `apply_noise` | **Ruido listo para 1b**: `bit_flip`, `salt_and_pepper`, `gaussian_clipped`. Parámetro `noise_level` en [0, 1]. |
| `evaluation.py` — `evaluate_autoencoder` | Métricas de píxel (exact rate, max error, within-1-pixel). Acepta `target_X` separado del input. |
| `visualization.py` | `plot_latent_scatter`, `plot_reconstructions`, `plot_generated_letter`. |
| `dataset.py` — `load_font_dataset` | Parsea `font.h`, devuelve el dataset como `FontDataset` con `X`, `labels`, etc. |
| `serialization.py` | `save_model_npz` / `load_model_npz`. |

### El denoising ya está cableado en `train_autoencoder`

`training.py` ya maneja denoising de forma transparente a través del config. Los parámetros
relevantes en el JSON de config son:

```json
"denoising": {
  "noise_type": "bit_flip",
  "noise_level": 0.1,
  "train_with_clean_target": true,
  "noise_on_train_only": true
}
```

Con `noise_level > 0`, el entrenador corrompe la entrada de cada batch automáticamente y
reconstruye contra el target limpio (`train_with_clean_target: true`). Para el autoencoder
básico de 1a se corre con `noise_level: 0.0` (sin ruido). Para 1b solo hay que cambiar ese valor.

### Harness de experimentos reutilizable

`experiments/shared/` tiene el scaffolding completo para correr variantes con múltiples seeds
y producir `summary.csv` + `comparison.csv` + `comparison.png`:

```python
from experiments.shared.experiment_utils import (
    load_base_config, run_variant_multi_seed, finalize_experiment
)
```

Los barridos de 1b en `experiments/denoising/` pueden seguir exactamente el mismo patrón
que `experiments/basic/loss.py`, `optimizer_lr.py`, etc.

### Modelo entrenado de 1a (punto de partida opcional)

El mejor modelo de 1a está guardado y se puede cargar:

```python
from autoencoder_core import load_model_npz
model = load_model_npz('autoencoder_basic/output/final_1a/model.npz')
```

Arquitectura: `35 → 24 → 8 → [2] → 8 → 24 → 35`, tanh + sigmoid, Adam lr=3e-3.
Logra reconstrucción exacta de los 32 patrones con 0 errores de píxel.

---

## 1b — Qué hay que hacer

La consigna pide implementar un **Denoising Autoencoder** sobre el mismo dataset `font.h`:

1. **Elegir la arquitectura** conveniente para denoising y justificar la elección.
2. **Barrer niveles de ruido** (`noise_level`) y estudiar hasta qué punto el modelo puede
   eliminar el ruido.

### Punto de entrada sugerido

Crear `autoencoder_denoising/` (o similar) con:
- Un config JSON como `autoencoder_basic/configs/base.json` pero con `noise_level > 0`.
- Un `main.py` que llame a `run_basic_autoencoder` de `autoencoder_basic/main.py` directamente
  (ya soporta denoising), o uno propio análogo.
- Scripts de barrido en `experiments/denoising/` siguiendo el patrón de `experiments/basic/`.

Los tipos de ruido ya implementados en `noise.py`:

| `noise_type` | Descripción |
|---|---|
| `bit_flip` | Flipea píxeles con probabilidad `noise_level` |
| `salt_and_pepper` | Fuerza píxeles a 0 o 1 aleatorio con probabilidad `noise_level` |
| `gaussian_clipped` | Suma ruido gaussiano (σ=`noise_level`) y clipea a [0,1] |

---

## Cómo correr lo que ya existe

```bash
# Corrida de entrega 1a (ya existe en output/final_1a/)
python3 autoencoder_basic/main.py

# Re-correr todos los barridos de 1a
bash experiments/basic/run_all.sh formal   # 10 seeds, ~lento
bash experiments/basic/run_all.sh quick    # 5 seeds, para iterar

# Barridos individuales
python3 experiments/basic/loss.py --mode formal
python3 experiments/basic/architecture.py --mode formal
python3 experiments/basic/optimizer_lr.py --mode formal
python3 experiments/basic/regularization_threshold.py --mode formal
```

## Instalación

```bash
pip install -r requirements.txt  # numpy>=1.26, matplotlib>=3.8
```
