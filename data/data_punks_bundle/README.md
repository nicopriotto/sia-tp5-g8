# Dataset: CryptoPunks (para el VAE — Punto 2 del TP5)

Conjunto de datos elegido para el Autoencoder Variacional: los **CryptoPunks**, pixel art
de **24×24 píxeles**, **10.000** imágenes únicas numeradas #0–#9999.

## ¿Por qué CryptoPunks?

- Son pixel art de baja resolución (como `font.h`), así que el pipeline es parecido.
- Pocos colores planos (solo 219 colores distintos en toda la colección) → fácil de aprender.
- 10.000 muestras con variabilidad estructural → buen caso para que el VAE genere caras nuevas
  "creíbles" (Punto 2c).

## Fuente

Imagen compuesta oficial de Larva Labs (`punks.png`, 2400×2400 RGBA, verificada por SHA-256):
contiene los 10.000 punks en una grilla de **100×100**, cada celda de 24×24.

```bash
# Descargar la grilla (no está versionada, ver .gitignore)
curl -L -o data/data_punks_bundle/raw/punks.png \
  https://raw.githubusercontent.com/larvalabs/cryptopunks/master/punks.png
```

Se eligió la grilla oficial en vez de datasets de Kaggle/HuggingFace porque estos suelen venir
**reescalados** (p. ej. 512×512), lo que obligaría a downscalear de vuelta (lossy). La grilla
tiene los píxeles canónicos nativos en 24×24 → el recorte es **sin pérdida**.

## Cómo se generan los datos

```bash
python3 data/data_punks_bundle/preprocess.py                 # PNGs + tensor RGB, fondo teal (por defecto)
python3 data/data_punks_bundle/preprocess.py --mode all      # genera rgba, rgb y gray
python3 data/data_punks_bundle/preprocess.py --mode rgb gray --no-pngs   # solo tensores elegidos
python3 data/data_punks_bundle/preprocess.py --bg black      # cambiar el fondo (teal/black/white o #RRGGBB)
```

### Color de fondo

Los CryptoPunks originales tienen **fondo transparente** (no un color propio). Para el
autoencoder no conviene la transparencia: el alfa es un canal extra que la red tendría que
reconstruir sin aportar info útil, y "transparente" no tiene RGB que generar. Por eso se compone
sobre un **color plano** (fondo uniforme → la red no malgasta el latente 2D modelando el fondo).

Por defecto se usa el **teal `#638596`**, que es el fondo que **OpenSea** renderiza detrás de cada
punk. Así las muestras generadas por el VAE se ven como CryptoPunks "de verdad" (canónicos), lo que
ayuda a justificar el Punto 2c ("juzgar que la muestra pertenece al conjunto"). Es una convención de
OpenSea, no un dato del asset original (que es transparente).

### Pasos del script (`data/data_punks_bundle/preprocess.py`)

1. **Cargar** `punks.png` → array `(2400, 2400, 4)` uint8 (R, G, B, A en 0–255).
2. **Recortar** la grilla con `reshape` + `transpose` (sin loops, sin pérdida) →
   `(10000, 24, 24, 4)`. El layout es row-major, así que:

   > **`punks[i]` es el CryptoPunk #i**, con `i = fila*100 + col` = ID oficial.

3. **Derivar los canales** según el modo:
   - **rgba** `(10000,24,24,4)` — tal cual, incluye alfa (transparencia).
   - **rgb** `(10000,24,24,3)` — se compone el alfa sobre el fondo elegido (`fg*alfa + bg*(1-alfa)`)
     y se descarta el alfa. Los píxeles transparentes (alfa=0) toman el color de fondo (teal por
     defecto); los semitransparentes (alfa=128) se mezclan; los opacos quedan intactos.
   - **gray** `(10000,24,24,1)` — luminancia Rec.601: `0.299·R + 0.587·G + 0.114·B`.
4. **Guardar** cada tensor como `.npy` en **uint8** (0–255), que es la representación natural y
   compacta del color (float32 lo inflaría 4× sin ganar información).

## Salidas

| Ruta | Contenido | Forma | Tamaño aprox. |
|---|---|---|---|
| `data/data_punks_bundle/raw/punks.png` | grilla oficial (fuente) | 2400×2400×4 | 848 KB |
| `data/data_punks_bundle/punks/punk_XXXX.png` | 10.000 PNG sueltos (RGB, ya compuestos sobre el fondo) | 24×24×3 | ~30 MB |
| `data/data_punks_bundle/tensors/punks_rgba.npy` | tensor RGBA uint8 | (10000,24,24,4) | ~23 MB |
| `data/data_punks_bundle/tensors/punks_rgb.npy` | tensor RGB uint8 (principal) | (10000,24,24,3) | ~17 MB |
| `data/data_punks_bundle/tensors/punks_gray.npy` | tensor gris uint8 | (10000,24,24,1) | ~6 MB |

Nada de esto está versionado (ver `.gitignore`): todo se regenera con el script.
El nombre de archivo de cada PNG es el **ID del punk** (`punk_4521.png` = CryptoPunk #4521).

## Qué guarda cada tensor

Para `punks_rgb.npy` con forma `(10000, 24, 24, 3)`:

- **eje 0** (10000): qué punk → `tensor[4521]` = CryptoPunk #4521.
- **ejes 1, 2** (24, 24): posición del píxel (fila = alto, columna = ancho).
- **eje 3** (3): canales `[R, G, B]` de ese píxel, en 0–255.

Ej.: `tensor[4521, 10, 8, 0]` = rojo del píxel (fila 10, col 8) del punk #4521.
No hay metadata ni etiquetas: la identidad/ID está implícita en el índice del eje 0.

## Cómo cargarlo para entrenar

`uint8` para **almacenar**, `float32` para **computar**. Se normaliza y aplana al cargar:

```python
import numpy as np

X = np.load("data/data_punks_bundle/tensors/punks_rgb.npy").astype(np.float32) / 255.0  # [0,1]
X = X.reshape(X.shape[0], -1)   # (10000, 1728) -> vector plano para el VAE
```

### Dimensiones de entrada según el modo

| Modo | Forma imagen | Dim. de entrada (aplanada) |
|---|---|---|
| rgba | 24×24×4 | **2304** |
| rgb  | 24×24×3 | **1728** ← elegida (color, sin el alfa redundante) |
| gray | 24×24×1 | **576** |
