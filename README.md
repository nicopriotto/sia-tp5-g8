# TP5 — Autoencoders

Autoencoder y denoising autoencoder sobre las imágenes binarias de `font.h` (32 caracteres de
5×7 píxeles, aplanados a vectores de 35 valores en {0, 1}), y un autoencoder variacional (VAE)
generativo sobre los CryptoPunks (pixel art de 24×24 píxeles RGB).

El código se apoya en `numpy` (cálculo), `matplotlib` (gráficos) y `Pillow` (lectura/escritura de
imágenes). La red, el backpropagation y los optimizadores están implementados a mano, sin
frameworks de deep learning.

## Estructura

```
autoencoder_core/        red, entrenamiento, ruido, métricas y visualización compartidos
autoencoder_basic/       autoencoder básico con espacio latente 2D (1a)
autoencoder_denoising/   denoising autoencoder (1b)
autoencoder_vae/         autoencoder variacional sobre cryptopunks (2)
experiments/             barridos multi-seed de hiperparámetros
data/raw/font.h          dataset de letras (1)
data/tensors/            tensores de cryptopunks para el VAE (2)
docs/                    consigna y material teórico
```

## Autoencoder básico

Un autoencoder es un perceptrón multicapa simétrico que comprime la entrada hasta un cuello
de botella (el espacio latente) y luego la reconstruye. El encoder mapea cada patrón de 35
píxeles a un código latente; el decoder reconstruye los 35 píxeles desde ese código. Se
entrena minimizando la diferencia entre la entrada y su reconstrucción, de modo que el código
latente capture la estructura del conjunto.

Arquitectura:

```
35 → 24 → 8 → 2 → 8 → 24 → 35
```

Activación `tanh` en las capas ocultas y `sigmoid` a la salida (probabilidad por píxel, que se
umbraliza en 0.5 para obtener el bit). Se entrena con Adam (learning rate 3e-3), binary
cross-entropy, batches de 8 y weight decay L2. Con esta configuración reconstruye los 32
patrones sin errores de píxel.

El espacio latente de dos dimensiones permite graficar los 32 caracteres en un plano y generar
caracteres nuevos: eligiendo un punto del espacio latente que no corresponde a ningún patrón de
entrenamiento y pasándolo por el decoder se obtiene un patrón inédito.

Correr:

```bash
python3 autoencoder_basic/main.py
```

Genera en `autoencoder_basic/output/<config>/`: métricas, historia de entrenamiento, scatter
del espacio latente, reconstrucciones por carácter, carácter generado y pesos del modelo.

## Denoising autoencoder

El denoising autoencoder aprende a reconstruir un patrón limpio a partir de una versión
corrompida con ruido. Durante el entrenamiento, en cada batch se corrompe la entrada y se la
compara contra el patrón original sin ruido. Así la red aprende a descartar el ruido y a
recuperar la estructura subyacente.

A diferencia del básico, un cuello de botella de dos dimensiones no alcanza: bajo ruido las
entradas caen fuera de la variedad aprendida y la reconstrucción se degrada por debajo de la
propia entrada ruidosa. Por eso se usa un espacio latente más amplio:

```
35 → 32 → 24 → 16 → 24 → 32 → 35
```

El resto de la configuración es la del autoencoder básico. El ruido se inyecta solo en el
entrenamiento; la selección del mejor checkpoint se hace sobre la reconstrucción limpia, ya que
el ruido durante el entrenamiento actúa como regularizador.

Tipos de ruido disponibles (`autoencoder_core/noise.py`), parametrizados por `noise_level`:

- `bit_flip`: invierte cada píxel con probabilidad `noise_level`.
- `salt_and_pepper`: fuerza píxeles a 0 o 1 al azar con probabilidad `noise_level`.
- `gaussian_clipped`: suma ruido gaussiano de desvío `noise_level` y recorta a [0, 1].

La evaluación mide la capacidad de denoising: corrompe el dataset limpio con muchas
realizaciones de ruido, reconstruye y compara contra el patrón original, promediando el error
de píxel sobre las realizaciones.

Correr:

```bash
python3 autoencoder_denoising/main.py
```

Genera en `autoencoder_denoising/output/<config>/`: métricas de denoising, reconstrucciones
limpias, tríos (original / con ruido / reconstruido) y pesos del modelo.

## Autoencoder variacional (VAE)

El punto 2 extiende el autoencoder a un esquema variacional sobre un dataset nuevo: los
**CryptoPunks**, pixel art de 24×24 píxeles RGB (1728 valores por imagen, normalizados a
[0, 1]). La preparación del dataset está en `data/README.md`.

Un autoencoder común aprende un espacio latente irregular y con huecos: un punto al azar no
decodifica en algo válido, así que no sirve para generar. El VAE lo resuelve haciendo que el
encoder produzca, en vez de un punto, una distribución `N(mu, sigma²)` por imagen. El código se
muestrea con el *reparametrization trick* `z = mu + exp(0.5·logvar)·eps` (con `eps ~ N(0, I)`), y
un término de divergencia KL empuja esas distribuciones hacia el prior `N(0, I)`. El latente
queda continuo y compacto, de modo que muestrear `z ~ N(0, I)` y decodificar produce punks nuevos.

La función de costo combina reconstrucción y regularización:

```
loss = reconstrucción + beta · KL
```

`beta` controla el equilibrio: valores altos ordenan más el latente (mejor para generar) a costa
de reconstrucciones más borrosas; valores bajos reconstruyen más nítido pero el latente pierde
estructura.

Arquitectura (latente de 10 dimensiones; el encoder agrega un segundo cabezal para `logvar`):

```
1728 → 512 → 128 → [mu, logvar] (10) → z → 128 → 512 → 1728
```

El modelo (`autoencoder_vae/model.py`) hereda del autoencoder base y reutiliza optimizadores,
activaciones y el loop de entrenamiento; solo agrega los dos cabezales, el muestreo y el término
KL. La selección del checkpoint se hace por la loss total (no por error de píxel, que no aplica a
imágenes a color continuas).

Correr:

```bash
python3 autoencoder_vae/main.py                                      # entrena (config base.json)
python3 autoencoder_vae/main.py autoencoder_vae/configs/quick.json   # iteración rápida (subset)
python3 autoencoder_vae/generation.py autoencoder_vae/output/base    # genera punks nuevos (2c)
python3 autoencoder_vae/visualization.py autoencoder_vae/output/base # figuras de análisis
```

`main.py` entrena y guarda métricas, historia y pesos. `generation.py` samplea del prior y
guarda los punks generados (PNGs individuales + grilla). `visualization.py` produce las figuras de
reconstrucción, muestras generadas, interpolación latente y proyección PCA del espacio latente.

## Experimentos

Los barridos repiten cada configuración con varias seeds y agregan los resultados en
`summary.csv`, `comparison.csv` y `comparison.png`.

```bash
# Autoencoder básico: loss, arquitectura, optimizador/learning rate y regularización
bash experiments/basic/run_all.sh quick      # 5 seeds
bash experiments/basic/run_all.sh formal     # 10 seeds

# Denoising: tipo de ruido y barrido de niveles de ruido
bash experiments/denoising/run_all.sh quick
bash experiments/denoising/run_all.sh formal

# VAE: barrido de beta y de dimensión latente
bash experiments/vae/run_all.sh quick
bash experiments/vae/run_all.sh formal
```

El barrido `experiments/denoising/noise_level.py` recorre varios niveles para cada tipo de
ruido y produce una curva por tipo (`curve_<noise_type>.png`) que muestra el error de entrada
contra el de salida y la tasa de reconstrucción exacta a medida que aumenta el ruido.

Los barridos del VAE (`experiments/vae/beta.py` y `latent_dim.py`) comparan la loss de
reconstrucción, la KL y la loss total entre valores de `beta` y dimensiones latentes; usan un
subconjunto de punks y menos épocas para que el barrido sea factible.

## Configuración

Cada corrida se define con un archivo JSON (`autoencoder_basic/configs/`,
`autoencoder_denoising/configs/`, `autoencoder_vae/configs/`) que fija dataset, arquitectura,
entrenamiento, evaluación y, según el caso, ruido o `beta`. Los experimentos parten del config
base y sobrescriben los campos que barren.

## Instalación

```bash
pip install -r requirements.txt
```
