# TP5 — Autoencoders

Autoencoder y denoising autoencoder sobre las imágenes binarias de `font.h`: 32 caracteres
de 5×7 píxeles, aplanados a vectores de 35 valores en {0, 1}.

El código se apoya únicamente en `numpy` (cálculo) y `matplotlib` (gráficos). La red, el
backpropagation y los optimizadores están implementados a mano, sin frameworks de deep learning.

## Estructura

```
autoencoder_core/        red, entrenamiento, ruido, métricas y visualización compartidos
autoencoder_basic/       autoencoder básico con espacio latente 2D (1a)
autoencoder_denoising/   denoising autoencoder (1b)
experiments/             barridos multi-seed de hiperparámetros
data/raw/font.h          dataset de entrada
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
```

El barrido `experiments/denoising/noise_level.py` recorre varios niveles para cada tipo de
ruido y produce una curva por tipo (`curve_<noise_type>.png`) que muestra el error de entrada
contra el de salida y la tasa de reconstrucción exacta a medida que aumenta el ruido.

## Configuración

Cada corrida se define con un archivo JSON (`autoencoder_basic/configs/`,
`autoencoder_denoising/configs/`) que fija dataset, arquitectura, entrenamiento, evaluación y
ruido. Los experimentos parten del config base y sobrescriben los campos que barren.

## Instalación

```bash
pip install -r requirements.txt
```
