# Ejercicio 1a — Autoencoder básico: resultados

## ¿Aprendió las 32 letras?

**Sí.** La corrida final (`autoencoder_basic/output/final_1a/`) entrenó sobre el dataset completo
(`all_32`, `font.h`, 32 patrones de 5×7 = 35 píxeles binarios) y obtuvo:

| Métrica | Valor |
|---|---|
| `exact_reconstruction_rate` | **1.0** (32/32 exactas) |
| `max_pixel_error` | **0** |
| `mean_pixel_error` | **0.0** |
| `all_patterns_within_one_pixel` | `true` |
| Épocas entrenadas / mejor época | 2000 / 1956 |

No fue necesario usar subconjunto.  
Evidencia directa: `autoencoder_basic/output/final_1a/metrics.json`.

---

## Arquitectura y técnica de optimización elegidas

### Arquitectura

MLP simétrico con cuello de botella en 2 dimensiones:

```
Input(35) → Dense(24) → Dense(8) → Latent(2) → Dense(8) → Dense(24) → Output(35)
```

- Activación capas ocultas: **tanh**
- Activación salida: **sigmoid** (produce probabilidades por píxel en [0,1])
- Inicialización: Xavier uniforme

### Optimización y función de pérdida

| Parámetro | Valor |
|---|---|
| Optimizador | **Adam** |
| Learning rate | **3×10⁻³** |
| Función de pérdida | Binary cross-entropy |
| Batch size | 8 |
| L2 weight decay | 1×10⁻⁵ |
| Epochs máx / patience | 2000 / 100 |

### Justificación (barridos formales, 10 seeds cada uno)

**Learning rate es la palanca dominante** (ver `experiments/output/basic/optimizer_lr/comparison.csv`):

| Variante | full_success_mean | max_pixel_error_mean |
|---|---|---|
| Adam lr=1×10⁻³ (baseline) | 0.0 | 10.1 |
| Adam lr=3×10⁻³ | **0.8** | **0.9** |
| Adam lr=1×10⁻² | 0.6 | 2.1 |
| SGD-momentum (todos lr) | 0.0 | ≥13 |

Adam lr=3×10⁻³ maximiza `full_success_rate_mean` y minimiza `max_pixel_error_mean` con la menor
varianza entre seeds exitosas.

Los barridos de arquitectura (`experiments/output/basic/architecture/comparison.csv`),
función de pérdida (`loss/comparison.csv`) y regularización (`regularization_threshold/comparison.csv`)
se realizaron con el learning rate base de 1×10⁻³; sus resultados sirven como estudio comparativo
de esos factores pero reflejan el baseline anterior a la optimización de lr. Con lr=3×10⁻³
la arquitectura base `[24,8]`/`tanh` demostró ser suficiente para alcanzar el objetivo sin necesidad
de capas adicionales.

La seed ganadora (38200) fue seleccionada con el ranking de la task 06:
`all_patterns_within_one_pixel` → `exact_reconstruction_rate` → menor `max_pixel_error` → menor seed.

---

## Representación en el espacio latente 2D

Los 32 caracteres proyectados al espacio latente de 2 dimensiones:

📄 `autoencoder_basic/output/final_1a/latent_scatter.png`

Cada punto está anotado con el label del carácter correspondiente.

---

## Reconstrucciones por letra

Imagen con (original | probabilidad | reconstrucción binaria) para los 32 caracteres:

📄 `autoencoder_basic/output/final_1a/reconstructions.png`

Todos los 32 patrones tienen **error de píxel = 0** (pixel_error_por_patron: todos ceros en
`metrics.json`).

---

## Letra nueva generada

El modelo puede generar patrones que **no pertenecen al conjunto de entrenamiento** interpolando
o extrapolando en el espacio latente de 2 dimensiones.

Método: búsqueda en grilla de 41×41 puntos sobre el espacio latente; se selecciona el punto
con mayor confianza cuyo patrón decodificado binarizado no coincide con ningún patrón conocido.

| Atributo | Valor |
|---|---|
| Método | `latent_grid_search` |
| Punto latente generado | `[3.507, 0.422]` |

📄 `autoencoder_basic/output/final_1a/generated_letters.png`

La imagen muestra el scatter del espacio latente con los 32 patrones de entrenamiento (azul)
y la estrella roja marcando el punto latente generado, junto con el patrón 5×7 resultante.

---

## Artefactos de la entrega

Todos en `autoencoder_basic/output/final_1a/`:

| Archivo | Descripción |
|---|---|
| `metrics.json` | Métricas completas de la corrida |
| `training_history.csv` | Loss y métricas por época |
| `latent_scatter.png` | Espacio latente 2D con labels |
| `reconstructions.png` | Original vs. reconstrucción por letra |
| `generated_letters.png` | Latent scatter + letra nueva generada |
| `model.npz` | Pesos del modelo entrenado |
| `resolved_config.json` | Configuración completa utilizada |
| `subset_manifest.json` | Descripción del dataset (all_32) |
