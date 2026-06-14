# Documento de Hiperparámetros para Autoencoder 1a

Este documento propone un barrido experimental para el autoencoder básico del ejercicio 1a y deja definidos, además, los parámetros de ruido que pueden reutilizarse en 1b (Denoising Autoencoder). El foco está en comparar configuraciones de manera sistemática, no en desarrollar teoría extensa.

## Restricciones fijas de la consigna

| Elemento | Valor fijo | Observación |
| --- | --- | --- |
| `input_dim` | `35` | Cada patrón es una grilla binaria `5x7`, aplanada como vector. |
| `output_dim` | `35` | La reconstrucción debe tener la misma dimensionalidad que la entrada. |
| `latent_dim` | `2` | La consigna pide representación en espacio latente bidimensional. |
| `dataset` | `32` caracteres | Patrones del archivo `font.h`, codificados como `32 x 7` enteros donde cada fila usa `5` bits útiles. |
| Objetivo de reconstrucción | `<= 1` píxel incorrecto por patrón | Meta principal para juzgar si la configuración es satisfactoria. |

## Supuestos de modelado

- Representación binaria `0/1` para los patrones `5x7`.
- En `font.h`, cada carácter está guardado como `7` valores hexadecimales; para entrenar el autoencoder hay que expandir cada fila a `5` bits y aplanar a un vector de longitud `35`.
- Autoencoder `MLP` denso, no convolucional.
- Decodificador con salida `sigmoid` para producir probabilidades por píxel en `[0,1]`.
- Error binario por píxel calculado umbralando la salida reconstruida.

## Hiperparámetros barribles

### Arquitectura

| Hiperparámetro | Qué controla | Dominio propuesto | Default sugerido | Por qué conviene estudiarlo |
| --- | --- | --- | --- | --- |
| `encoder_hidden_layers` | Profundidad y ancho del codificador antes del cuello de botella | `{[16], [24,8], [32,16,8]}` | `[24,8]` | Permite medir cuánto ayuda una compresión más gradual frente a una arquitectura mínima. |
| `decoder_hidden_layers` | Capacidad de reconstrucción desde el espacio latente | `{mirror(encoder), [8,24], [8,16,32]}` | `mirror(encoder)` | Sirve para comparar decodificadores simétricos contra variantes manuales con el mismo orden de magnitud de capacidad. |
| `hidden_activation` | No linealidad en capas ocultas | `{tanh, relu, leaky_relu}` | `tanh` | Cambia estabilidad, saturación y suavidad del mapeo hacia un latente de 2 dimensiones. |

Nota: `mirror(encoder)` significa invertir la secuencia del encoder. Por ejemplo, si `encoder_hidden_layers = [32,16,8]`, entonces `decoder_hidden_layers = [8,16,32]`.

### Optimización

| Hiperparámetro | Qué controla | Dominio propuesto | Default sugerido | Por qué conviene estudiarlo |
| --- | --- | --- | --- | --- |
| `optimizer` | Regla de actualización de pesos | `{adam, sgd_momentum}` | `adam` | Permite contrastar convergencia rápida vs. una dinámica más sensible al ajuste fino del learning rate. |
| `learning_rate` | Tamaño del paso de optimización | `{1e-4, 3e-4, 1e-3, 3e-3, 1e-2}` | `1e-3` | Es el hiperparámetro más influyente sobre velocidad de aprendizaje y estabilidad. |
| `batch_size` | Cantidad de patrones por actualización | `{4, 8, 16, 32}` | `8` | Cambia el ruido del gradiente; en datasets pequeños puede afectar mucho la capacidad de memorizar o generalizar. |
| `epochs_max` | Límite superior de entrenamiento | `{500, 1000, 2000, 5000}` | `2000` | Permite distinguir configuraciones que aprenden rápido de las que necesitan entrenamiento prolongado. |
| `early_stopping_patience` | Tolerancia a mesetas antes de cortar entrenamiento | `{50, 100, 200, none}` | `100` | Evita barridos innecesariamente largos y ayuda a detectar configuraciones inestables o estancadas. |
| `weight_init` | Escala inicial de los pesos | `{xavier_uniform, he_uniform}` | `xavier_uniform` | Interactúa con la activación oculta y puede facilitar o dificultar la convergencia desde el inicio. |

### Función de error y criterio binario

| Hiperparámetro | Qué controla | Dominio propuesto | Default sugerido | Por qué conviene estudiarlo |
| --- | --- | --- | --- | --- |
| `loss_function` | Objetivo numérico minimizado durante entrenamiento | `{binary_cross_entropy, mean_squared_error}` | `binary_cross_entropy` | `binary_cross_entropy` se alinea naturalmente con salidas binarias probabilísticas; `mean_squared_error` puede producir entrenamiento más suave pero menos calibrado para clasificación por píxel. |
| `output_activation` | Transformación final del decoder | `{sigmoid}` | `sigmoid` | Conviene fijarla para ambas losses para mantener comparabilidad y obtener salidas en `[0,1]`. |
| `pixel_threshold` | Umbral para convertir probabilidades reconstruidas en bits | `{0.4, 0.5, 0.6}` | `0.5` | Afecta directamente el conteo de píxeles incorrectos, que es la métrica operativa de la consigna. |

Comparación explícita entre losses:

- `binary_cross_entropy`: recomendada cuando cada píxel se interpreta como variable binaria independiente. Suele ser la primera candidata para alcanzar error binario bajo.
- `mean_squared_error`: útil como baseline comparable; puede reconstruir formas razonables aun cuando la calibración probabilística sea peor que con `binary_cross_entropy`.

### Regularización y estabilidad

| Hiperparámetro | Qué controla | Dominio propuesto | Default sugerido | Por qué conviene estudiarlo |
| --- | --- | --- | --- | --- |
| `l2_weight_decay` | Penalización sobre magnitud de los pesos | `{0, 1e-5, 1e-4, 1e-3}` | `1e-5` | Ayuda a evitar soluciones numéricamente frágiles sin imponer una regularización excesiva en un dataset pequeño. |
| `gradient_clip_norm` | Cota sobre la norma del gradiente | `{none, 1.0, 5.0}` | `none` | Sirve como mecanismo de estabilidad cuando alguna combinación de activación, init y learning rate explota. |
| `dropout` | Apagado aleatorio de unidades ocultas | `{0.0, 0.1, 0.2}` | `0.0` | Debe tratarse como exploratorio: puede mejorar robustez, pero también perjudicar memorización exacta en un conjunto tan chico. |

### Parámetros DAE-ready para reutilizar en 1b

Estos parámetros no son obligatorios para el autoencoder básico de 1a, pero conviene dejarlos definidos desde ahora para que el barrido de 1b sea comparable.

| Hiperparámetro | Qué controla | Dominio propuesto | Default sugerido | Por qué conviene estudiarlo |
| --- | --- | --- | --- | --- |
| `noise_type` | Tipo de corrupción aplicada a la entrada | `{bit_flip, salt_and_pepper, gaussian_clipped}` | `bit_flip` | Permite comparar ruido consistente con datos binarios contra variantes más agresivas o continuas. |
| `noise_level` | Intensidad del ruido aplicado | `{0.0, 0.05, 0.1, 0.15, 0.2, 0.3}` | `0.1` | Marca el punto en el que la reconstrucción deja de ser estable y muestra la tolerancia real del modelo al ruido. |
| `train_with_clean_target` | Si el target de entrenamiento sigue siendo la imagen limpia | `{true}` | `true` | Debe fijarse en `true` para mantener el esquema clásico de denoising autoencoder. |
| `noise_on_train_only` | Si el ruido se usa solo en entrenamiento o también en evaluación | `{true, false}` | `true` | Ayuda a separar robustez aprendida durante entrenamiento de robustez observada bajo test distorsionado. |

### Evaluación y reproducibilidad

| Hiperparámetro | Qué controla | Dominio propuesto | Default sugerido | Por qué conviene estudiarlo |
| --- | --- | --- | --- | --- |
| `train_split_strategy` | Qué parte del dataset se usa para ajustar el modelo | `{all_32, curated_subset}` | `all_32` | La consigna prioriza aprender los 32 patrones; `curated_subset` solo sirve para diagnosticar límites de capacidad o entrenamiento. |
| `seed` | Inicialización y orden aleatorio del experimento | `{0, 7, 42, 123}` | `42` | Permite medir sensibilidad de resultados en un problema pequeño y potencialmente inestable. |
| `selection_metric` | Criterio para elegir la mejor corrida | `{mean_pixel_error, exact_reconstruction_rate, max_pixel_error}` | `exact_reconstruction_rate` | Obliga a explicitar si se prioriza promedio, peor caso o tasa de reconstrucción perfecta. |

Nota sobre `curated_subset`: si se usa esta opción, conviene fijar un subconjunto explícito y reportarlo tal cual en el experimento, idealmente de tamaño `8`, `16` o `24`, para que la comparación sea reproducible.

## Barridos recomendados

Orden sugerido para no mezclar demasiados factores a la vez:

1. `loss`
   Comparar primero `binary_cross_entropy` vs. `mean_squared_error`, manteniendo arquitectura y optimizador fijos.
2. `arquitectura`
   Con la mejor loss preliminar, barrer `encoder_hidden_layers`, `decoder_hidden_layers` y `hidden_activation`.
3. `optimizer/LR`
   Fijada una arquitectura razonable, comparar `adam` vs. `sgd_momentum` y luego ajustar `learning_rate`.
4. `ruido`
   Recién después de estabilizar 1a, reutilizar la mejor base y barrer `noise_type` y `noise_level` para 1b.

## Configuración base sugerida

Si hace falta partir de una corrida inicial antes del barrido:

| Parámetro | Valor inicial sugerido |
| --- | --- |
| `encoder_hidden_layers` | `[24,8]` |
| `decoder_hidden_layers` | `mirror(encoder)` |
| `hidden_activation` | `tanh` |
| `optimizer` | `adam` |
| `learning_rate` | `1e-3` |
| `batch_size` | `8` |
| `epochs_max` | `2000` |
| `early_stopping_patience` | `100` |
| `weight_init` | `xavier_uniform` |
| `loss_function` | `binary_cross_entropy` |
| `output_activation` | `sigmoid` |
| `pixel_threshold` | `0.5` |
| `l2_weight_decay` | `1e-5` |
| `gradient_clip_norm` | `none` |
| `dropout` | `0.0` |
| `train_split_strategy` | `all_32` |
| `seed` | `42` |
| `selection_metric` | `exact_reconstruction_rate` |

Con esta base, el documento queda listo para usarse como guía de experimentación aun si la implementación todavía no existe.
