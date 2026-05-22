# Super-Resolución de Imágenes con Autoencoder Convolucional (x4)

Proyecto académico del Máster en IA / Big Data sobre super-resolución de
imágenes con un autoencoder convolucional tipo **U-Net** y *upsampling* final
mediante **PixelShuffle**. Dataset: **DIV2K**. Factor de escala: **x4**
(32×32 → 128×128). Entrenamiento con pérdida combinada L1 + perceptual VGG19
y evaluación con PSNR / SSIM / LPIPS frente a baselines bicúbico, EDSR-base,
DRLN y Real-ESRGAN x4plus.

El proyecto incluye además una **web app** (FastAPI + JS) que carga los tres
checkpoints internos (v1, v2, v3) y los baselines externos en paralelo para
comparar resultados sobre fotos del usuario.

## Resultados

Evaluación sobre el split de *test* (50 imágenes de DIV2K no vistas durante
el entrenamiento ni durante la selección del *checkpoint*). Mismas posiciones
de recorte deterministas para todas las filas:

| Método                 | Parámetros | PSNR (dB) ↑ |  SSIM ↑  |  LPIPS ↓ |
|------------------------|-----------:|------------:|---------:|---------:|
| Bicubic                |         —  |      28.20  |  0.7333  |  0.3631  |
| v1 (U-Net + BN + sigmoid)        |   6.86 M  | 28.14 | 0.7492 | 0.3089 |
| v2 (sin BN, residual sobre bicúbico) | 6.86 M | 29.33 | 0.7691 | 0.2731 |
| **v3 (RCAB + ICNR)**   |    10.44 M  |  **29.36**  | **0.7703** | **0.2701** |
| EDSR-base (HF)         |    ~1.5 M  |      30.10  |  0.7976  |  0.3115  |
| DRLN (HF)              |    ~34 M   |      30.32  |  0.8034  |  0.3010  |
| Real-ESRGAN x4plus     |    ~17 M   |      24.77  |  0.6470  |  0.3513  |

**Lectura rápida:**

- **v1 → v2** (+1.19 dB PSNR, −0.036 LPIPS) sin tocar la cuenta de parámetros:
  todo viene de quitar BatchNorm, sustituir sigmoid por clamp y aprender un
  residual sobre el upsample bicúbico (cabeza zero-init).
- **v2 → v3** mejora marginal en métricas (+0.03 dB) pero **convergencia ~2× más
  rápida** (best epoch 26 vs 54): los RCAB + ICNR mejoran la trayectoria de
  optimización más que el techo.
- **v3 vs EDSR/DRLN**: a 1-2 dB del SOTA con un dataset y entrenamiento mucho
  más modestos. Razonable para un proyecto académico.
- **Paradoja Real-ESRGAN**: tiene métricas peores sobre DIV2K (entrenado con
  degradaciones sintéticas complejas, no con bicúbico puro) pero produce
  salidas **visiblemente más nítidas** en fotos reales — un caso de manual
  sobre por qué PSNR no es la métrica adecuada para SR perceptual.

## Estructura

```
super-res/
├── src/
│   ├── data.py           # DIV2KDataset + DataLoader (cache en RAM, augmentaciones sincronizadas)
│   ├── model.py          # UNetSR v1 (BN + sigmoid, baseline original)
│   ├── model_v2.py       # UNetSRv2 (sin BN, sin sigmoid, residual sobre bicúbico)
│   ├── model_v3.py       # UNetSRv3 (RCAB + channel attention + ICNR init)
│   ├── losses.py         # L1 + perceptual VGG19
│   ├── metrics.py        # PSNR / SSIM / LPIPS
│   ├── realesrgan_arch.py # RRDBNet (carga pesos oficiales RealESRGAN_x4plus)
│   ├── train.py          # Loop de entrenamiento + AMP + checkpoints
│   ├── evaluate.py       # Evaluación cuantitativa y grids visuales
│   └── utils.py          # Semilla y curvas
├── scripts/
│   ├── check_env.py            # Verificación de Python/PyTorch/CUDA/GPU
│   ├── download_div2k.py       # Descarga + extracción DIV2K (~3.7 GB)
│   ├── eval_baselines.py       # Evalúa EDSR / DRLN / Real-ESRGAN en test
│   ├── gen_test_images.py      # Genera parches LR para probar en la webapp
│   ├── smoke_model.py          # Test rápido de forward del modelo
│   ├── smoke_losses_metrics.py # Test de losses + métricas
│   └── visualize_batch.py      # Visualiza un batch HR/LR
├── webapp/
│   ├── main.py           # FastAPI: endpoints upscale + health
│   ├── inference.py      # Carga v1/v2/v3 + baselines, upscale_image()
│   ├── static/           # index.html + app.js + style.css
│   └── README.md
├── data/                 # Dataset DIV2K (no versionado)
├── checkpoints/          # Pesos v1 (no versionado)
├── checkpoints_v2/       # Pesos v2 (no versionado)
├── checkpoints_v3/       # Pesos v3 (no versionado)
├── outputs/              # Curvas/métricas/ejemplos de v1 + baselines_metrics.json
├── outputs_v2/, outputs_v3/  # Idem por iteración
├── upscaling.ipynb       # Notebook final con la narrativa completa (v1 → v2 → v3)
├── requirements.txt
├── requirements-web.txt  # FastAPI + super-image + huggingface_hub
└── README.md
```

## Requisitos

- WSL2 Ubuntu / Linux
- Python 3.11
- GPU NVIDIA con CUDA 12.x (probado en RTX 2060, 6 GiB VRAM)
- Driver NVIDIA reciente (`nvidia-smi` operativo)
- ~16 GiB de RAM (el dataset HR se cachea en memoria para evitar cuellos de botella de I/O)
- ~12 GiB de disco (3.7 GB DIV2K + checkpoints v1/v2/v3 + outputs + pesos baselines descargados en caché)

## Instalación

```bash
# 1. Situarse en el directorio del proyecto
cd super-res

# 2. Crear entorno virtual (NO conda)
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Actualizar pip
pip install --upgrade pip

# 4. Instalar dependencias (PyTorch con CUDA 12.4)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124

# 5. Dependencias adicionales para la webapp y los baselines externos
pip install -r requirements-web.txt

# 6. Verificar entorno + GPU
python scripts/check_env.py
```

Salida esperada de `check_env.py`: versión de PyTorch, `CUDA available: True`,
nombre de la GPU y VRAM disponible.

## Uso

Todos los comandos se ejecutan desde la raíz del repositorio. El prefijo
`PYTHONPATH=.` es necesario para que el paquete `src` sea importable cuando
se invocan los módulos como scripts (`python -m src.train` etc.).

### 1. Descargar DIV2K

```bash
python scripts/download_div2k.py
```

Descarga y extrae `DIV2K_train_HR` (800 imágenes, 3.3 GB) y `DIV2K_valid_HR`
(100 imágenes, 430 MB). El script es idempotente: omite descargas y
extracciones si los PNG ya están presentes.

### 2. Entrenar el modelo

El flag `--model` selecciona la arquitectura. Cada versión escribe sus
artefactos en directorios distintos para poder compararlas:

```bash
# v1 (baseline original, U-Net con BatchNorm y sigmoid)
PYTHONPATH=. python -m src.train --model v1 --epochs 100 --batch-size 16 --lr 1e-4 \
    --checkpoint-dir checkpoints --outputs-dir outputs

# v2 (sin BN, residual sobre bicúbico)
PYTHONPATH=. python -m src.train --model v2 --epochs 100 --batch-size 16 --lr 1e-4 \
    --checkpoint-dir checkpoints_v2 --outputs-dir outputs_v2

# v3 (RCAB + ICNR)
PYTHONPATH=. python -m src.train --model v3 --epochs 100 --batch-size 16 --lr 1e-4 \
    --checkpoint-dir checkpoints_v3 --outputs-dir outputs_v3
```

Flags relevantes (`python -m src.train --help` para la lista completa):

- `--loss {l1,charbonnier}` — pérdida píxel-a-píxel base.
- `--weight-l1 1.0 --weight-perceptual 0.1` — pesos de la pérdida combinada.
- `--hr-size 128` — tamaño del parche HR.
- `--degradation {bicubic,mixed}` — bicúbico puro o mezcla con ruido + JPEG.
- `--patience N` — *early stopping* sobre PSNR de validación.

Configuración por defecto: Adam (β = 0.9, 0.999) + `CosineAnnealingLR`
(`T_max=epochs`), mixed precision (`torch.cuda.amp`) cuando hay GPU, pesos de
pérdida L1=1.0 / perceptual=0.1.

Salidas (por iteración):

- `<checkpoint-dir>/best.pt` → mejor PSNR de validación.
- `<checkpoint-dir>/last.pt` → último estado de entrenamiento.
- `<outputs-dir>/train_curves.png` → curvas de loss/PSNR/SSIM/LPIPS.
- `<outputs-dir>/train_history.json` → historial completo de métricas.

ETA orientativo en RTX 2060: ~57 s/época para v1-v2, ~85 s/época para v3
(más parámetros). v3 converge en ~26 epochs (≈40 min con `--patience 20`).

### 3. Evaluar contra el baseline bicúbico

```bash
PYTHONPATH=. python -m src.evaluate --checkpoint checkpoints_v3/best.pt \
    --num-examples 12 --split test --outputs-dir outputs_v3
```

El loader detecta automáticamente la versión del modelo desde el checkpoint
(`v1` / `v2` / `v3`). Salidas en `<outputs-dir>/`:

- `metrics_table.txt`, `metrics.json` → métricas agregadas modelo vs bicúbico.
- `examples_metrics.json` → métricas por ejemplo.
- `examples/example_NN.png` → grids [LR (nearest x4) | Bicubic | Modelo | HR].

### 4. Evaluar baselines externos (EDSR / DRLN / Real-ESRGAN)

```bash
PYTHONPATH=. python scripts/eval_baselines.py
```

Descarga los pesos preentrenados (EDSR-base y DRLN vía Hugging Face Hub con
el paquete `super-image`; Real-ESRGAN x4plus vía release de GitHub de
xinntao, ~64 MB) y los evalúa sobre exactamente el mismo split y los mismos
parches deterministas que `src/evaluate.py`. Escribe
`outputs/baselines_metrics.json` con `psnr`/`ssim`/`lpips` por baseline,
listo para que el notebook construya la tabla comparativa unificada.

### 5. Web app de comparación

```bash
PYTHONPATH=. uvicorn webapp.main:app --host 0.0.0.0 --port 8000
```

Abrir [http://localhost:8000](http://localhost:8000). La app carga al
arranque los tres checkpoints (v1, v2, v3) + EDSR + DRLN + Real-ESRGAN, y
permite subir una foto (PNG/JPEG/WebP, hasta 512 px de lado, 10 MB) para
comparar los resultados de cualquier par de modelos con un *slider*
antes/después. Ver [`webapp/README.md`](webapp/README.md) para detalles.

Para generar imágenes de prueba a tamaños fijos desde DIV2K_valid_HR:

```bash
python scripts/gen_test_images.py --n 6 --sizes 64 128 256
# → outputs/test_inputs/
```

### 6. Cuaderno final

Abrir `upscaling.ipynb` en JupyterLab o VS Code. El cuaderno carga los
artefactos generados (`checkpoints*/best.pt`, `outputs*/metrics.json`,
`outputs/baselines_metrics.json`, ejemplos PNG) y reproduce el análisis
completo de las tres iteraciones + la comparativa contra baselines externos.
Es totalmente ejecutable de principio a fin sin necesidad de reentrenar.

```bash
jupyter lab upscaling.ipynb
```

## Iteraciones del modelo

Cada versión cambia exclusivamente la arquitectura para que la ganancia
medida sea atribuible al cambio en sí (mismo dataset, misma loss, mismo
optimizador, mismas semillas).

### v1 — Baseline propio (`src.model.UNetSR`)

U-Net clásico con `DoubleConv` (Conv-BN-ReLU-Conv-BN-ReLU), cabeza
PixelShuffle (x2 + x2) y `Sigmoid` final. 6.86 M parámetros. Empata con la
bicúbica en PSNR y la mejora ligeramente en SSIM/LPIPS.

### v2 — Quick-win de literatura SR (`src.model_v2.UNetSRv2`)

Tres cambios "micro", sin tocar la cuenta de parámetros:

1. **Sin BatchNorm**. BN normaliza estadísticas que el modelo necesita para
   reconstruir textura de alta frecuencia (EDSR/RCAN/SwinIR la descartan).
2. **Sin Sigmoid final**: se reemplaza por `clamp(0, 1)` aplicado a la salida.
   Sigmoid satura gradientes cerca de 0 y 1, justo donde viven los píxeles
   muy brillantes/oscuros.
3. **Residual sobre bicúbico** con cabeza zero-init: la red predice
   `out = bicubic(x) + conv_path(x)`. En la época 0 el output ya es la
   bicúbica (PSNR arranca en ~29 dB en vez de ~24 dB), y la capacidad se
   dedica únicamente al detalle de alta frecuencia.

Resultado: +1.19 dB PSNR vs v1, +0.020 SSIM, −0.036 LPIPS.

### v3 — Channel attention + ICNR (`src.model_v3.UNetSRv3`)

1. **Bloques RCAB** (Zhang et al., RCAN 2018) en lugar de `DoubleConv`:
   `Conv → ReLU → Conv → ChannelAttention` + skip residual interno. Añade
   ~50 % de parámetros (10.44 M) y atención SE por canal.
2. **Inicialización ICNR** para las convs de PixelShuffle (Aitken et al.
   2017): elimina el patrón en damero inicial de la sub-pixel convolution.

Resultado: mejora marginal en métricas finales (+0.03 dB) pero **converge en
la mitad de epochs** que v2. Con `--patience 20` se entrena en la mitad de
tiempo.

## Decisiones técnicas relevantes

- **Cache en RAM de DIV2K**: el `DIV2KDataset` precarga las 800 imágenes
  HR como tensores `uint8` (~6.4 GiB) y aprovecha el *copy-on-write* del
  `fork` de Linux para compartir esas páginas con los *workers* del
  `DataLoader`. Elimina el cuello de botella de decodificar PNG en cada
  `__getitem__` y mantiene la GPU saturada.
- **Cabeza de upscaling con PixelShuffle (x2 + x2)** en lugar de
  *transposed convolutions*: evita el característico patrón en damero y
  produce salidas más limpias. v3 añade además **ICNR** para que el
  damero tampoco aparezca en la inicialización.
- **Pérdida combinada (L1 + perceptual VGG19)**: la L1 estabiliza el
  entrenamiento y evita los outputs borrosos típicos de la MSE; la
  perceptual empuja a generar texturas plausibles. Peso `0.1` para la
  perceptual elegido tras observar que pesos mayores degradaban demasiado
  la fidelidad numérica.
- **Splits disjuntos val/test**: las 100 imágenes de `DIV2K_valid_HR` se
  reparten en 50 para validación (selección del mejor *checkpoint*) y 50
  para test (evaluación final). Mismas posiciones de recorte deterministas
  para baselines internos y externos, garantizando comparación justa.
- **Pixel range [0, 1]** con clamp/sigmoid en la cabeza: convención
  coherente con LPIPS, SSIM y el cálculo del PSNR con `data_range=1`.

## Reproducibilidad

Las semillas están fijadas (`seed=1337` por defecto) en `set_seed()` y se
propagan a `random`, `numpy` y `torch` (CPU+CUDA). Los recortes de
validación y test son deterministas porque sus posiciones se sortean con
una `random.Random(seed)` independiente. Las aumentaciones de
entrenamiento usan una RNG sembrada por `seed * 1_000_003 + idx`, de modo
que cada muestra recibe siempre la misma transformación dada una semilla.

## Limitaciones y posibles extensiones

- Entrenamiento limitado a parches HR de 128×128 por restricciones de
  VRAM (RTX 2060, 6 GiB). Subir a 256×256 daría más contexto al modelo.
- No se exploró una pérdida adversarial (GAN). Real-ESRGAN demuestra que
  ese camino es el que mejora la percepción real, a costa de un
  entrenamiento bastante más complejo (degradaciones sintéticas + discriminator).
- Arquitecturas más modernas (SwinIR, HAT, DiffIR) superarían claramente
  este U-Net base, pero quedan fuera del alcance del trabajo.
- Inferencia *tiled* (con overlap + feathering) para procesar fotos
  mayores de 512 px sin OOM. Está pendiente como TODO en la webapp.
