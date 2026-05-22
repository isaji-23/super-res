# Web App Super-Resolución x4

Interfaz web para probar `checkpoints/best.pt` con tus propias fotos.

## Arrancar

```bash
# Desde la raíz del proyecto (super-res/)
source .venv/bin/activate

# Primera vez: instalar dependencias web
pip install -r requirements-web.txt

# Lanzar servidor
PYTHONPATH=. uvicorn webapp.main:app --host 0.0.0.0 --port 8000
```

Abrir: **http://localhost:8000**

## Uso

1. Arrastra una imagen PNG/JPEG/WebP (máx 512 px, máx 10 MB).
2. Click **Super-resolver**.
3. Compara Bicubic vs Modelo con el slider.
4. Click **Descargar SR** para guardar el resultado.

## Límites

- Tamaño máximo de entrada: 512 px (configurable con `SR_MAX_INPUT=N`).
- El modelo fue entrenado con parches 32×32 → funciona mejor con imágenes pequeñas.
- En GPU RTX 2060 6 GB: ~50-200 ms por imagen.
- En CPU: puede tardar 5-15 s.

## Smoke test (sin servidor)

```bash
PYTHONPATH=. python -m webapp.inference ruta/imagen.png
```

Guarda `imagen_sr.png` junto al original.
