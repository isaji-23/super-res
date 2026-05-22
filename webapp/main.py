"""FastAPI web app for UNetSR super-resolution."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from webapp.inference import (
    _MAX_FILE_BYTES,
    get_status,
    load_model,
    upscale_image,
)

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
STATIC_DIR = __file__.replace("main.py", "static")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Super-Resolución x4", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(f"{STATIC_DIR}/index.html")


@app.get("/api/health")
async def health():
    return get_status()


@app.post("/api/upscale")
async def upscale(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Tipo no soportado: {file.content_type}. Usa PNG, JPEG o WebP.",
        )

    raw = await file.read()
    if len(raw) > _MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Archivo demasiado grande ({len(raw)//1024} KB). Máximo 10 MB.",
        )

    try:
        import io
        img = Image.open(io.BytesIO(raw))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen.")

    try:
        result = upscale_image(img)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return result
