"""Pipeline F1: inspector → router → carriles → acabado → dataset.

Carriles (orden no negociable): daño → rostros → color → upscale.
F1 activa solo upscale (Real-ESRGAN, si resolucion=insuficiente); los
restantes quedan documentados para F2 con ComfyUI (skill print-bot-enhance).
"""
import datetime
import os
import json
import logging
import subprocess
from pathlib import Path

from photo_inspector import inspect
import watermark

logger = logging.getLogger("photo_pipeline")

DATASET_DIR = Path(os.getenv("PHOTOS_DATASET_DIR", "/root/photos_dataset"))
UPSCALE_BIN = "/opt/print-enhance-venv/bin/realesrgan-ncnn-vulkan"
UPSCALE_MODEL = "/opt/realesrgan-models/realesrgan-x4plus/RealESRGAN_x4plus.pth"

# Umbral duro secundario (el diagnóstico 'resolucion' prima sobre este piso)
MIN_PX_LONG = 1200


def route(diag: dict, width: int) -> dict:
    """Reglas deterministas — el inspector NO decide, solo diagnostica."""
    lanes = {
        "danio": diag.get("danos") != "sin_danio",          # F2 (BOPBTL)
        "rostros": bool(diag.get("rostros_degradados")),    # F2 (CodeFormer)
        "color": diag.get("color") in ("bn", "deslavado"),  # F2 (DDColor)
        "upscale": diag.get("resolucion") == "insuficiente" or width < MIN_PX_LONG,
    }
    return lanes


def upscale_cpu(src: str, dest: str, outscale: float = 2.0) -> str:
    """Real-ESRGAN x4 CPU (venv torch). Lento: solo cuando el router lo pide."""
    venv_bin = "/opt/print-enhance-venv/bin/python"
    script = (
        "from realesrgan import RealESRGANer; from basicsr.archs.rrdbnet_arch import RRDBNet;"
        "from PIL import Image; import numpy as np, sys;"
        "rr=RRDBNet(num_in_ch=3,num_out_ch=3,num_feat=64,num_block=23,num_grow_ch=32,scale=4);"
        "up=RealESRGANer(scale=4,model_path='%s',model=rr,tile=128);"
        "img=np.array(Image.open(sys.argv[1]).convert('RGB'));"
        "out=up.enhance(img,outscale=%s)[0];"
        "Image.fromarray(out.astype('uint8')).save(sys.argv[2])"
    ) % (UPSCALE_MODEL, outscale)
    subprocess.run([venv_bin, "-c", script, src, dest], check=True, timeout=3600)
    return dest


def _workdir(ts) -> Path:
    d = Path("/tmp/print_enhance") / ts.strftime("%Y%m%d_%H%M%S")
    d.mkdir(parents=True, exist_ok=True)
    return d


def run(input_path: str, date: str | None = None, out_dir: str | None = None) -> dict:
    """Ejecuta el pipeline sobre input_path. NUNCA escribe junto al original
    (los outputs van a /tmp/print_enhance/<ts>/; el dataset tiene copias)."""
    from PIL import Image

    ts = datetime.datetime.now()
    meta = {"ts": ts.isoformat(), "carriles": {}, "fecha": None, "inspector": None}

    # 1. Inspector
    diag = inspect(input_path)
    meta["inspector"] = diag.pop("_inspector", None)

    width = Image.open(input_path).size[0]
    lanes = route(diag, width)
    meta["carriles"] = lanes
    work = _workdir(ts) / "work_0.jpg"
    _copy(Path(input_path), work)

    # 2. Carriles en serie (F1: solo upscale; F2 añade los demás en este orden)
    if lanes.get("danio"):
        logger.info("carril danio: pendiente F2 (BOPBTL), se omite")
    if lanes.get("rostros"):
        logger.info("carril rostros: pendiente F2 (CodeFormer), se omite")
    if lanes.get("color"):
        logger.info("carril color: pendiente F2 (DDColor), se omite")
    if lanes.get("upscale"):
        up = _workdir(ts) / "work_up.jpg"
        try:
            work = upscale_cpu(str(work), str(up))
            meta["carriles"]["upscale_hecho"] = True
        except Exception as e:
            logger.warning(f"upscale falló, siguiendo sin él: {e}")

    # 3. Acabado: fecha
    if not date:
        date = watermark.has_exif_date(input_path)
    if date:
        final = watermark.stamp(str(work), date, str(_workdir(ts) / "final.jpg"))
        meta["fecha"] = date
    else:
        final = str(_workdir(ts) / "final.jpg")
        if Path(work) != Path(final):
            import shutil
            shutil.move(str(work), final)

    # 4. Dataset (original + final + sidecar)
    folder = DATASET_DIR / ts.strftime("%Y-%m") / ts.strftime("%Y%m%d_%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    ds_orig = folder / "original.jpg"
    ds_final = folder / "final.jpg"
    _copy(Path(input_path), ds_orig)
    _copy(Path(final), ds_final)
    meta.update({
        "diagnostic": diag,
        "dataset_orig": str(ds_orig),
        "dataset_final": str(ds_final),
        "input_path": input_path,
        "final_path": final,
    })
    (folder / "sidecar.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str)
    )
    logger.info(f"dataset grabado: {folder}")
    return meta


def _copy(src: Path, dst: Path):
    if src.resolve() != dst.resolve():
        dst.write_bytes(src.read_bytes())