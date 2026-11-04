"""Inspector visual de fotos para print_bot — decidió con Christian el 04-nov-2026.

Cadena de visión (no re-ordenar):
  1. glm-5.3-flash:cloud  (nube 1, LEGACY $20)
  2. kimi-k2.6:cloud      (nube 2)
  3. qwen3.5:9b local     (CPU, batch — lento, solo fallback)

Output: JSON estricto via "format" (structured outputs de Ollama).
Nunca llava/moondream: alucinan (ver skill local-vision-models).
"""
import base64
import json
import logging
import urllib.request
from pathlib import Path

logger = logging.getLogger("photo_inspector")

DIAG_SCHEMA = {
    "type": "object",
    "properties": {
        "danos": {"type": "string", "enum": ["sin_danio", "leve", "severo"]},
        "rostros_degradados": {"type": "boolean"},
        "color": {"type": "string", "enum": ["bn", "deslavado", "sano"]},
        "ruido": {"type": "string", "enum": ["bajo", "alto"]},
        "resolucion": {"type": "string", "enum": ["suficiente", "insuficiente"]},
        "notas": {"type": "string"},
    },
    "required": ["danos", "rostros_degradados", "color", "ruido", "resolucion", "notas"],
}

INSPECTOR_CHAIN = ["glm-5.3-flash:cloud", "kimi-k2.6:cloud", "qwen3.5:9b"]

PROMPT = (
    "Analiza esta foto para impresión fotográfica. Responde ÚNICAMENTE con un "
    "objeto JSON, sin texto antes ni después, con estas claves exactas: "
    "danos (sin_danio|leve|severo), rostros_degradados (bool, true solo si hay "
    "rostros difusos), color (bn|deslavado|sano), ruido (bajo|alto), "
    "resolucion (suficiente|insuficiente), notas (español, máx 15 palabras). "
    "No pienses en voz alta: entrega directamente el JSON."
)


def _ollama_chat(payload: dict, timeout: int) -> dict:
    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def inspect(path: str, chain: list = None) -> dict:
    """Devuelve el diagnóstico JSON del primer modelo de la cadena que responda."""
    chain = chain or INSPECTOR_CHAIN
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    last_err = None
    for model in chain:
        local = not model.endswith(":cloud")
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT, "images": [b64]}],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 900},
            "think": False,
        }
        if local:
            payload["format"] = DIAG_SCHEMA  # structured outputs: solo soportado local
        attempts = 1 if local else 2  # cloud: 1 retry antes de saltar de modelo
        for attempt in range(attempts):
            try:
                out = _ollama_chat(payload, timeout=120 if not local else 600)
                content = out["message"]["content"]
                if not content.strip():
                    raise ValueError("contenido vacío (runner crash o think corto)")
                diag = None
                if local:
                    diag = json.loads(content)
                else:
                    # cloud: puede venir con prosa o ```json; probar cada '{'
                    dec = json.JSONDecoder()
                    for i, ch in enumerate(content):
                        if ch == "{":
                            try:
                                cand, _ = dec.raw_decode(content[i:])
                                if isinstance(cand, dict) and "danos" in cand:
                                    diag = cand
                                    break
                            except ValueError:
                                continue
                    if diag is None:
                        raise ValueError("sin JSON válido en la respuesta")
                for k in DIAG_SCHEMA["required"]:
                    assert k in diag, f"falta {k}"
                diag["_inspector"] = model
                logger.info(f"Inspector {model}: {diag}")
                return diag
            except Exception as e:
                last_err = e
                logger.warning(f"Inspector {model} intento {attempt + 1}: {e}")
    raise RuntimeError(f"Ningún modelo de la cadena respondió: {last_err}")