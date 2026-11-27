"""
Cliente HTTP para integración de Estación 2 con la API v3 de Estación 5 (E5).

Contrato activo: v3 (una mandarina por mensaje).
Ruta de ingesta para E2: POST /api/v1/ingest/e2
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import json
import os
import socket
import time
import urllib.error
import urllib.request

DEFAULT_E5_BASE_URL = "http://192.168.0.10:8000"
E2_INGEST_PATH = "/api/v1/ingest/e2"

# Mapeo oficial de clases locales de E2 -> categorías v3 de E5
VISUAL_QUALITY_MAP: Dict[str, str] = {
    "buen_estado": "alta",
    "marginalmente_buena": "media",
    "mal_estado": "baja",
}

# Mapeo por defecto de defectos detectados según clase visual
DEFECTS_MAP: Dict[str, List[str]] = {
    "buen_estado": [],
    "marginalmente_buena": ["defecto_leve"],
    "mal_estado": ["defecto_visible"],
}


def utc_timestamp_iso() -> str:
    """Retorna el timestamp actual en formato ISO 8601 UTC con zona horaria ('Z')."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def map_prediction_to_e2_datos(
    predicted_class: str,
    confidence: Optional[float] = None,
    confidence_percent: Optional[float] = None,
    defects: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Mapea la salida del modelo de Estación 2 al objeto `datos` del contrato v3 de E5.

    - `calidad_visual`: 'alta' | 'media' | 'baja'
    - `confianza_visual`: float en rango [0.0, 1.0]
    - `defectos_detectados`: lista de strings
    """
    if predicted_class not in VISUAL_QUALITY_MAP:
        raise ValueError(
            f"Clase predicha '{predicted_class}' no tiene mapeo v3 en VISUAL_QUALITY_MAP "
            f"({list(VISUAL_QUALITY_MAP.keys())})."
        )

    if confidence is not None:
        conf_val = float(confidence)
    elif confidence_percent is not None:
        conf_val = float(confidence_percent) / 100.0
    else:
        raise ValueError("Se requiere 'confidence' o 'confidence_percent' para calcular confianza_visual.")

    conf_val = round(max(0.0, min(1.0, conf_val)), 4)

    if defects is not None:
        detected_defects = list(defects)
    else:
        detected_defects = list(DEFECTS_MAP.get(predicted_class, []))

    return {
        "calidad_visual": VISUAL_QUALITY_MAP[predicted_class],
        "confianza_visual": conf_val,
        "defectos_detectados": detected_defects,
    }


def build_e2_payload(
    id_lote: str,
    id_fruta: str,
    predicted_class: str,
    confidence: Optional[float] = None,
    confidence_percent: Optional[float] = None,
    defects: Optional[List[str]] = None,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Construye el mensaje completo de ingesta E2 conforme al contrato v3 de E5."""
    if not id_lote or not id_fruta:
        raise ValueError("id_lote e id_fruta son obligatorios para construir el payload v3 de E2.")

    return {
        "id_lote": str(id_lote),
        "id_fruta": str(id_fruta),
        "estacion": "E2",
        "timestamp": timestamp or utc_timestamp_iso(),
        "datos": map_prediction_to_e2_datos(
            predicted_class=predicted_class,
            confidence=confidence,
            confidence_percent=confidence_percent,
            defects=defects,
        ),
    }


class E5Client:
    """
    Cliente HTTP sin dependencias externas para enviar lecturas E2 a la API de E5.
    Implementa timeout finito y reintentos con espera creciente reutilizando
    exactamente el mismo JSON (mismo timestamp) para preservar idempotencia.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout_sec: float = 3.0,
        max_retries: int = 3,
        backoff_base_sec: float = 0.5,
        enabled: bool = True,
    ):
        resolved_url = base_url or os.environ.get("E5_API_URL", DEFAULT_E5_BASE_URL)
        self.base_url = resolved_url.rstrip("/")
        self.ingest_url = f"{self.base_url}{E2_INGEST_PATH}"
        self.timeout_sec = timeout_sec
        self.max_retries = max(0, int(max_retries))
        self.backoff_base_sec = max(0.0, float(backoff_base_sec))
        self.enabled = enabled

    def send_e2_ingest(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Envía `POST /api/v1/ingest/e2` con el payload v3.
        Reintenta ante timeouts o errores 5xx/red manteniendo intacto el JSON original.
        No reintenta ante errores de contrato/conflicto (409, 422, 4xx).
        """
        if not self.enabled:
            return {"ok": False, "skipped": True, "reason": "E5Client deshabilitado"}

        # Serializar una sola vez para garantizar idempotencia exacta en reintentos
        body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        last_error: Optional[str] = None
        attempts = self.max_retries + 1

        for attempt in range(attempts):
            req = urllib.request.Request(
                self.ingest_url,
                data=body_bytes,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    status = resp.getcode()
                    raw_body = resp.read().decode("utf-8", errors="replace")
                    parsed = json.loads(raw_body) if raw_body else {}
                    return {
                        "ok": True,
                        "status_code": status,
                        "response": parsed,
                        "attempts": attempt + 1,
                    }
            except urllib.error.HTTPError as err:
                status = err.code
                err_body = err.read().decode("utf-8", errors="replace")
                # 4xx (incluyendo 409 Conflict y 422 Unprocessable Entity) no son recuperables con reintento
                if 400 <= status < 500:
                    return {
                        "ok": False,
                        "status_code": status,
                        "error": err_body,
                        "attempts": attempt + 1,
                    }
                last_error = f"HTTP {status}: {err_body}"
            except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as err:
                last_error = str(err)

            if attempt < attempts - 1 and self.backoff_base_sec > 0:
                time.sleep(self.backoff_base_sec * (2 ** attempt))

        return {
            "ok": False,
            "status_code": None,
            "error": last_error or "Error desconocido enviando a E5",
            "attempts": attempts,
        }

