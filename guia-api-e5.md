# Guía de integración con la API de E5

**Contrato activo:** v3, una mandarina por mensaje. Los JSON de
[`docs/formato_datos/`](../formato_datos/README.md) son la fuente de verdad.
**Transporte:** HTTP + JSON sobre la red local de la planta.

## Conexión

En la Raspberry Pi la URL será `http://192.168.0.10:8000`. En desarrollo:
`http://localhost:8000`. E5 no tiene autenticación; la API debe quedar en la
LAN de la planta y **no** debe publicarse mediante port-forwarding a Internet.
El puerto del host debe enlazarse a la IP local de la Pi.

La documentación interactiva está en `/docs`. Las estaciones que usan Python,
C++, Arduino u otro cliente HTTP no necesitan CORS; CORS solo regula solicitudes
de navegadores.

## Ingesta

Cada estación envía un `POST` a su ruta y reutiliza el mismo `id_fruta` asignado
por E1:

| Estación | Ruta |
|---|---|
| E1 — Captura | `POST /api/v1/ingest/e1` |
| E2 — Calidad visual | `POST /api/v1/ingest/e2` |
| E3 — Diámetro y tamaño | `POST /api/v1/ingest/e3` |
| E4 — Decisión comercial | `POST /api/v1/ingest/e4` |
| E6 — Cinta | `POST /api/v1/ingest/e6` |

E1–E4 envían un mensaje por mandarina. E6 envía un mensaje al cambiar el estado
y un *heartbeat* cada 10 segundos. Todos los ejemplos siguientes describen la
mandarina `M-0001` del lote `LOTE-2026-11-A`.

### Ejemplos

```json
{
  "id_lote": "LOTE-2026-11-A",
  "id_fruta": "M-0001",
  "estacion": "E1",
  "timestamp": "2026-11-25T10:15:30Z",
  "datos": {
    "validez_imagen": true,
    "estado_captura": "aceptada",
    "posicion_correcta": true
  }
}
```

```json
{
  "id_lote": "LOTE-2026-11-A",
  "id_fruta": "M-0001",
  "estacion": "E2",
  "timestamp": "2026-11-25T10:15:36Z",
  "datos": {
    "calidad_visual": "alta",
    "confianza_visual": 0.92,
    "defectos_detectados": []
  }
}
```

```json
{
  "id_lote": "LOTE-2026-11-A",
  "id_fruta": "M-0001",
  "estacion": "E3",
  "timestamp": "2026-11-25T10:15:41Z",
  "datos": {
    "diametro_mm": 64,
    "tamano": "mediano"
  }
}
```

```json
{
  "id_lote": "LOTE-2026-11-A",
  "id_fruta": "M-0001",
  "estacion": "E4",
  "timestamp": "2026-11-25T10:15:45Z",
  "datos": {
    "categoria_final": "exportacion",
    "probabilidad": 0.88,
    "requiere_revision_manual": false
  }
}
```

```json
{
  "id_lote": "LOTE-2026-11-A",
  "estacion": "E6",
  "timestamp": "2026-11-25T10:15:46Z",
  "datos": {
    "estado_cinta": "normal",
    "falla_mecanica_detectada": false,
    "id_fruta_en_cinta": "M-0001"
  }
}
```

El `timestamp` debe ser ISO 8601 con zona horaria. E3 informa `tamano`; E5
almacena el resultado y **no** vuelve a calcularlo. Los rangos 60/70 mm son la
propuesta del contrato E5 y requieren confirmación de E3.

### Respuestas y reintentos

- `201 Created`: lectura guardada. El JSON de respuesta incluye `lectura_id` y
  `duplicado: false`.
- `201 Created` con `duplicado: true`: reintento exacto ya recibido; no crea otra
  lectura ni otra alerta.
- `422 Unprocessable Entity`: estructura, estación, valor o timestamp inválido.
- `409 Conflict`: `id_fruta` ya está asociado a otro `id_lote`.

Ante un timeout, reintentar **el mismo JSON sin cambiar el timestamp** para que
la clave de idempotencia detecte el duplicado. Usar timeout finito y reintentos
con espera creciente. No enviar imágenes por esta API; E5 recibe únicamente
metadatos y resultados.

Ejemplo mínimo con Python:

```python
import requests

url = "http://192.168.0.10:8000/api/v1/ingest/e3"
mensaje = {
    "id_lote": "LOTE-2026-11-A",
    "id_fruta": "M-0001",
    "estacion": "E3",
    "timestamp": "2026-11-25T10:15:41Z",
    "datos": {"diametro_mm": 64, "tamano": "mediano"},
}
respuesta = requests.post(url, json=mensaje, timeout=3)
respuesta.raise_for_status()
print(respuesta.status_code, respuesta.json())
```

## Consultas para E4 y otras estaciones

| Método y ruta | Respuesta / uso |
|---|---|
| `GET /api/v1/lotes` | Lotes y `total_frutas` |
| `GET /api/v1/lotes/{id_lote}/frutas?limit=20` | Mandarinas recientes; `limit` entre 1 y 100 |
| `GET /api/v1/frutas/{id_fruta}` | Trazabilidad de E1–E4, salida y alertas |
| `GET /api/v1/metricas?lote={id_lote}` | Producción del turno/del último periodo, rechazo y categorías |
| `GET /api/v1/alertas?activas=true` | Objeto `{ "alertas": [...] }`; `false` incluye resueltas |
| `PATCH /api/v1/alertas/{id}` | Resolver una alerta |
| `GET /api/v1/estaciones/estado` | Señal de estaciones y estado de E6 |
| `GET /api/v1/sistema` | CPU, RAM, temperatura y uso de almacenamiento |
| `GET /api/v1/predicciones` | Métricas de evaluación y pronóstico entrenado; `503` si aún no hay modelo |

Las rutas de grupos (`/lotes/{id}/grupos`, `/grupos/{id}`) fueron reemplazadas
por rutas de mandarinas. Una lectura E4 cierra la mandarina; si llegan mensajes
fuera de orden, se consolidan por `id_fruta`.

## Cambios que requieren las estaciones actuales

La revisión de los repositorios encontró protocolos de comunicación internos
que todavía no son el contrato v3 de E5. Esta guía describe qué debe adaptarse;
no cambia el código de los otros repositorios.

| Repo | Estado observado | Ajuste antes de la integración con E5 |
|---|---|---|
| [estacion1](https://github.com/PackIA-Coquimbo/estacion1) | E1 deja manifiestos por fruta y su emisor entrega imágenes a E2 mediante archivos/memoria compartida. | Además del flujo de imágenes E1→E2, enviar a E5 `id_lote`, `id_fruta`, timestamp con zona y los tres campos de captura de `datos`. |
| [estacion2](https://github.com/PackIA-Coquimbo/estacion2) | El listener entrega `station_id`, `predicted_class` y `confidence_percent` por memoria compartida. | Mapear la clasificación a `calidad_visual`, confianza de 0 a 1 y defectos; incorporar los identificadores del manifiesto E1 y enviar `POST /api/v1/ingest/e2`. |
| [estacion3](https://github.com/PackIA-Coquimbo/estacion3) | Envía a `POST /fruta` un payload con `estacion`, `id`, `calibre`, `diametro_mm` y `hora`; `revision` puede llevar diámetro nulo. | Cambiar a la ruta v3 y mapear a `id_fruta`, timestamp ISO con zona y `datos.diametro_mm`/`datos.tamano`. E5 exige diámetro positivo y no acepta `revision` como tamaño; ese caso necesita acuerdo de contrato con E3 antes de poder representarse. |
| [estacion4](https://github.com/PackIA-Coquimbo/estacion4) | El README consume lotes/grupos y envía una decisión por grupo; el modelo de lotes que ya tiene es propio de E4. | Cambiar las consultas a frutas/trazabilidad v3 y emitir un `POST /api/v1/ingest/e4` por mandarina. No reutilizar su modelo de lote como predictor de E5. |
| [estacion6](https://github.com/PackIA-Coquimbo/estacion6) | El repositorio contiene un prototipo Arduino; no se encontró un emisor HTTP/contrato de telemetría implementado. | Implementar el JSON E6 y heartbeat de 10 s de `docs/formato_datos/E6.json`. |

Hay diferencias específicas aún por cerrar: E3 documenta cortes 58,5/65 mm,
mientras el contrato E5 propone 60/70 mm; además, E2 usa nombres de clases
distintos de las categorías v3. E5 no debe transformar silenciosamente estos
valores: cada equipo confirma el mapeo y luego actualiza su emisor.

## Red, disponibilidad y datos

- E5 debe escuchar en la interfaz LAN `192.168.0.10`, puerto `8000`; las demás
  estaciones deben estar en la misma red.
- No reenviar el puerto desde el router hacia Internet. La API aún no incorpora
  autenticación; CORS no es autenticación ni control para clientes HTTP.
- SQLite usa WAL y conserva lecturas crudas. No hay purga automática: programar
  respaldos verificados fuera de la tarjeta/SSD y comprobar el uso del disco en
  `GET /api/v1/sistema`. El endpoint activa `almacenamiento_alerta` desde 80 %.
- En la Raspberry Pi, configura `BACKUP_DIR` en `.env` con el punto de montaje
  del disco externo, por ejemplo `/mnt/e5-respaldo`; luego ejecuta
  `docker compose -f docker-compose.yml exec backend python tools/backup_db.py --destino /backups`.
- El modelo de E5 es un baseline autorregresivo que pronostica frutas/hora y
  porcentaje de rechazo. Requiere al menos 24 horas con datos clasificados;
  conviene acumular varios días antes de evaluar su utilidad. No se entrega un
  modelo entrenado ni métricas de producción mientras no exista ese historial.
