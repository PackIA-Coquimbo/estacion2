# Estación 2 en producción (Raspberry Pi 5)

Carpeta del proyecto: `/home/icci/estacion2` (rama `feat/promedio-fotos-naranja` + integración E5).

## 1. Instalar (una sola vez)

```bash
cd /home/icci/estacion2
sudo bash scripts/setup_raspberry.sh
```

Instala paquetes, crea `.venv-pi`, deja la CPU en `performance`, instala y habilita
el servicio `estacion2` (arranca solo en cada encendido).

## 2. Configurar

```bash
sudo nano /etc/default/estacion2
```

| Variable | Uso |
|---|---|
| `E5_API_URL` | URL de la Estación 5 (por defecto `http://192.168.0.10:8000`) |
| `E2_ARGS` | Opciones extra: `--no-e5`, `--photo-timeout 3`, `--threads 4`, `--model weights/orange_model_int8.onnx` |

### Modelo

Por defecto se usa `weights/orange_model.onnx` (FP32, ~92 ms por foto en Pi 5).
El INT8 (`orange_model_int8.onnx`, ~30 ms por foto) es más rápido pero en una prueba
con una mandarina con moho la calificó como `buen_estado` (77.8 %), mientras que
FP32 la calificó correctamente como `mal_estado`. Usar INT8 solo tras validarlo
con fotos reales de la cámara de E1.

Después de cambiarlo: `sudo systemctl restart estacion2`.

## 3. Operar

```bash
sudo systemctl start estacion2      # iniciar
sudo systemctl stop estacion2       # detener
sudo systemctl restart estacion2    # reiniciar
systemctl status estacion2          # estado
journalctl -u estacion2 -f          # logs en vivo
```

## 4. Contrato con la Estación 1 (cómo entrega las 2 fotos)

Bandeja en RAM: `/dev/shm/estacion2_inbox/`

1. (Opcional, recomendado) Manifiesto con los IDs de E5: `<uid>-<n>.json`
   ```json
   {"id_lote": "LOTE-2026-11-A", "id_fruta": "M-0001"}
   ```
2. Las 2 fotos: `<uid>-<n>.1.jpg` y `<uid>-<n>.2.jpg`.

Escribir cada archivo primero como `*.tmp` y luego renombrarlo (escritura atómica).
Desde Python, E1 puede usar directamente:

```python
from src.station_listener import StationCommunicator
c = StationCommunicator()   # /dev/shm
c.send_orange_photos_from_station1("7712323", 1, [foto1, foto2],
                                   id_lote="LOTE-2026-11-A", id_fruta="M-0001")
resultado = c.wait_for_orange_result_in_station1("7712323-1", timeout_sec=5)
```

Resultado: `/dev/shm/estacion2_results/<uid>-<n>.json` (promedio de las 2 fotos,
`calidad_visual`, `confianza_visual`, detalle por foto). Si falta una foto, a los
2 s se califica con la que haya (`"complete": false`).

Sin manifiesto, se envía a E5 `id_lote=<uid>` e `id_fruta=<uid>-<n>`.
El envío a E5 corre en segundo plano con reintentos: si E5 está caído, la cinta no se frena.

## 5. Probar a mano

```bash
# Servicio detenido, en primer plano:
sudo systemctl stop estacion2
.venv-pi/bin/python src/station_listener.py --no-e5

# En otra terminal, simular E1:
cp pruebas/7712323-1.*.jpg /dev/shm/estacion2_inbox/
cat /dev/shm/estacion2_results/7712323-1.json

# Calificar fotos sueltas sin servicio:
.venv-pi/bin/python src/inference.py --image a-1.1.jpg a-1.2.jpg --onnx weights/orange_model.onnx --json

# Tests
.venv-pi/bin/python -m pytest tests -q --ignore=tests/smoke_test.py
```

## 6. Red

La Pi y E5 deben estar en la misma LAN de la planta. Verificar:
`curl http://192.168.0.10:8000/docs` debe responder.
