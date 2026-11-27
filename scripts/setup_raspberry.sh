#!/usr/bin/env bash
# ==============================================================================
# Instalación de la Estación 2 en Raspberry Pi OS (Bookworm / Trixie)
# Proyecto: Estación 2 - Análisis Visual de Calidad (OrangeQuality-AI)
#
#   1. Instala paquetes del sistema (nombres válidos en Debian 12 y 13).
#   2. Crea el entorno virtual .venv-pi (como el usuario dueño del proyecto).
#   3. CPU governor en 'performance' (servicio systemd, persistente).
#   4. Desactiva swap en la MicroSD si existe dphys-swapfile (zram se mantiene).
#   5. Instala el servicio systemd 'estacion2' y su configuración /etc/default/estacion2.
#
# Uso:  sudo bash scripts/setup_raspberry.sh
# ==============================================================================
set -euo pipefail

GREEN='\033[0;32m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}[ERROR] Ejecuta con sudo: sudo bash $0${NC}"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_USER="${SUDO_USER:-$(stat -c '%U' "$PROJECT_ROOT")}"
VENV_DIR="$PROJECT_ROOT/.venv-pi"

echo -e "${BLUE}== Estación 2: instalación en $PROJECT_ROOT (usuario: $APP_USER) ==${NC}"

# ---------------------------------------------------------------- 1. Paquetes
echo -e "\n${GREEN}[1/5] Paquetes del sistema...${NC}"
apt-get update -y
apt-get install -y python3-pip python3-venv python3-numpy v4l-utils
# Librerías que OpenCV necesita; el nombre cambia entre versiones de Debian
for pkg in libgl1 libglib2.0-0t64 libglib2.0-0; do
    apt-get install -y "$pkg" 2>/dev/null || true
done
apt-get install -y python3-picamera2 2>/dev/null || true

# ---------------------------------------------------------------- 2. Python
echo -e "\n${GREEN}[2/5] Entorno virtual $VENV_DIR ...${NC}"
if [ ! -x "$VENV_DIR/bin/python" ]; then
    sudo -u "$APP_USER" python3 -m venv --system-site-packages "$VENV_DIR"
fi
sudo -u "$APP_USER" "$VENV_DIR/bin/pip" install --no-cache-dir -r "$PROJECT_ROOT/requirements-pi.txt"
sudo -u "$APP_USER" "$VENV_DIR/bin/python" -c \
    "import onnxruntime, numpy, cv2; print('  onnxruntime', onnxruntime.__version__, '| numpy', numpy.__version__, '| opencv', cv2.__version__)"

if [ ! -f "$PROJECT_ROOT/weights/orange_model.onnx" ]; then
    echo -e "${RED}[ERROR] Falta weights/orange_model.onnx${NC}"
    exit 1
fi

# ---------------------------------------------------------------- 3. CPU
echo -e "\n${GREEN}[3/5] CPU governor 'performance'...${NC}"
install -m 644 "$SCRIPT_DIR/cpu-performance.service" /etc/systemd/system/cpu-performance.service
systemctl daemon-reload
systemctl enable --now cpu-performance.service
echo "  governor actual: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/d)"

# ---------------------------------------------------------------- 4. Swap
echo -e "\n${GREEN}[4/5] Swap...${NC}"
if command -v dphys-swapfile >/dev/null 2>&1; then
    dphys-swapfile swapoff || true
    systemctl disable dphys-swapfile 2>/dev/null || true
    echo "  dphys-swapfile (swap en MicroSD) desactivado."
fi
swapon --show || true

# ---------------------------------------------------------------- 5. Servicio
echo -e "\n${GREEN}[5/5] Servicio systemd 'estacion2'...${NC}"
sed -e "s#__USER__#$APP_USER#g" -e "s#__DIR__#$PROJECT_ROOT#g" \
    "$SCRIPT_DIR/estacion2.service" > /etc/systemd/system/estacion2.service
if [ ! -f /etc/default/estacion2 ]; then
    install -m 644 "$SCRIPT_DIR/estacion2.env" /etc/default/estacion2
    echo "  Configuración creada en /etc/default/estacion2 (edita E5_API_URL si corresponde)."
else
    echo "  /etc/default/estacion2 ya existe; no se sobrescribe."
fi
systemctl daemon-reload
systemctl enable estacion2.service

echo -e "\n${BLUE}== Instalación completa ==${NC}"
echo -e "Iniciar:     ${YELLOW}sudo systemctl start estacion2${NC}"
echo -e "Ver logs:    ${YELLOW}journalctl -u estacion2 -f${NC}"
echo -e "Bandeja E1:  ${YELLOW}/dev/shm/estacion2_inbox/${NC}   Resultados: ${YELLOW}/dev/shm/estacion2_results/${NC}"
