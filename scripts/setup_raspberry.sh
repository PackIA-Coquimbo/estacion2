#!/usr/bin/env bash
# ==============================================================================
# Script de Aprovisionamiento y Afinamiento de Raspberry Pi OS
# Proyecto: Estación 2 - Análisis Visual de Calidad (OrangeQuality-AI)
#
# Configura el sistema operativo Linux para máximo rendimiento sostenido:
# 1. Instala dependencias del sistema y librerías multimedia.
# 2. Configura el CPU Governor en modo 'performance' (inmediato y persistente).
# 3. Desactiva el archivo de swap en la MicroSD para evitar degradación.
# 4. Habilita zram para compresión de memoria volátil en RAM.
# 5. Instala las dependencias ligeras de Python (requirements-pi.txt).
# ==============================================================================

set -euo pipefail

# Colores para salida en consola
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # Sin color

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}  ESTACIÓN 2: OPTIMIZACIÓN DE RASPBERRY PI OS PARA INFERENCIA VISUAL  ${NC}"
echo -e "${BLUE}======================================================================${NC}"

# 1. Comprobar permisos de superusuario
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}[ERROR] Este script debe ejecutarse con privilegios sudo o root.${NC}"
    echo "Uso: sudo bash $0"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 2. Actualizar repositorios e instalar paquetes base del sistema
echo -e "\n${GREEN}[PASO 1/5] Actualizando repositorios APT e instalando paquetes del sistema...${NC}"
apt-get update -y
apt-get install -y \
    python3-pip \
    python3-numpy \
    python3-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    cpufrequtils \
    zram-tools \
    v4l-utils

# Intentar instalar picamera2 desde apt si está disponible en Raspberry Pi OS
apt-get install -y python3-picamera2 || true

# 3. Configurar CPU Governor en Performance (Inmediato y Persistente)
echo -e "\n${GREEN}[PASO 2/5] Configurando CPU Governor en modo 'performance'...${NC}"
# Aplicación inmediata a todos los núcleos disponibles
if command -v cpufreq-set >/dev/null 2>&1; then
    cpufreq-set -g performance || true
fi

# Configuración persistente tras reinicios en Debian/Raspberry Pi OS
cat << 'EOF' > /etc/default/cpufrequtils
ENABLE="true"
GOVERNOR="performance"
MAX_SPEED="0"
MIN_SPEED="0"
EOF

# Reiniciar servicio cpufrequtils si existe
systemctl restart cpufrequtils 2>/dev/null || true

echo -e "${GREEN}  -> CPU Governor configurado en 'performance'.${NC}"
if command -v cpufreq-info >/dev/null 2>&1; then
    echo -e "${BLUE}  -> Verificación de núcleos:${NC}"
    cpufreq-info | grep -E "analyzing CPU|current policy" | head -n 8 || true
fi

# 4. Desactivar Swap en MicroSD para evitar degradación de celda
echo -e "\n${GREEN}[PASO 3/5] Desactivando Swap en MicroSD (prevención de desgaste de disco)...${NC}"
if command -v dphys-swapfile >/dev/null 2>&1; then
    dphys-swapfile swapoff || true
    systemctl disable dphys-swapfile 2>/dev/null || true
    echo -e "${GREEN}  -> dphys-swapfile desactivado.${NC}"
fi

# Activar zram en RAM si está disponible
if [ -f /etc/default/zramswap ]; then
    sed -i 's/^#*ALLOCATION=.*/ALLOCATION=256/' /etc/default/zramswap || true
    systemctl enable zramswap 2>/dev/null || true
    systemctl restart zramswap 2>/dev/null || true
    echo -e "${GREEN}  -> zramswap configurado para compresión de swap en RAM.${NC}"
fi

# 5. Preparar entorno Python e instalar dependencias ligeras
echo -e "\n${GREEN}[PASO 4/5] Instalando dependencias de Python (requirements-pi.txt)...${NC}"
REQUIREMENTS_FILE="$PROJECT_ROOT/requirements-pi.txt"

if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo -e "${RED}[ERROR] No se encontró el archivo $REQUIREMENTS_FILE.${NC}"
    exit 1
fi

# Detección de PEP 668 (Debian 12 Bookworm requiere venv o --break-system-packages)
PIP_FLAGS="--no-cache-dir"
if python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
    # En entornos Bookworm, instalar usando virtualenv del proyecto o flags permitidos
    VENV_DIR="$PROJECT_ROOT/.venv-pi"
    if [ ! -d "$VENV_DIR" ]; then
        echo -e "${BLUE}  -> Creando entorno virtual ligero: $VENV_DIR${NC}"
        python3 -m venv --system-site-packages "$VENV_DIR"
    fi
    "$VENV_DIR/bin/pip" install $PIP_FLAGS -r "$REQUIREMENTS_FILE"
    echo -e "${GREEN}  -> Dependencias instaladas en: $VENV_DIR${NC}"
else
    pip install $PIP_FLAGS -r "$REQUIREMENTS_FILE" || pip3 install $PIP_FLAGS -r "$REQUIREMENTS_FILE"
fi

# 6. Verificación de Salud, Temperatura y Diagnóstico
echo -e "\n${GREEN}[PASO 5/5] Diagnóstico de temperatura y hardware...${NC}"
if command -v vcgencmd >/dev/null 2>&1; then
    TEMP=$(vcgencmd measure_temp 2>/dev/null || echo "No disponible")
    THROTTLE=$(vcgencmd get_throttled 2>/dev/null || echo "No disponible")
    echo -e "${BLUE}  -> Temperatura actual del SoC: ${YELLOW}${TEMP}${NC}"
    echo -e "${BLUE}  -> Estado de estrangulamiento térmico: ${YELLOW}${THROTTLE}${NC}"
else
    # Fallback para kernel genérico leyendo de sysfs
    if [ -f /sys/class/thermal/thermal_zone0/temp ]; then
        MILLI=$(cat /sys/class/thermal/thermal_zone0/temp)
        CELSIUS=$((MILLI / 1000))
        echo -e "${BLUE}  -> Temperatura actual del SoC: ${YELLOW}${CELSIUS}°C${NC}"
    fi
fi

# Verificar canal /dev/shm
if [ -d /dev/shm ]; then
    SHM_SPACE=$(df -h /dev/shm | awk 'NR==2 {print $4}')
    echo -e "${GREEN}  -> Canal de memoria RAM (/dev/shm) disponible. Espacio libre: ${SHM_SPACE}${NC}"
fi

echo -e "\n${BLUE}======================================================================${NC}"
echo -e "${GREEN}  ¡CONFIGURACIÓN COMPLETADA EXITOSAMENTE! (DoD FASE 5 CUMPLIDO)        ${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo -e "Para ejecutar la Estación 2 en modo Headless de alto rendimiento:"
echo -e "  ${YELLOW}python src/inference.py --onnx weights/orange_model_int8.onnx --webcam 0 --headless${NC}"
echo -e "O escuchar eventos de la Estación 1 vía RAM:"
echo -e "  ${YELLOW}python src/station_listener.py${NC}"

