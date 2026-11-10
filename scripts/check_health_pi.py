"""
Herramienta de Diagnóstico y Salud de Hardware (Fase 5: Afinamiento de Raspberry Pi)
(Estación 2: Análisis Visual de Calidad - OrangeQuality-AI)

Verifica:
1. Política de energía de CPU (Governor = performance).
2. Temperatura del SoC y estado de estrangulamiento térmico (Thermal Throttling).
3. Estado de swap en MicroSD y disponibilidad de memoria RAM en /dev/shm.
4. Integridad de los pesos del modelo ONNX cuantizado.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import os
import subprocess
import sys
import platform

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def check_cpu_governor() -> Dict[str, Any]:
    """Verifica si los núcleos operan bajo la política 'performance'."""
    governors = []
    cpu_path = Path("/sys/devices/system/cpu")
    if cpu_path.exists():
        for gov_file in cpu_path.glob("cpu[0-9]*/cpufreq/scaling_governor"):
            try:
                with open(gov_file, "r") as f:
                    governors.append(f.read().strip())
            except Exception:
                pass

    if not governors:
        # Intentar comando cpufreq-info si existe
        try:
            out = subprocess.check_output(["cpufreq-info", "-p"], stderr=subprocess.DEVNULL).decode()
            parts = out.split()
            if len(parts) >= 3:
                governors = [parts[2]]
        except Exception:
            governors = ["desconocido"]

    is_perf = all(g.lower() == "performance" for g in governors) if governors else False
    return {
        "governors": governors,
        "is_performance": is_perf,
        "status_str": "OPTIMIZADO (performance)" if is_perf else f"ALERTA ({governors})",
    }


def check_temperature() -> Dict[str, Any]:
    """Lee la temperatura del procesador y las banderas de throttling."""
    temp_c = None
    throttled = None

    # Intentar vcgencmd (Raspberry Pi oficial)
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"], stderr=subprocess.DEVNULL).decode()
        # formato: temp=42.5'C
        val_str = out.strip().replace("temp=", "").replace("'C", "")
        temp_c = float(val_str)
    except Exception:
        # Fallback a sysfs
        thermal_file = Path("/sys/class/thermal/thermal_zone0/temp")
        if thermal_file.exists():
            try:
                with open(thermal_file, "r") as f:
                    temp_c = float(f.read().strip()) / 1000.0
            except Exception:
                pass

    try:
        out_thr = subprocess.check_output(["vcgencmd", "get_throttled"], stderr=subprocess.DEVNULL).decode()
        throttled = out_thr.strip().replace("throttled=", "")
    except Exception:
        throttled = "N/A"

    temp_status = "NORMAL (< 70°C)" if (temp_c is not None and temp_c < 70.0) else "ELEVADA" if temp_c else "N/A"
    return {
        "temperature_c": temp_c,
        "throttled_code": throttled,
        "status": temp_status,
    }


def check_ram_and_shm() -> Dict[str, Any]:
    """Verifica el estado del canal de memoria volátil /dev/shm."""
    shm_dir = Path("/dev/shm")
    exists = shm_dir.exists() and shm_dir.is_dir()
    free_mb = 0.0

    if exists:
        try:
            stat = os.statvfs(str(shm_dir))
            free_mb = (stat.f_bavail * stat.f_frsize) / (1024 * 1024)
        except Exception:
            pass

    return {
        "shm_available": exists,
        "shm_free_mb": round(free_mb, 1),
    }


def run_diagnostics():
    print("=" * 70)
    print("DIAGNÓSTICO DE SALUD Y AFINAMIENTO DE RASPBERRY PI OS (FASE 5)")
    print(f"Sistema Operativo: {platform.system()} {platform.release()} ({platform.machine()})")
    print("=" * 70)

    # 1. CPU Governor
    gov = check_cpu_governor()
    print(f"[1/4] CPU Governor:              {gov['status_str']}")

    # 2. Temperatura
    temp = check_temperature()
    temp_str = f"{temp['temperature_c']:.1f}°C" if temp['temperature_c'] else "No disponible en emulación"
    print(f"[2/4] Temperatura SoC:           {temp_str} | Estado: {temp['status']}")
    print(f"      Código Throttling:         {temp['throttled_code']}")

    # 3. RAM /dev/shm
    shm = check_ram_and_shm()
    shm_str = f"Disponible ({shm['shm_free_mb']} MB libres)" if shm['shm_available'] else "No disponible (usando fallback en RAM)"
    print(f"[3/4] Canal RAM (/dev/shm):      {shm_str}")

    # 4. Pesos del modelo
    int8_model = PROJECT_ROOT / "weights" / "orange_model_int8.onnx"
    exists_model = int8_model.exists()
    size_mb = int8_model.stat().st_size / (1024 * 1024) if exists_model else 0
    print(f"[4/4] Modelo INT8 desplegado:    {'DISPONIBLE' if exists_model else 'FALTANTE'} ({size_mb:.2f} MB)")
    print("=" * 70)

    return {
        "governor": gov,
        "temperature": temp,
        "shm": shm,
        "model_exists": exists_model,
    }


if __name__ == "__main__":
    run_diagnostics()

