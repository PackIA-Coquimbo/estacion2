"""
Suite de Pruebas Automatizadas - Fase 5: Automatización y Afinamiento de Raspberry Pi OS
(Estación 2: Análisis Visual de Calidad - OrangeQuality-AI)

Verifica todos los criterios de la Definición de Terminado (DoD) de la Fase 5:
1. Existencia y completitud del script de provisionamiento scripts/setup_raspberry.sh.
2. Comprobación de directivas de CPU Governor ('performance') y persistencia.
3. Desactivación de Swap en MicroSD para evitar degradación.
4. Compatibilidad de dependencias ligeras de requirements-pi.txt.
5. Formato de saltos de línea LF (Unix) para evitar errores de interpretación en bash.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_health_pi import run_diagnostics, check_ram_and_shm


def test_setup_script_structure():
    """Valida la sintaxis, directivas y comandos clave del script de provisionamiento."""
    print("[1/4] Verificando script scripts/setup_raspberry.sh...")
    script_path = PROJECT_ROOT / "scripts" / "setup_raspberry.sh"
    assert script_path.exists(), f"El archivo {script_path} debe existir."

    with open(script_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Shebang y control de errores
    assert content.startswith("#!/usr/bin/env bash") or content.startswith("#!/bin/bash"), "Shebang inválido."
    assert "set -euo pipefail" in content or "set -e" in content, "Debe contener manejo estricto de errores (set -e)."

    # Verificación de privilegios
    assert "EUID" in content or "root" in content, "Debe validar permisos de root/sudo."

    # Paquetes clave requeridos por FASES_OPTIMIZACION_PI.md
    for pkg in ("python3-pip", "python3-numpy", "libgl1-mesa-glx", "cpufrequtils", "zram-tools"):
        assert pkg in content, f"Falta el paquete requerido: {pkg}"

    # Governor de CPU
    assert "cpufreq-set -g performance" in content, "Debe configurar cpufreq-set en performance."
    assert "GOVERNOR=\"performance\"" in content or "GOVERNOR='performance'" in content, "Debe persistir el governor."

    # Desactivar swap en MicroSD
    assert "dphys-swapfile swapoff" in content, "Debe desactivar dphys-swapfile."

    # Instalación de requirements-pi.txt
    assert "requirements-pi.txt" in content, "Debe instalar requirements-pi.txt."

    print("  -> Comandos de instalación, afinamiento de CPU y mitigación de swap validados.")


def test_unix_line_endings():
    """Verifica que el script de bash no contenga saltos de línea Windows CRLF."""
    print("[2/4] Verificando codificación y saltos de línea Unix (LF)...")
    script_path = PROJECT_ROOT / "scripts" / "setup_raspberry.sh"
    with open(script_path, "rb") as f:
        raw_bytes = f.read()

    assert b"\r\n" not in raw_bytes, "El script contiene retornos de carro Windows (CRLF). Debe ser LF."
    print("  -> Saltos de línea LF validados correctamente.")


def test_systemd_service_file():
    """Verifica la plantilla del servicio systemd para la Estación 2."""
    print("[3/4] Verificando archivo de servicio systemd (scripts/estacion2.service)...")
    service_path = PROJECT_ROOT / "scripts" / "estacion2.service"
    assert service_path.exists(), "Debe existir estacion2.service."

    with open(service_path, "r", encoding="utf-8") as f:
        service_content = f.read()

    assert "[Unit]" in service_content
    assert "[Service]" in service_content
    assert "[Install]" in service_content
    assert "Restart=" in service_content
    print("  -> Plantilla de servicio systemd validada.")


def test_diagnostics_tool():
    """Verifica la ejecución de la herramienta de diagnóstico de salud."""
    print("[4/4] Verificando scripts/check_health_pi.py...")
    diag = run_diagnostics()
    assert "governor" in diag
    assert "temperature" in diag
    assert "shm" in diag
    assert "model_exists" in diag
    assert diag["model_exists"] is True, "El modelo INT8 debe existir en weights/."
    print("  -> Diagnóstico de salud ejecutado con éxito.")


def run_all_tests():
    print("=" * 65)
    print("EJECUTANDO TEST SUITE - FASE 5: AUTOMATION & OS TUNING")
    print("=" * 65)
    test_setup_script_structure()
    test_unix_line_endings()
    test_systemd_service_file()
    test_diagnostics_tool()
    print("=" * 65)
    print("TODAS LAS PRUEBAS DE LA FASE 5 PASARON EXITOSAMENTE (DoD CUMPLIDO)")
    print("=" * 65)


if __name__ == "__main__":
    run_all_tests()

