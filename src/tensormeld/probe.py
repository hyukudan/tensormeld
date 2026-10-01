"""Read-only host inventory. Detected hardware is NOT a qualified backend."""
from __future__ import annotations

import csv
import ctypes
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Any


def _command(argv: list[str], timeout: float = 5.0) -> str | None:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                                check=False, shell=False, encoding="utf-8", errors="replace")
        return result.stdout[:262144] if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def memory() -> dict[str, Any]:
    try:
        import psutil
        v = psutil.virtual_memory()
        return {"source": "psutil.virtual_memory", "provider_version": psutil.__version__,
                "total_bytes": int(v.total), "available_bytes": int(v.available)}
    except (ImportError, OSError, RuntimeError):
        pass
    if platform.system() == "Linux":
        values = {}
        for line in (_read(Path("/proc/meminfo")) or "").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] in ("MemTotal:", "MemAvailable:"):
                values[parts[0][:-1]] = int(parts[1]) * 1024
        return {"source": "/proc/meminfo", "total_bytes": values.get("MemTotal"),
                "available_bytes": values.get("MemAvailable")}
    if platform.system() == "Windows":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32),
                        ("total_phys", ctypes.c_uint64), ("avail_phys", ctypes.c_uint64),
                        ("total_page", ctypes.c_uint64), ("avail_page", ctypes.c_uint64),
                        ("total_virtual", ctypes.c_uint64), ("avail_virtual", ctypes.c_uint64),
                        ("avail_extended", ctypes.c_uint64)]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        fn = ctypes.WinDLL("kernel32", use_last_error=True).GlobalMemoryStatusEx
        fn.argtypes = [ctypes.POINTER(MemoryStatus)]
        fn.restype = ctypes.c_int
        if fn(ctypes.byref(status)):
            return {"source": "GlobalMemoryStatusEx", "total_bytes": status.total_phys,
                    "available_bytes": status.avail_phys}
    return {"source": "unavailable", "total_bytes": None, "available_bytes": None}


def nvidia_devices() -> list[dict]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return []
    output = _command([executable, "--query-gpu=name,memory.total,memory.free,driver_version",
                       "--format=csv,noheader,nounits"])
    devices = []
    for row in csv.reader(io.StringIO(output or "")):
        if len(row) != 4:
            continue
        name, total, free, driver = (x.strip() for x in row)
        try:
            total_bytes, free_bytes = int(total) * 1024**2, int(free) * 1024**2
        except ValueError:
            total_bytes = free_bytes = None
        devices.append({"vendor": "nvidia", "name": name, "driver_version": driver,
                        "reported_memory_bytes": total_bytes, "reported_free_bytes": free_bytes,
                        "source": "nvidia-smi", "qualified": False})
    return devices


def amd_devices_linux() -> list[dict]:
    if platform.system() != "Linux":
        return []
    devices, seen = [], set()
    for path in sorted(Path("/sys/class/drm").glob("card[0-9]*/device")):
        resolved = str(path.resolve())
        if resolved in seen or _read(path / "vendor") != "0x1002":
            continue
        seen.add(resolved)
        devices.append({"vendor": "amd", "pci_device_id": _read(path / "device"),
                        "source": "sysfs", "qualified": False,
                        "memory_pool": "unknown: UMA versus discrete must be qualified"})
    return devices


def interfaces() -> list[dict]:
    try:
        import psutil
        stats = psutil.net_if_stats()
        return [{"name": name, "state": "up" if value.isup else "down",
                 "negotiated_mbps": value.speed if value.speed > 0 else None,
                 "duplex": str(value.duplex), "mtu": value.mtu,
                 "source": "psutil.net_if_stats", "provider_version": psutil.__version__}
                for name, value in sorted(stats.items())]
    except (ImportError, OSError, RuntimeError):
        pass
    if platform.system() == "Windows":
        executable = shutil.which("powershell.exe") or shutil.which("pwsh")
        if executable:
            command = ("Get-NetAdapter -Physical | Select-Object Name, InterfaceDescription, "
                       "Status, LinkSpeed | ConvertTo-Json -Compress")
            raw = _command([executable, "-NoProfile", "-NonInteractive", "-Command", command])
            try:
                data = json.loads(raw or "[]")
                return data if isinstance(data, list) else [data] if isinstance(data, dict) else []
            except json.JSONDecodeError:
                return []
    if platform.system() == "Linux":
        result = []
        for path in sorted(Path("/sys/class/net").glob("*")):
            speed = _read(path / "speed")
            result.append({"name": path.name, "state": _read(path / "operstate"),
                           "negotiated_mbps": int(speed) if speed and speed.isdigit() else None,
                           "duplex": _read(path / "duplex"), "mtu": _read(path / "mtu")})
        return result
    return []


def probe() -> dict[str, Any]:
    return {"schema_version": 1, "observed_at_utc": datetime.now(timezone.utc).isoformat(),
            "os": {"system": platform.system(), "release": platform.release(),
                   "machine": platform.machine(), "python": platform.python_version()},
            "logical_cpus": os.cpu_count(), "host_memory": memory(),
            "devices": nvidia_devices() + amd_devices_linux(), "interfaces": interfaces(),
            "tools_detected": {tool: shutil.which(tool) is not None
                               for tool in ("nvidia-smi", "rocminfo", "amd-smi", "vulkaninfo")},
            "warnings": [
                "Read-only inventory: no model or driver was installed; no backend was qualified.",
                "Do not add host RAM to iGPU memory on a unified-memory device.",
                "Negotiated network speed is not measured tensor-path payload bandwidth.",
                "AMD Windows GPU telemetry and usable UMA allocation limits are not implemented.",
            ]}
