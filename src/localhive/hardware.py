"""Detect host hardware so we only recommend models that actually fit.

Targets WSL2 Ubuntu, native Linux, and macOS (Apple Silicon / Intel).
Everything here is stdlib-only so it runs before any pip install.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Optional

GIB = 1024 ** 3


def _run(cmd: list[str], timeout: int = 10) -> str:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _meminfo() -> dict[str, int]:
    """Parse /proc/meminfo (Linux / WSL2)."""
    info: dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                nums = re.findall(r"\d+", rest)
                if nums:
                    info[key.strip()] = int(nums[0]) * 1024
    except OSError:
        pass
    return info


def _is_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        with open("/proc/version", "r", encoding="utf-8") as fh:
            return "microsoft" in fh.read().lower()
    except OSError:
        return False


@dataclass
class Gpu:
    name: str
    vram_bytes: int
    backend: str  # cuda | rocm | metal | none

    @property
    def vram_gib(self) -> float:
        return self.vram_bytes / GIB


@dataclass
class Hardware:
    os_name: str
    distro: str
    kernel: str
    arch: str
    is_wsl: bool
    cpu: str
    physical_cores: int
    logical_cores: int
    total_ram_bytes: int
    available_ram_bytes: int
    swap_bytes: int
    gpus: list[Gpu] = field(default_factory=list)

    @property
    def total_ram_gib(self) -> float:
        return self.total_ram_bytes / GIB

    @property
    def available_ram_gib(self) -> float:
        return self.available_ram_bytes / GIB

    @property
    def best_gpu(self) -> Optional[Gpu]:
        if not self.gpus:
            return None
        return max(self.gpus, key=lambda g: g.vram_bytes)

    @property
    def unified_memory(self) -> bool:
        """Apple Silicon shares RAM between CPU and GPU."""
        return self.arch in ("arm64", "aarch64") and self.os_name == "Darwin"


def _detect_gpus_linux() -> list[Gpu]:
    gpus: list[Gpu] = []

    # --- NVIDIA (CUDA) -------------------------------------------------
    if shutil.which("nvidia-smi"):
        out = _run([
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[1].isdigit():
                gpus.append(
                    Gpu(name=parts[0], vram_bytes=int(parts[1]) * 1024 * 1024,
                        backend="cuda")
                )

    # --- AMD (ROCm) ----------------------------------------------------
    if not gpus and shutil.which("rocm-smi"):
        out = _run(["rocm-smi", "--showmeminfo", "vram", "--csv"])
        name = "AMD GPU"
        for line in out.splitlines():
            if "card" in line.lower() and "," in line:
                cells = [c.strip() for c in line.split(",")]
                nums = [int(c) for c in cells[1:] if c.isdigit()]
                if nums:
                    gpus.append(Gpu(name=name, vram_bytes=max(nums), backend="rocm"))
                    break

    # --- Intel Arc / iGPU via sysfs ------------------------------------
    if not gpus and os.path.isdir("/sys/class/drm"):
        for entry in sorted(os.listdir("/sys/class/drm")):
            if not entry.startswith("card") or "-" in entry:
                continue
            base = f"/sys/class/drm/{entry}/device"
            try:
                with open(f"{base}/mem_info_vram_total", "r") as fh:
                    vram = int(fh.read().strip())
                gpus.append(Gpu(name=f"Intel GPU ({entry})", vram_bytes=vram,
                                backend="sycl"))
            except (OSError, ValueError):
                continue
    return gpus


def _detect_gpus_macos() -> list[Gpu]:
    out = _run(["system_profiler", "SPDisplaysDataType"], timeout=20)
    name = "Apple GPU"
    vram = 0
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("Chipset Model:"):
            name = s.split(":", 1)[1].strip()
        elif "VRAM" in s and ":" in s:
            nums = re.findall(r"\d+", s.split(":", 1)[1])
            if nums:
                unit = s.lower()
                v = int(nums[0])
                if "gb" in unit:
                    vram = v * GIB
                elif "mb" in unit:
                    vram = v * 1024 * 1024
    return [Gpu(name=name, vram_bytes=vram, backend="metal")]


def detect() -> Hardware:
    system = platform.system()
    machine = platform.machine()
    cpu = platform.processor() or machine

    physical = logical = os.cpu_count() or 0

    total_ram = available_ram = 0
    swap = 0
    distro = ""

    if system == "Linux":
        mem = _meminfo()
        total_ram = mem.get("MemTotal", 0)
        available_ram = mem.get("MemAvailable", mem.get("MemFree", 0))
        swap = mem.get("SwapTotal", 0)
        gpus = _detect_gpus_linux()

        pretty = ""
        try:
            with open("/etc/os-release", "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("PRETTY_NAME="):
                        pretty = line.split("=", 1)[1].strip().strip('"')
        except OSError:
            pass
        distro = pretty

        lscpu = _run(["lscpu"])
        for line in lscpu.splitlines():
            if line.startswith("Model name:"):
                cpu = line.split(":", 1)[1].strip()
            elif line.startswith("CPU(s):"):
                logical = int(re.findall(r"\d+", line)[0] or logical)
            elif line.startswith("Core(s) per socket:"):
                per_socket = int(re.findall(r"\d+", line)[0] or 0)
                sockets = 1
                for l2 in lscpu.splitlines():
                    if l2.startswith("Socket(s):"):
                        sockets = int(re.findall(r"\d+", l2)[0] or 1)
                physical = per_socket * sockets or physical
    elif system == "Darwin":
        total_ram = int(_run(["sysctl", "-n", "hw.memsize"]) or 0)
        available_ram = total_ram  # macOS has no cheap MemAvailable
        cpu = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or cpu
        logical = int(_run(["sysctl", "-n", "hw.logicalcpu"]) or logical)
        physical = int(_run(["sysctl", "-n", "hw.physicalcpu"]) or logical)
        gpus = _detect_gpus_macos()
        # Apple Silicon uses unified memory; expose it as usable VRAM.
        if machine in ("arm64", "aarch64") and gpus:
            gpus[0] = Gpu(name=gpus[0].name, vram_bytes=total_ram, backend="metal")
        distro = f"macOS {platform.mac_ver()[0]}"
    else:
        gpus = []

    return Hardware(
        os_name=system,
        distro=distro,
        kernel=platform.release(),
        arch=machine,
        is_wsl=_is_wsl(),
        cpu=cpu,
        physical_cores=physical,
        logical_cores=logical,
        total_ram_bytes=total_ram,
        available_ram_bytes=available_ram,
        swap_bytes=swap,
        gpus=gpus,
    )


def to_dict(hw: Hardware) -> dict:
    """JSON-friendly view (used by 'localhive hardware --json')."""
    return {
        "os": hw.os_name,
        "distro": hw.distro,
        "kernel": hw.kernel,
        "arch": hw.arch,
        "is_wsl": hw.is_wsl,
        "cpu": hw.cpu,
        "physical_cores": hw.physical_cores,
        "logical_cores": hw.logical_cores,
        "total_ram_bytes": hw.total_ram_bytes,
        "available_ram_bytes": hw.available_ram_bytes,
        "swap_bytes": hw.swap_bytes,
        "unified_memory": hw.unified_memory,
        "gpus": [asdict(g) for g in hw.gpus],
    }


def summary(hw: Hardware) -> str:
    lines = []
    where = "WSL2" if hw.is_wsl else hw.os_name
    lines.append(f"  Platform : {hw.distro} [{where}] {hw.arch}")
    lines.append(f"  CPU      : {hw.cpu}")
    lines.append(f"  Cores    : {hw.physical_cores} physical / {hw.logical_cores} logical")
    lines.append(
        f"  Memory   : {hw.total_ram_gib:.1f} GiB total, "
        f"{hw.available_ram_gib:.1f} GiB available"
    )
    if hw.gpus:
        for g in hw.gpus:
            lines.append(f"  GPU      : {g.name} - {g.vram_gib:.1f} GiB ({g.backend})")
    else:
        lines.append("  GPU      : none detected - CPU inference only")
    if hw.unified_memory:
        lines.append("  Note     : Apple Silicon unified memory (CPU+GPU share RAM)")
    elif hw.is_wsl:
        lines.append(
            "  Note     : WSL2 memory is capped by .wslconfig; "
            "see README for raising it"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    hw = detect()
    print("Detected hardware:")
    print(summary(hw))
    print()
    print(f"  CUDA offload budget : {hw.best_gpu.vram_gib if hw.best_gpu else 0:.1f} GiB")
    print(f"  System RAM budget   : {hw.total_ram_gib:.1f} GiB")
