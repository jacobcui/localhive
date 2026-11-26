"""The model catalog plus the "will this actually run here?" logic.

A model is recommended only when it fits in memory with room to spare
for the KV cache, the compute buffer and the OS. We deliberately keep
the rules conservative: a model that *loads* but swaps is not runnable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .hardware import GIB, Hardware

# --------------------------------------------------------------------------
# Tunables. These are the knobs that decide "runnable" vs "not".
# --------------------------------------------------------------------------

#: Fraction of RAM the OS + other processes get to keep (WSL2 + desktop).
OS_RESERVE_FRACTION = 0.25
#: Absolute floor for the OS reserve, so big-RAM boxes do not give it all away.
OS_RESERVE_MIN_BYTES = 2 * GIB
#: Extra headroom for llama.cpp compute buffers / graph.
COMPUTE_OVERHEAD_BYTES = int(1.2 * GIB)
#: Default context we plan for when sizing the KV cache.
DEFAULT_CONTEXT = 8192
#: Context beyond which we warn that the KV cache grows steeply.
COMFORTABLE_CONTEXT = 32768

#: Fraction of total VRAM that is actually usable after the driver/display.
VRAM_USABLE_FRACTION = 0.90
#: Fraction of total VRAM we are willing to hand to a single model.
VRAM_TARGET_FRACTION = 0.85


@dataclass
class Model:
    """A single downloadable GGUF."""

    key: str                      # short name used on the CLI
    repo: str                     # huggingface repo id
    filename: str                 # file inside the repo
    size_bytes: int
    params_b: float               # billions of parameters
    quant: str                    # IQ2_M, Q4_K_M, ...
    bits_per_weight: float
    kv_bytes_per_token: int       # KV cache growth per token at f16
    family: str = ""
    native_context: int = 262144
    multimodal: bool = False
    mmproj: Optional[str] = None  # vision projector file, if any
    mtp: Optional[str] = None     # multi-token-prediction draft head, if any
    notes: str = ""

    @property
    def size_gib(self) -> float:
        return self.size_bytes / GIB

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/main/{self.filename}"

    @property
    def display_name(self) -> str:
        return f"{self.repo.split('/')[-1]}-{self.quant}"

    def kv_bytes(self, context: int) -> int:
        return self.kv_bytes_per_token * context


# --------------------------------------------------------------------------
# Fit verdicts
# --------------------------------------------------------------------------

VERDICT_ORDER = {"perfect": 0, "good": 1, "tight": 2, "cpu-only": 3, "too-big": 4}


@dataclass
class Fit:
    verdict: str                  # perfect | good | tight | cpu-only | too-big
    reason: str
    required_bytes: int
    usable_bytes: int
    context: int
    gpu_layers: int               # -1 means "all layers on GPU"
    offload_ratio: float          # 0.0 = CPU only, 1.0 = fully on GPU
    #: How comfortable the fit is, independent of the verdict wording.
    #: 0 = nothing is left on the table (CPU-only but plenty of RAM spare),
    #: higher = closer to the edge. Used to pick the best build per family
    #: so we never recommend a 22G "tight" build over an 8G "perfect" one.
    pressure: float = 0.0

    @property
    def runnable(self) -> bool:
        return self.verdict != "too-big"

    @property
    def icon(self) -> str:
        return {
            "perfect": "[##]",
            "good": "[#-]",
            "tight": "[!!]",
            "cpu-only": "[cc]",
            "too-big": "[XX]",
        }[self.verdict]


def usable_memory(hw: Hardware) -> tuple[int, int]:
    """Return (usable_system_ram, usable_vram) in bytes."""
    reserve = max(int(hw.total_ram_bytes * OS_RESERVE_FRACTION), OS_RESERVE_MIN_BYTES)
    usable_ram = max(hw.total_ram_bytes - reserve, 0)

    gpu = hw.best_gpu
    if hw.unified_memory:
        # Unified memory: CPU and GPU address the same physical pool, so we
        # hold back the OS reserve once rather than once per engine.
        usable_vram = int(usable_ram * VRAM_TARGET_FRACTION)
    elif gpu is None:
        usable_vram = 0
    else:
        usable_vram = int(gpu.vram_bytes * VRAM_USABLE_FRACTION * VRAM_TARGET_FRACTION)
    return usable_ram, usable_vram


def evaluate(model: Model, hw: Hardware, context: int = DEFAULT_CONTEXT) -> Fit:
    """Decide whether *model* is runnable on *hw* at *context*."""
    usable_ram, usable_vram = usable_memory(hw)

    weights = model.size_bytes
    kv = model.kv_bytes(context)
    need_ram = weights + kv + COMPUTE_OVERHEAD_BYTES
    need_vram = weights + kv

    if hw.unified_memory:
        # One pool. Fits if the whole working set fits in usable RAM.
        if need_ram <= usable_ram * 0.80:
            return Fit("perfect", "comfortable in unified memory", need_ram,
                       usable_ram, context, -1, 1.0,
                       pressure=need_ram / max(usable_ram, 1))
        if need_ram <= usable_ram:
            return Fit("good", "fits with modest headroom", need_ram,
                       usable_ram, context, -1, 1.0,
                       pressure=need_ram / max(usable_ram, 1))
        if need_ram <= hw.total_ram_bytes:
            return Fit("tight", "fits but will pressure memory", need_ram,
                       usable_ram, context, -1, 1.0,
                       pressure=need_ram / max(usable_ram, 1))
        return Fit("too-big", "exceeds unified memory", need_ram,
                   hw.total_ram_bytes, context, 0, 0.0, pressure=1.0)

    if usable_vram > 0:
        if need_vram <= usable_vram * 0.80:
            return Fit("perfect", "fully offloadable to GPU", need_vram,
                       usable_vram, context, -1, 1.0,
                       pressure=need_vram / max(usable_vram, 1))
        if need_vram <= usable_vram:
            return Fit("good", "fits in VRAM with headroom", need_vram,
                       usable_vram, context, -1, 1.0,
                       pressure=need_vram / max(usable_vram, 1))
        if need_ram <= usable_ram:
            ratio = max(0.0, min(1.0, (usable_vram - kv) / max(weights, 1)))
            layers = -1 if ratio >= 0.99 else int(round(ratio * 100))
            return Fit("tight", f"partial offload ~{ratio*100:.0f}% of layers",
                       need_ram, usable_ram, context, layers, ratio,
                       pressure=0.5 + need_ram / max(usable_ram, 1) / 2)
        return Fit("too-big", "exceeds both VRAM and RAM", need_ram,
                   max(usable_ram, usable_vram), context, 0, 0.0, pressure=1.0)

    # CPU only.
    if need_ram <= usable_ram * 0.80:
        return Fit("cpu-only", "runs on CPU comfortably", need_ram,
                   usable_ram, context, 0, 0.0,
                   pressure=need_ram / max(usable_ram, 1))
    if need_ram <= usable_ram:
        return Fit("cpu-only", "runs on CPU", need_ram, usable_ram, context, 0, 0.0,
                   pressure=0.5 + need_ram / max(usable_ram, 1) / 2)
    return Fit("too-big", "exceeds usable RAM", need_ram, usable_ram, context, 0, 0.0,
                   pressure=1.0)


def suggested_context(model: Model, hw: Hardware) -> int:
    """Largest context (rounded to a power of two) that still fits."""
    usable_ram, usable_vram = usable_memory(hw)
    budget = usable_ram if hw.unified_memory or usable_vram == 0 else usable_vram
    if hw.unified_memory:
        budget = int(budget * 0.80)
    room = budget - model.size_bytes - COMPUTE_OVERHEAD_BYTES
    if room <= 0 or model.kv_bytes_per_token <= 0:
        return 2048
    ctx = 2048
    while ctx * 2 <= COMFORTABLE_CONTEXT and model.kv_bytes(ctx * 2) <= room:
        ctx *= 2
    return min(ctx, model.native_context)


def rank(model: Model, hw: Hardware, context: int = DEFAULT_CONTEXT):
    """Sort key: best verdict, then least memory pressure, then capability.

    "Capability" is the parameter count, so within equally comfortable
    builds we still steer the user toward the bigger, better model.
    """
    fit = evaluate(model, hw, context)
    return (VERDICT_ORDER[fit.verdict], round(fit.pressure, 2), -model.params_b)
