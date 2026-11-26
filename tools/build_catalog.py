#!/usr/bin/env python3
"""Regenerate src/localhive/catalog.py from the HuggingFace API.

    python3 tools/build_catalog.py > src/localhive/catalog.py

Reads exact file sizes from the API so the fit calculations stay honest.
Adding a publisher is one entry in PUBLISHERS below.

Content labelling
-----------------
Every repo declares its content type explicitly rather than having it
sniffed from filenames. "std" is a normal safety-tuned or vendor model;
"unc" is an uncensored community fine-tune. The CLI defaults to showing
only "std", so an uncensored build is never something you get by accident.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass

API = "https://huggingface.co/api/models/{repo}?blobs=true"


@dataclass(frozen=True)
class Repo:
    """One HuggingFace repo we pull quants from."""

    repo: str
    family: str          # grouping label, e.g. "Llama-3.2-3B"
    params_b: float
    native_context: int
    kv_bytes_per_token: int
    multimodal: bool = False
    kind: str = "dense"  # "dense" | "moe"
    content: str = "std"  # "std" | "unc"


#: The catalog. Standard models first; uncensored builds are opt-in.
PUBLISHERS: list[Repo] = [
    # ------------------------------------------------------------------
    # Mainstream / vendor models  (content = std, the default view)
    # ------------------------------------------------------------------
    Repo("bartowski/Llama-3.2-3B-Instruct-GGUF",
         "Llama-3.2-3B", 3.0, 131072, 16384),
    Repo("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
         "Llama-3.1-8B", 8.0, 131072, 32768),
    Repo("bartowski/Qwen2.5-7B-Instruct-GGUF",
         "Qwen2.5-7B", 7.0, 32768, 28672),
    # Note: Qwen/Qwen2.5-7B-Instruct-GGUF also exists, but it carries the
    # same quants as bartowski's with less coverage, so including both just
    # doubles up one family in the listing. One source per family.
    Repo("unsloth/Qwen3-8B-GGUF",
         "Qwen3-8B", 8.0, 40960, 32768),
    Repo("bartowski/Mistral-7B-Instruct-v0.3-GGUF",
         "Mistral-7B", 7.0, 32768, 28672),
    Repo("bartowski/gemma-2-9b-it-GGUF",
         "Gemma-2-9B", 9.0, 8192, 40960),
    Repo("bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
         "Qwen2.5-Coder-7B", 7.0, 32768, 28672),

    # ------------------------------------------------------------------
    # Uncensored community fine-tunes  (content = unc, opt-in only)
    # ------------------------------------------------------------------
    Repo("HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive",
         "Qwen3.6-35B-A3B", 35.0, 262144, 65536, True, "moe", "unc"),
    Repo("HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF",
         "Qwen3.8-27B", 27.0, 262144, 81920, True, "dense", "unc"),
    Repo("HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive",
         "Qwen3.5-9B", 9.0, 262144, 40960, True, "dense", "unc"),
    Repo("HauhauCS/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive",
         "Qwen3.6-27B", 27.0, 262144, 81920, True, "dense", "unc"),
    Repo("HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive",
         "Gemma-4-E4B", 8.0, 131072, 32768, True, "dense", "unc"),
    Repo("HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced",
         "Gemma4-12B", 12.0, 131072, 49152, True, "dense", "unc"),
    Repo("HauhauCS/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-MTP",
         "Gemma4-26B-A4B", 26.0, 131072, 24576, True, "moe", "unc"),
    Repo("HauhauCS/Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-MTP",
         "Gemma4-31B", 31.0, 131072, 65536, True, "dense", "unc"),
    Repo("HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive",
         "Qwen3.5-4B", 4.0, 262144, 20480, True, "dense", "unc"),
    Repo("HauhauCS/Qwen3.5-2B-Uncensored-HauhauCS-Aggressive",
         "Qwen3.5-2B", 2.0, 262144, 12288, True, "dense", "unc"),
    Repo("HauhauCS/Gemma-4-E2B-Uncensored-HauhauCS-Aggressive",
         "Gemma-4-E2B", 5.0, 131072, 24576, True, "dense", "unc"),
]

#: bits per weight, derived from real published file sizes
QUANT_BPW = {
    "IQ1_S": 1.7, "IQ2_XXS": 2.2, "IQ2_XS": 2.4, "IQ2_M": 2.7,
    "IQ3_XXS": 3.1, "IQ3_XS": 3.3, "IQ3_M": 3.7,
    "IQ4_XS": 4.25, "IQ4_NL": 4.5,
    "Q2_K": 3.3, "Q2_K_S": 3.2, "Q2_K_P": 3.4,
    "Q3_K_S": 3.6, "Q3_K_M": 3.9, "Q3_K_L": 4.1, "Q3_K_P": 3.9,
    "Q4_0": 4.6, "Q4_0_4_4": 4.6, "Q4_0_4_8": 4.6, "Q4_0_8_8": 4.6,
    "Q4_K_S": 4.6, "Q4_K_M": 4.85, "Q4_K_L": 5.0, "Q4_K_P": 5.1,
    "Q5_0": 5.55, "Q5_K_S": 5.55, "Q5_K_M": 5.6, "Q5_K_P": 5.65,
    "Q6_K": 6.6, "Q6_K_L": 6.9, "Q6_K_P": 6.6,
    "Q8_0": 8.5, "Q8_K_P": 8.5,
    "BF16": 16.0, "F16": 16.0, "F32": 32.0,
}

# Matches the quant token at the end of a GGUF filename. Covers llama.cpp
# (Q4_K_M, IQ3_XS), the HauhauCS "P" variants (Q4_K_P), and the LM Studio
# repack suffixes bartowski publishes (Q4_0_4_4, Q4_0_8_8), in any case.
QUANT_RE = re.compile(
    r"("
    r"IQ\d+_[A-Z0-9_]+"
    r"|Q\d+(_K)?(_[A-Z0-9]+)*"
    r"|BF16|F16|F32"
    r")",
    re.IGNORECASE,
)


def fetch(repo: str) -> list[tuple[str, int]]:
    req = urllib.request.Request(API.format(repo=repo),
                                 headers={"User-Agent": "localhive/0.1"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    out = []
    for sib in data.get("siblings", []):
        name = sib.get("rfilename", "")
        if name.endswith(".gguf") and sib.get("size"):
            out.append((name, int(sib["size"])))
    return out


def classifier(filename: str) -> str:
    """'model' | 'mmproj' | 'mtp'

    Anything that is not a known companion file is treated as a model. A
    filename we cannot parse is far more likely to be an unusual-but-valid
    quant than a non-model, and silently dropping it means the user never
    gets told the model exists.
    """
    base = filename[:-5] if filename.endswith(".gguf") else filename
    low = base.lower()
    if low.startswith("mmproj"):
        return "mmproj"
    if low.startswith("mtp-") or low.startswith("mtp_") or "fastmtp" in low:
        return "mtp"
    # llama.cpp splits very large models across shards; those are not
    # independently loadable, so skip them.
    if re.search(r"-\d{5}-of-\d{5}\.gguf$", filename):
        return "shard"
    return "model"


def quant_of(filename: str) -> str:
    """Best-effort quant label. Falls back to the last dash token."""
    base = filename[:-5] if filename.endswith(".gguf") else filename
    tok = base.rsplit("-", 1)[-1]
    if QUANT_RE.fullmatch(tok):
        return tok.upper()
    return tok.upper()


def main() -> int:
    entries = []
    used_keys: set[str] = set()
    failures = []

    for spec in PUBLISHERS:
        try:
            files = fetch(spec.repo)
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            failures.append((spec.repo, str(exc)))
            continue

        models = []
        companions: dict[str, str] = {}
        for name, size in files:
            kind = classifier(name)
            if kind in ("mmproj", "mtp"):
                companions[kind] = name
            elif kind == "model":
                models.append((name, size))

        for name, size in sorted(models, key=lambda x: x[1]):
            quant = quant_of(name)
            key = f"{spec.family.lower()}-{quant.lower().replace('_', '')}"
            base = key
            i = 2
            while key in used_keys:
                key = f"{base}-{i}"
                i += 1
            used_keys.add(key)
            entries.append({
                "key": key, "repo": spec.repo, "filename": name, "size": size,
                "params": spec.params_b, "quant": quant,
                "bpw": QUANT_BPW.get(quant, 4.5),
                "kv": spec.kv_bytes_per_token,
                "family": spec.family, "ctx": spec.native_context,
                "mm": spec.multimodal, "tag": spec.kind,
                "content": spec.content,
                **companions,
            })

    entries.sort(key=lambda e: (e["content"] != "std", e["family"], e["size"]))

    n_std = sum(1 for e in entries if e["content"] == "std")
    n_unc = len(entries) - n_std

    lines = [
        '"""Static catalog of GGUF builds.',
        "",
        "Generated by tools/build_catalog.py from the HuggingFace API.",
        "File sizes are exact byte counts as published upstream, so the",
        "fit calculations do not have to guess.",
        "",
        "Each entry declares its content type; the CLI shows standard",
        "models by default and uncensored builds only when asked.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from .models import Model",
        "",
        f"#: {n_std} standard models, {n_unc} uncensored.",
        'STANDARD = "standard"',
        'UNCENSORED = "uncensored"',
        "",
        "CATALOG: list[Model] = [",
    ]
    for e in entries:
        content = "uncensored" if e["content"] == "unc" else "standard"
        parts = [
            f'key="{e["key"]}"',
            f'repo="{e["repo"]}"',
            f'filename="{e["filename"]}"',
            f'size_bytes={e["size"]}',
            f'params_b={e["params"]}',
            f'quant="{e["quant"]}"',
            f'bits_per_weight={e["bpw"]}',
            f'kv_bytes_per_token={e["kv"]}',
            f'family="{e["family"]}"',
            f'native_context={e["ctx"]}',
            f'multimodal={e["mm"]}',
        ]
        if "mmproj" in e:
            parts.append(f'mmproj="{e["mmproj"]}"')
        if "mtp" in e:
            parts.append(f'mtp="{e["mtp"]}"')
        parts.append(f'notes="{e["tag"]}"')
        parts.append(f'content="{content}"')
        lines.append("    Model(")
        lines.extend(f"        {p}," for p in parts)
        lines.append("    ),")
    lines += [
        "]",
        "",
        "",
        "BY_KEY: dict[str, Model] = {m.key: m for m in CATALOG}",
        "",
        "",
        "def aliases() -> dict[str, str]:",
        '    """Loose names users actually type -> canonical key."""',
        "    out: dict[str, str] = {}",
        "    for m in CATALOG:",
        "        out[m.key] = m.key",
        "        out[m.filename] = m.key",
        "        out[m.display_name.lower()] = m.key",
        "    return out",
    ]
    print("\n".join(lines))

    print(f"# generated {len(entries)} entries "
          f"({n_std} standard, {n_unc} uncensored)", file=sys.stderr)
    for repo, err in failures:
        print(f"warning: skipped {repo}: {err}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
