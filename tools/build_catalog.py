#!/usr/bin/env python3
"""Regenerate src/localhive/catalog.py from the HuggingFace API.

Run this when HauhauCS publishes new quants:

    python3 tools/build_catalog.py > src/localhive/catalog.py

It reads exact file sizes from the API so the fit calculations stay honest.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request

API = "https://huggingface.co/api/models/{repo}?blobs=true"

#: repo -> (family, params_b, native_context, kv_bytes_per_token, multimodal, tag)
REPOS: dict[str, tuple[str, float, int, int, bool, str]] = {
    "HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive":
        ("Qwen3.6-35B-A3B", 35.0, 262144, 65536, True, "moe"),
    "HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF":
        ("Qwen3.8-27B", 27.0, 262144, 81920, True, "dense"),
    "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive":
        ("Qwen3.5-9B", 9.0, 262144, 40960, True, "dense"),
    "HauhauCS/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive":
        ("Qwen3.6-27B", 27.0, 262144, 81920, True, "dense"),
    "HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive":
        ("Gemma-4-E4B", 8.0, 131072, 32768, True, "dense"),
    "HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced":
        ("Gemma4-12B", 12.0, 131072, 49152, True, "dense"),
    "HauhauCS/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-MTP":
        ("Gemma4-26B-A4B", 26.0, 131072, 24576, True, "moe"),
    "HauhauCS/Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-MTP":
        ("Gemma4-31B", 31.0, 131072, 65536, True, "dense"),
    "HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive":
        ("Qwen3.5-4B", 4.0, 262144, 20480, True, "dense"),
    "HauhauCS/Qwen3.5-2B-Uncensored-HauhauCS-Aggressive":
        ("Qwen3.5-2B", 2.0, 262144, 12288, True, "dense"),
    "HauhauCS/Gemma-4-E2B-Uncensored-HauhauCS-Aggressive":
        ("Gemma-4-E2B", 5.0, 131072, 24576, True, "dense"),
}

#: bits per weight, derived from the real file sizes
QUANT_BPW = {
    "IQ2_M": 2.7, "IQ3_XS": 3.3, "IQ3_M": 3.7, "IQ4_XS": 4.25, "IQ4_NL": 4.5,
    "Q2_K_P": 3.4, "Q3_K_P": 3.9, "Q3_K_M": 3.9, "Q4_K_M": 4.85,
    "Q4_K_P": 5.1, "Q5_K_M": 5.6, "Q5_K_P": 5.65, "Q6_K_P": 6.6,
    "Q6_K": 6.6, "Q8_0": 8.5, "Q8_K_P": 8.5, "BF16": 16.0,
}

# Matches the quant token at the end of a GGUF filename. Covers llama.cpp
# (Q4_K_M, IQ3_XS), the HauhauCS "P" variants (Q4_K_P), and the LM Studio
# repack suffixes bartowski publishes (Q4_0_4_4, Q4_0_8_8), in any case.
QUANT_RE = re.compile(
    r"("
    r"IQ\d+_[A-Z0-9_]+"          # IQ3_XS, IQ4_NL, IQ2_M
    r"|Q\d+(_K)?(_[A-Z0-9]+)*"   # Q4_0, Q4_K_M, Q4_K_P, Q4_0_4_4
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
    """'model' | 'mmproj' | 'mtp' | 'skip'

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
    return "model"


def quant_of(filename: str) -> str:
    """Best-effort quant label. Falls back to the last dash token."""
    base = filename[:-5] if filename.endswith(".gguf") else filename
    tok = base.rsplit("-", 1)[-1]
    if QUANT_RE.fullmatch(tok):
        return tok.upper()
    return tok.upper() if tok.isupper() or len(tok) <= 6 else "unknown"


def main() -> int:
    entries = []
    used_keys: set[str] = set()

    for repo, meta in REPOS.items():
        family, params, ctx, kv, mm, tag = meta
        try:
            files = fetch(repo)
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            print(f"warning: skipping {repo}: {exc}", file=sys.stderr)
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
            quant = name.rsplit("-", 1)[-1][:-5]
            key = f"{family.lower()}-{quant.lower().replace('_', '')}"
            base = key
            i = 2
            while key in used_keys:
                key = f"{base}-{i}"
                i += 1
            used_keys.add(key)
            entries.append({
                "key": key, "repo": repo, "filename": name, "size": size,
                "params": params, "quant": quant,
                "bpw": QUANT_BPW.get(quant, 4.5), "kv": kv,
                "family": family, "ctx": ctx, "mm": mm, "tag": tag,
                **companions,
            })

    entries.sort(key=lambda e: (e["family"], e["size"]))

    wrote_mmproj = "mmproj" in entries[0] if entries else False
    _ = wrote_mmproj

    lines = [
        '"""Static catalog of HauhauCS GGUF builds.',
        "",
        "Generated by tools/build_catalog.py from the HuggingFace API.",
        "File sizes are exact byte counts as published upstream, so the",
        "fit calculations do not have to guess.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from .models import Model",
        "",
        "#: Upstream publisher - all of these are HauhauCS fine-tunes.",
        'PUBLISHER = "HauhauCS"',
        "",
        "CATALOG: list[Model] = [",
    ]
    for e in entries:
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
    print(f"# generated {len(entries)} entries", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
