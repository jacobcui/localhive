<!-- Banner: keep the pitch under one line so it wraps cleanly in search results. -->
<div align="center">

<img src="docs/logo.svg" alt="localhive" width="112" height="112">

# localhive

**Stop guessing whether a model will fit in your RAM.**

`localhive` reads your actual CPU, RAM and GPU, compares that against the exact
byte size of every candidate model, and shows you only the ones that will
genuinely run.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Platforms](https://img.shields.io/badge/platform-WSL2%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)](#requirements)
[![Tests](https://img.shields.io/badge/tests-27%20passing-brightgreen.svg)](tests/test_fit.py)

<img src="docs/list-models.svg" alt="make list-models output showing only models that fit this machine" width="840">

</div>

---

## The problem

You find a great model. You download 12 GB. You run it. Your machine swaps to
death, or the GPU falls back to CPU and it crawls — or it just OOMs.

Every "run LLMs locally" guide assumes you already know that a Q4_K_M of a 30B
model needs roughly 18 GB resident before the KV cache. So you guess, and you
guess wrong.

`localhive` does the arithmetic for you, before you spend the bandwidth.

## Quick start

```bash
git clone https://github.com/jacobcui/localhive.git
cd localhive

make install                              # llama.cpp + venv + PATH
make list-models                          # what can THIS box actually run?
make install-model MODEL=qwen3.6-35b-a3b-iq2m
localhive serve qwen3.6-35b-a3b-iq2m      # OpenAI-compatible API on :7000
```

Then point any OpenAI client at `http://127.0.0.1:7000/v1`.

<details>
<summary><b>Requirements</b></summary>

* **Python 3.9+** — no third-party packages, the core is stdlib only
* **llama.cpp** — `make install` fetches it (Homebrew on macOS, prebuilt
  release or source build on Linux/WSL2)
* **Disk** — models land in `~/.localhive/models`; a single quant is 1–40 GB
* **OS** — WSL2 Ubuntu, native Linux, or macOS (Apple Silicon or Intel)

</details>

---

## `make list-models`

Only shows what fits. Everything else is hidden, not silently recommended.

```console
$ make list-models

  Platform : Ubuntu 26.04 LTS [WSL2] x86_64
  CPU      : Intel(R) Core(TM) i7-12700H
  Memory   : 32.0 GiB total, 29.4 GiB available
  GPU      : none detected - CPU inference only

  Budget   : 24.0 GiB usable RAM, CPU only  (planning context 8,192)

        MODEL                          SIZE  VERDICT       CTX  DETAIL
  ------------------------------------------------------------------------------------------------
  [cc]  gemma4-31b-q4km             17.40G  CPU        8,192  runs on CPU comfortably | vision+MTP
  [cc]  qwen3.6-35b-a3b-iq2m        10.86G  CPU        8,192  runs on CPU comfortably | vision+MoE
  [cc]  gemma4-12b-q4km              6.87G  CPU        8,192  runs on CPU comfortably | vision+MTP
  [cc]  qwen3.5-2b-q4km              1.18G  CPU        8,192  runs on CPU comfortably | vision

  58 of 64 catalog builds fit at 8,192 context.
  Largest hidden (too big): qwen3.6-35b-a3b-q8kp - exceeds usable RAM
```

One row per family by default — the **best** quant that fits, not the biggest.

<details>
<summary><b>Verdicts, and what they mean</b></summary>

| | Verdict | Meaning |
| --- | --- | --- |
| `[##]` | **PERFECT** | fits with real headroom (≤80% of budget) |
| `[#-]` | **GOOD** | fits, modest headroom |
| `[!!]` | **TIGHT** | fits, but will pressure memory — or needs partial GPU offload |
| `[cc]` | **CPU** | runs on CPU. Fine for MoE (few active params), slow for dense |
| `[XX]` | **TOO BIG** | hidden unless you pass `--all` |

</details>

<details>
<summary><b>Options</b></summary>

| Flag | Meaning |
| --- | --- |
| `CONTEXT=32768` | plan for a longer context — bigger KV cache, fewer models fit |
| `ARGS=--verbose` | every quant, not just the best per family |
| `ARGS=--all` | include models that do **not** fit, so you can see why |
| `ARGS=--json` | machine-readable, for scripting |

```bash
make list-models CONTEXT=32768              # "I want long context"
make list-models ARGS=--verbose             # "show me every quant"
make list-models ARGS=--all                 # "why is X missing?"
```

</details>

## `make install-model MODEL=<name>`

Takes a name straight from `list-models`.

```console
$ make install-model MODEL=qwen3.6-35b-a3b-iq2m

  Source   : HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
  Size     : 10.9 GiB on disk
  Target   : /home/jacob/.localhive/models
  Fit      : [cc] cpu-only - runs on CPU comfortably

  qwen3.6-35b-a3b-iq2m    [######################------]  78.4% 8.5 GiB/10.9 GiB 31.2 MiB/s
```

* **Refuses models that don't fit** — `ARGS=--force` to override
* **Checks free disk space** before downloading
* **Resumable-safe** — an already-complete file is detected by size and skipped
* **Pulls the vision projector** too, when the build has one (`ARGS=--no-mmproj` to skip)

---

## How "does it fit" is decided

Exact file size, not an estimate from parameter count:

```
weights = exact byte size of the .gguf
kv      = kv_bytes_per_token x context
need    = weights + kv + 1.2 GiB compute overhead
```

Against a budget that leaves the OS alive:

```
usable_ram  = total_ram  - max(25% of RAM, 2 GiB)
usable_vram = vram x 0.90 (driver) x 0.85 (per-model cap)
```

* **Discrete GPU** — fully offloaded if `need <= 0.80 x usable_vram`, otherwise
  the report tells you what fraction of layers will fit, so you know it's a
  partial offload before you commit.
* **Apple Silicon** — CPU and GPU share one physical pool, so the OS reserve is
  subtracted **once**, not twice. Getting this wrong is why most sizing advice
  is wrong on Macs.
* **CPU only** — `need <= 0.80 x usable_ram` is "comfortable", up to
  `usable_ram` is "runs", beyond that the model is hidden.

Constants live at the top of [`src/localhive/models.py`](src/localhive/models.py) — tune freely.

<details>
<summary><b>Why "the best fit per family" isn't just the biggest model</b></summary>

Ranking uses memory *pressure*, not raw size. On a 32 GiB box, a 21.8 GB build
that technically fits but sits at the edge loses to a 10.9 GB build that runs
comfortably. Recommending the tight one because it's "better" is how you end up
with a machine that swaps the moment you paste in a long prompt.

</details>

---

## WSL2 notes

WSL2 does **not** give you your Windows RAM. It gets a fraction (50% by
default, 8 GB on older builds), and `list-models` reports *that* number — so
the recommendations are correctly conservative rather than optimistic.

To raise it, create `%UserProfile%\.wslconfig` on the Windows side:

```ini
[wsl2]
memory=24GB
processors=8
swap=8GB
```

Then `wsl --shutdown` from PowerShell and reopen. Re-run `make list-models`.

> **GPU passthrough:** do *not* install an NVIDIA driver inside WSL — the
> Windows driver provides `/usr/lib/wsl/lib/libcuda.so`. `make install` detects
> a visible GPU via `nvidia-smi` and fetches the CUDA build automatically.

---

## Model catalog

> **The catalog is currently 100% uncensored models from a single publisher**
> ([HauhauCS](https://huggingface.co/HauhauCS)): 64 quantisations across 11
> families, from a 2B that runs on anything to a 35B MoE. There are no
> gated, safety-tuned, or vendor-official models in it.
>
> That is a deliberate curation choice, not a technical limit — `localhive`
> does not care what a model says. If you want a different publisher, see
> [Adding your own models](#adding-your-own-models) below; it is one line.

If you are here for the *sizing* logic rather than the model list, jump to
[How "does it fit" is decided](#how-does-it-fit-is-decided) — that part is
publisher-agnostic.

| | |
| --- | --- |
| **Model** | `Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive` |
| **Origin** | [HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive](https://huggingface.co/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive) |
| **Base** | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (Apache-2.0) |
| **File** | `Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf` |
| **Size** | 11,659,235,456 bytes (10.86 GiB) |

It's a 35B **Mixture-of-Experts** model (~3B active per token), which is why it
stays usable even on CPU. `list-models` marks these `MoE`.

> **Note:** the catalog is a curated list, not a search engine. It contains one
> publisher's builds by default. Adding another is a single dict entry — see below.

### Adding your own models

The catalog is **generated, not hand-written**:

```bash
python3 tools/build_catalog.py > src/localhive/catalog.py
```

To add a publisher, add an entry to `REPOS` at the top of
[`tools/build_catalog.py`](tools/build_catalog.py) and regenerate. Sizes and
filenames come straight from the HuggingFace API, so the fit maths never works
from a stale guess.

Prefer to skip the catalog entirely? Point `serve` at any local GGUF:

```bash
llama-server --model /path/to/anything.gguf --ctx-size 8192
```

---

## All targets

| Target | Description |
| --- | --- |
| `make install` | llama.cpp + python venv + PATH shim |
| `make check` | verify the environment end to end |
| `make list-models` | **models that fit this machine** |
| `make install-model MODEL=X` | **download model X** |
| `make serve MODEL=X` | run llama-server (OpenAI-compatible) |
| `make installed` | models already on disk |
| `make hardware` | show detected CPU / RAM / GPU |
| `make where` | print the models directory |
| `make link` / `unlink` | add / remove the `localhive` PATH shim |
| `make reinstall` | force a fresh dependency install |
| `make test` | run the test suite |
| `make clean-venv` | remove the virtualenv |

## CLI

`make` is a thin wrapper; the `localhive` CLI does the work:

```bash
localhive list-models --context 32768 --verbose
localhive list-models --json                    # for scripting
localhive install-model qwen3.6-35b-a3b-iq2m
localhive serve qwen3.6-35b-a3b-iq2m --port 7000
localhive hardware --json
localhive list-installed
```

<details>
<summary><b><code>command not found: localhive</code>?</b></summary>

The entry point lives in the project venv. `make install` runs `make link`,
which writes a shim to `~/.local/bin/localhive` that forwards to this checkout —
nothing is installed system-wide. If it's still missing, run `make link` and
open a new shell, or call `.venv/bin/localhive` directly.

Every "Next:" hint the tool prints names a command that **exists**, so they are
always safe to copy-paste.

</details>

## Project layout

```
src/localhive/
  hardware.py    CPU / RAM / GPU / WSL detection, stdlib only
  models.py      the fit maths and verdict rules
  catalog.py     generated model list (exact byte sizes)
  cli.py         list-models, install-model, serve
scripts/
  install_llama.sh          picks the right llama.cpp build per platform
  build_llama_from_source.sh  fallback when no prebuilt release matches
tools/
  build_catalog.py          regenerates catalog.py from the HF API
docs/
  logo.svg                  the mark (see docs/logo.md)
  list-models.svg           the hero screenshot
```

## License

[MIT](LICENSE) — the code. Model weights carry their own licenses; check the
upstream HuggingFace repo before redistributing them.
