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
[![Tests](https://img.shields.io/badge/tests-63%20passing-brightgreen.svg)](tests/test_fit.py)

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

> **WSL2 / Debian note:** `python3 -m venv` can create a venv **without pip**
> when `python3-venv` is not installed. `make` detects this, recreates the
> venv and bootstraps pip automatically. If pip still cannot be bootstrapped:
>
> ```bash
> sudo apt update && sudo apt install -y python3-venv python3-pip
> rm -rf .venv && make
> ```
>
> `make check` diagnoses the environment, and `make doctor-deps` repairs a
> broken venv in place.

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

  Showing  : standard models (64 hidden by content filter)

        MODEL                          SIZE  VERDICT       CTX  DETAIL
  ------------------------------------------------------------------------------------------------
  [cc]  gemma-2-9b-iq2xs             2.86G  CPU         8,192  runs on CPU comfortably
  [cc]  llama-3.1-8b-iq2m            2.75G  CPU         8,192  runs on CPU comfortably
  [cc]  qwen2.5-coder-7b-iq2m        2.59G  CPU         8,192  runs on CPU comfortably
  [cc]  qwen2.5-7b-iq2m              2.59G  CPU         8,192  runs on CPU comfortably
  [cc]  qwen3-8b-iq1s                2.12G  CPU         8,192  runs on CPU comfortably
  [cc]  mistral-7b-iq1s              1.50G  CPU         8,192  runs on CPU comfortably
  [cc]  llama-3.2-3b-iq3m            1.49G  CPU         8,192  runs on CPU comfortably

  164 of 231 catalog builds fit at 8,192 context.
  Showing the best fit per family (7 rows).
```

One row per family by default — the **best** quant that fits, not the biggest.
The header always states which content filter is in effect and how many
models it hid, so you are never left wondering where something went.

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
| `uncensored=yes` | show uncensored builds instead of standard models |
| `all=yes` | show both standard and uncensored together |

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

**Standard models by default. Uncensored builds are opt-in.**

`make list-models` shows normal, safety-tuned and vendor models — Llama,
Qwen, Mistral, Gemma, Qwen-Coder — across 7 families and 4 publishers. If
you just want "a model that runs", that is what you get with no flags.

Uncensored community fine-tunes are one flag away and never appear by
accident:

```bash
make list-models                     # standard models (the default)
make list-models uncensored=yes      # uncensored builds only
make list-models all=yes             # both together

make install-model MODEL=qwen3.6-35b-a3b-iq2m uncensored=yes
```

Every row is labelled. A `CONTENT` column appears whenever a listing mixes
both types, and uncensored rows carry a tag in `DETAIL` otherwise — so a
model stays identifiable after you copy its name out of the terminal:

```console
$ make list-models all=yes
        MODEL                          SIZE  CONTENT     VERDICT       CTX  DETAIL
  ----------------------------------------------------------------------------------------------------
  [##]  qwen2.5-7b-iq2m              2.59G  standard    PERFECT     8,192  comfortable in unified memory
  [##]  llama-3.2-3b-iq3m            1.49G  standard    PERFECT     8,192  comfortable in unified memory
  [##]  qwen3.6-35b-a3b-iq2m        10.86G  uncensored  PERFECT     8,192  comfortable in unified memory | MoE
```

Standard models sort first, since they are the default. `make installed`
labels whatever is already on your disk, and `install-model` prints a
`Content: uncensored` line for anything uncensored.

If you try to install an uncensored model without the flag, it is not
silently allowed — you get told exactly what to do:

```console
$ make install-model MODEL=qwen3.6-35b-a3b-iq2m
error: no model matches 'qwen3.6-35b-a3b-iq2m'.
       'qwen3.6-35b-a3b-iq2m' is an uncensored model. Re-run with --uncensored to use it.
```

<details>
<summary><b>How content is classified</b></summary>

Every catalog entry declares its type in a `content` field — it is **not**
sniffed from the filename. Filename heuristics break the moment a publisher
names something unusually, and silently mislabelling a model is worse than
not labelling it at all.

A test asserts that any model whose filename says "uncensored" is labelled
as such, so the two cannot drift apart.

</details>

<details>
<summary><b>What's in each view</b></summary>

| View | Families | Examples |
| --- | --- | --- |
| Standard (default) | 7 | Llama-3.2-3B, Llama-3.1-8B, Qwen2.5-7B, Qwen2.5-Coder-7B, Qwen3-8B, Mistral-7B, Gemma-2-9B |
| Uncensored (`uncensored=yes`) | 11 | Qwen3.6-35B-A3B, Qwen3.8-27B, Gemma4-31B, … |

Roughly 231 quants in total, and every one carries an exact byte size read
from the HuggingFace API.

</details>

| | |
| --- | --- |
| **Model** | `Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive` |
| **Origin** | [HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive](https://huggingface.co/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive) |
| **Base** | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (Apache-2.0) |
| **File** | `Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf` |
| **Size** | 11,659,235,456 bytes (10.86 GiB) |

It's a 35B **Mixture-of-Experts** model (~3B active per token), which is why it
stays usable even on CPU. `list-models` marks these `MoE`.

> **Note:** the catalog is a curated list, not a search engine. It covers
> well-known publishers (bartowski, unsloth, and HauhauCS), not all of
> HuggingFace. Adding another is a single entry — see below.

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
| `make ps` | list running servers |
| `make stop PORT=7000` | stop one server |
| `make stop all=yes` | stop every server |
| `make restart MODEL=X` | stop whatever holds the port, then serve |
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

## Serving, and stopping what you started

`serve` replaces itself with `llama-server`, so the server keeps running
after the command returns. Forgetting about one and then hitting a cryptic
bind error on the next run is easy:

```console
E srv start: couldn't bind HTTP server socket, hostname: 127.0.0.1, port: 7000
```

So there is a way to see and stop them:

```bash
make ps                  # what is running?
make stop PORT=7000      # stop one
make stop all=yes        # stop everything
```

```console
$ make ps
Running localhive servers

      PID   PORT  MODEL                                  CTX  STATUS
  ----------------------------------------------------------------------------------
    10954   7000  llama-3.2-3b-iq3m                   32,768  ready
    11016   7001  qwen2.5-7b-iq2m                     32,768  ready
```

You should not need either command to diagnose a failure, because `serve`
now checks the port **before** starting and tells you exactly what to do:

```console
$ make serve MODEL=qwen2.5-7b-iq2m PORT=7000
  Port 7000 is already in use.
    pid 10954  Llama-3.2-3B-Instruct-IQ3_M.gguf
    http://127.0.0.1:7000

  Either stop it:
    localhive stop 7000
  or serve this model on a different port:
    localhive serve qwen2.5-7b-iq2m --port 7001
```

<details>
<summary><b>Servers started outside localhive</b></summary>

`ps` only lists servers localhive started, because that is the only set it
can describe accurately. To find a `llama-server` you launched by hand, or
one started before this feature existed:

```bash
make ps all=yes               # probe ports 7000-7099
localhive ps --all --scan 8000-8100
```

Discovery works by probing `/health` and reading `/v1/models`, so it will
not mistake an unrelated service for a model server. Stopping is safe
regardless: the pid is verified to actually be a `llama-server` before
anything is signalled, so a recycled pid is never killed by mistake.

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
