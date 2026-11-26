# localhive

Run local GGUF models, sized to the machine you are actually sitting in
front of.

The core idea: `make list-models` reads your real CPU, RAM and GPU, compares
that against the **exact byte size** of every candidate model, and shows you
only the ones that will genuinely run. No guessing, no "it loads but swaps
itself to death".

Works on **WSL2 Ubuntu**, native Linux and macOS.

---

## Quick start

```bash
git clone git@github.com:jacobcui/localhive.git
cd localhive

make install                         # llama.cpp + python venv
make list-models                     # what can THIS box run?
make install-model MODEL=qwen3.6-35b-a3b-iq2m
make serve MODEL=qwen3.6-35b-a3b-iq2m
```

## The two targets you asked for

### `make list-models`

```console
$ make list-models

localhive - models that fit this machine

  Platform : Ubuntu 26.04 LTS [WSL2] x86_64
  CPU      : Intel(R) Core(TM) i7-12700H
  Cores    : 14 physical / 20 logical
  Memory   : 32.0 GiB total, 29.4 GiB available
  GPU      : none detected - CPU inference only
  Note     : WSL2 memory is capped by .wslconfig; see README for raising it

  Budget   : 24.0 GiB usable RAM, CPU only  (planning context 8,192)

        MODEL                          SIZE  VERDICT       CTX  DETAIL
  ------------------------------------------------------------------------------------------------
  [cc]  gemma4-31b-q4km             17.40G  CPU        8,192  runs on CPU comfortably | vision+MTP
  [cc]  gemma4-26b-a4b-q4km         15.64G  CPU        8,192  runs on CPU comfortably | vision+MoE+MTP
  [cc]  qwen3.6-35b-a3b-iq2m        10.86G  CPU        8,192  runs on CPU comfortably | vision+MoE
  ...

  58 of 64 catalog builds fit at 8,192 context.
```

Options:

| Flag | Meaning |
| --- | --- |
| `CONTEXT=32768` | plan for a longer context (bigger KV cache, fewer models fit) |
| `ARGS=--verbose` | show every quant, not just the best fit per family |
| `ARGS=--all` | also list models that do **not** fit, so you can see why |

Verdicts:

| | Verdict | Meaning |
| --- | --- | --- |
| `[##]` | **PERFECT** | fits with real headroom |
| `[#-]` | **GOOD** | fits, modest headroom |
| `[!!]` | **TIGHT** | fits but will pressure memory, or needs partial GPU offload |
| `[cc]` | **CPU** | runs on CPU (fine for MoE models, slow for dense ones) |
| `[XX]` | **TOO BIG** | hidden unless you pass `--all` |

### `make install-model MODEL=<name>`

Takes a name straight from `list-models` and installs it to
`~/.localhive/models`.

```console
$ make install-model MODEL=qwen3.6-35b-a3b-iq2m

Installing qwen3.6-35b-a3b-iq2m

  Source   : HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
  File     : Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf
  Size     : 10.9 GiB on disk
  Target   : /home/jacob/.localhive/models
  Fit      : [cc] cpu-only - runs on CPU comfortably

  qwen3.6-35b-a3b-iq2m    [######################------]  78.4% 8.5 GiB/10.9 GiB 31.2 MiB/s
```

It refuses to install something that does not fit (override with
`ARGS=--force`), checks free disk space first, and is safe to re-run —
an already-complete file is detected by size and skipped.

---

## How "does it fit" is decided

For each model we compute the resident working set:

```
weights = exact size of the .gguf file
kv      = kv_bytes_per_token x context
need    = weights + kv + compute_overhead   # 1.2 GiB for llama.cpp buffers
```

Then we compare against a budget that keeps the OS alive:

```
usable_ram = total_ram - max(25% of RAM, 2 GiB)
usable_vram = vram x 0.90 (driver) x 0.85 (per-model cap)
```

* **dedicated GPU** — fully on GPU if `need <= 0.80 x usable_vram`,
  otherwise we report the fraction of layers that will fit with partial offload.
* **Apple Silicon** — CPU and GPU share one physical pool, so the OS reserve
  is subtracted **once**, not twice.
* **CPU only** — `need <= 0.80 x usable_ram` is "comfortable", up to
  `usable_ram` is "runs", beyond that the model is hidden.

The constants live at the top of `src/localhive/models.py` and are easy to tune.

---

## WSL2 notes

WSL2 gives the distro a **fraction of the Windows machine's RAM** (default:
50%, or 8 GB on older builds). That is the number `list-models` reports, so
the recommendations are correctly conservative.

To give WSL2 more memory, create `%UserProfile%\.wslconfig` on the Windows side:

```ini
[wsl2]
memory=24GB
processors=8
swap=8GB
```

Then `wsl --shutdown` from PowerShell and reopen. Re-run `make list-models`
to see the bigger set of models that now fit.

**GPU:** do *not* install an NVIDIA driver inside WSL — the Windows driver
provides `/usr/lib/wsl/lib/libcuda.so`. `scripts/install_llama.sh` detects a
visible GPU via `nvidia-smi` and installs the CUDA build automatically.

---

## All targets

| Target | Description |
| --- | --- |
| `make install` | llama.cpp + python venv + localhive |
| `make check` | verify the environment end to end |
| `make list-models` | models that fit this machine |
| `make install-model MODEL=X` | download model X |
| `make installed` | models already on disk |
| `make serve MODEL=X` | run llama-server (OpenAI-compatible API) |
| `make hardware` | show detected CPU / RAM / GPU |
| `make where` | print the models directory |
| `make test` | run the test suite |
| `make clean-venv` | remove the virtualenv |

## CLI

`make` is a thin wrapper; the `localhive` CLI does the work and can be used
directly once the venv exists:

```bash
.venv/bin/localhive list-models --context 32768 --verbose
.venv/bin/localhive install-model qwen3.6-35b-a3b-iq2m
.venv/bin/localhive serve qwen3.6-35b-a3b-iq2m --port 7000
.venv/bin/localhive hardware --json
```

## Model source

All catalogued models are GGUF quantisations published by
[HauhauCS](https://huggingface.co/HauhauCS). The model this repo was built
around is:

| | |
| --- | --- |
| **Model** | `Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive` |
| **Origin** | https://huggingface.co/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive |
| **Base** | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (Apache-2.0) |
| **File** | `Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf` |
| **Size** | 11,659,235,456 bytes (10.86 GiB) |

It is a 35B-parameter **Mixture-of-Experts** model (A3B = ~3B active per
token), which is why it stays fast even when it does not fit entirely in VRAM.
`list-models` flags these as `MoE`.

Roughly 64 quantisations across 11 model families are catalogued. Sizes are
exact byte counts taken from the HuggingFace API, so the fit maths does not
have to guess.

The catalog is generated, not hand-written:

```bash
python3 tools/build_catalog.py > src/localhive/catalog.py
```

## License

MIT
