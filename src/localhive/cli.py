"""localhive - list, install and serve local GGUF models.

The point of this tool is that ``list-models`` only ever shows models that
will actually run on the machine you are sitting in front of. It reads
your real RAM/CPU/GPU, compares that against the exact byte size of every
candidate GGUF, and hides the ones that would swap themselves to death.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import catalog, hardware, models
from .models import Model

DEFAULT_MODELS_DIR = Path.home() / ".localhive" / "models"

_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def c(text: str, code: str) -> str:
    if not _COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def bold(t: str) -> str:
    return c(t, "1")


def dim(t: str) -> str:
    return c(t, "2")


def green(t: str) -> str:
    return c(t, "32")


def yellow(t: str) -> str:
    return c(t, "33")


def red(t: str) -> str:
    return c(t, "31")


def cyan(t: str) -> str:
    return c(t, "36")


VERDICT_LABEL = {
    "perfect": ("PERFECT", green),
    "good": ("GOOD", green),
    "tight": ("TIGHT", yellow),
    "cpu-only": ("CPU", cyan),
    "too-big": ("TOO BIG", red),
}


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024:
            return f"{n:,.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} PiB"


def self_command(subcommand: str) -> str:
    """A command line the user can actually paste.

    If 'localhive' is on PATH we print the bare name; otherwise we print an
    absolute path to this very script so the hint is never a lie.
    """
    on_path = shutil.which("localhive")
    if on_path:
        return f"localhive {subcommand}"
    argv0 = Path(sys.argv[0]).resolve()
    if argv0.name.startswith("localhive") or argv0.name == "__main__.py":
        if argv0.exists() and os.access(argv0, os.X_OK):
            return f"{argv0} {subcommand}"
        return f"{sys.executable} -m localhive.cli {subcommand}"
    return f"localhive {subcommand}"


def resolve(name: str) -> Model:
    """Map whatever the user typed onto a catalog entry."""
    want = name.strip()
    aliases = catalog.aliases()

    if want in aliases:
        return catalog.BY_KEY[aliases[want]]

    low = want.lower()
    for m in catalog.CATALOG:
        if low in (m.filename.lower(), m.filename[:-5].lower(), m.key.lower()):
            return m

    hits = [
        m for m in catalog.CATALOG
        if low in m.key.lower()
        or low in m.filename.lower()
        or low in m.family.lower()
        or low in m.quant.lower()
    ]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise SystemExit(
            f"error: no model matches {name!r}.\n"
            f"       run '{self_command('list-models')}' to see what is available."
        )

    hits.sort(key=lambda m: m.size_bytes)
    print(yellow(f"note: {name!r} matched {len(hits)} builds; using the smallest."),
          file=sys.stderr)
    print(dim("      candidates: " + ", ".join(m.key for m in hits[:8])), file=sys.stderr)
    return hits[0]


# ---------------------------------------------------------------------------
# list-models
# ---------------------------------------------------------------------------

def cmd_list_models(args: argparse.Namespace) -> int:
    hw = hardware.detect()
    context = args.context

    if args.json:
        scored = [(m, models.evaluate(m, hw, context)) for m in catalog.CATALOG]
        fits = [(m, f) for m, f in scored if f.runnable]
        payload = {
            "context": context,
            "hardware": hardware.to_dict(hw),
            "models": [
                {
                    "key": m.key,
                    "family": m.family,
                    "quant": m.quant,
                    "filename": m.filename,
                    "repo": m.repo,
                    "size_bytes": m.size_bytes,
                    "size_gib": round(m.size_gib, 2),
                    "multimodal": m.multimodal,
                    "mixture_of_experts": m.notes == "moe",
                    "fit": {
                        "verdict": f.verdict,
                        "runnable": f.runnable,
                        "reason": f.reason,
                        "required_bytes": f.required_bytes,
                        "usable_bytes": f.usable_bytes,
                        "gpu_layers": f.gpu_layers,
                        "offload_ratio": round(f.offload_ratio, 3),
                    },
                }
                for m, f in sorted(
                    scored, key=lambda p: (models.VERDICT_ORDER[p[1].verdict], -p[0].size_bytes)
                )
            ],
            "summary": {"total": len(scored), "runnable": len(fits)},
        }
        print(json.dumps(payload, indent=2))
        return 0

    print(bold("localhive - models that fit this machine"))
    print()
    print(hardware.summary(hw))
    print()

    usable_ram, usable_vram = models.usable_memory(hw)
    print(
        dim(
            f"  Budget   : {human(usable_ram)} usable RAM"
            + (f", {human(usable_vram)} for GPU offload" if usable_vram else ", CPU only")
            + f"  (planning context {context:,})"
        )
    )
    print()

    scored = [(m, models.evaluate(m, hw, context)) for m in catalog.CATALOG]
    fits = [(m, f) for m, f in scored if f.runnable]
    hidden = [(m, f) for m, f in scored if not f.runnable]

    shown = list(scored) if args.all else list(fits)

    if not args.all and not args.verbose:
        by_family: dict[str, tuple[Model, models.Fit]] = {}
        for m, f in fits:
            cur = by_family.get(m.family)
            if cur is None or models.rank(m, hw, context) < models.rank(cur[0], hw, context):
                by_family[m.family] = (m, f)
        shown = list(by_family.values())

    shown.sort(key=lambda p: (models.VERDICT_ORDER[p[1].verdict], -p[0].size_bytes))

    if not shown:
        print(red("  No models fit this machine at the requested context."))
        print(dim("  Try a smaller --context, or free up memory."))
        return 1

    print(bold(f"  {'':<5} {'MODEL':<26} {'SIZE':>8}  {'VERDICT':<8} {'CTX':>8}  DETAIL"))
    print(dim("  " + "-" * 96))

    for m, f in shown:
        label, colour = VERDICT_LABEL[f.verdict]
        caps = []
        if m.multimodal:
            caps.append("vision")
        if m.notes == "moe":
            caps.append("MoE")
        if m.mtp:
            caps.append("MTP")
        detail = f.reason
        if caps:
            detail += " | " + "+".join(caps)
        print(
            f"  {f.icon:<5} {m.key:<26} {m.size_gib:>6.2f}G  "
            f"{colour(label):<8} {f.context:>8,}  {dim(detail)}"
        )

    print()
    print(f"  {green(str(len(fits)))} of {len(catalog.CATALOG)} catalog builds fit"
          f" at {context:,} context.")
    if not args.all and not args.verbose and len(fits) > len(shown):
        print(dim(f"  Showing the best fit per family ({len(shown)} rows)."))
        print(dim("  Use --verbose for every quant, --all to include models that do not fit."))
    if hidden and not args.all:
        biggest = max(hidden, key=lambda p: p[0].size_bytes)
        print(dim(f"  Largest hidden (too big): {biggest[0].key} - {biggest[1].reason}"))
    print()
    print(dim("  Next: ") + cyan(self_command("install-model <model>")))
    return 0


# ---------------------------------------------------------------------------
# install-model
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path, label: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    req = urllib.request.Request(url, headers={"User-Agent": "localhive/0.1"})
    try:
        with urllib.request.urlopen(req) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            start = time.time()
            last = 0.0
            with open(tmp, "wb") as fh:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    now = time.time()
                    if now - last > 0.5 or done == total:
                        last = now
                        speed = done / max(now - start, 1e-6)
                        if total:
                            pct = done / total * 100
                            filled = int(28 * done / total)
                            bar = "#" * filled + "-" * (28 - filled)
                            sys.stderr.write(
                                f"\r  {label:<28} [{bar}] {pct:5.1f}% "
                                f"{human(done)}/{human(total)} {human(speed)}/s   "
                            )
                        else:
                            sys.stderr.write(
                                f"\r  {label:<28} {human(done)} {human(speed)}/s   "
                            )
                        sys.stderr.flush()
        sys.stderr.write("\n")
    except urllib.error.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"error: download failed for {url}\n"
            f"       HTTP {exc.code} {exc.reason}\n"
            f"       Check the repo is public and the filename still exists."
        ) from exc
    except urllib.error.URLError as exc:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"error: cannot reach huggingface.co: {exc.reason}") from exc
    except KeyboardInterrupt:
        tmp.unlink(missing_ok=True)
        print("\ninterrupted - partial download removed", file=sys.stderr)
        raise SystemExit(130)

    tmp.replace(dest)


def cmd_install_model(args: argparse.Namespace) -> int:
    model = resolve(args.model)
    hw = hardware.detect()
    fit = models.evaluate(model, hw, args.context)
    dest_dir = Path(args.dir).expanduser()

    print(bold(f"Installing {model.key}"))
    print()
    print(f"  Source   : {cyan(model.repo)}")
    print(f"  File     : {model.filename}")
    print(f"  Size     : {human(model.size_bytes)} on disk")
    print(f"  Target   : {dest_dir}")
    print(f"  Fit      : {fit.icon} {fit.verdict} - {fit.reason}")
    print()

    if not fit.runnable and not args.force:
        print(red("  This model does not fit this machine."))
        print(dim(f"  {fit.reason}. Re-run with --force if you really want it."))
        return 1

    dest_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest_dir).free
    if free < model.size_bytes * 1.05:
        print(red(f"  Not enough disk: need {human(model.size_bytes)}, have {human(free)} free."))
        return 1

    dest = dest_dir / model.filename
    if dest.exists() and not args.force:
        if dest.stat().st_size == model.size_bytes:
            print(green(f"  Already installed: {dest}"))
        else:
            print(yellow(f"  Exists but size differs - re-downloading: {dest}"))
            _download(model.url, dest, model.key)
    else:
        print(f"  Downloading {model.url}")
        _download(model.url, dest, model.key)

    if model.mmproj and not args.no_mmproj:
        mm_dest = dest_dir / model.mmproj
        mm_url = model.url.rsplit("/", 1)[0] + "/" + model.mmproj
        if mm_dest.exists() and not args.force:
            print(green(f"  Already installed: {mm_dest.name}"))
        else:
            print(dim(f"  Fetching vision projector {model.mmproj}"))
            _download(mm_url, mm_dest, mm_dest.name)

    meta_path = dest_dir / (model.filename + ".localhive.json")
    meta_path.write_text(
        json.dumps(
            {
                "key": model.key,
                "repo": model.repo,
                "filename": model.filename,
                "size_bytes": model.size_bytes,
                "quant": model.quant,
                "family": model.family,
                "mmproj": model.mmproj,
                "mtp": model.mtp,
                "suggested_context": models.suggested_context(model, hw),
                "installed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
            indent=2,
        )
        + "\n"
    )

    print()
    print(green("  Installed:"))
    print(f"    {dest}")
    if model.mmproj and not args.no_mmproj:
        print(f"    {dest_dir / model.mmproj}")
    print()
    print(dim("  Next: ") + cyan(self_command(f"serve {model.key}")))
    return 0


# ---------------------------------------------------------------------------
# list-installed / serve / where / hardware
# ---------------------------------------------------------------------------

def _installed(dest_dir: Path) -> list[Path]:
    if not dest_dir.is_dir():
        return []
    return sorted(p for p in dest_dir.glob("*.gguf") if not p.name.startswith("mmproj"))


def cmd_list_installed(args: argparse.Namespace) -> int:
    dest_dir = Path(args.dir).expanduser()
    files = _installed(dest_dir)
    print(bold(f"Installed models in {dest_dir}"))
    print()
    if not files:
        print(dim("  (none)"))
        print()
        print(dim("  Next: ") + cyan(self_command("list-models")))
        return 0
    total = 0
    for p in files:
        size = p.stat().st_size
        total += size
        meta = p.with_suffix(p.suffix + ".localhive.json")
        key = ""
        if meta.exists():
            try:
                key = json.loads(meta.read_text()).get("key", "")
            except (OSError, json.JSONDecodeError):
                key = ""
        print(f"  {human(size):>12}  {p.name}" + (dim(f"  ({key})") if key else ""))
    print()
    print(dim(f"  {len(files)} file(s), {human(total)} total"))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    dest_dir = Path(args.dir).expanduser()
    model = resolve(args.model)
    path = dest_dir / model.filename
    if not path.exists():
        raise SystemExit(
            f"error: {path} not found.\n"
            f"       install it first: {self_command(f'install-model {model.key}')}"
        )

    server = shutil.which("llama-server")
    if server is None:
        raise SystemExit(
            "error: llama-server not found in PATH.\n"
            "       run 'make install' in the localhive checkout."
        )

    hw = hardware.detect()
    ctx = args.context or models.suggested_context(model, hw)
    fit = models.evaluate(model, hw, ctx)

    mmproj = None
    if model.mmproj and (dest_dir / model.mmproj).exists():
        mmproj = dest_dir / model.mmproj

    cmd = [
        server,
        "--model", str(path),
        "--ctx-size", str(ctx),
        "--host", args.host,
        "--port", str(args.port),
    ]
    if fit.gpu_layers == -1:
        cmd += ["--n-gpu-layers", "999"]
    elif fit.gpu_layers > 0:
        cmd += ["--n-gpu-layers", str(fit.gpu_layers)]
    if mmproj:
        cmd += ["--mmproj", str(mmproj)]

    print(bold(f"Serving {model.key}"))
    print(f"  Fit   : {fit.icon} {fit.verdict} - {fit.reason}")
    print(f"  Ctx   : {ctx:,}")
    print(f"  URL   : {cyan(f'http://{args.host}:{args.port}')}")
    print()
    print(dim("  " + " ".join(cmd)))
    print()
    sys.stdout.flush()
    os.execv(server, cmd)
    return 0  # unreachable


def cmd_where(args: argparse.Namespace) -> int:
    print(Path(args.dir).expanduser())
    return 0


def cmd_hardware(args: argparse.Namespace) -> int:
    hw = hardware.detect()
    if args.json:
        print(json.dumps(hardware.to_dict(hw), indent=2))
        return 0
    print(bold("Detected hardware"))
    print(hardware.summary(hw))
    usable_ram, usable_vram = models.usable_memory(hw)
    print()
    print(f"  Usable RAM : {human(usable_ram)}")
    print(f"  GPU budget : {human(usable_vram) if usable_vram else 'none (CPU inference)'}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="localhive",
        description="List, install and serve local GGUF models sized to your machine.",
    )
    p.add_argument("--dir", default=str(DEFAULT_MODELS_DIR),
                   help=f"models directory (default: {DEFAULT_MODELS_DIR})")
    sub = p.add_subparsers(dest="command", required=True)

    lm = sub.add_parser("list-models", aliases=["list"],
                        help="show models that fit this machine")
    lm.add_argument("--context", "-c", type=int, default=models.DEFAULT_CONTEXT,
                    help=f"context length to plan for (default: {models.DEFAULT_CONTEXT})")
    lm.add_argument("--all", action="store_true", help="also show models that do NOT fit")
    lm.add_argument("--verbose", "-v", action="store_true",
                    help="show every quant, not just the best per family")
    lm.add_argument("--json", action="store_true",
                    help="machine-readable output for scripting")
    lm.set_defaults(func=cmd_list_models)

    im = sub.add_parser("install-model", aliases=["install"],
                        help="download a model into the models directory")
    im.add_argument("model", help="model key from list-models, or a filename")
    im.add_argument("--context", "-c", type=int, default=models.DEFAULT_CONTEXT)
    im.add_argument("--force", action="store_true", help="re-download / ignore fit check")
    im.add_argument("--no-mmproj", dest="no_mmproj", action="store_true",
                    help="skip the vision projector")
    im.set_defaults(func=cmd_install_model)

    li = sub.add_parser("list-installed", aliases=["installed"],
                        help="show models already on disk")
    li.set_defaults(func=cmd_list_installed)

    sv = sub.add_parser("serve", help="run llama-server for an installed model")
    sv.add_argument("model", help="model key")
    sv.add_argument("--port", type=int, default=7000)
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--context", "-c", type=int, default=0,
                    help="context length (default: auto-fit)")
    sv.set_defaults(func=cmd_serve)

    wh = sub.add_parser("where", help="print the models directory")
    wh.set_defaults(func=cmd_where)

    hw = sub.add_parser("hardware", help="show detected hardware")
    hw.add_argument("--json", action="store_true")
    hw.set_defaults(func=cmd_hardware)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
