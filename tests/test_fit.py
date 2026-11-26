"""Tests for the fit calculations and the CLI surface.

These are the rules that decide what the user is allowed to see in
'list-models', so they are worth pinning down.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from localhive import catalog, models  # noqa: E402
from localhive.hardware import GIB, Gpu, Hardware  # noqa: E402


def make_hw(ram_gib, vram_gib=0, wsl=True, arch="x86_64", os_name="Linux"):
    gpus = []
    if vram_gib:
        gpus = [Gpu(name="Test GPU", vram_bytes=int(vram_gib * GIB), backend="cuda")]
    return Hardware(
        os_name=os_name,
        distro="Ubuntu 26.04 LTS",
        kernel="6.6.0-microsoft",
        arch=arch,
        is_wsl=wsl,
        cpu="Test CPU",
        physical_cores=8,
        logical_cores=16,
        total_ram_bytes=int(ram_gib * GIB),
        available_ram_bytes=int(ram_gib * GIB * 0.9),
        swap_bytes=2 * GIB,
        gpus=gpus,
    )


# --- catalog sanity --------------------------------------------------------

def test_catalog_is_not_empty():
    assert len(catalog.CATALOG) > 20


def test_catalog_keys_are_unique():
    keys = [m.key for m in catalog.CATALOG]
    assert len(keys) == len(set(keys))


def test_catalog_has_no_projector_or_draft_files():
    """mmproj / mtp companions must never be listed as installable models."""
    for m in catalog.CATALOG:
        assert not m.filename.startswith("mmproj"), m.filename
        assert not m.filename.startswith("mtp"), m.filename
        assert "mmproj" not in m.filename.lower()
        assert m.size_bytes > 1024 ** 3, f"{m.key} looks like a companion file"


def test_every_model_has_a_url():
    for m in catalog.CATALOG:
        assert m.url.startswith("https://huggingface.co/")
        assert m.url.endswith(m.filename)


def test_known_model_is_present():
    """The model the user already downloaded must be in the catalog."""
    match = [m for m in catalog.CATALOG
             if m.filename == "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf"]
    assert len(match) == 1
    assert match[0].size_bytes == 11659235456


# --- fit logic -------------------------------------------------------------

def test_small_machine_hides_big_models():
    hw = make_hw(8, 0)
    visible = [m for m in catalog.CATALOG if models.evaluate(m, hw).runnable]
    hidden = [m for m in catalog.CATALOG if not models.evaluate(m, hw).runnable]
    assert hidden, "an 8 GiB machine must not be shown 20 GiB models"
    assert all(m.size_gib < 8 for m in visible)


def test_32gb_wsl_fits_the_35b_iq2m():
    """32 GiB WSL2 is the user's laptop spec; the IQ2_M build must fit."""
    hw = make_hw(32, 0)
    m = catalog.BY_KEY["qwen3.6-35b-a3b-iq2m"]
    fit = models.evaluate(m, hw)
    assert fit.runnable


def test_context_growth_can_push_a_model_out():
    hw = make_hw(32, 0)
    m = catalog.BY_KEY["qwen3.6-35b-a3b-iq2m"]
    assert models.evaluate(m, hw, 8192).runnable
    # A huge context should eventually stop fitting.
    assert not models.evaluate(m, hw, 262144).runnable


def test_unified_memory_uses_one_pool():
    hw = make_hw(64, 0, wsl=False, arch="arm64", os_name="Darwin")
    assert hw.unified_memory
    usable_ram, usable_vram = models.usable_memory(hw)
    assert usable_vram > 0
    assert usable_vram <= usable_ram


def test_best_per_family_prefers_comfortable_over_big():
    """We must never push a 'tight' 22G build ahead of a 'perfect' 5G one."""
    hw = make_hw(32, 0)
    scored = {m.key: models.evaluate(m, hw) for m in catalog.CATALOG}

    def best_in(family):
        fam = [m for m in catalog.CATALOG if m.family == family]
        fam = [m for m in fam if scored[m.key].runnable]
        fam.sort(key=lambda m: models.rank(m, hw))
        return fam[0]

    pick = best_in("Qwen3.6-35B-A3B")
    assert pick.size_gib < 20, f"picked {pick.key} ({pick.size_gib:.1f}G) on a 32 GiB box"


def test_too_big_models_are_flagged():
    hw = make_hw(8, 0)
    big = catalog.BY_KEY["qwen3.6-35b-a3b-q8kp"]
    fit = models.evaluate(big, hw)
    assert fit.verdict == "too-big"
    assert not fit.runnable


def test_gpu_offload_reported():
    hw = make_hw(64, 24)
    m = catalog.BY_KEY["qwen3.5-9b-q4km"]
    fit = models.evaluate(m, hw)
    assert fit.gpu_layers == -1 or fit.gpu_layers > 0
    assert fit.offload_ratio > 0


# --- CLI -------------------------------------------------------------------

def test_resolve_by_key():
    from localhive import cli
    assert cli.resolve("qwen3.6-35b-a3b-iq2m").key == "qwen3.6-35b-a3b-iq2m"


def test_resolve_by_filename():
    from localhive import cli
    m = cli.resolve("Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf")
    assert m.key == "qwen3.6-35b-a3b-iq2m"


def test_resolve_unknown_exits():
    from localhive import cli
    with pytest.raises(SystemExit):
        cli.resolve("definitely-not-a-model-xyz")


def test_list_models_runs(capsys, monkeypatch):
    """Smoke-test the real command against the real machine."""
    from localhive import cli
    rc = cli.main(["list-models"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "localhive" in out


def test_list_models_all_flag(capsys):
    from localhive import cli
    rc = cli.main(["list-models", "--all"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "TOO BIG" in out or "PERFECT" in out


def test_hardware_json(capsys):
    from localhive import cli
    rc = cli.main(["hardware", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    import json
    data = json.loads(out)
    assert data["total_ram_bytes"] > 0
    assert "gpus" in data

def test_self_command_is_runnable():
    """The 'Next:' hint must name something that exists, not a bare name."""
    import shutil as _shutil
    from localhive import cli
    cmd = cli.self_command("list-models")
    prog = cmd.split()[0]
    assert _shutil.which(prog) or os.path.exists(prog), (
        f"hint points at a command that does not exist: {cmd!r}"
    )


def test_next_hint_does_not_lie(capsys):
    """Whatever list-models prints as 'Next:' must be an existing program."""
    import shutil as _shutil
    from localhive import cli
    cli.main(["list-models"])
    out = capsys.readouterr().out
    line = [l for l in out.splitlines() if "Next:" in l]
    assert line, "list-models should print a next step"
    # strip ANSI colour codes if present
    import re
    clean = re.sub(r"\x1b\[[0-9;]*m", "", line[0])
    cmd = clean.split("Next:")[1].strip().split()[0]
    assert _shutil.which(cmd) or os.path.exists(cmd), f"bad hint: {cmd!r}"

def test_list_models_json_is_valid(capsys):
    """--json is documented in the README, so it has to actually work."""
    import json as _json
    from localhive import cli, catalog
    rc = cli.main(["list-models", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    data = _json.loads(out)
    assert data["summary"]["total"] == len(catalog.CATALOG)
    assert len(data["models"]) == len(catalog.CATALOG)
    assert data["hardware"]["total_ram_bytes"] > 0
    for m in data["models"]:
        assert set(m["fit"]) >= {"verdict", "runnable", "reason"}


def test_json_runnable_count_matches_verdicts(capsys):
    import json as _json
    from localhive import cli
    cli.main(["list-models", "--json"])
    data = _json.loads(capsys.readouterr().out)
    flagged = sum(1 for m in data["models"] if m["fit"]["runnable"])
    assert flagged == data["summary"]["runnable"]


def test_readme_documents_only_real_flags():
    """Every --flag the README shows for list-models must exist in the parser."""
    import re
    from localhive import cli
    parser = cli.build_parser()
    readme = os.path.join(os.path.dirname(__file__), "..", "README.md")
    with open(readme, encoding="utf-8") as fh:
        text = fh.read()
    documented = set(re.findall(r"localhive list-models((?:\s+--[a-z-]+)+)", text))
    flags = set()
    for chunk in documented:
        flags |= set(re.findall(r"--[a-z-]+", chunk))
    # --context takes a value; the rest are store_true
    known = {"--context", "--verbose", "--json", "--all", "-v", "-c"}
    unknown = flags - known
    assert not unknown, f"README documents flags that do not exist: {unknown}"
    # and prove they parse
    for f in sorted(flags - {"--context"}):
        parser.parse_args(["list-models", f])

def test_readme_test_badge_matches_reality():
    """The tests badge must not claim a number we do not have."""
    import re
    import subprocess
    readme = os.path.join(os.path.dirname(__file__), "..", "README.md")
    with open(readme, encoding="utf-8") as fh:
        m = re.search(r"badge/tests-(\d+)%20passing", fh.read())
    assert m, "tests badge missing from README"
    claimed = int(m.group(1))

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", os.path.dirname(__file__), "-q", "--collect-only"],
        capture_output=True, text=True,
    )
    found = re.search(r"(\d+) tests? collected", proc.stdout)
    assert found, proc.stdout[-500:]
    assert claimed == int(found.group(1)), (
        f"README badge claims {claimed} tests, pytest collects {found.group(1)}"
    )

def test_classifier_keeps_unknown_quants():
    """A quant we do not recognise must still be offered, not silently dropped.

    Regression: the old classifier only matched a narrow regex, so LM Studio
    repacks like Q4_0_4_4 and lowercase f16 were classified as 'skip' and
    vanished from the catalog with no warning.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_catalog",
        os.path.join(os.path.dirname(__file__), "..", "tools", "build_catalog.py"),
    )
    bc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bc)

    real_models = [
        "Llama-3.2-3B-Instruct-Q4_0_4_4.gguf",
        "Llama-3.2-3B-Instruct-Q4_0_8_8.gguf",
        "Llama-3.2-3B-Instruct-f16.gguf",
        "Llama-3.2-3B-Instruct-Q4_K_L.gguf",
        "Something-Weird-XYZ123.gguf",
    ]
    for name in real_models:
        assert bc.classifier(name) == "model", f"{name} was dropped"

    # companions must still be recognised
    assert bc.classifier("mmproj-Foo-f16.gguf") == "mmproj"
    assert bc.classifier("mtp-gemma-4-12B-it.gguf") == "mtp"
    assert bc.classifier("MTP-something.gguf") == "mtp"


def test_catalog_has_no_companion_files():
    """mmproj / mtp must never appear as installable entries."""
    from localhive import catalog
    for m in catalog.CATALOG:
        low = m.filename.lower()
        assert not low.startswith("mmproj"), m.filename
        assert not low.startswith("mtp-"), m.filename
        assert "fastmtp" not in low, m.filename


def test_catalog_matches_its_generator_shape():
    """catalog.py must be exactly what tools/build_catalog.py emits.

    Guards against someone hand-editing the generated file, which would make
    the documented regenerate command produce a surprise diff.
    """
    import ast
    path = os.path.join(os.path.dirname(__file__), "..", "src", "localhive", "catalog.py")
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)

    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "Model"
    ]
    from localhive import catalog as c
    assert len(calls) == len(c.CATALOG)

    # the generator emits double-quoted strings; the file should match
    assert "key='" not in src, "catalog.py has single-quoted keys; regenerate it"
    assert 'key="' in src, "catalog.py should be generator output"
