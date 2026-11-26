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
    # uncensored now requires an explicit scope
    assert cli.resolve("qwen3.6-35b-a3b-iq2m", "uncensored").key == "qwen3.6-35b-a3b-iq2m"


def test_resolve_by_filename():
    from localhive import cli
    m = cli.resolve("Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf",
                    "uncensored")
    assert m.key == "qwen3.6-35b-a3b-iq2m"


def test_resolve_unknown_exits():
    from localhive import cli
    with pytest.raises(SystemExit):
        cli.resolve("definitely-not-a-model-xyz", "all")


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
    rc = cli.main(["list-models", "--json", "--all-content"])
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
    import sys as _sys

    # The module must be registered in sys.modules before exec_module, or
    # dataclasses cannot resolve the string annotations produced by
    # "from __future__ import annotations".
    name = "_localhive_build_catalog_test"
    spec = importlib.util.spec_from_file_location(
        name,
        os.path.join(os.path.dirname(__file__), "..", "tools", "build_catalog.py"),
    )
    bc = importlib.util.module_from_spec(spec)
    _sys.modules[name] = bc
    try:
        spec.loader.exec_module(bc)
    finally:
        _sys.modules.pop(name, None)

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

# --- content filtering ------------------------------------------------------

def test_catalog_has_both_content_types():
    from localhive import catalog, models
    std = models.filter_by_content(catalog.CATALOG, "standard")
    unc = models.filter_by_content(catalog.CATALOG, "uncensored")
    assert std, "catalog must contain standard models by default"
    assert unc, "uncensored models should still be available"
    assert len(std) + len(unc) == len(models.filter_by_content(catalog.CATALOG, "all"))


def test_default_content_is_standard():
    from localhive import models
    assert models.DEFAULT_CONTENT == "standard"


def test_filter_standard_excludes_uncensored():
    from localhive import catalog, models
    std = models.filter_by_content(catalog.CATALOG, "standard")
    assert all(not m.is_uncensored for m in std)
    assert len(std) < len(catalog.CATALOG)


def test_filter_uncensored_only():
    from localhive import catalog, models
    unc = models.filter_by_content(catalog.CATALOG, "uncensored")
    assert all(m.is_uncensored for m in unc)


def test_filter_all_returns_everything():
    from localhive import catalog, models
    assert len(models.filter_by_content(catalog.CATALOG, "all")) == len(catalog.CATALOG)


def test_content_is_declared_not_sniffed():
    """Content must be an explicit field, not guessed from the filename."""
    from localhive import catalog
    for m in catalog.CATALOG:
        assert m.content in ("standard", "uncensored"), m.key
        # a model whose name says 'uncensored' must be labelled as such
        if "uncensored" in m.filename.lower():
            assert m.is_uncensored, f"{m.key} name says uncensored but content=s standard"


def test_catalog_spans_multiple_publishers():
    """The catalog must not be a single publisher's shelf."""
    from localhive import catalog
    owners = {m.repo.split("/")[0] for m in catalog.CATALOG}
    assert len(owners) >= 3, f"only {len(owners)} publisher(s): {owners}"


def test_resolve_hides_uncensored_by_default():
    from localhive import cli
    with pytest.raises(SystemExit) as exc:
        cli.resolve("qwen3.6-35b-a3b-iq2m", "standard")
    # the error should tell the user how to get it
    assert "uncensored" in str(exc.value).lower()


def test_resolve_finds_uncensored_when_asked():
    from localhive import cli
    m = cli.resolve("qwen3.6-35b-a3b-iq2m", "uncensored")
    assert m.is_uncensored


def test_resolve_standard_model_without_flag():
    from localhive import cli
    from localhive import catalog, models
    std = models.filter_by_content(catalog.CATALOG, "standard")
    m = cli.resolve(std[0].key, "standard")
    assert not m.is_uncensored


def test_list_models_default_hides_uncensored(capsys):
    from localhive import cli
    cli.main(["list-models", "--json"])
    import json as _json
    data = _json.loads(capsys.readouterr().out)
    assert data["content_filter"] == "standard"
    assert all(m["content"] == "standard" for m in data["models"])


def test_list_models_uncensored_flag(capsys):
    from localhive import cli
    cli.main(["list-models", "--json", "--uncensored"])
    import json as _json
    data = _json.loads(capsys.readouterr().out)
    assert data["content_filter"] == "uncensored"
    assert all(m["content"] == "uncensored" for m in data["models"])


def test_list_models_all_content_flag(capsys):
    from localhive import cli
    from localhive import catalog
    cli.main(["list-models", "--json", "--all-content"])
    import json as _json
    data = _json.loads(capsys.readouterr().out)
    assert data["content_filter"] == "all"
    assert len(data["models"]) == len(catalog.CATALOG)

def test_download_resumes_from_part_file(tmp_path, monkeypatch):
    """A .part file must be resumed, not restarted, and not corrupted.

    Regression: the downloader opened the .part with "wb", so an interrupted
    3 GB download started again from zero.
    """
    import urllib.request
    from localhive import cli

    payload = b"0123456789" * 1000          # 10 KB "file"
    already = 4000
    part = tmp_path / "m.gguf.part"
    part.write_bytes(payload[:already])

    seen = {}

    class FakeResp:
        status = 206
        headers = {"Content-Length": str(len(payload) - already)}

        def read(self, n=-1):
            self._sent = getattr(self, "_sent", False)
            if self._sent:
                return b""
            self._sent = True
            return payload[already:]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, *a, **k):
        seen["range"] = req.headers.get("Range")
        return FakeResp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    cli._download("https://example.invalid/m.gguf", tmp_path / "m.gguf", "test")

    assert seen.get("range") == f"bytes={already}-", "resume Range header not sent"
    got = (tmp_path / "m.gguf").read_bytes()
    assert got == payload, "resumed file does not match the original"
    assert not part.exists(), ".part should be gone after a completed download"


def test_download_restarts_when_server_ignores_range(tmp_path, monkeypatch):
    """If the server returns 200 instead of 206, start over rather than append."""
    import urllib.request
    from localhive import cli

    payload = b"abcdefghij" * 500
    part = tmp_path / "n.gguf.part"
    part.write_bytes(b"GARBAGE" * 1000)     # stale partial content

    class FakeResp:
        status = 200                        # server ignored our Range
        headers = {"Content-Length": str(len(payload))}

        def read(self, n=-1):
            if getattr(self, "_done", False):
                return b""
            self._done = True
            return payload

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: FakeResp())
    cli._download("https://example.invalid/n.gguf", tmp_path / "n.gguf", "test")

    got = (tmp_path / "n.gguf").read_bytes()
    assert got == payload, "stale .part content leaked into the final file"


def test_download_keeps_part_on_interrupt(tmp_path, monkeypatch):
    """Ctrl-C must keep the partial file so the next run can resume."""
    import urllib.request
    from localhive import cli

    class Boom(Exception):
        pass

    class FakeResp:
        status = 200
        headers = {"Content-Length": "999999"}

        def read(self, n=-1):
            raise KeyboardInterrupt

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: FakeResp())
    with pytest.raises(SystemExit):
        cli._download("https://example.invalid/o.gguf", tmp_path / "o.gguf", "test")

    assert (tmp_path / "o.gguf.part").exists(), "partial file was deleted; resume impossible"

# --- server tracking --------------------------------------------------------

def test_server_state_roundtrip(tmp_path, monkeypatch):
    from localhive import servers as S

    monkeypatch.setattr(S, "RUN_DIR", tmp_path)
    monkeypatch.setattr(S, "_state_file", lambda p: tmp_path / f"{p}.json")
    monkeypatch.setattr(S, "_pid_is_llama", lambda pid: True)

    srv = S.Server(pid=os.getpid(), port=7000, host="127.0.0.1",
                   model_key="k", model_file="f.gguf", context=8192,
                   started_at=0.0)
    S.record(srv)
    got = S.running()
    assert len(got) == 1
    assert got[0].port == 7000
    assert got[0].model_key == "k"

    S.forget(7000)
    assert S.running() == []


def test_stale_records_are_pruned(tmp_path, monkeypatch):
    """A record whose process died must not be reported as running."""
    from localhive import servers as S

    monkeypatch.setattr(S, "RUN_DIR", tmp_path)
    monkeypatch.setattr(S, "_state_file", lambda p: tmp_path / f"{p}.json")
    monkeypatch.setattr(S, "is_alive", lambda pid: False)
    monkeypatch.setattr(S, "_pid_is_llama", lambda pid: True)

    S.record(S.Server(pid=999999, port=7002, host="127.0.0.1",
                      model_key="gone", model_file="g.gguf", context=0,
                      started_at=0.0))
    assert S.running() == []
    # and the stale file is cleaned up
    assert not (tmp_path / "7002.json").exists()


def test_record_pid_reuse_is_not_reported(tmp_path, monkeypatch):
    """If the pid now belongs to a non-llama process, treat it as gone."""
    from localhive import servers as S

    monkeypatch.setattr(S, "RUN_DIR", tmp_path)
    monkeypatch.setattr(S, "_state_file", lambda p: tmp_path / f"{p}.json")
    monkeypatch.setattr(S, "is_alive", lambda pid: True)
    monkeypatch.setattr(S, "_pid_is_llama", lambda pid: False)

    S.record(S.Server(pid=os.getpid(), port=7003, host="127.0.0.1",
                      model_key="stale", model_file="s.gguf", context=0,
                      started_at=0.0))
    assert S.running() == []


def test_corrupt_state_file_is_ignored(tmp_path, monkeypatch):
    from localhive import servers as S

    monkeypatch.setattr(S, "RUN_DIR", tmp_path)
    monkeypatch.setattr(S, "_state_file", lambda p: tmp_path / f"{p}.json")
    (tmp_path / "7004.json").write_text("{not json")
    assert S.running() == []
    assert not (tmp_path / "7004.json").exists()


def test_port_in_use_detects_a_real_listener():
    """Bind a socket, then confirm port_in_use sees it."""
    import socket
    from localhive import servers as S

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
        assert S.port_in_use(port) is True
    # after close, it should be free again
    assert S.port_in_use(port) is False


def test_serve_refuses_busy_port(capsys):
    """serve must not start on a port something already holds."""
    import socket
    from localhive import cli

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
        rc = cli.main(["serve", "llama-3.2-3b-iq3m", "--port", str(port)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "already in use" in out.lower()
    assert "stop" in out.lower(), "should tell the user how to free the port"


def test_ps_and_stop_are_registered():
    from localhive import cli
    parser = cli.build_parser()
    for cmd in ("ps", "stop"):
        ns = parser.parse_args([cmd])
        assert callable(ns.func)


def test_stop_without_args_explains_itself():
    from localhive import cli
    with pytest.raises(SystemExit) as exc:
        cli.main(["stop"])
    assert "port" in str(exc.value).lower()
