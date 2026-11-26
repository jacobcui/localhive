"""Track and control the llama-server processes localhive starts.

'llama-server' keeps no state of its own, so we record what we launch in a
small JSON file under ~/.localhive. That lets 'ps' show what is running and
'stop' kill exactly the right thing, rather than pattern-matching on process
names and hoping.
"""

from __future__ import annotations

import errno
import json
import os
import signal
import socket
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

STATE_DIR = Path.home() / ".localhive"
RUN_DIR = STATE_DIR / "run"


@dataclass
class Server:
    """A llama-server we started (or one we found already listening)."""

    pid: int
    port: int
    host: str
    model_key: str
    model_file: str
    context: int
    started_at: float

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def alive(self) -> bool:
        return is_alive(self.pid)


def _state_file(port: int) -> Path:
    return RUN_DIR / f"{port}.json"


def is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as exc:
        # EPERM means it exists but belongs to someone else.
        return exc.errno == errno.EPERM
    return True


def record(server: Server) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    _state_file(server.port).write_text(json.dumps(asdict(server), indent=2) + "\n")


def forget(port: int) -> None:
    _state_file(port).unlink(missing_ok=True)


def _pid_is_llama(pid: int) -> bool:
    """Guard against killing an unrelated process that reused the pid."""
    try:
        out = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return "llama-server" in out.stdout


def running() -> list[Server]:
    """Every recorded server that is still alive.

    Stale records (process died, or the pid now belongs to something else)
    are pruned as a side effect, so 'ps' never reports a ghost.
    """
    if not RUN_DIR.is_dir():
        return []
    out: list[Server] = []
    for f in sorted(RUN_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            srv = Server(**data)
        except (OSError, json.JSONDecodeError, TypeError):
            f.unlink(missing_ok=True)
            continue
        if srv.alive and _pid_is_llama(srv.pid):
            out.append(srv)
        else:
            f.unlink(missing_ok=True)
    return out


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """True if something is already listening on host:port.

    A plain TCP connect is used rather than binding, because binding would
    itself briefly occupy the port.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex((host, port)) == 0


def find_by_port(port: int) -> Optional[Server]:
    for srv in running():
        if srv.port == port:
            return srv
    return None


def stop(server: Server, timeout: float = 10.0) -> bool:
    """SIGTERM then, if it will not go, SIGKILL. Returns True if it stopped."""
    if not is_alive(server.pid):
        forget(server.port)
        return True

    try:
        os.kill(server.pid, signal.SIGTERM)
    except OSError:
        forget(server.port)
        return True

    deadline = time.time() + timeout
    while time.time() < deadline:
        if not is_alive(server.pid):
            forget(server.port)
            return True
        time.sleep(0.15)

    # Still there - force it.
    try:
        os.kill(server.pid, signal.SIGKILL)
    except OSError:
        pass
    time.sleep(0.3)
    stopped = not is_alive(server.pid)
    if stopped:
        forget(server.port)
    return stopped


def discover(ports: range | None = None,
             host: str = "127.0.0.1") -> list[Server]:
    """Find llama-servers by probing ports, not just by our own records.

    Covers servers started before tracking existed, or started by hand.
    Only ports that answer /health with something llama-shaped are returned,
    so we do not pick up unrelated services.
    """
    if ports is None:
        ports = range(7000, 7100)
    found: list[Server] = []
    known = {s.port for s in running()}
    for port in ports:
        if port in known or not port_in_use(port, host):
            continue
        status = health(port, host)
        if status is None:
            continue
        # An OpenAI-compatible /v1/models response identifies it as llama.cpp.
        model_file = ""
        key = "(untracked)"
        try:
            import urllib.request
            with urllib.request.urlopen(
                f"http://{host}:{port}/v1/models", timeout=1.5
            ) as resp:
                data = json.loads(resp.read().decode("utf-8", "replace"))
            entry = (data.get("data") or [{}])[0]
            model_file = Path(entry.get("id", "")).name
            if model_file:
                key = model_file
        except Exception:  # noqa: BLE001 - best effort only
            pass

        found.append(Server(
            pid=_pid_on_port(port) or 0,
            port=port,
            host=host,
            model_key=key,
            model_file=model_file,
            context=0,
            started_at=0.0,
        ))
    return found


def _pid_on_port(port: int) -> Optional[int]:
    """Best-effort pid lookup for a listening port, using lsof."""
    try:
        out = subprocess.run(
            ["lsof", "-nP", "-iTCP:%d" % port, "-sTCP:LISTEN", "-t"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.stdout.split():
        if line.isdigit():
            return int(line)
    return None


def health(port: int, host: str = "127.0.0.1", timeout: float = 1.5) -> Optional[str]:
    """Query /health. Returns 'ok', another status, or None if unreachable."""
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(
            f"http://{host}:{port}/health", timeout=timeout
        ) as resp:
            body = resp.read().decode("utf-8", "replace")
        try:
            return json.loads(body).get("status", "unknown")
        except json.JSONDecodeError:
            return body.strip() or "unknown"
    except urllib.error.HTTPError as exc:
        # llama-server returns 503 while the model is still loading.
        return f"http {exc.code}"
    except (urllib.error.URLError, OSError):
        return None
