# SPDX-License-Identifier: Apache-2.0
"""End-to-end worker liveness: a live worker is never reaped, and a reaped
worker recovers.

Runs a real MP server process with a short reap timeout and a real
engine-driven SHM worker adapter on CPU tensors. No accelerator is needed:
the worker's SHM pin is patched, and the slow re-pin a large pool can take
is modelled by a sleep.
"""

# Standard
from collections.abc import Iterator
from pathlib import Path
import os
import signal
import socket
import subprocess
import sys
import time

# Third Party
import pytest
import torch
import zmq

# First Party
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    LMCacheMPWorkerAdapter,
    LoadStoreOp,
    ParallelStrategy,
)
from lmcache.v1.client.transfer_context.shm import EngineDrivenContextShm
from lmcache.v1.common.token_codec import pack_token_ids
from lmcache.v1.transport.futures import MessagingFuture
import lmcache

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="Named SHM pool requires Linux"
)

SOURCE_ROOT = Path(lmcache.__file__).resolve().parents[1]
REAP_TIMEOUT_SECONDS = 6.0
HEARTBEAT_INTERVAL_SECONDS = 1.0
MQ_TIMEOUT_SECONDS = 5.0
SLOW_PIN_SECONDS = 10.0
BLOCK_SIZE = 4
CHUNK_SIZE = 16

# The CLI enforces a 30 s reap-timeout floor; lower it after parsing so the
# reaper runs on a test time scale (it scans every timeout / 4).
_SERVER_MAIN = f"""
import runpy
import lmcache.v1.multiprocess.config as config
parse = config.parse_args_to_mp_server_config
def parse_with_short_reap(args):
    mp_config = parse(args)
    mp_config.worker_reap_timeout_seconds = {REAP_TIMEOUT_SECONDS}
    mp_config.worker_registration_grace_seconds = {REAP_TIMEOUT_SECONDS}
    return mp_config
config.parse_args_to_mp_server_config = parse_with_short_reap
runpy.run_module("lmcache.v1.multiprocess.server", run_name="__main__")
"""


class _Server:
    def __init__(self, process: subprocess.Popen, url: str, log: Path) -> None:
        self.process = process
        self.url = url
        self.log = log

    def log_count(self, text: str) -> int:
        return self.log.read_text(errors="replace").count(text)


def _unused_tcp_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def server(tmp_path: Path) -> Iterator[_Server]:
    port = _unused_tcp_port()
    log = tmp_path / "server.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(SOURCE_ROOT), env.get("PYTHONPATH")) if p
    )
    shm_suffix = f"reap_recovery_{os.getpid()}_{port}"
    with log.open("w") as log_file:
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                _SERVER_MAIN,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--l1-size-gb",
                "0.0625",
                "--no-l1-use-lazy",
                "--supported-transfer-mode",
                "engine_driven",
                "--shm-name",
                shm_suffix,
                "--eviction-policy",
                "LRU",
                "--chunk-size",
                str(CHUNK_SIZE),
                "--worker-reap-timeout-seconds",
                "30",
                "--disable-observability",
            ],
            cwd=tmp_path,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    srv = _Server(process, f"tcp://127.0.0.1:{port}", log)
    try:
        deadline = time.monotonic() + 90
        while "cache server is running on" not in log.read_text(errors="replace"):
            if process.poll() is not None or time.monotonic() > deadline:
                pytest.fail(f"MP server did not start:\n{log.read_text()}")
            time.sleep(0.2)
        yield srv
    finally:
        os.kill(process.pid, signal.SIGCONT)
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        shm_path = Path("/dev/shm") / f"lmcache_l1_pool_{shm_suffix}"
        if shm_path.exists():
            shm_path.unlink()


@pytest.fixture
def adapter(server: _Server) -> Iterator[LMCacheMPWorkerAdapter]:
    worker = LMCacheMPWorkerAdapter(
        server_url=server.url,
        context=zmq.Context.instance(),
        model_name="reap-test",
        vllm_block_size=BLOCK_SIZE,
        parallel_strategy=ParallelStrategy(
            vllm_world_size=1,
            vllm_worker_id=0,
            tp_size=1,
            pp_size=1,
            n_servers=1,
        ),
        mq_timeout=MQ_TIMEOUT_SECONDS,
        heartbeat_interval=HEARTBEAT_INTERVAL_SECONDS,
    )
    kv_caches = {f"layer_{i}": torch.randn(2, 16, BLOCK_SIZE, 2, 8) for i in range(2)}
    worker.register_kv_caches(kv_caches, layout_hints={})
    try:
        yield worker
    finally:
        worker.shutdown()


def _patch_slow_pin(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Make every later SHM pin succeed slowly; return a call record."""
    calls: list[int] = []

    def pin(self: EngineDrivenContextShm) -> None:
        calls.append(1)
        time.sleep(SLOW_PIN_SECONDS)
        self._pinned = True

    monkeypatch.setattr(EngineDrivenContextShm, "_pin_shm_buffer", pin)
    return calls


def _store(adapter: LMCacheMPWorkerAdapter, n: int) -> tuple[bool, float]:
    """Store one chunk; return (stored, seconds the submit took)."""
    request_id = f"req-{n}"
    op = LoadStoreOp(
        token_bytes=pack_token_ids(list(range(n * 100, n * 100 + CHUNK_SIZE))),
        block_ids=[[0, 1, 2, 3]],
        start=0,
        end=CHUNK_SIZE,
    )
    start = time.monotonic()
    adapter.submit_store_request(request_id, op, None)
    elapsed = time.monotonic() - start
    future = adapter.store_futures.get(request_id)
    return future is not None and bool(future.result(timeout=10)), elapsed


def _store_until_ok(
    adapter: LMCacheMPWorkerAdapter, first: int, timeout: float
) -> tuple[int, float]:
    """Store every 0.5 s until one succeeds; return (attempts, slowest submit)."""
    deadline = time.monotonic() + timeout
    attempts, slowest = 0, 0.0
    while True:
        ok, elapsed = _store(adapter, first + attempts)
        attempts += 1
        slowest = max(slowest, elapsed)
        if ok:
            return attempts, slowest
        assert time.monotonic() < deadline, "stores never succeeded again"
        time.sleep(0.5)


@pytest.mark.parametrize("reuse_pool", [True, False], ids=["reuse", "repin"])
def test_slow_recovery_is_not_reaped(
    server: _Server,
    adapter: LMCacheMPWorkerAdapter,
    monkeypatch: pytest.MonkeyPatch,
    reuse_pool: bool,
) -> None:
    """A server stall longer than the ping timeout triggers recovery; a
    recovery that outlasts the reap timeout must not get the worker reaped,
    and stores must work again afterwards."""
    # Registration already mapped the pool; any re-pin is the slow one.
    pins = _patch_slow_pin(monkeypatch)
    if not reuse_pool:
        # Model a pool that must be mapped and pinned again.
        monkeypatch.setattr(
            EngineDrivenContextShm,
            "_adopt_shared_pool",
            lambda self: False,
            raising=False,
        )
    assert _store(adapter, 0)[0]

    os.kill(server.process.pid, signal.SIGSTOP)
    time.sleep(2 * HEARTBEAT_INTERVAL_SECONDS + 0.5)
    os.kill(server.process.pid, signal.SIGCONT)

    # Long enough for the slow pin and two reap scans past the timeout.
    attempts, slowest = _store_until_ok(
        adapter, 1, timeout=SLOW_PIN_SECONDS + REAP_TIMEOUT_SECONDS + 10
    )
    time.sleep(REAP_TIMEOUT_SECONDS / 2)

    assert server.log_count("Reaped non-GPU instance") == 0
    assert adapter.is_healthy
    assert _store(adapter, 1000)[0]
    # Stores during recovery are dropped at once, not left to time out.
    assert slowest < 1.0, f"a store took {slowest:.1f}s during recovery"
    assert len(pins) == (0 if reuse_pool else 1)


def test_reaped_worker_reregisters_on_store(
    server: _Server,
    adapter: LMCacheMPWorkerAdapter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live worker the server has reaped re-registers when a store is
    rejected as unregistered, and the following stores succeed."""
    assert _store(adapter, 0)[0]

    # Keep the heartbeat healthy without reaching the server, so only the
    # store path can notice the reap.
    def silent_ping(instance_id: int | None) -> MessagingFuture[bool]:
        future: MessagingFuture[bool] = MessagingFuture()
        future.set_result(True)
        return future

    monkeypatch.setattr(adapter.req_client, "ping", silent_ping)
    deadline = time.monotonic() + 3 * REAP_TIMEOUT_SECONDS
    while server.log_count("Reaped non-GPU instance") == 0:
        assert time.monotonic() < deadline, "server never reaped the worker"
        time.sleep(0.2)
    assert adapter.is_healthy

    attempts, slowest = _store_until_ok(adapter, 1, timeout=10)

    assert slowest < 1.0, f"a store took {slowest:.1f}s after the reap"
    assert server.log_count("Registered non-GPU context") == 2
    assert adapter.is_healthy
