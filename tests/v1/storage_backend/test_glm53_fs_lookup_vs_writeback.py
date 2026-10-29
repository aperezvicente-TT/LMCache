# SPDX-License-Identifier: Apache-2.0
"""glm53 (Stage 12, program 2026-10-27T09:28:17Z): the post-1M admission stall.

The k4b window's 1M store wrote ~8,200 chunk files to the fs remote tier. Every aiofiles call of those writes ran on
the remote loop's default executor, and so did every lookup's exists(): each lookup waited out the write-back (the
3-s lookup timeout, TTFT P50 112 s). Now the FS connector writes on its own pool, capped by the bytes handed to it,
and answers existence from an in-memory index or a stat pool of its own. A write-back that cannot finish must not
delay a lookup."""

# Standard
import asyncio
import os
import shutil
import tempfile
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.connector.fs_connector import FSConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.remote_backend import RemoteBackend
from tests.v1.utils import create_test_memory_obj


def _key(i: int) -> CacheEngineKey:
    return CacheEngineKey(model_name="m", world_size=1, worker_id=0, chunk_hash=1_000_003 * i + 7,
                          dtype=torch.bfloat16)  # fmt: skip


def _metadata():
    return LMCacheMetadata(model_name="m", world_size=1, local_world_size=1, worker_id=0, local_worker_id=0,
                           kv_dtype=torch.bfloat16, kv_shape=(2, 2, 256, 8, 128))  # fmt: skip


@pytest.fixture
def fs_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def loop():
    # First Party
    from lmcache.utils import start_loop_in_thread_with_exceptions

    lp = asyncio.new_event_loop()
    t = threading.Thread(target=start_loop_in_thread_with_exceptions, args=(lp,), name="test-remote-loop")
    t.start()
    yield lp
    lp.call_soon_threadsafe(lp.stop)
    t.join(timeout=5.0)


def _fs(b) -> FSConnector:
    """the FS connector under RemoteBackend's InstrumentedRemoteConnector wrapper (which forwards every call)"""
    conn = b.connection
    return conn.getWrappedConnector() if hasattr(conn, "getWrappedConnector") else conn


@pytest.fixture
def make_backend(fs_dir, loop, memory_allocator):
    made = []

    def _make(**extra):
        config = LMCacheEngineConfig.from_defaults(chunk_size=256, remote_url=f"fs://host:0/{fs_dir}",
                                                   remote_serde="naive", lmcache_instance_id="t",
                                                   extra_config=dict(extra) or None)  # fmt: skip
        cpu = LocalCPUBackend(LMCacheEngineConfig.from_legacy(chunk_size=256), _metadata(),
                              memory_allocator=memory_allocator)  # fmt: skip
        b = RemoteBackend(config=config, metadata=_metadata(), loop=loop, local_cpu_backend=cpu, dst_device="cpu")
        assert isinstance(_fs(b), FSConnector)
        made.append(b)
        return b

    yield _make
    for b in made:
        b.close()


def _timed(fn, *a):
    t0 = time.perf_counter()
    r = fn(*a)
    return r, time.perf_counter() - t0


def _wait(pred, timeout=20.0):
    t0 = time.monotonic()
    while not pred():
        if time.monotonic() - t0 > timeout:
            raise AssertionError("timed out")
        time.sleep(0.01)


def test_a_lookup_never_waits_for_the_write_back(fs_dir, make_backend, monkeypatch):
    b = make_backend(fs_connector_write_threads=2, fs_connector_write_inflight_mb=1)
    conn = _fs(b)
    go = threading.Event()
    real = FSConnector._put_sync

    def held_write(self, final_path, temp_path, memory_obj):  # the share's write-back, held until go
        go.wait(30)
        real(self, final_path, temp_path, memory_obj)

    monkeypatch.setattr(FSConnector, "_put_sync", held_write)
    old = 10_000  # a file of an earlier window (on disk, not in this process's index)
    obj = create_test_memory_obj()
    with open(conn._get_file_path(_key(old)), "wb") as f:
        f.write(b"x")
    n = 400  # ~ a 1M store's first 400 chunks: none can land while `go` is clear
    for i in range(n):
        b.submit_put_task(_key(i), obj)
    _wait(lambda: conn.fs_stats["write_waits"] > 0)  # the cap is holding chunks back
    hit_old, t_old = _timed(b.contains, _key(old))
    miss, t_miss = _timed(b.contains, _key(n + 5))
    prefix, t_pre = _timed(b.batched_contains, [_key(old), _key(n + 6)])
    assert hit_old and not miss and prefix == 1
    assert max(t_old, t_miss, t_pre) < 0.5, (t_old, t_miss, t_pre)  # was: behind every queued write
    # the cap: chunks handed to the write pool stay within it (one chunk always fits)
    chunk = len(obj.byte_array)
    assert conn.fs_stats["write_inflight_max"] <= (1 << 20) + chunk
    go.set()
    _wait(lambda: conn.fs_stats["writes"] == n)
    _wait(lambda: not b.put_tasks)
    files = {f for f in os.listdir(fs_dir) if f.endswith(".data")}
    assert len(files) == n + 1
    stats0 = conn.fs_stats["stats"]
    assert b.batched_contains([_key(i) for i in range(n)]) == n  # every written key from the index
    assert conn.fs_stats["stats"] == stats0


def test_the_async_lookup_path_has_prefix_semantics_and_uses_the_index(fs_dir, make_backend):
    b = make_backend()
    conn = _fs(b)
    obj = create_test_memory_obj()
    for i in range(10):
        b.submit_put_task(_key(i), obj)
    _wait(lambda: conn.fs_stats["writes"] == 10)
    keys = [_key(i) for i in range(10)] + [_key(99), _key(3)]
    fut = asyncio.run_coroutine_threadsafe(b.batched_async_contains("l", keys), b.loop)
    assert fut.result(5) == 10
    assert conn.fs_stats["index_hits"] >= 10


def test_a_file_removed_under_the_index_is_a_miss_after_its_failed_get(fs_dir, make_backend):
    b = make_backend()
    conn = _fs(b)
    obj = create_test_memory_obj()
    b.submit_put_task(_key(1), obj)
    _wait(lambda: conn.fs_stats["writes"] == 1)
    assert b.contains(_key(1))
    os.remove(conn._get_file_path(_key(1)))  # e.g. a prune between windows
    assert b.contains(_key(1))  # the index still says so ...
    assert b.get_blocking(_key(1)) is None  # ... until the read finds no file
    assert not b.contains(_key(1))


def test_a_failed_write_is_not_indexed(fs_dir, make_backend, monkeypatch):
    b = make_backend()
    conn = _fs(b)

    def boom(self, final_path, temp_path, memory_obj):
        raise OSError("ENOSPC")

    monkeypatch.setattr(FSConnector, "_put_sync", boom)
    b.submit_put_task(_key(2), create_test_memory_obj())
    _wait(lambda: not b.put_tasks)
    assert not b.contains(_key(2)) and b.put_failed_count == 1


def test_the_writes_land_bitwise(fs_dir, make_backend):
    b = make_backend()
    obj = create_test_memory_obj()
    obj.tensor.copy_(torch.arange(obj.tensor.numel(), dtype=torch.float32).reshape(obj.tensor.shape).to(obj.tensor.dtype))
    b.submit_put_task(_key(5), obj)
    _wait(lambda: _fs(b).fs_stats["writes"] == 1)
    got = b.get_blocking(_key(5))
    assert got is not None and torch.equal(got.tensor, obj.tensor)


def test_stale_temp_files_are_swept_at_start_and_fresh_ones_kept(fs_dir, make_backend):
    """a killed process's in-flight temp files (14's NIT c): only those older than fs_connector_tmp_sweep_min"""
    old, fresh = os.path.join(fs_dir, "a.tmp"), os.path.join(fs_dir, "b.tmp")
    for p in (old, fresh):
        with open(p, "wb") as f:
            f.write(b"x")
    os.utime(old, (time.time() - 3600, time.time() - 3600))
    b = make_backend(fs_connector_tmp_sweep_min=10)
    _wait(lambda: "tmp_swept" in _fs(b).fs_stats)
    assert not os.path.exists(old) and os.path.exists(fresh) and _fs(b).fs_stats["tmp_swept"] == 1
