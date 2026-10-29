# SPDX-License-Identifier: Apache-2.0
"""glm53 (program 2026-10-28T18:59:10Z / 19:01:29Z): the fs remote tier's optional size cap.

The fleet's cold tier had no cap and grew ~500 GB in 23 min under one users16 cell. An external pruner is unsafe: the
connector's in-memory index answers "hit" for a file deleted underneath, and the request whose promised chunk is gone
fails under kv_load_failure_policy "fail". The cap evicts through the connector, least recently used first, dropping
each key from the index before its file. It never evicts a file used within the grace window. Off by default."""

# Standard
import os
import time

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.connector.fs_connector import FSConnector
from tests.v1.storage_backend.test_glm53_fs_lookup_vs_writeback import (  # noqa: F401 (fixtures)
    _fs,
    _key,
    _wait,
    fs_dir,
    loop,
    make_backend,
)
from tests.v1.utils import create_test_memory_obj


@pytest.fixture(autouse=True)
def _no_cap_env(monkeypatch):
    for k in ("GLM53_LMCACHE_REMOTE_GB", "GLM53_LMCACHE_REMOTE_FREE_FLOOR_GB", "GLM53_LMCACHE_REMOTE_GRACE_S"):
        monkeypatch.delenv(k, raising=False)


def _files(d):
    return sorted(f for f in os.listdir(d) if f.endswith(".data"))


def _chunk_bytes(conn):
    obj = create_test_memory_obj()
    return len(obj.byte_array) + (conn.remote_metadata_bytes if conn.save_chunk_meta else 0)


def _capped(make_backend, monkeypatch, cap_chunks=0, floor_chunks=0, grace="0"):
    """a connector with the cap from the ENV (read at construction), sized in test chunks (a probe gives the size)"""
    chunk = _chunk_bytes(_fs(make_backend()))
    monkeypatch.setenv("GLM53_LMCACHE_REMOTE_GB", str(cap_chunks * chunk / 1e9))
    monkeypatch.setenv("GLM53_LMCACHE_REMOTE_FREE_FLOOR_GB", str(floor_chunks * chunk / 1e9))
    monkeypatch.setenv("GLM53_LMCACHE_REMOTE_GRACE_S", grace)
    b = make_backend()
    return b, _fs(b), chunk


def test_off_by_default_nothing_is_evicted(fs_dir, make_backend):
    b = make_backend()
    conn = _fs(b)
    assert not conn._lru_on
    obj = create_test_memory_obj()
    for i in range(20):
        b.submit_put_task(_key(i), obj)
    _wait(lambda: conn.fs_stats["writes"] == 20)
    time.sleep(0.2)
    assert len(_files(fs_dir)) == 20 and conn.fs_stats["evicted_files"] == 0


def test_the_cap_evicts_the_least_recently_used_and_the_index_drops_them(fs_dir, make_backend, monkeypatch):
    b, conn, chunk = _capped(make_backend, monkeypatch, cap_chunks=10.4)  # (a GB cap scaled to the test chunk)
    obj = create_test_memory_obj()
    for i in range(8):
        b.submit_put_task(_key(i), obj)
    _wait(lambda: conn.fs_stats["writes"] == 8)
    assert b.contains(_key(0)) and b.contains(_key(1))  # keys 0 and 1 used again: now the most recent
    for i in range(8, 14):
        b.submit_put_task(_key(i), obj)
    _wait(lambda: conn.fs_stats["writes"] == 14)
    _wait(lambda: conn.fs_stats.get("evicted_files", 0) >= 5)
    _wait(lambda: conn._lru_bytes <= 9 * chunk)
    kept = {i for i in range(14) if os.path.exists(conn._get_file_path(_key(i)))}
    assert {0, 1}.issubset(kept), kept  # used recently: kept
    assert 2 not in kept and 3 not in kept, kept  # the least recently used went first
    for i in range(14):  # no stale promise: an evicted key is a MISS for the index and the lookup alike
        assert b.contains(_key(i)) == (i in kept), i
    assert len(_files(fs_dir)) == len(kept) and len(kept) <= 9


def test_the_grace_window_protects_recently_used_files(fs_dir, make_backend, monkeypatch):
    b, conn, chunk = _capped(make_backend, monkeypatch, cap_chunks=4, grace="3600")
    obj = create_test_memory_obj()
    for i in range(8):
        b.submit_put_task(_key(i), obj)
    _wait(lambda: conn.fs_stats["writes"] == 8)
    conn._evict_kick.set()
    time.sleep(0.5)
    assert len(_files(fs_dir)) == 8  # all used within the grace window: over the cap for now, nothing promised vanishes


def test_files_of_an_earlier_engine_are_found_at_start_and_evicted_oldest_first(fs_dir, make_backend, monkeypatch):
    b0 = make_backend()
    obj = create_test_memory_obj()
    for i in range(6):
        b0.submit_put_task(_key(i), obj)
    _wait(lambda: _fs(b0).fs_stats["writes"] == 6)
    now = time.time()
    for i in range(6):  # key 0 the oldest
        p = _fs(b0)._get_file_path(_key(i))
        os.utime(p, (now - 3600 + i, now - 3600 + i))
    chunk = _chunk_bytes(_fs(b0))
    monkeypatch.setenv("GLM53_LMCACHE_REMOTE_GB", str(4.4 * chunk / 1e9))
    monkeypatch.setenv("GLM53_LMCACHE_REMOTE_GRACE_S", "60")
    b = make_backend()  # a new engine on the same dir (the cap from the env)
    conn = _fs(b)
    assert conn._lru_on and conn._cap_bytes > 0
    _wait(lambda: conn.fs_stats.get("lru_scanned", 0) == 6)
    _wait(lambda: conn.fs_stats.get("evict_passes", 0) >= 1)
    kept = {i for i in range(6) if os.path.exists(conn._get_file_path(_key(i)))}
    assert kept == {3, 4, 5} or kept == {2, 3, 4, 5}, kept  # the oldest went first, down to 90 pct of the cap
    assert not b.contains(_key(0))


def test_the_free_space_floor_evicts_while_the_filesystem_is_short(fs_dir, make_backend, monkeypatch):
    b, conn, chunk = _capped(make_backend, monkeypatch, floor_chunks=3)
    state = {"free": 10**15}  # plenty while writing
    monkeypatch.setattr(conn, "_free_bytes", lambda: state["free"])
    obj = create_test_memory_obj()
    for i in range(6):
        b.submit_put_task(_key(i), obj)
    _wait(lambda: conn.fs_stats["writes"] == 6)
    _wait(lambda: conn.fs_stats["lru_scanned"] >= 0 and len(conn._lru) == 6)
    state["free"] = 0
    conn._evict_pass()  # free 0 < floor: evict until 110 pct of the floor is freed
    left = _files(fs_dir)
    assert len(left) == 2, left  # 4 chunks freed >= 3.3 chunks
    assert conn.fs_stats["evicted_files"] == 4


def test_a_stat_during_eviction_is_a_miss(fs_dir, make_backend):
    b = make_backend()
    conn = _fs(b)
    obj = create_test_memory_obj()
    b.submit_put_task(_key(1), obj)
    _wait(lambda: conn.fs_stats["writes"] == 1)
    p = conn._get_file_path(_key(1))
    conn._index_drop(p)
    with conn._known_lock:
        conn._evicting.add(str(p))
    assert not conn._stat(p)  # the file is still there, but the cap is removing it
    with conn._known_lock:
        conn._evicting.discard(str(p))
    assert conn._stat(p)


def test_the_cap_reads_its_keys_from_extra_config_too(fs_dir, make_backend):
    b = make_backend(fs_connector_cap_gb=2.5, fs_connector_free_floor_gb=0, fs_connector_evict_grace_s=30)
    conn = _fs(b)
    assert conn._lru_on and conn._cap_bytes == int(2.5e9) and conn._grace_s == 30
    assert isinstance(conn, FSConnector)
