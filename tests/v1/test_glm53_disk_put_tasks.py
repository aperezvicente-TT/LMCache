# SPDX-License-Identifier: Apache-2.0
"""glm53 (Stage 12, program 2026-10-26T23:48:49Z, the store path): LocalDiskWorker's pending-put tracking is O(1)
(a key -> count dict; it was a List, scanned by every insert / exists / remove: 198,611 CacheEngineKey.__eq__ calls in
one 32K-token store with the disk tier behind). The list's semantics are kept: a key inserted twice stays pending until
removed twice; removing an absent key only warns."""

# Standard
import time

# Third Party
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.storage_backend.local_disk_backend import LocalDiskWorker


def _key(i):
    return CacheEngineKey(model_name="m", world_size=1, worker_id=0, chunk_hash=i, dtype=torch.bfloat16)


def _worker():
    w = LocalDiskWorker.__new__(LocalDiskWorker)
    # Standard
    import threading

    w.put_lock, w.put_tasks = threading.Lock(), {}
    return w


def test_multiset_semantics_are_kept():
    w = _worker()
    k = _key(1)
    assert not w.exists_in_put_tasks(k)
    w.insert_put_task(k)
    w.insert_put_task(k)
    w.remove_put_task(k)
    assert w.exists_in_put_tasks(k)  # inserted twice, removed once
    w.remove_put_task(k)
    assert not w.exists_in_put_tasks(k) and w.put_tasks == {}
    w.remove_put_task(k)  # absent: a warning, no error
    assert w.put_tasks == {}


def test_many_pending_puts_stay_constant_time():
    w = _worker()
    keys = [_key(i) for i in range(20000)]
    t0 = time.perf_counter()
    for k in keys:
        w.insert_put_task(k)
    for k in keys:
        assert w.exists_in_put_tasks(k)
    for k in keys:
        w.remove_put_task(k)
    assert w.put_tasks == {} and time.perf_counter() - t0 < 2.0  # the list took O(n^2): minutes at 20K
