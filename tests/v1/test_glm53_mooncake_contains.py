# SPDX-License-Identifier: Apache-2.0
"""glm53 (Stage 12, form B v2 rehearsal 2026-10-26): the Mooncake connector's lookup. One batch_is_exist RPC for the
whole prefix when the binding has it (the per-chunk is_exist loop cost 2,048 RPCs / 0.32 s for a 128K-token D lookup
over RDMA loopback, mooncake/mc_pd_bench.py), the prefix ends at the first miss, and a negative status (a failed RPC)
is a miss, not a hit. Host only: a fake store, the connector built without its __init__."""

# Standard
import asyncio

# First Party
from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
    MooncakestoreConnector,
)


class _Key:
    def __init__(self, s):
        self.s = s

    def to_string(self):
        return self.s


class _Store:
    def __init__(self, status, batch=True):
        self.status, self.calls = status, []
        if batch:
            self.batch_is_exist = self._batch

    def _batch(self, keys):
        self.calls.append(("batch", len(keys)))
        return [self.status.get(k, 0) for k in keys]

    def is_exist(self, key):
        self.calls.append(("one", key))
        return self.status.get(key, 0)


def _conn(store):
    c = MooncakestoreConnector.__new__(MooncakestoreConnector)
    c.store = store
    return c


def _contains(c, names):
    return asyncio.run(c.batched_async_contains("lookup", [_Key(n) for n in names]))


def test_one_batched_rpc_counts_the_prefix_to_the_first_miss():
    store = _Store({"k0": 1, "k1": 1, "k2": 0, "k3": 1})
    assert _contains(_conn(store), ["k0", "k1", "k2", "k3"]) == 2
    assert store.calls == [("batch", 4)]


def test_a_failed_rpc_status_is_a_miss():
    store = _Store({"k0": 1, "k1": -800, "k2": 1})
    assert _contains(_conn(store), ["k0", "k1", "k2"]) == 1
    assert _contains(_conn(_Store({"k0": -800}, batch=False)), ["k0"]) == 0  # (was a hit: `not -800` is False)


def test_without_batch_is_exist_the_per_key_loop_remains():
    store = _Store({"k0": 1, "k1": 1}, batch=False)
    assert _contains(_conn(store), ["k0", "k1", "k2"]) == 2
    assert [c[0] for c in store.calls] == ["one", "one", "one"]


def test_no_keys_no_rpc():
    store = _Store({})
    assert _contains(_conn(store), []) == 0 and store.calls == []


def test_exists_is_a_bool_and_an_error_is_false():
    c = _conn(_Store({"a": 1, "b": -800}))
    assert c.exists_sync(_Key("a")) is True and c.exists_sync(_Key("b")) is False
    assert asyncio.run(c.exists(_Key("b"))) is False


def test_the_sync_lookup_hook_uses_the_batched_rpc():
    store = _Store({"k0": 1, "k1": 1, "k2": 1})
    c = _conn(store)
    assert c.support_batched_contains()
    assert c.batched_contains([_Key("k0"), _Key("k1"), _Key("k2"), _Key("k3")]) == 3
    assert store.calls == [("batch", 4)]
    assert not _conn(_Store({}, batch=False)).support_batched_contains()  # the base class's per-key path then


class _PutStore:
    def __init__(self, rcs):
        self.rcs, self.calls = rcs, []

    def batch_put_from(self, keys, ptrs, sizes, cfg):
        self.calls.append(len(keys))
        return list(self.rcs[: len(keys)])

    def put_from(self, key, ptr, size, cfg):
        self.calls.append(1)
        return self.rcs[0]


class _Obj:
    raw_tensor = object()
    data_ptr = 4096

    def get_size(self):
        return 1536


def _put_conn(store):
    # Standard
    from types import SimpleNamespace

    c = _conn(store)
    c.config = SimpleNamespace(transfer_timeout=5)
    c.replica_config = None
    return c


def test_a_failed_batched_put_is_counted_not_silent():
    c = _put_conn(_PutStore([0, -704, 0, -704]))
    asyncio.run(c._batched_put_zero_copy([_Key(f"k{i}") for i in range(4)], [_Obj() for _ in range(4)]))
    assert c.put_failed_keys == 2
    c2 = _put_conn(_PutStore([0, 0]))
    asyncio.run(c2._batched_put_zero_copy([_Key("a"), _Key("b")], [_Obj(), _Obj()]))
    assert getattr(c2, "put_failed_keys", 0) == 0


def test_a_failed_single_put_is_counted():
    c = _put_conn(_PutStore([-800]))
    asyncio.run(c._put_without_metadata("k0", _Obj()))
    assert c.put_failed_keys == 1
