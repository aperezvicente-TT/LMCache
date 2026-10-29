# SPDX-License-Identifier: Apache-2.0
"""glm53 (Stage 12, program 2026-10-26T22:24:05Z, form B v2: Mooncake as the KV transfer): the connector's setup
dict. Mooncake's dict API (0.3.11+) reads rdma_devices / master_server_addr; the connector's legacy device_name /
master_server_address keys are forwarded under those names too, the mooncake_ prefixed keys still win, and a non-zero
setup() status raises instead of logging success (it used to fail only at the first put). Host only: a fake store."""

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
    MooncakeStoreConfig,
    setup_mooncake_store,
)


class _DictStore:
    """MooncakeDistributedStore's dict API: setup(dict) -> int status"""

    def __init__(self, rc=0):
        self.rc, self.calls = rc, []

    def setup(self, cfg):
        self.calls.append(dict(cfg))
        return self.rc


class _PositionalStore:
    """a pre-#1445 build: setup(*7 args) only"""

    def __init__(self, rc=0):
        self.rc, self.calls = rc, []

    def setup(self, *args):
        if len(args) == 1 and isinstance(args[0], dict):
            raise TypeError("setup(): incompatible function arguments")
        self.calls.append(args)
        return self.rc


def _cfg(extra):
    return MooncakeStoreConfig(setup_config=MooncakeStoreConfig._build_setup_dict(extra))


def test_legacy_keys_are_forwarded_under_the_dict_api_names():
    setup = _cfg({"device_name": "bnxt_re0", "master_server_address": "10.30.1.2:51051"}).setup_config
    assert setup["rdma_devices"] == "bnxt_re0" and setup["master_server_addr"] == "10.30.1.2:51051"
    # the legacy names stay for the positional fallback
    assert setup["device_name"] == "bnxt_re0" and setup["master_server_address"] == "10.30.1.2:51051"


def test_prefixed_keys_win_over_the_aliases():
    setup = _cfg({"master_server_address": "10.0.0.1:1", "mooncake_master_server_addr": "10.0.0.2:2",
                  "device_name": "mlx5_0", "mooncake_rdma_devices": "bnxt_re0"}).setup_config  # fmt: skip
    assert setup["master_server_addr"] == "10.0.0.2:2" and setup["rdma_devices"] == "bnxt_re0"


def test_no_device_given_sets_no_rdma_devices():
    setup = _cfg({"mooncake_protocol": "tcp"}).setup_config
    assert "rdma_devices" not in setup and setup["device_name"] == "" and setup["protocol"] == "tcp"


def test_dict_setup_success_passes_the_aliased_dict():
    store = _DictStore(0)
    setup_mooncake_store(store, _cfg({"master_server_address": "10.30.1.2:51051"}))
    assert store.calls[0]["master_server_addr"] == "10.30.1.2:51051"


@pytest.mark.parametrize("store_cls", [_DictStore, _PositionalStore])
def test_a_failed_setup_raises(store_cls):
    store = store_cls(-1)
    with pytest.raises(RuntimeError, match="setup failed: rc=-1") as e:
        setup_mooncake_store(store, _cfg({"master_server_address": "10.30.1.2:51051", "metadata_server": "http://x"}))
    assert "10.30.1.2" not in str(e.value) and "http://x" not in str(e.value)  # addresses stay redacted


def test_positional_fallback_success():
    store = _PositionalStore(0)
    setup_mooncake_store(store, _cfg({"device_name": "bnxt_re0", "global_segment_size": "1024"}))
    (args,) = store.calls
    assert args[2] == 1024 and args[5] == "bnxt_re0"


def test_a_setup_returning_none_is_accepted():
    """older bindings return None: no status to check"""
    store = _DictStore(None)
    setup_mooncake_store(store, _cfg({}))
    assert store.calls
