# SPDX-License-Identifier: Apache-2.0
"""glm53 (Stage 12, program 2026-10-26T16:12:35Z B2): the tile-object key namespace. A row-major object (the k line,
no namespace) stored on any tier is a MISS for a tile-mode engine ("#tt-tile-v2"), because the namespace is part of
every key: the in-memory key equality / hash and the key string the disk and fs:// tiers name their objects by."""

# Standard
from types import SimpleNamespace

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm.utils import key_namespace
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.token_database import ChunkedTokenDatabase

MODEL = "/opt/glm53/share/llm-wksp/models/models--zai-org--GLM-5.3-Flash/snapshots/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"


def _db(ns: str) -> ChunkedTokenDatabase:
    cfg = LMCacheEngineConfig.from_defaults(chunk_size=128, extra_config={"key_namespace": ns} if ns else None)
    md = LMCacheMetadata(model_name=MODEL + key_namespace(cfg), world_size=1, local_world_size=1, worker_id=0,
                         local_worker_id=0, kv_dtype=torch.bfloat16, kv_shape=(11, 1, 128, 1, 576), use_mla=True)  # fmt: skip
    return ChunkedTokenDatabase(cfg, md)


def _keys(db, tokens):
    return [r[2] for r in db.process_tokens(tokens=tokens)]


def test_a_row_major_object_is_a_miss_under_the_tile_namespace():
    tokens = list(range(1000, 1000 + 4 * 128))
    plain, tile = _keys(_db(""), tokens), _keys(_db("#tt-tile-v2"), tokens)
    assert len(plain) == len(tile) == 4
    # the same chunk hashes (the token chain is unchanged) ...
    assert [k.chunk_hash for k in plain] == [k.chunk_hash for k in tile]
    # ... but no key matches: a dict-backed tier (LocalCPUBackend's hot cache) keyed by the row-major stores misses
    store = {k: b"row-major" for k in plain}
    assert not any(k in store for k in tile)
    # and the object names of the disk / fs:// tiers (CacheEngineKey.to_string) are disjoint too
    names = {k.to_string() for k in plain}
    assert not any(k.to_string() in names for k in tile)
    assert all("#tt-tile-v2@" in k.to_string() for k in tile)


def test_the_namespaced_key_round_trips_through_its_string():
    (k,) = _keys(_db("#tt-tile-v2"), list(range(128)))
    assert CacheEngineKey.from_string(k.to_string()) == k


def test_no_namespace_keeps_the_k_line_keys_unchanged():
    tokens = list(range(5, 5 + 128))
    (k,) = _keys(_db(""), tokens)
    assert k.model_name == MODEL


@pytest.mark.parametrize("bad", ["@tile", "tile%2"])
def test_a_namespace_with_a_key_separator_is_refused(bad):
    with pytest.raises(ValueError, match="key separator"):
        key_namespace(SimpleNamespace(extra_config={"key_namespace": bad}))
