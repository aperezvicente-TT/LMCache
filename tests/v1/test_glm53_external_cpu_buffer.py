# SPDX-License-Identifier: Apache-2.0
"""glm53 lever (c) step 2 (program 2026-10-26T03:59:21Z): LMCache's CPU tier over a caller-owned buffer (the TT
plugin's pinned page pool), named through extra_config["external_cpu_buffer"]. The objects must live INSIDE the buffer
(the page-DMA reads them in place), the buffer is never freed by LMCache, and after LMCache closes the owner can unmap
it (PinnedPagePool.close returns False while any view still exports the mapping)."""

import mmap

import pytest
import torch

from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventManager
from lmcache.v1.memory_allocators.external_buffer_allocator import (
    ExternalBufferMemoryAllocator,
    register_external_cpu_buffer,
    resolve_external_cpu_buffer,
    unregister_external_cpu_buffer,
)
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.storage_manager import StorageManager

MIB = 1024**2
SHAPE = torch.Size([11, 256, 544])  # the served object: 11 latent layers x a 256-token chunk x 1,088 B (bf16)


def _pool(nbytes):
    """an anonymous mapping exported as a torch view, as PinnedPagePool(backing="anon").tensor()"""
    mm = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    return mm, torch.frombuffer(mm, dtype=torch.uint8)


def _inside(obj, buf):
    lo, n = buf.data_ptr(), buf.numel()
    t = obj.tensor
    return lo <= t.data_ptr() and t.data_ptr() + t.numel() * t.element_size() <= lo + n


def test_registry_resolves_names_and_refuses_unknown_or_bad_buffers():
    mm, buf = _pool(4 * MIB)
    register_external_cpu_buffer("t_reg", buf)
    try:
        assert resolve_external_cpu_buffer("t_reg").data_ptr() == buf.data_ptr()
        assert resolve_external_cpu_buffer(None) is None and resolve_external_cpu_buffer("") is None
        assert resolve_external_cpu_buffer(buf) is buf
        register_external_cpu_buffer("t_reg", buf)  # the same buffer again: idempotent
        _, other = _pool(MIB)
        with pytest.raises(ValueError):
            register_external_cpu_buffer("t_reg", other)
        with pytest.raises(KeyError):
            resolve_external_cpu_buffer("t_missing")  # never a silent fallback to LMCache's own arena
        with pytest.raises(ValueError):
            register_external_cpu_buffer("t_bad", torch.zeros(16, dtype=torch.bfloat16))
    finally:
        unregister_external_cpu_buffer("t_reg")
    with pytest.raises(KeyError):
        resolve_external_cpu_buffer("t_reg")


def test_objects_live_inside_the_buffer_and_free_reuses_them():
    mm, buf = _pool(16 * MIB)
    a = ExternalBufferMemoryAllocator(buf)
    objs = []
    while (o := a.allocate(SHAPE, torch.bfloat16, MemoryFormat.KV_MLA_FMT)) is not None:
        objs.append(o)
    assert len(objs) == 16 * MIB // (SHAPE.numel() * 2)  # 5 x 3,063,808 B in 16 MiB
    assert all(_inside(o, buf) for o in objs)
    objs[2].tensor.fill_(1.0)  # writes land in the caller's memory
    off = objs[2].tensor.data_ptr() - buf.data_ptr()
    assert torch.all(buf[off : off + SHAPE.numel() * 2].view(torch.bfloat16) == 1.0)
    ptr = objs[2].tensor.data_ptr()
    a.free(objs.pop(2))
    again = a.allocate(SHAPE, torch.bfloat16, MemoryFormat.KV_MLA_FMT)
    assert again is not None and again.tensor.data_ptr() == ptr
    raw = a.allocate(torch.Size([100]), torch.uint8, MemoryFormat.BINARY_BUFFER)  # byte arrays: not in the pool
    assert raw is not None and (raw.tensor is None or not _inside(raw, buf))
    assert str(a) == "ExternalBufferMemoryAllocator"


def test_storage_manager_uses_the_named_buffer_and_releases_it_on_close():
    mm, buf = _pool(32 * MIB)
    register_external_cpu_buffer("glm53_pool_test", buf)
    cfg = LMCacheEngineConfig.from_defaults(chunk_size=256, local_cpu=True, max_local_cpu_size=0.5,
                                            lmcache_instance_id="glm53_extbuf",
                                            extra_config={"external_cpu_buffer": "glm53_pool_test"})  # fmt: skip
    md = LMCacheMetadata(model_name="m", world_size=1, local_world_size=1, worker_id=0, local_worker_id=0,
                         kv_dtype=torch.bfloat16, kv_shape=(11, 1, 256, 1, 544), role="worker", use_mla=True)  # fmt: skip
    sm = StorageManager(config=cfg, metadata=md, event_manager=EventManager())
    try:
        alloc = sm.local_cpu_backend.memory_allocator
        assert isinstance(alloc, ExternalBufferMemoryAllocator) and alloc.buffer.data_ptr() == buf.data_ptr()
        o = sm.allocate(SHAPE, torch.bfloat16, fmt=MemoryFormat.KV_MLA_FMT)
        assert o is not None and _inside(o, buf)
        o.ref_count_down()
        del o
    finally:
        sm.close()
        unregister_external_cpu_buffer("glm53_pool_test")
    del alloc, buf
    mm.close()  # BufferError if LMCache still held a view of the mapping


_HOOK_POOL = {}


def hook_buffer(nbytes):
    """a "module:fn" hook, as page_pool.lmcache_external_buffer: the first nbytes of one long-lived mapping"""
    if "t" not in _HOOK_POOL:
        _HOOK_POOL["mm"], _HOOK_POOL["t"] = _pool(64 * MIB)
    return _HOOK_POOL["t"][:nbytes]


def test_module_fn_form_calls_the_hook_with_the_tier_size(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "glm53_extbuf_hook", sys.modules[__name__])
    buf = resolve_external_cpu_buffer("glm53_extbuf_hook:hook_buffer", 32 * MIB)
    assert buf.numel() == 32 * MIB and buf.data_ptr() == _HOOK_POOL["t"].data_ptr()
    with pytest.raises(ValueError):
        resolve_external_cpu_buffer("glm53_extbuf_hook:hook_buffer")  # no size
    with pytest.raises(ValueError):
        resolve_external_cpu_buffer("glm53_extbuf_hook:hook_buffer", 128 * MIB)  # the hook returns too little
    cfg = LMCacheEngineConfig.from_defaults(chunk_size=256, local_cpu=True, max_local_cpu_size=1 / 32,
                                            lmcache_instance_id="glm53_extbuf_fn",
                                            extra_config={"external_cpu_buffer": "glm53_extbuf_hook:hook_buffer"})  # fmt: skip
    md = LMCacheMetadata(model_name="m", world_size=1, local_world_size=1, worker_id=0, local_worker_id=0,
                         kv_dtype=torch.bfloat16, kv_shape=(11, 1, 256, 1, 544), role="worker", use_mla=True)  # fmt: skip
    sm = StorageManager(config=cfg, metadata=md, event_manager=EventManager())
    try:
        alloc = sm.local_cpu_backend.memory_allocator
        assert isinstance(alloc, ExternalBufferMemoryAllocator) and alloc.size == 32 * MIB
        o = sm.allocate(SHAPE, torch.bfloat16, fmt=MemoryFormat.KV_MLA_FMT)
        assert o is not None and _inside(o, _HOOK_POOL["t"])
        o.ref_count_down()
    finally:
        sm.close()
