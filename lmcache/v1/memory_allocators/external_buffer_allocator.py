# SPDX-License-Identifier: Apache-2.0
"""LMCache's CPU tier over a caller-owned host buffer (glm53 lever (c) step 2).

The TT plugin keeps LMCache's CPU-tier objects inside its pinned page pool (hugetlbfs pages the TT driver pins per
chip), so a hit's pages DMA straight from the objects with no host copy. The buffer is created before LMCache's engine
and handed over in one of two JSON-safe forms of ``extra_config["external_cpu_buffer"]``:
- ``"module:fn"`` -- LMCache imports ``module`` and calls ``fn(nbytes)``, which returns a CPU uint8 tensor of at least
  nbytes (the TT plugin: ``models.autoports.zai_org_glm_5_3_flash.tt.page_pool:lmcache_external_buffer``, the shared
  pinned pool's LMCache region; Stage 12's contract, 2026-10-26T04:23:59Z);
- ``"name"`` -- a buffer the owner published with ``register_external_cpu_buffer(name, tensor)``.
A tensor in an in-process config is taken as is.
LocalCPUBackend then allocates its pinned objects in that buffer instead of its own arena. The allocator never frees
the buffer; close() drops its references so the owner can unmap it.
"""

# Standard
from typing import Dict, Optional, Union
import threading

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.memory_allocators.buffer_allocator import BufferAllocator
from lmcache.v1.memory_allocators.mixed_memory_allocator import MixedMemoryAllocator
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator
from lmcache.v1.memory_management import AddressManager

logger = init_logger(__name__)

EXTRA_CONFIG_KEY = "external_cpu_buffer"

_registry: Dict[str, torch.Tensor] = {}
_registry_lock = threading.Lock()


def _check_buffer(buffer: torch.Tensor) -> torch.Tensor:
    if not isinstance(buffer, torch.Tensor):
        raise TypeError(f"external CPU buffer must be a torch.Tensor, got {type(buffer).__name__}")
    if buffer.device.type != "cpu" or buffer.dtype != torch.uint8 or buffer.dim() != 1:
        raise ValueError(
            f"external CPU buffer must be a 1-D uint8 CPU tensor (got {buffer.dtype} {tuple(buffer.shape)} on {buffer.device})"
        )
    if not buffer.is_contiguous() or buffer.numel() == 0:
        raise ValueError("external CPU buffer must be contiguous and non-empty")
    return buffer


def register_external_cpu_buffer(name: str, buffer: torch.Tensor) -> None:
    """Publish ``buffer`` (1-D uint8, CPU, contiguous) under ``name`` for ``extra_config["external_cpu_buffer"]``."""
    _check_buffer(buffer)
    with _registry_lock:
        if name in _registry and _registry[name].data_ptr() != buffer.data_ptr():
            raise ValueError(f"external CPU buffer {name!r} is already registered to another buffer")
        _registry[name] = buffer


def unregister_external_cpu_buffer(name: str) -> None:
    """Drop the registry's reference (the owner unmaps the buffer after LMCache closed)."""
    with _registry_lock:
        _registry.pop(name, None)


def resolve_external_cpu_buffer(
    value: Union[None, str, torch.Tensor], nbytes: Optional[int] = None
) -> Optional[torch.Tensor]:
    """extra_config value -> the buffer: None -> None, a tensor -> it, "module:fn" -> fn(nbytes) (at least nbytes),
    a name -> the registered buffer (KeyError if the owner did not register it: LMCache must not silently fall back to
    its own arena)."""
    if value is None or value == "":
        return None
    if isinstance(value, torch.Tensor):
        return _check_buffer(value)
    if ":" in value:
        # Standard
        import importlib

        mod, _, fn = value.partition(":")
        if nbytes is None:
            raise ValueError(f"extra_config[{EXTRA_CONFIG_KEY!r}] = {value!r} needs the tier size in bytes")
        buf = _check_buffer(getattr(importlib.import_module(mod), fn)(int(nbytes)))
        if buf.numel() < int(nbytes):
            raise ValueError(f"{value}({nbytes}) returned {buf.numel()} bytes, want >= {nbytes}")
        return buf
    with _registry_lock:
        if value not in _registry:
            raise KeyError(
                f"extra_config[{EXTRA_CONFIG_KEY!r}] = {value!r} names no registered buffer "
                f"(register_external_cpu_buffer before the LMCache engine is built)"
            )
        return _registry[value]


class ExternalBufferMemoryAllocator(MixedMemoryAllocator):
    """MixedMemoryAllocator whose pinned arena is a caller-owned buffer: objects are allocated in ``buffer`` (no
    allocation, no pinning, no free of the arena); BINARY_BUFFER objects still use the byte-array allocator."""

    def __init__(self, buffer: torch.Tensor, align_bytes: Optional[int] = None) -> None:
        # MixedMemoryAllocator.__init__ allocates its own arena: set its fields here instead
        self.numa_mapping = None
        self.use_hugepages = False
        self.shm_name = None
        self.align_bytes = AddressManager.ALIGN_BYTES if align_bytes is None else align_bytes
        if self.align_bytes <= 0 or self.align_bytes & (self.align_bytes - 1) != 0:
            raise ValueError("align_bytes must be a positive power of two")
        self.buffer = _check_buffer(buffer)
        self.size = self.buffer.numel()
        self._unregistered = False
        self.pin_allocator = TensorMemoryAllocator(self.buffer, align_bytes=self.align_bytes)
        self.host_mem_lock = threading.Lock()
        self.buffer_allocator = BufferAllocator("cpu")
        logger.info(
            "LMCache CPU tier over an external buffer: %.2f GB at 0x%x (align %d)",
            self.size / 1024**3,
            self.buffer.data_ptr(),
            self.align_bytes,
        )

    def close(self) -> None:
        """Drop the references to the caller's buffer (live memory objects still hold their own views)."""
        if self._unregistered:
            return
        empty = torch.empty(0, dtype=torch.uint8)
        with self.host_mem_lock:
            self.pin_allocator.buffer = empty
            self.buffer = empty
        self._unregistered = True

    def __str__(self) -> str:
        return "ExternalBufferMemoryAllocator"
