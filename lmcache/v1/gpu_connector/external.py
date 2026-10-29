# SPDX-License-Identifier: Apache-2.0
"""Externally provided vLLM GPU connectors (host-driven accelerators).

Some accelerators drive the device from the host with a CPU-only torch build
(e.g. Tenstorrent through ``vllm-tt-plugin``), so ``torch_device_type`` is
``"cpu"`` and no in-tree connector applies. The serving plugin names its
connector class as ``"module:Class"`` in ``extra_config["gpu_connector"]`` (or
the ``LMCACHE_GPU_CONNECTOR`` environment variable); the class must implement
``GPUConnectorInterface`` and a ``from_metadata(metadata, use_gpu, device,
layout_hints=...)`` factory, as the in-tree vLLM connectors do.
"""

# Standard
from typing import Any, Optional
import importlib
import os

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector.gpu_connectors import GPUConnectorInterface
from lmcache.v1.metadata import LMCacheMetadata

logger = init_logger(__name__)

EXTRA_CONFIG_KEY = "gpu_connector"
ENV_VAR = "LMCACHE_GPU_CONNECTOR"


def resolve_external_connector_spec(config: LMCacheEngineConfig) -> Optional[str]:
    """``"module:Class"`` from extra_config or the environment, or ``None``."""
    spec = None
    if config.extra_config is not None:
        spec = config.extra_config.get(EXTRA_CONFIG_KEY)
    return spec or os.environ.get(ENV_VAR) or None


def create_external_vllm_connector(
    config: LMCacheEngineConfig,
    metadata: LMCacheMetadata,
    use_gpu: bool,
    device: torch.device,
    layout_hints: Any = None,
) -> Optional[GPUConnectorInterface]:
    """Instantiate the configured external connector, or return ``None``."""
    spec = resolve_external_connector_spec(config)
    if not spec:
        return None
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr:
        raise ValueError(
            f"{EXTRA_CONFIG_KEY} / {ENV_VAR} must be 'module:Class', got {spec!r}"
        )
    cls = getattr(importlib.import_module(module_name), attr)
    if not (isinstance(cls, type) and issubclass(cls, GPUConnectorInterface)):
        raise TypeError(f"{spec} is not a GPUConnectorInterface subclass")
    logger.info("Using external GPU connector %s", spec)
    return cls.from_metadata(metadata, use_gpu, device, layout_hints=layout_hints)
