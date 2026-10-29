# SPDX-License-Identifier: Apache-2.0
"""glm53 (int27m28c2, 2026-10-22): a process whose LMCache engine was built but never closed (a failed vLLM engine
init) must still EXIT -- no non-daemon LMCache thread may park the interpreter in threading._shutdown (the TT mesh
stayed open there: an unclean device close). Runs the construction in a subprocess and bounds its exit."""

import subprocess
import sys
import textwrap

CHILD = textwrap.dedent(
    """
    import threading, torch
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.event_manager import EventManager
    from lmcache.v1.metadata import LMCacheMetadata
    from lmcache.v1.storage_backend.storage_manager import StorageManager
    cfg = LMCacheEngineConfig.from_defaults(chunk_size=128, local_cpu=True, max_local_cpu_size=0.01,
                                            lmcache_instance_id="glm53_failed_init")
    md = LMCacheMetadata(model_name="m", world_size=1, local_world_size=1, worker_id=0, local_worker_id=0,
                         kv_dtype=torch.bfloat16, kv_shape=(11, 1, 128, 1, 544), role="worker", use_mla=True)
    sm = StorageManager(config=cfg, metadata=md, event_manager=EventManager())
    live = [t for t in threading.enumerate() if t is not threading.main_thread() and t.is_alive()]
    print("NONDAEMON", sorted(t.name for t in live if not t.daemon), flush=True)
    # no sm.close(): the failed-init path; the interpreter must exit by itself
    """
)


def test_unclosed_storage_manager_lets_the_process_exit():
    r = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-2000:]
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("NONDAEMON")]
    assert line == ["NONDAEMON []"], (line, r.stderr[-2000:])
