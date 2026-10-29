# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple, no_type_check
import asyncio
import os
import threading
import time

# Third Party
import aiofiles

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.protocol import RemoteMetadata
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)


class FSConnector(RemoteConnector):
    """File system based connector that stores data in local files.

    Data is stored in the following format:
    - Each key is stored as a separate file
    - File content: metadata (remote_metadata_bytes) + serialized data
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        config: Optional[LMCacheEngineConfig],
        plugin_name: Optional[str] = None,
        base_paths_str: Optional[str] = None,
    ):
        """
        Args:
            loop: Asyncio event loop
            local_cpu_backend: Memory allocator interface
            config: Lmcache engine config
            plugin_name: Plugin instance name
                (e.g. "fs", "fs.primary")
            base_paths_str: Comma-separated base paths
                (legacy, passed from adapter when using
                fs:// URL)
        """
        # initialize base class
        super().__init__(
            local_cpu_backend.config,
            local_cpu_backend.metadata,
        )

        base_path = base_paths_str
        if base_path is None:
            # Resolve from extra_config
            extra_config = config.extra_config if config else None
            if extra_config is not None:
                key_prefix = plugin_name or "fs"
                base_path = extra_config.get(
                    "remote_storage_plugin.%s.base_path" % key_prefix
                )
            if base_path is None:
                if extra_config is not None:
                    base_path = extra_config.get("fs_base_path")
            if base_path is None:
                raise ValueError(
                    "FS connector requires base_path via URL or extra_config"
                )

        # Parse comma separated paths
        self.base_paths = (
            [Path(p.strip()) for p in base_path.split(",")]
            if "," in base_path
            else [Path(base_path)]
        )

        self.loop = loop
        self.local_cpu_backend = local_cpu_backend

        relative_tmp_dir = (
            None
            if config is None
            else config.get_extra_config_value("fs_connector_relative_tmp_dir", None)
        )
        self.relative_tmp_dir = None
        if relative_tmp_dir is not None:
            self.relative_tmp_dir = Path(relative_tmp_dir)
            assert not self.relative_tmp_dir.is_absolute()

        self.read_ahead_size = (
            None
            if config is None
            else config.get_extra_config_value("fs_connector_read_ahead_size", None)
        )

        self.use_odirect = (
            False
            if config is None
            else config.get_extra_config_value("fs_connector_use_odirect", False)
        )
        self.os_disk_bs = 0
        if self.use_odirect:
            # save_chunk_meta is useful if save_unfull_chunk is True, since partial
            # chunk will be saved. When loading partial chunk, we need to know
            # data size and shape. However, chunk meta is short (28 bytes) which
            # is not aligned to disk block size (512 or 4096).
            # Therefore, we disable O_DIRECT if save_chunk_meta is True.
            # TODO: support O_DIRECT for save_chunk_meta by
            # padding meta data to 4096.
            if self.save_chunk_meta:
                logger.warning("Cannot use O_DIRECT if save_chunk_meta enabled.")
                self.use_odirect = False
            else:
                stat = os.statvfs(self.base_paths[0])
                self.os_disk_bs = stat.f_bsize

        # glm53 (Stage 12, program 2026-10-27T09:28:17Z, the post-1M admission
        # stall): every aiofiles call runs on the loop's DEFAULT executor, so the
        # 4+ jobs of each chunk write (a 1M store: ~8,200 chunks) sat in front of
        # every lookup's exists() and every get. Writes now run as ONE job per
        # chunk on this connector's own write pool, with at most
        # fs_connector_write_inflight_mb of chunks handed to it at a time (the
        # rest wait as coroutines). Existence checks answer from an in-memory
        # index of the keys this process wrote or found, else stat on a small
        # pool of their own, so a lookup never queues behind bulk I/O.
        # The index trusts THIS process: a file deleted by another actor while the
        # server runs answers "hit" until a get misses it (and drops it). Today the
        # only deleters are serve_window.sh's pre-launch prune and remove_sync; a
        # future runtime pruner must go through remove_sync.
        def _xcfg(name, default):
            return (
                default if config is None else config.get_extra_config_value(name, default)
            )

        self._write_threads = max(1, int(_xcfg("fs_connector_write_threads", 4)))
        self._write_cap = max(1, int(_xcfg("fs_connector_write_inflight_mb", 256))) << 20
        self._stat_threads = max(1, int(_xcfg("fs_connector_stat_threads", 4)))
        self._write_pool = ThreadPoolExecutor(
            max_workers=self._write_threads, thread_name_prefix="lmc-fs-write"
        )
        self._stat_pool = ThreadPoolExecutor(
            max_workers=self._stat_threads, thread_name_prefix="lmc-fs-stat"
        )
        self._write_inflight = 0
        # FIFO admission to the write pool, all on the loop thread: a chunk waits
        # as (size, future) and the completing write admits from the head (one
        # wake-up per admitted chunk; a notify_all over thousands of waiting
        # puts was quadratic)
        self._write_waiters: deque = deque()
        self._known: set = set()
        self._known_lock = threading.Lock()  # also guards fs_stats (stat-pool threads count)
        self.fs_stats = {
            "index_hits": 0,
            "stats": 0,
            "stat_hits": 0,
            "writes": 0,
            "write_waits": 0,
            "write_inflight_max": 0,
        }

        logger.info(
            "Initialized FSConnector with base paths %s, "
            "relative tmp dir: %s, "
            "read ahead size: %s, "
            "use O_DIRECT: %s, "
            "write pool %d threads / %d MiB in flight, stat pool %d threads",
            self.base_paths,
            self.relative_tmp_dir,
            self.read_ahead_size,
            self.use_odirect,
            self._write_threads,
            self._write_cap >> 20,
            self._stat_threads,
        )
        # Create directories for all paths
        for path in self.base_paths:
            path.mkdir(parents=True, exist_ok=True)
            if self.relative_tmp_dir is not None:
                (path / self.relative_tmp_dir).mkdir(parents=False, exist_ok=True)
        # glm53: the temp files of writes a killed process left behind, swept once
        # in the background (a directory listing on NFS can take a while); only
        # those older than fs_connector_tmp_sweep_min minutes (0 = off), so a live
        # writer's in-flight temp file is never touched
        # glm53 (program 2026-10-28T18:59:10Z / 19:01:29Z: the fleet's cold tier had no size cap and grew ~500 GB in
        # 23 min under one users16 cell): an OPTIONAL cap, evicting through THIS connector so its index never answers
        # "hit" for a file it removed (an external pruner cannot: a request whose promised chunk vanished fails under
        # kv_load_failure_policy "fail"). GLM53_LMCACHE_REMOTE_GB (extra_config fs_connector_cap_gb): bytes of chunk
        # files this connector's dirs may hold; GLM53_LMCACHE_REMOTE_FREE_FLOOR_GB (fs_connector_free_floor_gb): evict
        # while the filesystem's free space is under it. Both 0 = off (today's behaviour). Least recently used first
        # (put, a lookup's index / stat hit and a get count as use; the files found at start by their mtime), never a
        # file used in the last GLM53_LMCACHE_REMOTE_GRACE_S (fs_connector_evict_grace_s, 120): a lookup that just
        # promised a chunk always finds it at its get. Down to 90 pct of the cap / up to 110 pct of the floor. Assumes
        # ONE engine per remote dir (the fleet's per-block dirs, the mesh's per-form stable dirs).
        def _env_or_cfg(env, name, default):
            v = os.environ.get(env)
            if v not in (None, ""):
                try:
                    return float(v)
                except ValueError:
                    logger.warning("FSConnector: ignoring %s=%r", env, v)
            return float(_xcfg(name, default))

        self._cap_bytes = int(_env_or_cfg("GLM53_LMCACHE_REMOTE_GB", "fs_connector_cap_gb", 0) * 1e9)
        self._floor_bytes = int(_env_or_cfg("GLM53_LMCACHE_REMOTE_FREE_FLOOR_GB", "fs_connector_free_floor_gb", 0) * 1e9)
        self._grace_s = _env_or_cfg("GLM53_LMCACHE_REMOTE_GRACE_S", "fs_connector_evict_grace_s", 120)
        self._lru_on = self._cap_bytes > 0 or self._floor_bytes > 0
        self._lru: OrderedDict = OrderedDict()  # path -> [bytes, last use (monotonic)], least recent first
        self._lru_bytes = 0
        self._evicting: set = set()  # paths between their index drop and their unlink: a stat answers "miss"
        self._evict_kick = threading.Event()
        self._closed = False
        self.fs_stats.update(evicted_files=0, evicted_bytes=0, evict_passes=0, lru_scanned=0)
        if self._lru_on:
            logger.info("FSConnector size cap: %.1f GB, free-space floor %.1f GB, grace %.0f s (LRU eviction)",
                        self._cap_bytes / 1e9, self._floor_bytes / 1e9, self._grace_s)  # fmt: skip
            threading.Thread(target=self._evict_loop, name="lmc-fs-evict", daemon=True).start()
        sweep_min = float(_xcfg("fs_connector_tmp_sweep_min", 10))
        if sweep_min > 0:
            threading.Thread(
                target=self._sweep_tmp, args=(sweep_min * 60,), name="lmc-fs-tmp-sweep",
                daemon=True,
            ).start()

    def _get_base_path(self, key: CacheEngineKey) -> Path:
        """Get file base path for the given key"""
        if len(self.base_paths) == 1:
            base_path = self.base_paths[0]
        else:
            # Calculate hash value and modulo to select path
            hash_val = abs(key.chunk_hash)
            idx = hash_val % len(self.base_paths)
            base_path = self.base_paths[idx]

        return base_path

    def _get_file_name(self, key: CacheEngineKey) -> str:
        return key.to_string().replace("/", "-SEP-") + ".data"

    def _get_file_path(self, key: CacheEngineKey) -> Path:
        """Get file path for the given key"""
        base_path = self._get_base_path(key)
        file_name = self._get_file_name(key)
        return base_path / file_name

    def _get_file_and_tmp_path(self, key: CacheEngineKey) -> Tuple[Path, Path]:
        """Get file and tmp path for the given key"""
        base_path = self._get_base_path(key)
        file_name = self._get_file_name(key)
        file_path = base_path / file_name
        if self.relative_tmp_dir is not None:
            tmp_path = base_path / self.relative_tmp_dir / file_name
        else:
            tmp_path = file_path.with_suffix(".tmp")
        return file_path, tmp_path

    # -- glm53: the key index + the stat pool (see __init__) --------------------

    def _sweep_tmp(self, min_age_s: float) -> None:
        dirs = []
        for base in self.base_paths:
            dirs.append(base if self.relative_tmp_dir is None else base / self.relative_tmp_dir)
        now, removed = time.time(), 0
        for d in dirs:
            try:
                with os.scandir(d) as it:
                    for e in it:
                        if not e.name.endswith(".tmp"):
                            continue
                        try:
                            if now - e.stat().st_mtime >= min_age_s:
                                os.unlink(e.path)
                                removed += 1
                        except OSError:
                            pass
            except OSError as e:
                logger.warning("FSConnector temp sweep of %s failed: %s", d, e)
        if removed:
            logger.info("FSConnector removed %d stale temp files (older than %.0f s)", removed, min_age_s)
        with self._known_lock:
            self.fs_stats["tmp_swept"] = removed

    def _index_has(self, file_path: Path) -> bool:
        with self._known_lock:
            hit = str(file_path) in self._known
            if hit and self._lru_on:
                self._lru_touch(str(file_path))
            return hit

    def _index_add(self, file_path: Path, size: Optional[int] = None) -> None:
        with self._known_lock:
            self._known.add(str(file_path))
            if self._lru_on:
                self._lru_touch(str(file_path), size)

    def _index_drop(self, file_path: Path) -> None:
        with self._known_lock:
            self._known.discard(str(file_path))
            if self._lru_on:
                self._lru_forget(str(file_path))

    # -- glm53: the optional size cap (see __init__); every _lru_* runs under _known_lock --

    def _lru_touch(self, path: str, size: Optional[int] = None) -> None:
        e = self._lru.get(path)
        if e is None:
            if size is None:
                return  # (a size comes with its put, its stat or the start scan)
            self._lru[path] = [int(size), time.monotonic()]
            self._lru_bytes += int(size)
        else:
            if size is not None and int(size) != e[0]:
                self._lru_bytes += int(size) - e[0]
                e[0] = int(size)
            e[1] = time.monotonic()
            self._lru.move_to_end(path)
        if (self._cap_bytes > 0 and self._lru_bytes > self._cap_bytes) and not self._evict_kick.is_set():
            self._evict_kick.set()

    def _lru_forget(self, path: str) -> None:
        e = self._lru.pop(path, None)
        if e is not None:
            self._lru_bytes -= e[0]

    def _free_bytes(self) -> int:
        st = os.statvfs(self.base_paths[0])
        return int(st.f_bavail) * int(st.f_frsize)

    def _scan_existing(self) -> None:
        """the chunk files already on disk (an earlier engine's), oldest mtime first, at the LRU's head"""
        found = []
        for base in self.base_paths:
            try:
                with os.scandir(base) as it:
                    for e in it:
                        if e.name.endswith(".data"):
                            try:
                                st = e.stat()
                                found.append((st.st_mtime, e.path, st.st_size))
                            except OSError:
                                pass
            except OSError as e:
                logger.warning("FSConnector cap: scan of %s failed: %s", base, e)
        found.sort()
        mono, wall = time.monotonic(), time.time()
        with self._known_lock:
            fresh: OrderedDict = OrderedDict()
            for mt, path, size in found:
                if path not in self._lru:
                    fresh[path] = [int(size), mono - max(0.0, wall - mt)]
            for path, v in self._lru.items():  # what this process used meanwhile: more recent
                fresh[path] = v
            self._lru = fresh
            self._lru_bytes = sum(v[0] for v in fresh.values())
            self.fs_stats["lru_scanned"] = len(found)
        self._evict_kick.set()

    def _evict_loop(self) -> None:
        self._scan_existing()
        while not self._closed:
            self._evict_kick.wait(30.0)  # (the floor is re-checked every 30 s without a put)
            self._evict_kick.clear()
            if self._closed:
                return
            try:
                self._evict_pass()
            except Exception as e:  # noqa: BLE001 -- never kill the eviction thread
                logger.warning("FSConnector cap: eviction pass failed: %s", e)

    def _evict_pass(self) -> None:
        free = self._free_bytes() if self._floor_bytes > 0 else None
        with self._known_lock:
            over_cap = self._cap_bytes > 0 and self._lru_bytes > self._cap_bytes
        under_floor = free is not None and free < self._floor_bytes
        if not (over_cap or under_floor):
            return
        t0, files, freed, freed_total = time.monotonic(), 0, 0, 0
        while not self._closed:
            with self._known_lock:
                done_cap = not over_cap or self._lru_bytes <= int(self._cap_bytes * 0.9)
                done_floor = not under_floor or (free is not None and free + freed >= int(self._floor_bytes * 1.1))
                if (done_cap and done_floor) or not self._lru:
                    break
                path, (size, used) = next(iter(self._lru.items()))
                if time.monotonic() - used < self._grace_s:
                    break  # the rest were used even more recently
                del self._lru[path]
                self._lru_bytes -= size
                self._known.discard(path)  # the index first: no lookup promises it from here on
                self._evicting.add(path)
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            except OSError as e:
                logger.warning("FSConnector cap: could not remove %s: %s", path, e)
            finally:
                with self._known_lock:
                    self._evicting.discard(path)
            files += 1
            freed += size
            freed_total += size
            if under_floor and files % 256 == 0:
                free, freed = self._free_bytes(), 0
        with self._known_lock:
            self.fs_stats["evicted_files"] += files
            self.fs_stats["evicted_bytes"] += freed_total
            self.fs_stats["evict_passes"] += 1
            left = self._lru_bytes
        logger.info("FSConnector cap: evicted %d files in %.1f s; %.2f GB held (cap %.1f GB)%s", files,
                    time.monotonic() - t0, left / 1e9, self._cap_bytes / 1e9,
                    "" if free is None else f"; free {free / 1e9:.1f} GB (floor {self._floor_bytes / 1e9:.1f} GB)")  # fmt: skip

    def _count(self, name: str, n: int = 1) -> None:
        with self._known_lock:
            self.fs_stats[name] += n

    def _stat(self, file_path: Path) -> bool:
        """one existence check off every bulk I/O queue (a stat-pool thread)"""
        try:
            size = os.stat(file_path).st_size
            ok = True
        except OSError:
            size, ok = None, False
        with self._known_lock:
            self.fs_stats["stats"] += 1
            if ok and str(file_path) in self._evicting:
                ok = False  # (the size cap is removing it: a miss, never a promise it cannot keep)
            if ok:
                self.fs_stats["stat_hits"] += 1
                self._known.add(str(file_path))
                if self._lru_on:
                    self._lru_touch(str(file_path), size)
        return ok

    async def exists(self, key: CacheEngineKey) -> bool:
        """Check if key exists in file system"""
        file_path = self._get_file_path(key)
        if self._index_has(file_path):
            self._count("index_hits")
            return True
        return await self.loop.run_in_executor(self._stat_pool, self._stat, file_path)

    def exists_sync(self, key: CacheEngineKey) -> bool:
        """Check if key exists in file system synchronized"""
        file_path = self._get_file_path(key)
        if self._index_has(file_path):
            self._count("index_hits")
            return True
        return self._stat(file_path)

    def _index_prefix(self, keys: List[CacheEngineKey]) -> int:
        n = 0
        while n < len(keys) and self._index_has(self._get_file_path(keys[n])):
            n += 1
        self._count("index_hits", n)
        return n

    _STAT_BATCH = 64

    def support_batched_contains(self) -> bool:
        return True

    def batched_contains(self, keys: List[CacheEngineKey]) -> int:
        """the prefix of keys that exist: the index first, then stats in batches
        of 64 on the stat pool up to the first miss (the lookup's sync path)"""
        n = self._index_prefix(keys)
        while n < len(keys):
            batch = keys[n : n + self._STAT_BATCH]
            futs = [
                self._stat_pool.submit(self._stat, self._get_file_path(k)) for k in batch
            ]
            for fut in futs:
                if not fut.result():
                    return n
                n += 1
        return n

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        """batched_contains for the async lookup path (prefix semantics)"""
        n = self._index_prefix(keys)
        while n < len(keys):
            batch = keys[n : n + self._STAT_BATCH]
            res = await asyncio.gather(
                *[
                    self.loop.run_in_executor(
                        self._stat_pool, self._stat, self._get_file_path(k)
                    )
                    for k in batch
                ]
            )
            for ok in res:
                if not ok:
                    return n
                n += 1
        return n

    def _get_with_odirect(self, file_path: Path) -> Optional[MemoryObj]:
        """Synchronous direct IO read, executed in a thread."""
        fd = -1
        memory_obj: Optional[MemoryObj] = None
        try:
            memory_obj = self.local_cpu_backend.allocate(
                self.meta_shapes, self.meta_dtypes, self.meta_fmt
            )
            if memory_obj is None:
                logger.debug("Memory allocation failed.")
                return None

            buffer = memory_obj.byte_array
            size = len(buffer)

            fblock_aligned = (
                self.os_disk_bs is not None
                and self.os_disk_bs > 0
                and size % self.os_disk_bs == 0
            )
            if not fblock_aligned:
                logger.warning(
                    "Cannot use O_DIRECT for %s, size is not aligned.", file_path
                )
                with open(file_path, "rb") as f:
                    num_read = f.readinto(buffer)
            else:
                fd = os.open(file_path, os.O_RDONLY | getattr(os, "O_DIRECT", 0))
                with os.fdopen(fd, "rb", buffering=0) as fdo:
                    # The fd is now managed by the file object, so we "forget" it
                    # to prevent closing it in the finally block.
                    fd = -1
                    num_read = fdo.readinto(buffer)

            memory_obj = self.reshape_partial_chunk(memory_obj, num_read)
            return memory_obj

        except Exception as e:
            logger.error("Failed to read from file %s: %s", file_path, str(e))
            if memory_obj is not None:
                memory_obj.ref_count_down()
            return None
        finally:
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Get data from file system"""
        file_path = self._get_file_path(key)

        if self.use_odirect and not self.save_chunk_meta:
            return await self.loop.run_in_executor(
                None, self._get_with_odirect, file_path
            )

        memory_obj = None
        try:
            async with aiofiles.open(file_path, "rb") as f:
                if self.save_chunk_meta:
                    # Read metadata buffer first to get shape, dtype, fmt
                    # to be able to allocate memory object for the data and read into it
                    md_buffer = bytearray(self.remote_metadata_bytes)
                    num_read = await f.readinto(md_buffer)
                    if num_read != len(md_buffer):
                        raise RuntimeError(
                            f"Partial read meta {len(md_buffer)} got {num_read}"
                        )

                    # Deserialize metadata and allocate memory
                    metadata = RemoteMetadata.deserialize(md_buffer)
                    memory_obj = self.local_cpu_backend.allocate(
                        metadata.shapes, metadata.dtypes, metadata.fmt
                    )
                else:
                    memory_obj = self.local_cpu_backend.allocate(
                        self.meta_shapes, self.meta_dtypes, self.meta_fmt
                    )
                if memory_obj is None:
                    logger.debug("Memory allocation failed during async disk load.")
                    return None

                # Read the actual data into allocated memory
                buffer = memory_obj.byte_array
                if self.save_chunk_meta:
                    # if save chunk meta, read meta will trigger
                    # read ahead if fs supported
                    num_read = await f.readinto(buffer)
                    if num_read != len(buffer):
                        raise RuntimeError(
                            f"Partial read data {len(buffer)} got {num_read}"
                        )
                else:
                    if self.read_ahead_size is None:
                        num_read = await f.readinto(buffer)
                    else:
                        if not isinstance(buffer, memoryview):
                            buffer = memoryview(buffer)

                        # trigger read head if fs supported
                        num_read_ahead = await f.readinto(
                            buffer[: self.read_ahead_size]
                        )
                        assert num_read_ahead <= self.read_ahead_size

                        # if num_read_ahead == self.read_ahead_size,
                        # means there may still be some remaining content
                        if num_read_ahead == self.read_ahead_size:
                            num_read_tail = await f.readinto(
                                buffer[self.read_ahead_size :]
                            )
                            assert num_read_tail is not None
                            num_read = num_read_ahead + num_read_tail
                        else:
                            num_read = num_read_ahead
                    # reshape and check
                    assert num_read is not None
                    memory_obj = self.reshape_partial_chunk(memory_obj, num_read)

            if self._lru_on:
                with self._known_lock:
                    self._lru_touch(str(file_path))
            return memory_obj

        except Exception as e:
            if not isinstance(e, FileNotFoundError):
                logger.error("Failed to read from file %s: %s", file_path, str(e))
            else:
                self._index_drop(file_path)  # removed under us: not a hit any more
            if memory_obj is not None:
                memory_obj.ref_count_down()
            return None

    def _put_with_odirect(self, file_path: Path, buffer: bytes) -> None:
        fd = -1
        try:
            fd = os.open(
                str(file_path),
                os.O_CREAT | os.O_WRONLY | getattr(os, "O_DIRECT", 0),
                0o644,
            )
            os.write(fd, buffer)
        except Exception as e:
            logger.error("Failed to write to file %s: %s", file_path, e)
            raise
        finally:
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass

    def _write_admit(self, size: int) -> None:
        self._write_inflight += size
        if self._write_inflight > self.fs_stats["write_inflight_max"]:
            self.fs_stats["write_inflight_max"] = self._write_inflight

    def _write_fits(self, size: int) -> bool:
        return self._write_inflight == 0 or self._write_inflight + size <= self._write_cap

    async def _write_slot(self, size: int) -> None:
        """wait (FIFO) until the chunk fits the in-flight cap (one chunk always
        fits); on return its bytes are counted in flight"""
        if not self._write_waiters and self._write_fits(size):
            self._write_admit(size)
            return
        fut = self.loop.create_future()
        self._write_waiters.append((size, fut))
        self._count("write_waits")
        try:
            await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():
                self._write_done(size)  # admitted, then cancelled: give the bytes back
            raise

    def _write_done(self, size: int) -> None:
        """a chunk left the write pool: admit waiting chunks from the head"""
        self._write_inflight -= size
        while self._write_waiters:
            sz, fut = self._write_waiters[0]
            if fut.cancelled():
                self._write_waiters.popleft()
                continue
            if not self._write_fits(sz):
                break
            self._write_waiters.popleft()
            self._write_admit(sz)
            fut.set_result(None)

    def _put_sync(self, final_path: Path, temp_path: Path, memory_obj: MemoryObj) -> None:
        """the whole chunk write on a write-pool thread: the file (metadata +
        data) at the temp path, then the atomic rename"""
        try:
            buffer = memory_obj.byte_array
            metadata = (
                RemoteMetadata(
                    len(buffer),
                    memory_obj.get_shapes(),
                    memory_obj.get_dtypes(),
                    memory_obj.get_memory_format(),
                )
                if self.save_chunk_meta
                else None
            )
            size = len(buffer)
            do_use_odirect = self.use_odirect
            if do_use_odirect:
                fblock_aligned = self.os_disk_bs > 0 and size % self.os_disk_bs == 0
                if not fblock_aligned:
                    logger.warning(
                        "Cannot use O_DIRECT for writing size %s, "
                        "which is not aligned to block size %s.",
                        size,
                        self.os_disk_bs,
                    )
                    do_use_odirect = False
            if do_use_odirect:
                self._put_with_odirect(temp_path, buffer)
            else:
                with open(temp_path, "wb") as f:
                    if metadata is not None:
                        f.write(metadata.serialize())
                    f.write(buffer)
            os.replace(temp_path, final_path)
        except Exception as e:
            logger.error("Failed to write file %s: %s", final_path, str(e))
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)  # Remove corrupted file
            except OSError:
                pass
            raise

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        """Store data to file system"""
        final_path, temp_path = self._get_file_and_tmp_path(key)
        size = len(memory_obj.byte_array)
        await self._write_slot(size)
        try:
            await self.loop.run_in_executor(
                self._write_pool, self._put_sync, final_path, temp_path, memory_obj
            )
            self._index_add(final_path, size)
            self._count("writes")
        finally:
            self._write_done(size)

    def remove_sync(self, key: CacheEngineKey) -> bool:
        """
        Remove the file associated with the given key.

        Args:
            key: The key to remove.

        Returns:
            bool: True if the file was successfully removed, False otherwise.
        """
        file_path = self._get_file_path(key)
        self._index_drop(file_path)
        try:
            os.remove(file_path)
            return True
        except OSError as e:
            logger.error("Failed to remove file %s: %s", file_path, e)
            return False

    @no_type_check
    async def list(self) -> List[str]:
        """List all keys in file system"""
        keys = []
        for base_path in self.base_paths:
            keys.extend([f.stem for f in base_path.glob("*.data")])
        return keys

    async def close(self):
        """Clean up resources"""
        self._closed = True
        self._evict_kick.set()
        self._write_pool.shutdown(wait=False)
        self._stat_pool.shutdown(wait=False)
        logger.info("Closed the file system connector (stats %s)", self.fs_stats)
