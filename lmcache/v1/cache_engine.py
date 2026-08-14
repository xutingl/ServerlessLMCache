# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.health_monitor.base import HealthMonitor

# Standard
import asyncio
import copy
import gc
import multiprocessing
import os
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.observability import LMCacheStatsLogger, LMCStatsMonitor
from lmcache.request_metrics_bridge import log_lmcache_count_metric
from lmcache.usage_context import InitializeUsageContext
from lmcache.utils import (
    CacheEngineKey,
    CacheStoreEvent,
    _lmcache_nvtx_annotate,
    compress_slot_mapping,
    convert_tokens_to_list,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventManager, EventStatus, EventType
from lmcache.v1.gpu_connector.gpu_connectors import (
    GPUConnectorInterface,
    LayerwiseOffloadHandle,
    LayerwiseRequestBatch,
)
from lmcache.v1.gpu_connector.utils import assert_layerwise_gpu_connector
from lmcache.v1.memory_management import CuFileMemoryAllocator  # noqa: E501
from lmcache.v1.memory_management import (  # noqa: E501
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    MixedMemoryAllocator,
    PagedTensorMemoryAllocator,
    TensorMemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.pin_monitor import PinMonitor
from lmcache.v1.storage_backend.storage_manager import StorageManager
from lmcache.v1.system_detection import NUMADetector, NUMAMapping
from lmcache.v1.token_database import (
    ChunkedTokenDatabase,
    LOOKUP_CHUNK_LENGTHS_CONFIG,
    SAVE_CHUNK_LENGTHS_CONFIG,
    SegmentTokenDatabase,
    TokenDatabase,
    extract_chunk_lengths,
)

logger = init_logger(__name__)


def _layerwise_put_workers() -> int:
    value = os.environ.get("PD_BACKEND_LAYERWISE_PUT_WORKERS")
    if value is None:
        return 1
    workers = int(value)
    if workers < 1:
        raise ValueError("PD_BACKEND_LAYERWISE_PUT_WORKERS must be positive")
    return workers


# Type aliases for processed chunks
# (cache_key, memory_obj, start_index, end_index)
ProcessedChunk = Tuple[CacheEngineKey, MemoryObj, int, int]
# (list of processed chunks, total kv size)
ProcessTokensInternalResult = Tuple[List[ProcessedChunk], int]


def _chunk_ids_for_spans(
    starts: list[int],
    ends: list[int],
    *,
    chunk_size: int,
    chunk_lengths: Optional[list[int]],
) -> list[int]:
    if chunk_lengths is None:
        return [start // chunk_size for start in starts]

    boundaries: dict[int, tuple[int, int]] = {}
    start = 0
    for chunk_id, length in enumerate(chunk_lengths):
        end = start + length
        boundaries[start] = (chunk_id, end)
        start = end

    chunk_ids: list[int] = []
    for span_start, span_end in zip(starts, ends, strict=False):
        chunk_info = boundaries.get(span_start)
        if chunk_info is None:
            raise ValueError(
                "Stored span does not align with custom chunk boundaries: "
                f"start={span_start}, chunk_lengths={chunk_lengths}"
            )
        chunk_id, expected_end = chunk_info
        if span_end != expected_end:
            raise ValueError(
                "Stored span end does not align with custom chunk boundaries: "
                f"span=({span_start}, {span_end}), expected_end={expected_end}, "
                f"chunk_lengths={chunk_lengths}"
            )
        chunk_ids.append(chunk_id)
    return chunk_ids


def _transfer_spec_with_chunk_ids(
    transfer_spec: Any,
    chunk_ids: list[int],
) -> Any:
    if transfer_spec is None:
        return None
    cloned_spec = copy.copy(transfer_spec)
    cloned_spec.chunk_ids = list(chunk_ids)
    return cloned_spec


class CacheEngineEndSignal:
    pass


class _LayerwiseStoreBatchTask:
    """One request's put work for a shared layer tile."""

    def __init__(
        self,
        wait_ready: Callable[[], None],
        put: Callable[[], None],
        abort: Callable[[], None],
    ) -> None:
        self.wait_ready = wait_ready
        self.put = put
        self.abort = abort


class LayerwiseStoreBatch:
    """Submit one cache-engine task for a layer tile spanning requests."""

    def __init__(
        self,
        gpu_batch: LayerwiseRequestBatch,
        executor: ThreadPoolExecutor,
    ) -> None:
        self.gpu_batch = gpu_batch
        self.group_size = gpu_batch.group_size
        self._executor = executor
        self._lock = threading.Lock()
        self._pending: list[_LayerwiseStoreBatchTask] = []
        self._pending_completion: Future[None] = Future()
        self._error: Optional[BaseException] = None

    def register_put_task(self, task: _LayerwiseStoreBatchTask) -> Future[None]:
        with self._lock:
            if self._error is not None:
                raise RuntimeError("layerwise store batch is aborted") from self._error
            self._pending.append(task)
            return self._pending_completion

    @staticmethod
    def _abort_tasks(tasks: list[_LayerwiseStoreBatchTask]) -> None:
        if not tasks:
            return
        try:
            tasks[0].wait_ready()
        except BaseException:
            pass
        for task in tasks:
            try:
                task.abort()
            except BaseException:
                logger.exception("Failed to clean up an aborted layerwise batch task")

    @staticmethod
    def _run_tasks(tasks: list[_LayerwiseStoreBatchTask]) -> None:
        try:
            # Every task in this tile resolves to the same shared D2H event.
            tasks[0].wait_ready()
        except BaseException:
            LayerwiseStoreBatch._abort_tasks(tasks)
            raise

        first_error: Optional[BaseException] = None
        for task in tasks:
            try:
                task.put()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    @staticmethod
    def _resolve_completion(
        submitted: Future[None],
        completion: Future[None],
    ) -> None:
        try:
            submitted.result()
        except BaseException as exc:
            completion.set_exception(exc)
        else:
            completion.set_result(None)

    def flush(self) -> int:
        with self._lock:
            tasks = self._pending
            completion = self._pending_completion
            self._pending = []
            self._pending_completion = Future()
            error = self._error

        if error is not None:
            raise RuntimeError("layerwise store batch is aborted") from error

        try:
            entry_count = self.gpu_batch.flush()
            if entry_count != len(tasks):
                raise RuntimeError(
                    "layerwise store batch registration mismatch: "
                    f"gpu_entries={entry_count}, put_tasks={len(tasks)}"
                )
            if not tasks:
                completion.set_result(None)
                return 0
            submitted = self._executor.submit(self._run_tasks, tasks)
        except BaseException as exc:
            self.gpu_batch.abort(exc)
            self._abort_tasks(tasks)
            completion.set_exception(exc)
            raise

        submitted.add_done_callback(
            lambda done: self._resolve_completion(done, completion)
        )
        return entry_count

    def abort(self, error: BaseException) -> None:
        with self._lock:
            if self._error is None:
                self._error = error
            tasks = self._pending
            completion = self._pending_completion
            self._pending = []
            self._pending_completion = Future()
        self.gpu_batch.abort(error)
        self._abort_tasks(tasks)
        if tasks and not completion.done():
            completion.set_exception(error)


class LMCacheEngine:
    """The main class for the cache engine.

    When storing the KV caches into the cache engine, it takes GPU KV
    caches from the serving engine and convert them into MemoryObjs that
    resides in the CPU. The MemoryObjs are then being stored into the
    StorageBackends in an asynchronous manner.

    When retrieving the KV caches from the cache engine, it fetches the
    MemoryObjs from the StorageBackends and convert them into GPU KV caches
    by GPUConnectors specialized for the serving engine.

    It also supports prefetching the KV caches from the StorageBackends.
    It relies on the StorageBackends to manage the requests of prefetching
    and real retrieval and avoid the conflicts.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        token_database: TokenDatabase,
        gpu_connector: Optional[GPUConnectorInterface],
        broadcast_fn: Callable[[torch.Tensor, int], None],
        broadcast_object_fn: Callable[[Any, int], Any],
    ):
        logger.info(f"Creating LMCacheEngine with config: {config}")
        self.config = config
        self.metadata = metadata
        self.token_database = token_database
        self.gpu_connector = gpu_connector
        self.broadcast_fn = broadcast_fn
        self.broadcast_object_fn = broadcast_object_fn
        # save_only_first_rank only works when use mla
        self.save_only_first_rank = (
            self.config.get_extra_config_value("save_only_first_rank", metadata.use_mla)
            and metadata.use_mla
        )

        if self.save_only_first_rank and self.gpu_connector is not None:
            self.broadcast_stream = (
                self.gpu_connector.load_stream
                if hasattr(self.gpu_connector, "load_stream")
                else torch.cuda.Stream()
            )

        self.enable_controller = config.enable_controller

        # NOTE: Unix systems use fork by default
        multiprocessing.set_start_method("spawn", force=True)

        # avoid circular import
        # First Party
        from lmcache.v1.cache_controller import LMCacheWorker

        self.lmcache_worker: Optional[LMCacheWorker] = None
        lmcache_worker_ids = config.get_lmcache_worker_ids(
            metadata.use_mla, metadata.world_size
        )
        # lmcache_worker_ids is empty means start on all workers
        if (
            self.enable_controller
            and self.metadata.role != "scheduler"
            and (not lmcache_worker_ids or metadata.worker_id in lmcache_worker_ids)
        ):
            self.lmcache_worker = LMCacheWorker(config, metadata, self)
        else:
            self.lmcache_worker = None
            logger.info(
                "LMCacheWorker is not initialized (related configs: "
                "enable_controller: %s, role: %s, worker_id: %s, worker_ids: %s).",
                self.enable_controller,
                self.metadata.role,
                self.metadata.worker_id,
                lmcache_worker_ids,
            )

        self.async_loading = config.enable_async_loading
        self.event_manager = EventManager()

        self.use_layerwise = config.use_layerwise
        self._layerwise_put_executor: Optional[ThreadPoolExecutor] = None
        if self.use_layerwise:
            layerwise_put_workers = _layerwise_put_workers()
            logger.info(
                "Creating layerwise put executor with max_workers=%d",
                layerwise_put_workers,
            )
            self._layerwise_put_executor = ThreadPoolExecutor(
                max_workers=layerwise_put_workers,
                thread_name_prefix="lmcache-layer-put",
            )

        # TODO: support save_only_first_rank when use layerwise
        # if use_layerwise is True, all ranks will initialize the storage_manager
        # if save_only_first_rank is False, all ranks will initialize
        # the storage_manager
        # if save_only_first_rank is True, only the first rank and
        # lookup server workers will initialize the storage_manager
        self.storage_manager: Optional[StorageManager] = None

        # KV events
        self.kv_events_enabled = False
        self.kv_events_enabled = config.enable_kv_events
        if self.kv_events_enabled:
            self.kv_events: List[CacheStoreEvent] = []
            logger.info("KV events are enabled.")
        else:
            logger.info("KV events are disabled.")

        # HACK: remove this in the future
        # NOTE (Jiayi): This is currently used to support
        # dropping the kv cache from the buffer in PD backend
        # at decoder.
        # PD receiver-side objects are one-shot handoff buffers. In kv_both mode
        # this worker can also receive PD objects, so they must be removed after
        # retrieve just like the pure receiver role.
        self.remove_after_retrieve = config.enable_pd and config.pd_role in (
            "receiver",
            "both",
        )

        # asymmetric store/retrieve location can be specified
        # this is typically used (but not limited) in PD system
        self.store_location = config.store_location
        self.retrieve_locations = config.retrieve_locations

        self.num_layers = metadata.kv_shape[0]
        self.fmt = None
        if self.use_layerwise:
            if metadata.use_mla:
                self.fmt = MemoryFormat.KV_MLA_FMT
            elif config.enable_blending:
                self.fmt = MemoryFormat.KV_2TD
            else:
                self.fmt = MemoryFormat.KV_T2D
        if metadata.use_mla:
            self.fmt = MemoryFormat.KV_MLA_FMT

        # NOTE(ApostaC): we haven't support lookup-cache yet
        self.lookup_cache: dict[CacheEngineKey, Any] = {}

        # lookup_id -> {location -> [pinned keys]}
        self.lookup_pins: dict[str, dict[str, list]] = defaultdict(
            lambda: defaultdict(list)
        )

        InitializeUsageContext(config, metadata)
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()
        # Initialize PinMonitor singleton with config
        PinMonitor.GetOrCreate(config)

        self.post_inited = False

        # Flag to control KVCache Check logging (can be toggled via API)
        self.kvcache_check_log_enabled = False

        gc.collect()
        if not config.py_enable_gc:
            gc.disable()

        # Health monitor reference (injected by LMCacheManager)
        self._health_monitor: Optional["HealthMonitor"] = None

        # Flag to indicate if initialization failed (irrecoverable error)
        self._init_failed = False

    def set_health_monitor(self, health_monitor: "HealthMonitor") -> None:
        """
        Set the health monitor reference.

        This is called by LMCacheManager after creating the HealthMonitor
        to inject the reference into the engine.

        Args:
            health_monitor: The HealthMonitor instance from LMCacheManager
        """
        self._health_monitor = health_monitor

    def is_healthy(self) -> bool:
        """
        Check if the LMCache system is healthy.

        This method returns False if:
        - Initialization failed (irrecoverable error)
        - HealthMonitor reports unhealthy

        If no health monitor is set and initialization succeeded,
        it returns True (assume healthy).

        Returns:
            bool: True if healthy, False otherwise
        """
        if self._init_failed:
            return False
        if self._health_monitor is not None:
            return self._health_monitor.is_healthy()
        return True

    def _get_req_id(self, kwargs: dict) -> str:
        """Extracts request ID from kwargs for logging."""
        return kwargs.get("req_id", "unspecified")

    def mark_init_failed(self, reason: str = "") -> None:
        """
        Mark the engine as having failed initialization.

        This is called by LMCacheManager when an irrecoverable error occurs
        during initialization or post_init. Once marked, is_healthy() will
        always return False, causing the system to fall back to recomputation.

        Args:
            reason: Optional reason string for logging
        """
        self._init_failed = True
        if reason:
            logger.error("LMCacheEngine marked as init failed: %s", reason)
        else:
            logger.error("LMCacheEngine marked as init failed")

    def post_init(self, **kwargs) -> None:
        if not self.post_inited:
            logger.info("Post initializing LMCacheEngine")
            lookup_server_worker_ids = self.config.get_lookup_server_worker_ids(
                self.metadata.use_mla, self.metadata.world_size
            )
            if (
                self.lmcache_worker is not None
                or self.use_layerwise
                or not self.save_only_first_rank
                or self.metadata.is_first_rank()
                or len(lookup_server_worker_ids) == 0
                or self.metadata.worker_id in lookup_server_worker_ids
            ):
                logger.info(
                    f"Initialize storage manager on rank {self.metadata.worker_id}, "
                    f"use layerwise: {self.use_layerwise},"
                    f"save only first rank: {self.save_only_first_rank}"
                )
                async_lookup_server = kwargs.get("async_lookup_server", None)
                self.storage_manager = StorageManager(
                    self.config,
                    self.metadata,
                    event_manager=self.event_manager,
                    lmcache_worker=self.lmcache_worker,
                    async_lookup_server=async_lookup_server,
                )
            self.post_inited = True

    def freeze(self, enabled: bool) -> None:
        """
        Set the freeze mode for the cache engine.

        When freeze mode is enabled:
        - All store operations will be skipped (no new data stored)
        - Only local_cpu backend will be used for retrieval
        - No admit/evict messages will be generated
        This protects the local_cpu hot cache from changes.

        Args:
            enabled (bool): Whether to enable freeze mode
        """
        if self.storage_manager is not None:
            self.storage_manager.set_freeze(enabled)

    def is_frozen(self) -> bool:
        """
        Get the current freeze mode status.

        Returns:
            bool: True if freeze mode is enabled, False otherwise
        """
        if self.storage_manager is not None:
            return self.storage_manager.is_frozen()
        return False

    def set_hot_cache(self, enabled: bool) -> None:
        """
        Dynamically enable or disable the LocalCPUBackend hot cache.

        When disabled, the existing hot cache entries will be cleared
        and no new data will be written to the hot cache.

        Args:
            enabled (bool): Whether to enable hot cache
        """
        if self.storage_manager is not None:
            self.storage_manager.set_hot_cache(enabled)

    def is_hot_cache_enabled(self) -> bool:
        """
        Get the current hot cache status of LocalCPUBackend.

        Returns:
            bool: True if hot cache is enabled, False otherwise
        """
        if self.storage_manager is not None:
            return self.storage_manager.is_hot_cache_enabled()
        return False

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def store(
        self,
        tokens: Optional[Union[torch.Tensor, list[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Store the tokens/hashes and mask into the cache engine.

        :param Optional[torch.Tensor] tokens: The tokens of the corresponding KV caches.

        :param Optional[List[int]] hashes: The hashes of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.
            Should include KV cache specific information (e.g., paged KV buffer
            and the page tables).

        :raises: ValueError if the number of Falses in the mask is not a
            multiple of the chunk size.
        """
        block_lease = kwargs.pop("block_lease", None)
        block_lease_released = False
        req_id = self._get_req_id(kwargs)

        def close_store_block_lease(phase: str) -> None:
            nonlocal block_lease_released
            if block_lease is None or block_lease_released:
                return
            close = getattr(block_lease, "close", None)
            if not callable(close):
                logger.warning(
                    "[pd-demo-block-pin] op=release_skip req_id=%s phase=%s "
                    "reason=invalid_store_lease",
                    req_id,
                    phase,
                )
                block_lease_released = True
                return
            release_start = time.perf_counter()
            close()
            release_ms = (time.perf_counter() - release_start) * 1000
            block_lease_released = True
            logger.info(
                "[pd-demo-block-pin] op=release req_id=%s phase=%s release_ms=%.4f",
                req_id,
                phase,
                release_ms,
            )

        # Health check: block operation if LMCache is unhealthy
        if not self.is_healthy():
            logger.warning("LMCache is unhealthy, skipping store operation")
            close_store_block_lease("unhealthy")
            return

        assert self.gpu_connector is not None, (
            "gpu_connector is required for store operation"
        )

        if self._is_passive():
            logger.debug(f"rank={self.metadata.worker_id} ignore store")
            close_store_block_lease("passive")
            return

        assert self.storage_manager is not None

        # Initialize num_to_store_tokens to avoid reference before assignment
        num_to_store_tokens = 0

        if mask is not None:
            num_to_store_tokens = torch.sum(mask).item()
        elif tokens is not None:
            num_to_store_tokens = len(tokens)
        elif hashes is not None:
            assert offsets is not None, (
                "Offsets should be set when hashes are provided during store"
            )
            num_to_store_tokens = sum(offsets)
            kwargs["slot_mapping"] = torch.tensor(
                kwargs["slot_mapping"], dtype=torch.long, device="cuda"
            )

        assert tokens is not None or hashes is not None, (
            "Either 'tokens' or 'hashes' must be provided."
        )

        # KVCache Check logging
        self._log_kvcache_for_check(
            operation="Store",
            kwargs=kwargs,
            token_count=num_to_store_tokens,
            require_req_id=False,
        )

        # Check if freeze mode is enabled
        if self.is_frozen():
            logger.debug(
                "Freeze mode enabled, skipping store operation for %d tokens",
                num_to_store_tokens,
            )
            close_store_block_lease("frozen")
            return

        store_stats = self.stats_monitor.on_store_request(num_to_store_tokens)

        starts: List[int] = []
        ends: List[int] = []
        keys: List[CacheEngineKey] = []
        memory_objs: List[MemoryObj] = []

        tot_kv_size = 0
        tot_token_num = 0

        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)
        save_chunk_lengths = extract_chunk_lengths(
            request_configs,
            SAVE_CHUNK_LENGTHS_CONFIG,
        )

        with store_stats.profile_process_tokens():
            prev_key = 0
            for start, end, key in self.token_database.process_tokens(
                tokens,
                hashes,
                offsets,
                mask,
                request_configs=request_configs,
                chunk_lengths=save_chunk_lengths,
                chunk_lengths_config_name=SAVE_CHUNK_LENGTHS_CONFIG,
            ):
                assert isinstance(key, CacheEngineKey)
                # Allocate the memory object
                num_tokens = end - start
                kv_shapes = self.metadata.get_shapes(num_tokens)
                kv_dtypes = self.metadata.get_dtypes()

                # TODO (Jiayi): should be batched in the future
                memory_obj = self.storage_manager.allocate(
                    kv_shapes,
                    kv_dtypes,
                    busy_loop=self.config.get_extra_config_value(
                        "force_store_wait", False
                    ),
                    fmt=self.fmt,
                )
                if memory_obj is None:
                    logger.warning(
                        "Local cpu memory under pressure so"
                        " choosing to store only "
                        f" {len(memory_objs)}"
                        " total chunks of KV cache."
                    )
                    break

                starts.append(start)
                ends.append(end)
                keys.append(key)
                memory_objs.append(memory_obj)
                tot_kv_size += memory_obj.get_size()
                tot_token_num += num_tokens

                # Create KV event
                if self.kv_events_enabled:
                    stored_event = CacheStoreEvent(
                        block_hashes=[key.chunk_hash],
                        parent_block_hash=None if start == 0 else prev_key,
                        token_ids=[],
                        block_size=num_tokens,
                        lora_id=None,
                        medium="cpu",
                        lora_name=None,
                    )
                    if tokens is not None:
                        stored_event.token_ids = convert_tokens_to_list(
                            tokens,
                            start,
                            end,
                        )
                        if isinstance(tokens, torch.Tensor):
                            stored_event.medium = tokens.device
                    elif hashes is not None:
                        stored_event.token_ids = hashes[start : end + 1]
                    logger.debug(
                        (
                            "Added kv cache event '%s' to kv cache events queue"
                            % stored_event
                        )
                    )
                    self.kv_events.append(stored_event)
                    prev_key = key.chunk_hash

        # memory_objs might be empty, directly return to avoid sending tokens
        if not memory_objs:
            close_store_block_lease("no_memory_objs")
            return

        with store_stats.profile_from_gpu():
            try:
                self.gpu_connector.batched_from_gpu(memory_objs, starts, ends, **kwargs)
            finally:
                close_store_block_lease("d2h_done")

        with store_stats.profile_put():
            chunk_ids = _chunk_ids_for_spans(
                starts,
                ends,
                chunk_size=self.config.chunk_size,
                chunk_lengths=save_chunk_lengths,
            )
            transfer_spec = _transfer_spec_with_chunk_ids(
                kwargs.get("transfer_spec", None),
                chunk_ids,
            )
            # TODO: we implicitly rely on batched_put to call ref_count_down
            # this management should be done in a cleaner way
            self.storage_manager.batched_put(
                keys,
                memory_objs,
                transfer_spec=transfer_spec,
                location=self.store_location,
            )

        self.stats_monitor.on_store_finished(
            store_stats,
            tot_token_num,
        )
        tot_time = store_stats.time_to_store()

        logger.info(
            "[req_id=%s] Stored %d out of total %d tokens. "
            "size: %.4f GB, cost %.4f ms, throughput: %.4f GB/s; "
            "offload_time: %.4f ms, put_time: %.4f ms",
            req_id,
            tot_token_num,
            num_to_store_tokens,
            tot_kv_size / 1024**3,
            tot_time * 1000,
            tot_kv_size / tot_time / 1024**3 if tot_time > 0 else 0,
            (store_stats.process_tokens_time + store_stats.from_gpu_time) * 1000,
            store_stats.put_time * 1000,
        )

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def create_layerwise_store_batch(self, total_tokens: int) -> LayerwiseStoreBatch:
        """Create a layer-major GPU and put batch for one scheduler forward."""

        if self.gpu_connector is None or self._layerwise_put_executor is None:
            raise RuntimeError("layerwise store batching requires a layerwise engine")
        create_gpu_batch = getattr(
            self.gpu_connector,
            "create_layerwise_request_batch",
            None,
        )
        if not callable(create_gpu_batch):
            raise RuntimeError(
                "GPU connector does not support layerwise request batches"
            )
        return LayerwiseStoreBatch(
            create_gpu_batch(total_tokens),
            self._layerwise_put_executor,
        )

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def store_layer(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Generator[None, None, None]:
        """
        Store the KV cache in a layerwise manner.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.

        return: A generator that yields as the GPU connector enqueues offload
            work. Layer groups may be submitted to storage as soon as their D2H
            completes, while fused saves retain one full-request transfer.
        """
        layerwise_block_leases = kwargs.pop("layerwise_block_leases", None)
        req_id = self._get_req_id(kwargs)
        detailed_offload_timing = os.environ.get("PD_BACKEND_LAYER_TIMING") == "1"

        def lease_ref_summary(lease: Any, limit: int = 8) -> tuple[int, str]:
            refs = getattr(lease, "refs", ())
            block_ids = [getattr(ref, "block_id", ref) for ref in refs]
            if len(block_ids) <= limit:
                return len(block_ids), str(block_ids)
            head = ", ".join(str(block_id) for block_id in block_ids[:limit])
            return len(block_ids), f"[{head}, ... total={len(block_ids)}]"

        def pop_block_lease(layer_id: int) -> Any:
            if layerwise_block_leases is None:
                return None
            popleft = getattr(layerwise_block_leases, "popleft", None)
            if not callable(popleft):
                logger.warning(
                    "[pd-demo-block-pin] op=bind_skip req_id=%s layer=%d "
                    "reason=invalid_lease_queue",
                    req_id,
                    layer_id,
                )
                return None
            try:
                lease = popleft()
            except IndexError:
                logger.warning(
                    "[pd-demo-block-pin] op=bind_skip req_id=%s layer=%d "
                    "reason=empty_lease_queue",
                    req_id,
                    layer_id,
                )
                return None
            if detailed_offload_timing:
                block_count, block_ids = lease_ref_summary(lease)
                logger.info(
                    "[pd-demo-block-pin] op=bind req_id=%s layer=%d blocks=%d "
                    "block_ids=%s remaining_leases=%s",
                    req_id,
                    layer_id,
                    block_count,
                    block_ids,
                    len(layerwise_block_leases),
                )
            return lease

        def close_block_lease(lease: Any, layer_id: int, phase: str) -> float:
            if lease is None:
                return 0.0
            close = getattr(lease, "close", None)
            if not callable(close):
                logger.warning(
                    "[pd-demo-block-pin] op=release_skip req_id=%s layer=%d "
                    "phase=%s reason=invalid_lease",
                    req_id,
                    layer_id,
                    phase,
                )
                return 0.0
            release_start = (
                time.perf_counter() if detailed_offload_timing else 0.0
            )
            close()
            release_elapsed = (
                time.perf_counter() - release_start
                if detailed_offload_timing
                else 0.0
            )
            if detailed_offload_timing:
                block_count, block_ids = lease_ref_summary(lease)
                logger.info(
                    "[pd-demo-block-pin] op=release req_id=%s layer=%d phase=%s "
                    "blocks=%d block_ids=%s release_ms=%.4f",
                    req_id,
                    layer_id,
                    phase,
                    block_count,
                    block_ids,
                    release_elapsed * 1000,
                )
            return release_elapsed

        def release_remaining_block_leases(reason: str) -> None:
            if layerwise_block_leases is None:
                return
            released = 0
            while True:
                try:
                    lease = layerwise_block_leases.popleft()
                except IndexError:
                    break
                close_block_lease(lease, -1, reason)
                released += 1
            if released and detailed_offload_timing:
                logger.info(
                    "[pd-demo-block-pin] op=release_remaining req_id=%s "
                    "reason=%s leases=%d",
                    req_id,
                    reason,
                    released,
                )

        # Health check: block operation if LMCache is unhealthy
        if not self.is_healthy():
            release_remaining_block_leases("unhealthy")
            logger.warning("LMCache is unhealthy, skipping store_layer operation")
            return

        assert self.storage_manager is not None
        assert self.gpu_connector is not None, (
            "gpu_connector is required for store_layer operation"
        )

        if mask is not None:
            num_to_store_tokens = torch.sum(mask).item()
        else:
            num_to_store_tokens = len(tokens)

        # KVCache Check logging
        self._log_kvcache_for_check(
            operation="Layerwise store",
            kwargs=kwargs,
            token_count=num_to_store_tokens,
            require_req_id=True,
        )

        monitor_req_id = self.stats_monitor.on_store_request(num_to_store_tokens)

        # Check if freeze mode is enabled
        if self.is_frozen():
            logger.debug(
                "Freeze mode enabled, skipping store_layer for %d tokens",
                num_to_store_tokens,
            )
            release_remaining_block_leases("frozen")
            # Still need to yield to avoid StopIteration
            for layer_id in range(self.num_layers):
                yield
            return

        starts = []
        ends = []
        keys = []
        memory_objs = []
        tot_token_num = 0
        kv_dtype = self.metadata.kv_dtype
        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)
        fuse_layerwise_offload = bool(kwargs.get("fuse_layerwise_offload", False))
        layerwise_store_batch = kwargs.get("layerwise_request_batch")
        if isinstance(layerwise_store_batch, LayerwiseStoreBatch):
            kwargs["layerwise_request_batch"] = layerwise_store_batch.gpu_batch

        store_timeline_enabled = (
            os.environ.get("PD_BACKEND_LAYER_TIMING") == "1"
            or os.environ.get("PD_BACKEND_STORE_TIMELINE") == "1"
        )
        store_timeline_seq = 0

        def log_store_timeline(
            phase: str,
            start_time: float,
            end_time: float,
            layer_id: int = -1,
            extra: str = "",
        ) -> None:
            nonlocal store_timeline_seq
            if not store_timeline_enabled:
                return
            logger.info(
                "[pd-demo-store-timeline] req_id=%s seq=%d phase=%s "
                "layer=%d t_start_ms=%.4f t_end_ms=%.4f dur_ms=%.4f%s",
                req_id,
                store_timeline_seq,
                phase,
                layer_id,
                (start_time - prepare_start) * 1000,
                (end_time - prepare_start) * 1000,
                (end_time - start_time) * 1000,
                f" {extra}" if extra else "",
            )
            store_timeline_seq += 1

        prepare_start = time.perf_counter()
        contains_time = 0.0
        allocation_time = 0.0
        prev_key = 0
        save_chunk_lengths = extract_chunk_lengths(
            request_configs,
            SAVE_CHUNK_LENGTHS_CONFIG,
        )
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens,
            mask=mask,
            request_configs=request_configs,
            chunk_lengths=save_chunk_lengths,
            chunk_lengths_config_name=SAVE_CHUNK_LENGTHS_CONFIG,
        ):
            assert isinstance(key, CacheEngineKey)

            keys_multi_layer = key.split_layers(self.num_layers)
            # Only check the first layer
            contains_start = time.perf_counter()
            key_exists = self.storage_manager.contains(
                keys_multi_layer[0], self.retrieve_locations
            )
            contains_time += time.perf_counter() - contains_start
            if key_exists:
                continue

            # Allocate the memory object
            num_tokens = end - start
            kv_shape_single_layer = self.gpu_connector.get_shape(num_tokens)

            allocation_start = time.perf_counter()
            memory_objs_multi_layer = self.storage_manager.batched_allocate(
                kv_shape_single_layer,
                kv_dtype,
                batch_size=self.num_layers,
                fmt=self.fmt,
                busy_loop=self.config.get_extra_config_value("force_store_wait", False),
            )
            allocation_time += time.perf_counter() - allocation_start

            if memory_objs_multi_layer is None:
                logger.warning(
                    "Local cpu memory under pressure so"
                    " choosing to not store the KV cache."
                )
                break

            starts.append(start)
            ends.append(end)
            keys.append(keys_multi_layer)
            memory_objs.append(memory_objs_multi_layer)
            tot_token_num += num_tokens

            # Create KV event
            if self.kv_events_enabled and tokens is not None:
                stored_event = CacheStoreEvent(
                    block_hashes=[key.chunk_hash],
                    parent_block_hash=None if start == 0 else prev_key,
                    token_ids=[],
                    block_size=num_tokens,
                    lora_id=None,
                    medium="cpu",
                    lora_name=None,
                )
                if tokens is not None:
                    stored_event.token_ids = convert_tokens_to_list(
                        tokens,
                        start,
                        end,
                    )
                    if isinstance(tokens, torch.Tensor):
                        stored_event.medium = tokens.device
                logger.debug(
                    f"Added kv cache event '{stored_event}' to kv cache events queue"
                )
                self.kv_events.append(stored_event)
                prev_key = key.chunk_hash

        if keys:
            # Transpose the keys and memory objects into layer major format
            memory_objs = [list(row) for row in zip(*memory_objs, strict=False)]
            keys = [list(row) for row in zip(*keys, strict=False)]
            batch_layerwise_puts = bool(kwargs.get("batch_layerwise_puts", False))
            yield_group_size = max(
                1,
                min(
                    self.num_layers,
                    int(kwargs.get("layerwise_yield_group_size", 1)),
                ),
            )
            put_group_size = (
                self.num_layers
                if fuse_layerwise_offload
                else yield_group_size
                if batch_layerwise_puts
                else 1
            )
            put_layer_groups = [
                list(
                    range(
                        group_start,
                        min(group_start + put_group_size, self.num_layers),
                    )
                )
                for group_start in range(0, self.num_layers, put_group_size)
            ]
            all_layer_batch_keys = [key for layer_keys in keys for key in layer_keys]
            all_layer_batch_memory_objs = [
                memory_obj
                for layer_memory_objs in memory_objs
                for memory_obj in layer_memory_objs
            ]
            put_group_keys = [
                [key for layer_id in layer_ids for key in keys[layer_id]]
                for layer_ids in put_layer_groups
            ]
            put_group_memory_objs = [
                [
                    memory_obj
                    for layer_id in layer_ids
                    for memory_obj in memory_objs[layer_id]
                ]
                for layer_ids in put_layer_groups
            ]

            # Calculate total KV size for logging
            tot_kv_size = sum(
                mo.get_size() for layer_objs in memory_objs for mo in layer_objs
            )

            assert_layerwise_gpu_connector(self.gpu_connector)

            prepare_end = time.perf_counter()
            prepare_time = prepare_end - prepare_start
            prepare_other_time = max(
                0.0, prepare_time - contains_time - allocation_time
            )
            log_store_timeline(
                "prepare_tokens",
                prepare_start,
                prepare_end,
                extra=(
                    f"chunks={len(starts)} contains_ms={contains_time * 1000:.4f} "
                    f"allocation_ms={allocation_time * 1000:.4f} "
                    f"prepare_other_ms={prepare_other_time * 1000:.4f}"
                ),
            )
            remote_prepare_submit_time = 0.0
            remote_prepare_layers = 0
            chunk_ids = _chunk_ids_for_spans(
                starts,
                ends,
                chunk_size=self.config.chunk_size,
                chunk_lengths=save_chunk_lengths,
            )
            base_transfer_spec = kwargs.get("transfer_spec")
            transfer_spec = _transfer_spec_with_chunk_ids(
                base_transfer_spec,
                chunk_ids * len(keys),
            )
            if transfer_spec is not None:
                remote_prepare_start = time.perf_counter()
                for backend_name, backend in (
                    self.storage_manager.storage_backends.items()
                ):
                    if self.store_location and backend_name != self.store_location:
                        continue
                    prepare_put = getattr(backend, "prepare_batched_put_task", None)
                    if prepare_put is None:
                        continue
                    if fuse_layerwise_offload:
                        prepare_put(
                            all_layer_batch_keys,
                            all_layer_batch_memory_objs,
                            transfer_spec=transfer_spec,
                        )
                        remote_prepare_layers += len(keys)
                        continue
                    prepare_grouped = getattr(
                        backend, "prepare_layerwise_put_tasks", None
                    )
                    if callable(prepare_grouped) and prepare_grouped(
                        put_group_keys,
                        put_group_memory_objs,
                        transfer_spec=transfer_spec,
                    ):
                        remote_prepare_layers += len(keys)
                        continue
                    for layer_ids, group_keys, group_memory_objs in zip(
                        put_layer_groups,
                        put_group_keys,
                        put_group_memory_objs,
                        strict=True,
                    ):
                        prepare_put(
                            group_keys,
                            group_memory_objs,
                            transfer_spec=_transfer_spec_with_chunk_ids(
                                base_transfer_spec,
                                chunk_ids * len(layer_ids),
                            ),
                        )
                        remote_prepare_layers += len(layer_ids)
                remote_prepare_end = time.perf_counter()
                remote_prepare_submit_time = (
                    remote_prepare_end - remote_prepare_start
                )
                log_store_timeline(
                    "remote_prepare_submit",
                    remote_prepare_start,
                    remote_prepare_end,
                    extra=(
                        f"layers={remote_prepare_layers} chunks={len(starts)}"
                    ),
                )
            t_start = time.perf_counter()
            io_time = 0.0
            gpu_offload_step_time = 0.0
            put_submit_time = 0.0
            put_ready_wait_time = 0.0
            put_future_wait_time = 0.0
            put_task_wall_time = 0.0
            block_lease_release_time = 0.0
            yield_resume_time = 0.0
            layer_connector_next_time = 0.0
            layer_step_time = 0.0
            mem_obj_generator_create_start = time.perf_counter()
            mem_obj_generator = self.gpu_connector.batched_from_gpu(
                memory_objs, starts, ends, **kwargs
            )
            mem_obj_generator_create_end = time.perf_counter()
            log_store_timeline(
                "mem_obj_generator_create",
                mem_obj_generator_create_start,
                mem_obj_generator_create_end,
            )

            put_futures: list[Future[None]] = []
            pending_layer_puts: list[tuple[int, LayerwiseOffloadHandle]] = []
            put_stats_lock = threading.Lock()

            def build_layer_put_task(
                entries: list[tuple[int, LayerwiseOffloadHandle, Any]],
                *,
                shared_request_batch: bool,
            ) -> _LayerwiseStoreBatchTask:
                nonlocal io_time
                nonlocal put_submit_time
                nonlocal put_ready_wait_time
                nonlocal put_task_wall_time
                nonlocal block_lease_release_time
                assert entries

                batch_keys = [
                    key for layer_id, _, _ in entries for key in keys[layer_id]
                ]
                batch_memory_objs = [
                    memory_obj
                    for layer_id, _, _ in entries
                    for memory_obj in memory_objs[layer_id]
                ]
                put_transfer_spec = _transfer_spec_with_chunk_ids(
                    base_transfer_spec,
                    chunk_ids * len(entries),
                )

                cleanup_lock = threading.Lock()
                cleaned = False

                def wait_ready() -> None:
                    nonlocal put_ready_wait_time
                    wait_start = time.perf_counter()
                    first_wait_error: Optional[BaseException] = None
                    # One request-batch tile has one shared D2H event. Fallback
                    # paths retain independent per-layer handles and wait all.
                    ready_entries = entries[:1] if shared_request_batch else entries
                    for _, offload_handle, _ in ready_entries:
                        try:
                            offload_handle.wait()
                        except BaseException as exc:
                            if first_wait_error is None:
                                first_wait_error = exc
                    wait_end = time.perf_counter()
                    ready_wait_elapsed = wait_end - wait_start

                    with put_stats_lock:
                        put_ready_wait_time += ready_wait_elapsed
                    if first_wait_error is not None:
                        raise first_wait_error

                def put() -> None:
                    nonlocal io_time
                    nonlocal put_submit_time
                    nonlocal put_task_wall_time
                    nonlocal block_lease_release_time
                    task_start = time.perf_counter()
                    lease_release_elapsed = 0.0
                    for layer_id, _, block_lease in entries:
                        lease_release_elapsed += close_block_lease(
                            block_lease,
                            layer_id,
                            "d2h_done",
                        )

                    put_start = time.perf_counter()
                    self.storage_manager.batched_put(
                        batch_keys,
                        batch_memory_objs,
                        transfer_spec=put_transfer_spec,
                        location=self.store_location,
                    )
                    put_end = time.perf_counter()
                    put_elapsed = put_end - put_start
                    task_wall_elapsed = put_end - task_start
                    with put_stats_lock:
                        put_submit_time += put_elapsed
                        io_time += task_wall_elapsed
                        put_task_wall_time += task_wall_elapsed
                        block_lease_release_time += lease_release_elapsed
                    if detailed_offload_timing:
                        logger.info(
                            "[req_id=%s] Layerwise async put step: "
                            "first_layer=%d layers=%d "
                            "lease_release_ms=%.4f put_submit_ms=%.4f task_ms=%.4f",
                            req_id,
                            entries[0][0],
                            len(entries),
                            lease_release_elapsed * 1000,
                            put_elapsed * 1000,
                            task_wall_elapsed * 1000,
                        )

                def abort() -> None:
                    nonlocal cleaned
                    with cleanup_lock:
                        if cleaned:
                            return
                        cleaned = True
                    for layer_id, _, block_lease in entries:
                        close_block_lease(
                            block_lease,
                            layer_id,
                            "put_submit_failed",
                        )
                    for memory_obj in batch_memory_objs:
                        memory_obj.ref_count_down()

                return _LayerwiseStoreBatchTask(wait_ready, put, abort)

            def submit_layer_puts(
                entries: list[tuple[int, LayerwiseOffloadHandle, Any]],
            ) -> None:
                uses_request_batch = (
                    isinstance(layerwise_store_batch, LayerwiseStoreBatch)
                    and all(
                        getattr(handle, "request_batch", None)
                        is layerwise_store_batch.gpu_batch
                        for _, handle, _ in entries
                    )
                )
                task = build_layer_put_task(
                    entries,
                    shared_request_batch=uses_request_batch,
                )
                if uses_request_batch:
                    try:
                        future = layerwise_store_batch.register_put_task(task)
                    except BaseException:
                        try:
                            task.wait_ready()
                        except BaseException:
                            pass
                        task.abort()
                        raise
                    put_futures.append(future)
                    return

                executor = self._layerwise_put_executor
                assert executor is not None

                def wait_and_put() -> None:
                    try:
                        task.wait_ready()
                    except BaseException:
                        task.abort()
                        raise
                    task.put()

                submit_start = time.perf_counter()
                try:
                    future = executor.submit(wait_and_put)
                except BaseException:
                    try:
                        task.wait_ready()
                    except BaseException:
                        pass
                    task.abort()
                    raise
                submit_end = time.perf_counter()
                put_futures.append(future)
                log_store_timeline(
                    "async_put_task_submit",
                    submit_start,
                    submit_end,
                    layer_id=entries[0][0],
                    extra=(
                        f"layers={len(entries)} futures={len(put_futures)}"
                    ),
                )

            offload_start = time.perf_counter()
            first_offload_handle = next(mem_obj_generator)
            offload_end = time.perf_counter()
            offload_elapsed = offload_end - offload_start
            gpu_offload_step_time += offload_elapsed
            log_store_timeline(
                "connector_next_initial",
                offload_start,
                offload_end,
                extra="description=enqueue_first_layer",
            )
            if detailed_offload_timing:
                logger.info(
                    "[req_id=%s] Layerwise offload step: "
                    "step=initial put_layer=-1 next_ms=%.4f",
                    req_id,
                    offload_elapsed * 1000,
                )

            if isinstance(first_offload_handle, LayerwiseOffloadHandle):
                offload_handle = first_offload_handle
                for layer_id in range(self.num_layers):
                    if layer_id > 0:
                        offload_start = time.perf_counter()
                        offload_handle = next(mem_obj_generator)
                        offload_end = time.perf_counter()
                        offload_elapsed = offload_end - offload_start
                        gpu_offload_step_time += offload_elapsed
                        layer_connector_next_time += offload_elapsed
                    else:
                        offload_elapsed = gpu_offload_step_time

                    if (
                        not isinstance(offload_handle, LayerwiseOffloadHandle)
                        or getattr(offload_handle, "layer_id", None) != layer_id
                    ):
                        raise RuntimeError(
                            "Layerwise offload handle order mismatch: "
                            f"expected layer {layer_id}, "
                            f"got {getattr(offload_handle, 'layer_id', None)}"
                        )
                    pending_layer_puts.append((layer_id, offload_handle))
                    if detailed_offload_timing:
                        logger.info(
                            "[req_id=%s] Layerwise offload step: "
                            "step=async_enqueue layer=%d next_ms=%.4f",
                            req_id,
                            layer_id,
                            offload_elapsed * 1000,
                        )

                    put_group_complete = (
                        (layer_id + 1) % put_group_size == 0
                        or layer_id == self.num_layers - 1
                    )
                    if put_group_complete:
                        group_entries = [
                            (
                                pending_layer_id,
                                pending_offload_handle,
                                pop_block_lease(pending_layer_id),
                            )
                            for pending_layer_id, pending_offload_handle in (
                                pending_layer_puts
                            )
                        ]
                        pending_layer_puts.clear()
                        submit_layer_puts(group_entries)

                    yield_group_complete = (
                        (layer_id + 1) % yield_group_size == 0
                        or layer_id == self.num_layers - 1
                    )
                    if yield_group_complete:
                        yield_start = time.perf_counter()
                        yield
                        yield_end = time.perf_counter()
                        yield_resume_time += yield_end - yield_start

                assert not pending_layer_puts

                first_error: Optional[Exception] = None
                future_wait_start = time.perf_counter()
                for future in put_futures:
                    try:
                        future.result()
                    except Exception as exc:
                        if first_error is None:
                            first_error = exc
                future_wait_end = time.perf_counter()
                put_future_wait_time += future_wait_end - future_wait_start
                log_store_timeline(
                    "async_put_tasks_wait",
                    future_wait_start,
                    future_wait_end,
                    extra=f"futures={len(put_futures)}",
                )

                finalize_start = time.perf_counter()
                next(mem_obj_generator)
                finalize_end = time.perf_counter()
                log_store_timeline(
                    "connector_finalize",
                    finalize_start,
                    finalize_end,
                )
                if first_error is not None:
                    raise first_error
            else:
                for layer_id in range(self.num_layers):
                    yield_start = time.perf_counter()
                    yield
                    yield_end = time.perf_counter()
                    yield_elapsed = yield_end - yield_start
                    yield_resume_time += yield_elapsed
                    t_io = time.perf_counter()
                    block_lease = pop_block_lease(layer_id)
                    offload_start = time.perf_counter()
                    try:
                        next(mem_obj_generator)
                    finally:
                        block_lease_release_time += close_block_lease(
                            block_lease,
                            layer_id,
                            "sync_next_done",
                        )
                    offload_end = time.perf_counter()
                    offload_elapsed = offload_end - offload_start
                    gpu_offload_step_time += offload_elapsed
                    layer_connector_next_time += offload_elapsed
                    if detailed_offload_timing:
                        logger.info(
                            "[req_id=%s] Layerwise offload step: "
                            "step=loop put_layer=%d next_ms=%.4f",
                            req_id,
                            layer_id,
                            offload_elapsed * 1000,
                        )
                    put_start = time.perf_counter()
                    self.storage_manager.batched_put(
                        keys[layer_id],
                        memory_objs[layer_id],
                        transfer_spec=_transfer_spec_with_chunk_ids(
                            base_transfer_spec,
                            chunk_ids,
                        ),
                        location=self.store_location,
                    )
                    put_end = time.perf_counter()
                    put_elapsed = put_end - put_start
                    put_submit_time += put_elapsed
                    io_time += put_end - t_io
                    layer_step_time += put_end - yield_end
                    log_store_timeline(
                        "layer_step",
                        yield_start,
                        put_end,
                        layer_id=layer_id,
                        extra=(
                            f"yield_wait_ms={yield_elapsed * 1000:.4f} "
                            f"connector_next_ms={offload_elapsed * 1000:.4f} "
                            f"put_submit_ms={put_elapsed * 1000:.4f} "
                            f"active_ms={(put_end - yield_end) * 1000:.4f}"
                        ),
                    )

            wall_end = time.perf_counter()
            wall_time = wall_end - t_start
            pipeline_other_time = max(
                0.0,
                wall_time
                - yield_resume_time
                - gpu_offload_step_time
                - put_ready_wait_time
                - put_submit_time,
            )
            log_store_timeline(
                "store_pipeline_total",
                t_start,
                wall_end,
                extra=(
                    f"initial_connector_next_ms="
                    f"{(gpu_offload_step_time - layer_connector_next_time) * 1000:.4f} "
                    f"layer_connector_next_ms={layer_connector_next_time * 1000:.4f} "
                    f"yield_wait_ms={yield_resume_time * 1000:.4f} "
                    f"put_ready_wait_ms={put_ready_wait_time * 1000:.4f} "
                    f"put_submit_ms={put_submit_time * 1000:.4f} "
                    f"put_future_wait_ms={put_future_wait_time * 1000:.4f} "
                    f"put_task_wall_ms={put_task_wall_time * 1000:.4f} "
                    f"block_lease_release_ms={block_lease_release_time * 1000:.4f} "
                    f"active_layer_ms={layer_step_time * 1000:.4f} "
                    f"other_ms={pipeline_other_time * 1000:.4f}"
                ),
            )
            logger.info(
                "[req_id=%s] Stored %d out of total %d tokens. "
                "size: %.4f GB, cost %.4f ms, "
                "io_time %.4f ms, gpu_offload_step_time %.4f ms, "
                "put_ready_wait_time %.4f ms, put_submit_time %.4f ms, "
                "put_future_wait_time %.4f ms, wall_time %.4f ms, "
                "throughput: %.4f GB/s",
                req_id,
                tot_token_num,
                len(tokens),
                tot_kv_size / 1024**3,
                io_time * 1000,
                io_time * 1000,
                gpu_offload_step_time * 1000,
                put_ready_wait_time * 1000,
                put_submit_time * 1000,
                put_future_wait_time * 1000,
                wall_time * 1000,
                tot_kv_size / io_time / 1024**3 if io_time > 0 else 0,
            )
            logger.info(
                "[req_id=%s] Layerwise store pipeline breakdown: "
                "chunks=%d, prepare_time=%.4f ms, contains_time=%.4f ms, "
                "allocation_time=%.4f ms, prepare_other_time=%.4f ms, "
                "remote_prepare_layers=%d, remote_prepare_submit_time=%.4f ms, "
                "yield_resume_time=%.4f ms, put_ready_wait_time=%.4f ms, "
                "put_submit_time=%.4f ms, put_future_wait_time=%.4f ms, "
                "put_task_wall_time=%.4f ms, block_lease_release_time=%.4f ms, "
                "pipeline_other_time=%.4f ms, "
                "total_time=%.4f ms",
                req_id,
                len(starts),
                prepare_time * 1000,
                contains_time * 1000,
                allocation_time * 1000,
                prepare_other_time * 1000,
                remote_prepare_layers,
                remote_prepare_submit_time * 1000,
                yield_resume_time * 1000,
                put_ready_wait_time * 1000,
                put_submit_time * 1000,
                put_future_wait_time * 1000,
                put_task_wall_time * 1000,
                block_lease_release_time * 1000,
                pipeline_other_time * 1000,
                (wall_end - prepare_start) * 1000,
            )
        else:
            release_remaining_block_leases("no_keys")
            # If no cache are found, we still need to yield to avoid
            # `StopIteration`
            for layer_id in range(self.num_layers):
                yield

        release_remaining_block_leases("store_layer_end")
        self.stats_monitor.on_store_finished(monitor_req_id, tot_token_num)
        yield

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def retrieve(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Retrieve the KV caches from the cache engine. And put the retrieved
        KV cache to the serving engine via the GPU connector.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.
            Should include KV cache specific information (e.g., paged KV buffer
            and the page tables).

        :return: the boolean mask indicating which tokens are retrieved. The
            length of the mask should be the same as the tokens. On CPU.

        :raises: ValueError if the number of Falses in the mask is not a
            multiple of the chunk size.
        """
        # Health check: block operation if LMCache is unhealthy
        if not self.is_healthy():
            logger.warning("LMCache is unhealthy, skipping retrieve operation")
            return torch.zeros(len(tokens), dtype=torch.bool)

        assert self.gpu_connector is not None, (
            "gpu_connector is required for retrieve operation"
        )

        # Get req_id for logging
        req_id = self._get_req_id(kwargs)

        tot_kv_size = 0

        if mask is not None:
            num_required_tokens = torch.sum(mask).item()
        else:
            num_required_tokens = len(tokens)

        # KVCache Check logging
        self._log_kvcache_for_check(
            operation="retrieve",
            kwargs=kwargs,
            token_count=num_required_tokens,
            require_req_id=True,
        )

        retrieve_stats = self.stats_monitor.on_retrieve_request(num_required_tokens)

        ret_mask = torch.zeros(len(tokens), dtype=torch.bool, device="cpu")

        reordered_chunks: List[ProcessedChunk] = []
        if not self._is_passive():
            with retrieve_stats.profile_process_tokens():
                if self.async_loading:
                    reordered_chunks, tot_kv_size = self._async_process_tokens_internal(  # noqa: E501
                        tokens,
                        mask,
                        ret_mask,
                        **kwargs,
                    )
                else:
                    reordered_chunks, tot_kv_size = self._process_tokens_internal(
                        tokens,
                        mask,
                        ret_mask,
                        **kwargs,
                    )

        if self.save_only_first_rank:
            with retrieve_stats.profile_broadcast():
                with torch.cuda.stream(self.broadcast_stream):
                    self._broadcast_or_receive_memory_objs(
                        reordered_chunks,
                        ret_mask,
                    )

                # if self.gpu_connector has load_stream, self.broadcast_stream is equals
                # to self.gpu_connector.load_stream, the broadcast and to_gpu operation
                # will execute sequentially within the stream.
                # if self.gpu_connector does not have load_stream, self.broadcast_stream
                # is created by torch.cuda.Stream(), we need to synchronize broadcast
                # operation, and then process to_cpu operation.
                if not hasattr(self.gpu_connector, "load_stream"):
                    self.broadcast_stream.synchronize()

        # NOTE(Jiayi): memory_obj doesn't have to be a pinned
        # cpu tensor for the sake of performance.
        # For example, disk->gpu is faster than disk->cpu->gpu.
        # RDMA is another example.
        if len(reordered_chunks) > 0:
            with retrieve_stats.profile_to_gpu():
                _, memory_objs, starts, ends = zip(*reordered_chunks, strict=False)
                self.gpu_connector.batched_to_gpu(
                    list(memory_objs), list(starts), list(ends), **kwargs
                )

        # TODO(Jiayi): Remove the following for loop with batched operations
        # TODO(Jiayi): Need to refactor the `remove_after_retrieve` logic.
        for key, memory_obj, _, _ in reordered_chunks:
            if self.remove_after_retrieve and not self._is_passive():
                assert self.storage_manager is not None
                self.storage_manager.remove(key, self.retrieve_locations)
            if not self.async_loading:
                memory_obj.ref_count_down()

        retrieved_tokens = torch.sum(ret_mask)
        log_lmcache_count_metric(req_id, "LmcacheRetrievedTokens", retrieved_tokens)
        self.stats_monitor.on_retrieve_finished(
            retrieve_stats,
            retrieved_tokens,
        )
        onload_time = retrieve_stats.time_to_retrieve()
        # The retrieved may be larger than the need_to_load
        # Example (page_size=16, chunk_size=256):
        #
        # chunks:  [0..255]                [256..511]
        # pages:   [0..15]...[240..255]    [256..271][272..287] ...
        #
        # num_computed_tokens = 288 => vLLM already has [0..287] (18 pages)
        # LMCache hit_prefix_tokens = 512 => cache covers [0..511] (2 chunks)
        #
        # Skip chunk 1, retrieve chunk 2, overwrite [256..287] (32-token overlap)
        # need_to_load: 512 - 288 = 224 tokens
        # retrieved: 256 tokens
        if not self._is_passive():
            logger.info(
                "[req_id=%s] Retrieved %d out of %d required tokens "
                "(from %d total tokens). size: %.4f gb, "
                "cost %.4f ms, throughput: %.4f GB/s;",
                req_id,
                retrieved_tokens,
                num_required_tokens,
                len(tokens),
                tot_kv_size / 1024**3,
                onload_time * 1000,
                tot_kv_size / onload_time / 1024**3 if onload_time > 0 else 0,
            )
        return ret_mask

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def retrieve_layer(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Generator[Optional[torch.Tensor], None, None]:
        """
        Retrieve the KV cache in a layerwise manner.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.

        return: A generator that yields Optional[torch.Tensor]. The tensor will
            be the boolean mask indicating which tokens are retrieved and will
            only be returned in the last iteration. In the first iteration,
            the generator retrieve the memory objects of the first layer from
            the storage backends. In the next iterations, it moves the KV cache
            of layer i from the memory objects (on CPU) to GPU and retrieves
            the memory objects of layer i+1 from the storage backends. In the
            last iteration, it moves the memory objects of the last layer to
            the GPU.
        """
        # Health check: block operation if LMCache is unhealthy
        if not self.is_healthy():
            logger.warning("LMCache is unhealthy, skipping retrieve_layer operation")
            yield torch.zeros(len(tokens), dtype=torch.bool)
            return

        assert self.storage_manager is not None
        assert self.gpu_connector is not None, (
            "gpu_connector is required for retrieve_layer operation"
        )

        # Get req_id for logging
        req_id = self._get_req_id(kwargs)
        load_profile_enabled = os.getenv("LMCACHE_LOAD_PROFILE", "false").lower() in (
            "1",
            "true",
            "yes",
        )
        load_profile_layers_enabled = load_profile_enabled and os.getenv(
            "LMCACHE_LOAD_PROFILE_LAYERS", "true"
        ).lower() in ("1", "true", "yes")

        if mask is not None:
            num_required_tokens = torch.sum(mask).item()
        else:
            num_required_tokens = len(tokens)
        monitor_req_id = self.stats_monitor.on_retrieve_request(num_required_tokens)

        ret_mask = torch.zeros(len(tokens), dtype=torch.bool, device="cpu")

        starts = []
        ends = []
        keys = []

        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)
        lookup_chunk_lengths = extract_chunk_lengths(
            request_configs,
            LOOKUP_CHUNK_LENGTHS_CONFIG,
        )

        location = None
        contains_time = 0.0
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens,
            mask=mask,
            request_configs=request_configs,
            chunk_lengths=lookup_chunk_lengths,
            chunk_lengths_config_name=LOOKUP_CHUNK_LENGTHS_CONFIG,
        ):
            assert isinstance(key, CacheEngineKey)

            keys_multi_layer = key.split_layers(self.num_layers)

            # NOTE: Only check the first layer
            contains_start = time.perf_counter()
            current_location = self.storage_manager.contains(
                keys_multi_layer[0], self.retrieve_locations
            )
            contains_time += time.perf_counter() - contains_start
            if current_location:
                if location is None:
                    location = current_location
                else:
                    # TODO(Jiayi): Support multi-location retrieval in the future
                    assert location == current_location, (
                        "All retrieved keys should be from the same location "
                        "when use layerwise retrieval."
                        "Please support multi-location retrieval in the future."
                    )
            else:
                break

            starts.append(start)
            ends.append(end)
            keys.append(keys_multi_layer)

            ret_mask[start:end] = True

        submit_time = 0.0
        yield_resume_time = 0.0
        result_wait_time = 0.0
        to_gpu_send_time = 0.0
        final_yield_wait = 0.0
        final_sync_time = 0.0

        if keys:
            # Transpose the keys into layer major format
            keys_layer_major = [list(row) for row in zip(*keys, strict=False)]
            if load_profile_enabled:
                logger.info(
                    "[req_id=%s] Layerwise load matched chunks=%d layers=%d "
                    "location=%s required_tokens=%s total_tokens=%s "
                    "contains_time=%.4f ms",
                    req_id,
                    len(keys),
                    self.num_layers,
                    location,
                    num_required_tokens,
                    len(tokens),
                    contains_time * 1000,
                )

            get_generator = self.storage_manager.layerwise_batched_get(
                keys_layer_major,
                location=location,
            )

            assert_layerwise_gpu_connector(self.gpu_connector)

            mem_obj_consumer = self.gpu_connector.batched_to_gpu(starts, ends, **kwargs)
            next(mem_obj_consumer)

            t_start = time.perf_counter()
            io_time = 0.0
            tot_kv_size = 0

            to_count_down = []
            to_release = []
            for layer_id in range(self.num_layers):
                layer_start = time.perf_counter()
                t_io = time.perf_counter()
                task = next(get_generator)
                submit_elapsed = time.perf_counter() - t_io
                submit_time += submit_elapsed
                io_time += submit_elapsed

                assert task is not None

                yield_start = time.perf_counter()
                if layer_id == 0:
                    # NOTE(Yuwei): For sglang integration we need to provide retrieved
                    # tokens number in the first layer loading since there is no lookup
                    yield torch.sum(ret_mask)
                else:
                    yield None
                yield_elapsed = time.perf_counter() - yield_start
                yield_resume_time += yield_elapsed

                t_io = time.perf_counter()
                mem_objs_layer = task.result()
                result_elapsed = time.perf_counter() - t_io
                result_wait_time += result_elapsed
                io_time += result_elapsed

                t_io = time.perf_counter()
                mem_obj_consumer.send(mem_objs_layer)
                send_elapsed = time.perf_counter() - t_io
                to_gpu_send_time += send_elapsed
                io_time += send_elapsed

                layer_bytes = 0
                for mo in mem_objs_layer:
                    obj_size = mo.get_size()
                    layer_bytes += obj_size
                    tot_kv_size += obj_size
                to_count_down.extend(mem_objs_layer)
                to_release.extend(
                    zip(keys_layer_major[layer_id], mem_objs_layer, strict=False)
                )
                if load_profile_layers_enabled:
                    logger.info(
                        "[req_id=%s] Layerwise load layer=%d chunks=%d "
                        "bytes=%d submit_time=%.4f ms yield_resume_time=%.4f ms "
                        "result_wait_time=%.4f ms to_gpu_send_time=%.4f ms "
                        "layer_wall_time=%.4f ms",
                        req_id,
                        layer_id,
                        len(mem_objs_layer),
                        layer_bytes,
                        submit_elapsed * 1000,
                        yield_elapsed * 1000,
                        result_elapsed * 1000,
                        send_elapsed * 1000,
                        (time.perf_counter() - layer_start) * 1000,
                    )

            for key, mem_obj in to_release:
                if self.remove_after_retrieve and not self._is_passive():
                    self.storage_manager.remove(key, self.retrieve_locations)
                mem_obj.ref_count_down()
        else:
            t_start = time.perf_counter()
            io_time = 0.0
            tot_kv_size = 0
            # If no cache are found, we still need to yield to avoid
            # `StopIteration`
            for layer_id in range(self.num_layers):
                yield None

        final_yield_start = time.perf_counter()
        yield None
        final_yield_wait = time.perf_counter() - final_yield_start

        # synchronize the last layer
        final_sync_start = time.perf_counter()
        next(mem_obj_consumer)
        final_sync_time = time.perf_counter() - final_sync_start

        # Unpin any disk-loaded staging objects now that the device-side sync
        # has been enqueued (mem_obj_consumer advanced past its sync point).
        # Without this, pin_count stays at 1 forever and the CPU staging pool
        # fills up, causing the next retrieve to deadlock inside allocate().
        for mem_obj in to_count_down:
            if mem_obj.is_pinned:
                mem_obj.unpin()

        wall_time = time.perf_counter() - t_start
        retrieved_tokens = torch.sum(ret_mask)
        log_lmcache_count_metric(req_id, "LmcacheRetrievedTokens", retrieved_tokens)
        self.stats_monitor.on_retrieve_finished(monitor_req_id, retrieved_tokens)
        if not self._is_passive():
            logger.info(
                "[req_id=%s] Retrieved %d out of %d required tokens "
                "(from %d total tokens). size: %.4f GB, "
                "io_time %.4f ms, wall_time %.4f ms, "
                "throughput: %.4f GB/s",
                req_id,
                retrieved_tokens,
                num_required_tokens,
                len(tokens),
                tot_kv_size / 1024**3,
                io_time * 1000,
                wall_time * 1000,
                tot_kv_size / io_time / 1024**3 if io_time > 0 else 0,
            )
            if load_profile_enabled:
                logger.info(
                    "[req_id=%s] Layerwise load profile summary: "
                    "chunks=%d layers=%d contains_time=%.4f ms "
                    "submit_time=%.4f ms yield_resume_time=%.4f ms "
                    "result_wait_time=%.4f ms to_gpu_send_time=%.4f ms "
                    "final_yield_wait=%.4f ms final_sync_time=%.4f ms "
                    "wall_time=%.4f ms",
                    req_id,
                    len(keys),
                    self.num_layers,
                    contains_time * 1000,
                    submit_time * 1000,
                    yield_resume_time * 1000,
                    result_wait_time * 1000,
                    to_gpu_send_time * 1000,
                    final_yield_wait * 1000,
                    final_sync_time * 1000,
                    wall_time * 1000,
                )

        yield ret_mask

    @_lmcache_nvtx_annotate
    def lookup(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        search_range: Optional[List[str]] = None,
        lookup_id: Optional[str] = None,
        pin: bool = False,
        request_configs: Optional[dict] = None,
    ) -> int:
        """
        Checks the existence of KV cache of the tokens from the cache engine.

        :param Optional[Union[torch.Tensor, List[int]]] tokens: the input tokens,
        with shape [seq_len]

        :param Optional[List[int]] hashes: the input hashes, with length [num_chunks]
        :param Optional[List[int]] offsets: the offsets of each chunk,
        with length [num_chunks]

        :param Optional[List[str]] search_range: The range of storage backends
        to search in. Should be a subset of
        ["LocalCPUBackend", "LocalDiskBackend"] for now.
        If None, search in all backends.

        :param Optional[str] lookup_id: The lookup ID to
            associate with the lookup. When pin is true, this argument is
            required to be not None.

        :param bool pin: If True, pin the KV cache in the storage.

        :param Optional[dict] request_configs: the configs of the request.

        :return: An int indicating how many prefix tokens exist inside LMCache.
        """
        # Health check: block operation if LMCache is unhealthy
        if not self.is_healthy():
            logger.warning("LMCache is unhealthy, skipping lookup operation")
            return 0

        assert self.storage_manager is not None

        if tokens is not None:
            lookup_stats = self.stats_monitor.on_lookup_request(len(tokens))
        else:
            assert offsets is not None
            assert hashes is not None
            lookup_stats = self.stats_monitor.on_lookup_request(sum(offsets))

        if search_range is None:
            search_range = self.retrieve_locations

        res = 0
        try:
            lookup_chunk_lengths = (
                extract_chunk_lengths(
                    request_configs,
                    LOOKUP_CHUNK_LENGTHS_CONFIG,
                )
                if tokens is not None
                else None
            )
            chunk_info_iterator = self.token_database.process_tokens(
                tokens=tokens,
                hashes=hashes,
                offsets=offsets,
                request_configs=request_configs,
                chunk_lengths=lookup_chunk_lengths,
                chunk_lengths_config_name=LOOKUP_CHUNK_LENGTHS_CONFIG,
            )

            # TODO: support batched_contains when layerwise is enabled
            if self.use_layerwise:
                for start, end, key in chunk_info_iterator:
                    assert isinstance(key, CacheEngineKey)

                    # TODO(Jiayi): Optimize by checking only the existence of the key
                    # of one layer
                    key_all_layers = key.split_layers(self.num_layers)

                    hit_chunks, block_mapping = self.storage_manager.batched_contains(
                        key_all_layers,  # type: ignore
                        search_range,
                        pin,
                    )
                    # Only all layers are hit and hit in one location,
                    # we consider this key as a hit
                    if hit_chunks == self.num_layers and len(block_mapping) == 1:
                        if pin:
                            assert lookup_id is not None, (
                                "lookup_id is required when pin is True"
                            )
                            location = next(iter(block_mapping.keys()))
                            self.lookup_pins[lookup_id][location].extend(key_all_layers)
                        res = end
                        continue
                    return res
            else:
                chunk_info_list = []
                keys = []
                for chunk_info in chunk_info_iterator:
                    assert isinstance(chunk_info[2], CacheEngineKey)
                    start, end, _ = chunk_info
                    chunk_info_list.append(chunk_info)
                    # chunk_info contains (start, end, key)
                    # chunk_info[2] is the key
                    keys.append(chunk_info[2])
                # hit chunks by prefix matching
                hit_chunks, block_mapping = self.storage_manager.batched_contains(
                    keys, search_range, pin
                )
                if pin and block_mapping:
                    assert lookup_id is not None, (
                        "lookup_id is required when pin is True"
                    )
                    self.lookup_pins[lookup_id] = block_mapping
                for idx, (start, end, key) in enumerate(chunk_info_list):
                    if idx < hit_chunks:
                        res = end
                        continue
                    return res

            # all tokens where found, return the maximal end
            return res
        finally:
            self.stats_monitor.on_lookup_finished(lookup_stats, res)
            # vllm lookup sets pin to True
            if pin:
                # touch_cache is tightly coupled with batched_contains
                self.storage_manager.touch_cache()

    @_lmcache_nvtx_annotate
    def move(
        self,
        tokens: Union[torch.Tensor, List[int]],
        old_position: str,
        new_position: tuple[str, str],
        event_id: str,
        do_copy: bool = True,
    ) -> int:
        """
        Perform cross-node move of the KV cache.
        """
        assert self.storage_manager is not None

        num_tokens = self.lookup(
            tokens,
            search_range=[old_position],
            lookup_id=event_id,
            pin=True,
        )

        if not num_tokens:
            logger.debug("Move is not performed as there are no tokens to move.")
            return 0

        block_mapping = self.lookup_pins[event_id]
        assert len(block_mapping) == 1
        keys = block_mapping[old_position]

        memory_objs = self.storage_manager.batched_get(
            keys=keys,
            location=old_position,
        )
        assert None not in memory_objs, "Failed to get memory objects to move"
        logger.debug(
            f"Trying to send {len(memory_objs)} memory objects to {new_position}"
        )

        # TODO: reduce loops
        token_dim = memory_objs[0].meta.fmt.token_dim()  # type: ignore
        offsets = [m.meta.shape[token_dim] for m in memory_objs]  # type: ignore

        transfer_spec = {
            "target_peer_init_url": new_position[0],
            "offsets": offsets,
        }

        logger.info(self.storage_manager.storage_backends)
        p2p_backend = self.storage_manager.storage_backends["P2PBackend"]

        future = asyncio.run_coroutine_threadsafe(
            p2p_backend.async_batched_submit_put_task(
                keys,
                memory_objs,  # type: ignore
                transfer_spec=transfer_spec,
            ),
            self.storage_manager.loop,
        )

        future.result()

        if not do_copy:
            self.storage_manager.batched_remove(keys, locations=[old_position])

        logger.debug(f"Moving {num_tokens} token from {old_position} to {new_position}")
        return num_tokens

    # TODO(Jiayi): Add layerwise support.
    @_lmcache_nvtx_annotate
    def async_lookup_and_prefetch(
        self,
        lookup_id: str,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        search_range: Optional[List[str]] = None,
        pin: bool = False,
        request_configs: Optional[dict] = None,
    ) -> None:
        """
        An async version of lookup + prefetch.

        There are three categories of backends:
        (1) sync lookup + sync retrieval (e.g., cpu)
        (2) sync lookup + async retrieval (e.g., disk)
        (3) async lookup + async retrieval (e.g., p2p)
        """
        assert self.storage_manager is not None

        keys: list[CacheEngineKey] = []
        cum_chunk_lengths = [0]

        if search_range is None:
            search_range = self.retrieve_locations
        lookup_chunk_lengths = extract_chunk_lengths(
            request_configs,
            LOOKUP_CHUNK_LENGTHS_CONFIG,
        )

        # TODO(Jiayi): make token database able to return list.
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens,
            hashes=hashes,
            offsets=offsets,
            request_configs=request_configs,
            chunk_lengths=lookup_chunk_lengths,
            chunk_lengths_config_name=LOOKUP_CHUNK_LENGTHS_CONFIG,
        ):
            assert isinstance(key, CacheEngineKey)
            keys.append(key)
            cum_chunk_lengths.append(end)

        asyncio.run_coroutine_threadsafe(
            self.storage_manager.async_lookup_and_prefetch(
                lookup_id, keys, cum_chunk_lengths, search_range, pin
            ),
            self.storage_manager.loop,
        )

    def cleanup_memory_objs(self, lookup_id: str) -> None:
        """
        Cleanup memory objects allocated during prefetch for an aborted lookup.

        Called by the scheduler when it determines that an aborted lookup
        has finished its prefetch tasks.
        """
        try:
            # Get the completed future from event_manager
            if (
                self.event_manager.get_event_status(EventType.LOADING, lookup_id)
                != EventStatus.DONE
            ):
                logger.debug(
                    "No completed event found for lookup_id=%s to clean up.", lookup_id
                )
                return
            future = self.event_manager.pop_event(EventType.LOADING, lookup_id)

            # Get memory objects from the future result
            memory_objs = future.result()
            # Flatten nested lists (each backend returns a list of chunks)
            memory_objs_flat = [mm for m in memory_objs for mm in m]

            # Release each memory object
            for key, memory_obj in memory_objs_flat:
                try:
                    logger.debug("Releasing memory object for lookup_id=%s", lookup_id)
                    memory_obj.unpin()
                    memory_obj.ref_count_down()
                except Exception as e:
                    logger.error(f"Error releasing memory object: {e}")
        except Exception as e:
            logger.error(
                f"Error during cleanup_memory_objs for lookup_id={lookup_id}: {e}"
            )

    # TODO(Jiayi): Need to handle the case where `tokens=None`.
    # In this case, we compress all tokens.
    # TODO(Jiayi): support other compression methods.
    @_lmcache_nvtx_annotate
    def compress(
        self,
        tokens: Union[torch.Tensor, List[int]],
        method: str,
        location: str,
        event_id: str,
    ) -> int:
        assert self.storage_manager is not None
        if method not in ["cachegen"]:
            logger.warning(f"Unsupported compression method: {method}.")
            return 0

        # First Party
        from lmcache.v1.storage_backend.naive_serde import CreateSerde

        serializer, _ = CreateSerde(method, self.metadata, self.config)

        num_tokens = self.lookup(
            tokens,
            search_range=[location],
            lookup_id=event_id,
            pin=True,
        )

        if not num_tokens:
            logger.debug("Move is not performed as there are no tokens to move.")
            return 0

        block_mapping = self.lookup_pins[event_id]
        assert len(block_mapping) == 1
        keys = block_mapping[location]

        memory_objs = self.storage_manager.batched_get(
            keys=keys,
            location=location,
        )
        assert None not in memory_objs, (
            "LMCacheEngine.compress: Failed to get memory objects to compress"
        )

        compressed_memory_objs = []
        for memory_obj in memory_objs:
            assert memory_obj is not None
            compressed_memory_obj = serializer.serialize(memory_obj)
            memory_obj.unpin()
            compressed_memory_objs.append(compressed_memory_obj)

        self.storage_manager.batched_remove(keys, locations=[location])

        self.storage_manager.batched_put(
            keys=keys,
            memory_objs=compressed_memory_objs,
            location=location,
        )

        return num_tokens

    @_lmcache_nvtx_annotate
    def decompress(
        self,
        tokens: Union[torch.Tensor, List[int]],
        method: str,
        location: str,
        event_id: str,
    ) -> int:
        assert self.storage_manager is not None
        if method not in ["cachegen"]:
            logger.warning(f"Unsupported decompression method: {method}.")
            return 0

        # First Party
        from lmcache.v1.storage_backend.naive_serde import CreateSerde

        _, deserializer = CreateSerde(method, self.metadata, self.config)

        num_tokens = self.lookup(
            tokens,
            search_range=[location],
            lookup_id=event_id,
            pin=True,
        )

        if not num_tokens:
            logger.debug("there are no tokens to decompress.")
            return 0

        block_mapping = self.lookup_pins[event_id]
        assert len(block_mapping) == 1
        keys = block_mapping[location]

        compressed_memory_objs = self.storage_manager.batched_get(
            keys=keys,
            location=location,
        )

        assert None not in compressed_memory_objs, (
            "LMCacheEngine.compress: Failed to get compressed "
            "memory objects to decompress"
        )

        memory_objs = []
        for compressed_memory_obj in compressed_memory_objs:
            assert compressed_memory_obj is not None
            memory_obj = deserializer.deserialize(compressed_memory_obj)
            compressed_memory_obj.unpin()
            memory_objs.append(memory_obj)

        self.storage_manager.batched_remove(keys, locations=[location])

        self.storage_manager.batched_put(
            keys=keys,
            memory_objs=memory_objs,
            location=location,
        )

        return num_tokens

    @_lmcache_nvtx_annotate
    def lookup_unpin(self, lookup_id: str) -> None:
        if lookup_id in self.lookup_pins:
            assert self.storage_manager is not None
            for location, keys in self.lookup_pins.pop(lookup_id).items():
                self.storage_manager.batched_unpin(keys, [location])

        elif (
            self.async_loading is not None
            and self.event_manager.get_event_status(EventType.LOADING, lookup_id)
            != EventStatus.NOT_FOUND
        ):
            self.cleanup_memory_objs(lookup_id)

    @_lmcache_nvtx_annotate
    def clear(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        locations: Optional[List[str]] = None,
        request_configs: Optional[dict] = None,
    ) -> int:
        # TODO: need to clear by request_configs
        if self.save_only_first_rank:
            if self.metadata.is_first_rank():
                num_removed = self._clear(tokens, locations, request_configs)
                return num_removed
            else:
                return 0
        return self._clear(tokens, locations, request_configs)

    @_lmcache_nvtx_annotate
    def get_kv_events(self) -> Iterable[CacheStoreEvent]:
        if self.kv_events_enabled and (events := self.kv_events):
            self.kv_events = []
            return events
        return []

    def _clear(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        locations: Optional[List[str]] = None,
        request_configs: Optional[dict] = None,
    ) -> int:
        assert self.storage_manager is not None
        assert isinstance(self.storage_manager, StorageManager)
        # Clear all caches if tokens is None
        if tokens is None or len(tokens) == 0:
            num_cleared = self.storage_manager.clear(locations)
            return num_cleared

        num_removed = 0
        # Only remove the caches for the given tokens
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens, request_configs=request_configs
        ):
            assert isinstance(key, CacheEngineKey)
            removed = self.storage_manager.remove(key, locations)
            num_removed += removed
        return num_removed

    @_lmcache_nvtx_annotate
    def health(
        self,
    ) -> int:
        """
        Check the health of the cache engine.
        return: 0 if healthy, otherwise the error code
        """
        assert self.storage_manager is not None
        return 0 if self.storage_manager.memcheck() else -1

    def close(self) -> None:
        """Close the cache engine and free all the resources"""
        logger.info("Closing LMCacheEngine...")

        if self.lmcache_worker is not None:
            try:
                logger.info("Closing lmcache_worker...")
                self.lmcache_worker.close()
                logger.info("lmcache_worker closed successfully")
            except Exception as e:
                logger.error(f"Error closing lmcache_worker: {e}")

        if self._layerwise_put_executor is not None:
            try:
                logger.info("Closing layerwise put executor...")
                self._layerwise_put_executor.shutdown(wait=True)
                self._layerwise_put_executor = None
                logger.info("layerwise put executor closed successfully")
            except Exception as e:
                logger.error(f"Error closing layerwise put executor: {e}")

        try:
            logger.info("Closing storage_manager...")
            if self.storage_manager is not None:
                self.storage_manager.close()
            logger.info("storage_manager closed successfully")
        except Exception as e:
            logger.error(f"Error closing storage_manager: {e}")

        logger.info("LMCacheEngine closed.")

    def _async_process_tokens_internal(
        self,
        tokens,
        mask,
        ret_mask,
        **kwargs,
    ) -> ProcessTokensInternalResult:
        """
        This function is used to get the memory objects from the event manager.

        Args:
            tokens: Input tokens to process
            mask: Mask indicating valid token positions
            ret_mask: Output mask updated with cache hit positions
            **kwargs: Additional keyword arguments
        """
        assert "req_id" in kwargs, "req_id is required for async loading"
        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)
        lookup_chunk_lengths = extract_chunk_lengths(
            request_configs,
            LOOKUP_CHUNK_LENGTHS_CONFIG,
        )

        tot_kv_size = 0
        chunks: List[ProcessedChunk] = []
        future = self.event_manager.get_event_future(
            EventType.LOADING, kwargs["req_id"]
        )
        # As mentioned in async_lookup_and_prefetch(), the future.result()
        # is key data pair for each chunk in each tier. So extract the key
        # and memory object pairs to memory_obj_map
        try:
            keyed_memory_objs = future.result()
            memory_obj_map: dict[CacheEngineKey, MemoryObj] = {}
        except Exception as e:
            logger.error(f"Error popping event for request {kwargs['req_id']}: {e}")
            return [], 0

        for backend_results in keyed_memory_objs:
            for key, memory_obj in backend_results:
                memory_obj_map[key] = memory_obj

        # TODO(Jiayi): hashing inside `process_tokens` can be skipped.
        used_keys: set[CacheEngineKey] = set()
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens,
            mask=mask,
            request_configs=request_configs,
            chunk_lengths=lookup_chunk_lengths,
            chunk_lengths_config_name=LOOKUP_CHUNK_LENGTHS_CONFIG,
        ):
            assert isinstance(key, CacheEngineKey)
            memory_obj = memory_obj_map.get(key)
            if memory_obj is None:
                # returned chunks are expected to be contiguous.
                # break at the first missing chunk.
                break
            chunks.append((key, memory_obj, start, end))
            tot_kv_size += memory_obj.get_size()
            ret_mask[start:end] = True
            used_keys.add(key)

        # NOTE: free the memory objects that are not hit.
        for key, mem_obj in memory_obj_map.items():
            if key not in used_keys:
                mem_obj.ref_count_down()

        return chunks, tot_kv_size

    def _process_tokens_internal(
        self,
        tokens,
        mask,
        ret_mask,
        **kwargs,
    ) -> ProcessTokensInternalResult:
        """Process tokens and populate the reordered lists.

        This function is used to process tokens and populate the reordered lists.

        Args:
            tokens: Input tokens to process
            mask: Mask indicating valid token positions
            ret_mask: Output mask updated with cache hit positions
            **kwargs: Additional keyword arguments
        """
        assert self.storage_manager is not None

        tot_kv_size = 0
        reordered_chunks: List[ProcessedChunk] = []
        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)
        lookup_chunk_lengths = extract_chunk_lengths(
            request_configs,
            LOOKUP_CHUNK_LENGTHS_CONFIG,
        )

        chunk_infos = []
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens,
            mask=mask,
            request_configs=request_configs,
            chunk_lengths=lookup_chunk_lengths,
            chunk_lengths_config_name=LOOKUP_CHUNK_LENGTHS_CONFIG,
        ):
            assert isinstance(key, CacheEngineKey)
            chunk_infos.append((key, start, end))

        # block_mapping: location -> [(CacheEngineKey, start, end)]
        if (
            "req_id" in kwargs
            and kwargs["req_id"] in self.lookup_pins
            and len(self.lookup_pins[kwargs["req_id"]]) == 1
        ):
            location = next(iter(self.lookup_pins[kwargs["req_id"]].keys()))
            block_mapping = {location: chunk_infos}
        else:
            block_mapping = self.storage_manager.get_block_mapping(chunk_infos)

        last_failed_block_start = None
        for location, blocks in block_mapping.items():
            keys = [key for key, _, _ in blocks]
            memory_objs = self.storage_manager.batched_get(
                keys=keys,
                location=location,
            )

            for (key, start, end), memory_obj in zip(blocks, memory_objs, strict=False):
                if memory_obj is None:
                    logger.warning(
                        "The cache block is in the storage, but it can't be retrieved"
                    )
                    if (
                        last_failed_block_start is None
                        or last_failed_block_start < start
                    ):
                        last_failed_block_start = start
                    break
                reordered_chunks.append((key, memory_obj, start, end))
                tot_kv_size += memory_obj.get_size()
                ret_mask[start:end] = True

        if last_failed_block_start is not None:
            ret_mask[last_failed_block_start:] = False

            reordered_chunks = [
                (key, memory_obj, start, end)
                for key, memory_obj, start, end in reordered_chunks
                if end < last_failed_block_start
            ]
        return reordered_chunks, tot_kv_size

    def _broadcast_or_receive_memory_objs(
        self,
        reordered_chunks,
        ret_mask,
    ):
        """
        Handles broadcasting or receiving memory objects in a distributed environment.

        This function implements the communication logic where:
        - The first rank (coordinator) broadcasts memory objects and metadata to others
        - Other ranks receive and reconstruct the memory objects

        Parameters:
        reordered_chunks: List of tuples containing [key, memory object, start, end]
        ret_mask: Boolean mask indicating which positions have been processed

        Side Effects:
        - On first rank:
          * Broadcasts chunk count and each chunk's combined metadata
          * Broadcasts tensor data
        - On other ranks:
          * Receives chunk data and populates reordered_chunks
          * Updates ret_mask to mark received positions as True
        """
        if self.metadata.is_first_rank():
            # Broadcast total chunk count
            chunk_count = len(reordered_chunks)
            self.broadcast_object_fn(chunk_count, self.metadata.first_rank)

            # Broadcast each chunk's data
            for key, memory_obj, start, end in reordered_chunks:
                # Combine (start, end) and metadata into single broadcast
                metadata_dict = memory_obj.metadata.to_dict()
                combined_metadata = (start, end, metadata_dict)
                self.broadcast_object_fn(combined_metadata, self.metadata.first_rank)

                # Broadcast tensor data
                raw_tensor = memory_obj.raw_tensor
                assert raw_tensor is not None
                tensor_to_broadcast = raw_tensor.to(f"cuda:{self.metadata.worker_id}")
                self.broadcast_fn(tensor_to_broadcast, self.metadata.first_rank)
        else:
            # Receive total chunk count
            chunk_count = self.broadcast_object_fn(None, self.metadata.first_rank)
            if chunk_count is None:
                logger.warning(
                    f"rank={self.metadata.worker_id} received None chunk_count"
                )
                return

            # Fill reordered_chunks with received data
            for _ in range(chunk_count):
                # Receive combined metadata (start, end, metadata_dict)
                combined_metadata = self.broadcast_object_fn(
                    None, self.metadata.first_rank
                )
                if combined_metadata is None:
                    logger.warning(
                        f"rank={self.metadata.worker_id} "
                        "received None combined_metadata"
                    )
                    break
                start, end, metadata_dict = combined_metadata
                ret_mask[start:end] = True

                # Create tensor and receive data
                metadata = MemoryObjMetadata.from_dict(metadata_dict)
                local_rank = self.metadata.worker_id % torch.cuda.device_count()
                raw_tensor = torch.empty(
                    torch.Size([metadata.get_size()]),
                    dtype=torch.uint8,
                    device=f"cuda:{local_rank}",
                )
                self.broadcast_fn(raw_tensor, self.metadata.first_rank)

                # Create temporary memory object (key not needed for other ranks)
                memory_obj = TensorMemoryObj(
                    raw_data=raw_tensor, metadata=metadata, parent_allocator=None
                )
                reordered_chunks.append((None, memory_obj, start, end))

    def _is_passive(self):
        """
        A 'passive' CacheEngine means that the node itself will not store/retrieve
        the data directly, but from the "active" worker (i.e., rank 0 in MLA)
        """
        return self.save_only_first_rank and not self.metadata.is_first_rank()

    def _get_slot_mapping_list(
        self,
        slot_mapping: Optional[Union[torch.Tensor, List[int]]],
    ) -> Optional[List[int]]:
        """
        Convert slot_mapping to list if it's a tensor, otherwise return as is.

        :param slot_mapping: The slot_mapping to convert,
            can be a torch.Tensor or List[int], or None
        :type slot_mapping: Optional[Union[torch.Tensor, List[int]]]
        :return: The slot_mapping as a List[int], or None if input is None
        :rtype: Optional[List[int]]
        """
        if slot_mapping is None:
            return None
        if isinstance(slot_mapping, torch.Tensor):
            return slot_mapping.tolist()
        # At this point, slot_mapping must be List[int]
        return slot_mapping

    def _log_kvcache_for_check(
        self,
        operation: str,
        kwargs: dict,
        token_count: int,
        require_req_id: bool = False,
    ) -> None:
        """
        Helper method to log KVCache Check information.

        This method centralizes the KVCache Check logging logic that was
        duplicated in multiple methods.

        Args:
            operation: The operation being performed (e.g., "Store", "retrieve")
            kwargs: The keyword arguments containing slot_mapping and req_id
            token_count: The number of tokens involved in the operation
            require_req_id: Whether req_id must be present (default: False)
        """
        if not self.kvcache_check_log_enabled:
            return

        slot_mapping = kwargs.get("slot_mapping")
        if slot_mapping is None:
            return

        if require_req_id:
            req_id = kwargs.get("req_id")
            if req_id is None:
                return
        else:
            req_id = kwargs.get("req_id", "unspecified")

        # Convert slot_mapping to list if it's a tensor
        slot_mapping_list = self._get_slot_mapping_list(slot_mapping)
        # slot_mapping_list should not be None when slot_mapping is not None
        assert slot_mapping_list is not None

        logger.info(
            "[KVCache Check] %s request %s, tokens=%d, slot_mapping: %s",
            operation,
            req_id,
            token_count,
            compress_slot_mapping(slot_mapping_list),
        )


class LMCacheEngineBuilder:
    _instances: Dict[str, LMCacheEngine] = {}
    _cfgs: Dict[str, LMCacheEngineConfig] = {}
    _metadatas: Dict[str, LMCacheMetadata] = {}
    _stat_loggers: Dict[str, LMCacheStatsLogger] = {}

    # TODO(Jiayi): Please remove this helper function in the future.
    # Currently, it's only used for testing.
    @staticmethod
    def _Create_memory_allocator(
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        numa_mapping: Optional[NUMAMapping] = None,
    ) -> MemoryAllocatorInterface:
        # NOTE: should remove this function after fixing the unit tests:
        # raise RuntimeError("_Create_memory_allocator is deprecated!")
        extra_config = config.extra_config
        enable_nixl_storage = extra_config is not None and extra_config.get(
            "enable_nixl_storage"
        )

        if enable_nixl_storage:
            # TODO(Jiayi): weird to import from transfer utils.
            # First Party
            from lmcache.v1.transfer_channel.transfer_utils import (
                get_correct_device,
            )

            corrected_device = get_correct_device(
                config.nixl_buffer_device,
                metadata.worker_id,
            )

            buffer = torch.empty(
                config.nixl_buffer_size,
                dtype=torch.uint8,
                device=corrected_device,
            )

            if corrected_device == "cpu":
                torch.cuda.cudart().cudaHostRegister(
                    buffer.data_ptr(), config.nixl_buffer_size, 0
                )
            else:
                logger.info(f"Setting cuda device to {corrected_device} ")
                torch.cuda.set_device(corrected_device)

            return PagedTensorMemoryAllocator(
                buffer,
                [torch.Size(metadata.kv_shape)],
                [metadata.kv_dtype],
                MemoryFormat.KV_2LTD,
            )

        if config.gds_path is not None:
            assert config.cufile_buffer_size is not None
            return CuFileMemoryAllocator(config.cufile_buffer_size * 1024**2)

        max_local_cpu_size = config.max_local_cpu_size
        # save_only_first_rank only works when use mla
        save_only_first_rank = (
            config.get_extra_config_value("save_only_first_rank", metadata.use_mla)
            and metadata.use_mla
        )
        if save_only_first_rank and metadata.is_first_rank():
            # Only the first rank will save the cache,
            # so we need to set it lager than other ranks
            first_rank_max_local_cpu_size = (
                config.extra_config.get(
                    "first_rank_max_local_cpu_size", max_local_cpu_size
                )
                if config.extra_config
                else max_local_cpu_size
            )
            return MixedMemoryAllocator(
                int(first_rank_max_local_cpu_size * 1024**3),
                numa_mapping=numa_mapping,
            )
        return MixedMemoryAllocator(
            int(max_local_cpu_size * 1024**3),
            numa_mapping=numa_mapping,
        )

    @staticmethod
    def _Create_token_database(
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
    ) -> TokenDatabase:
        if config.enable_blending:
            return SegmentTokenDatabase(config, metadata)
        return ChunkedTokenDatabase(config, metadata)

    @classmethod
    def get_or_create(
        cls,
        instance_id: str,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        gpu_connector: Optional[GPUConnectorInterface],
        broadcast_fn: Callable[[torch.Tensor, int], None],
        broadcast_object_fn: Callable[[Any, int], Any],
    ) -> LMCacheEngine:
        """
        Builds a new LMCacheEngine instance if it doesn't already exist for the
        given ID.

        raises: ValueError if the instance already exists with a different
            configuration.
        """
        logger.info(f"Creating LMCacheEngine instance {instance_id}")
        if instance_id not in cls._instances:
            numa_mapping = NUMADetector.get_numa_mapping(config)
            logger.info(f"NUMA mapping for instance {instance_id}: {numa_mapping}")
            token_database = cls._Create_token_database(config, metadata)
            stat_logger = LMCacheStatsLogger(
                metadata,
                log_interval=10,
                config=config,
            )

            engine = LMCacheEngine(
                config,
                metadata,
                token_database,
                gpu_connector,
                broadcast_fn,
                broadcast_object_fn,
            )

            cls._instances[instance_id] = engine
            cls._cfgs[instance_id] = config
            cls._metadatas[instance_id] = metadata
            cls._stat_loggers[instance_id] = stat_logger
            return engine
        else:
            if (
                cls._cfgs[instance_id] != config
                or cls._metadatas[instance_id] != metadata
            ):
                raise ValueError(
                    f"Instance {instance_id} already exists with a different "
                    f"configuration or metadata."
                )
            return cls._instances[instance_id]

    @classmethod
    def get(cls, instance_id: str) -> Optional[LMCacheEngine]:
        """Returns the LMCacheEngine instance associated with the instance ID,
        or None if not found."""
        return cls._instances.get(instance_id)

    @classmethod
    def destroy(cls, instance_id: str) -> None:
        """Close and delete the LMCacheEngine instance by the instance ID"""
        # TODO: unit test for this
        logger.info(f"Destroying LMCacheEngine instance: {instance_id}")

        if instance_id in cls._instances:
            stat_logger = cls._stat_loggers[instance_id]
            try:
                logger.info("Shutting down stats logger...")
                stat_logger.shutdown()
                logger.info("Stats logger shut down successfully")
            except Exception as e:
                logger.error(f"Error shutting down stats logger: {e}")

            engine = cls._instances[instance_id]
            try:
                logger.info("Closing cache engine...")
                engine.close()
                logger.info("Cache engine closed successfully")
            except Exception as e:
                logger.error(f"Error closing cache engine: {e}")

            try:
                logger.info("Cleaning up instance dictionaries...")
                cls._instances.pop(instance_id, None)
                cls._cfgs.pop(instance_id, None)
                cls._metadatas.pop(instance_id, None)
                cls._stat_loggers.pop(instance_id, None)
                logger.info("Instance dictionaries cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up instances: {e}")

            try:
                logger.info("Destroying stats monitor...")
                LMCStatsMonitor.DestroyInstance()
                logger.info("Stats monitor destroyed successfully")
            except Exception as e:
                logger.error(f"Error destroying stats monitor: {e}")

            logger.info(f"LMCacheEngine instance {instance_id} destroyed")
        else:
            logger.warning(f"Instance {instance_id} not found for destruction")
