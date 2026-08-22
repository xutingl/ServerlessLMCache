# SPDX-License-Identifier: Apache-2.0
# Standard
import abc
from contextlib import nullcontext
import os
import statistics
import threading
import time
from typing import Any, Callable, List, Optional, Tuple, Union

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import EngineType, _lmcache_nvtx_annotate
from lmcache.v1.compute.blend.utils import LMCBlenderBuilder
from lmcache.v1.gpu_connector.utils import (
    LayoutHints,
    assert_is_vllm_flash_attn_or_flash_infer,
    discover_gpu_kv_format,
    ensure_contiguous_kv_caches,
    get_block_size,
    get_elements_per_layer,
    get_head_size,
    get_num_blocks,
    get_page_buffer_size,
    get_tokens_per_layer,
    permute_kv_caches_to_contiguous,
)
from lmcache.v1.memory_management import GPUMemoryAllocator  # noqa: E501
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    TensorMemoryObj,
    TensorMemoryAllocator,
)
from lmcache.v1.metadata import LMCacheMetadata

if torch.cuda.is_available():
    # First Party
    import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)

_PD_LAYERWISE_MIN_STORE_GROUP_LAYERS = 4
_PD_LAYERWISE_TARGET_STORE_GROUP_BYTES = 64 * 1024**2


class LayerwiseOffloadHandle:
    def __init__(
        self,
        layer_id: int,
        wait_fn: Callable[[], None],
        request_batch: Optional[object] = None,
    ) -> None:
        self.layer_id = layer_id
        self._wait_fn = wait_fn
        self.request_batch = request_batch

    def wait(self) -> None:
        self._wait_fn()


class _DeferredLayerwiseBatchWaiter:
    """Resolve a layer handle after the model thread flushes its request batch."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._wait_fn: Optional[Callable[[], None]] = None
        self._error: Optional[BaseException] = None

    def resolve(self, wait_fn: Callable[[], None]) -> None:
        with self._condition:
            if self._wait_fn is not None or self._error is not None:
                raise RuntimeError("layerwise request batch waiter already resolved")
            self._wait_fn = wait_fn
            self._condition.notify_all()

    def fail(self, error: BaseException) -> None:
        with self._condition:
            if self._wait_fn is not None or self._error is not None:
                return
            self._error = error
            self._condition.notify_all()

    def wait(self) -> None:
        with self._condition:
            while self._wait_fn is None and self._error is None:
                self._condition.wait()
            error = self._error
            wait_fn = self._wait_fn
        if error is not None:
            raise error
        assert wait_fn is not None
        wait_fn()


class _LayerwiseRequestBatchEntry:
    def __init__(
        self,
        *,
        group_start: int,
        group_end: int,
        memory_objs: List[List[MemoryObj]],
        slot_mapping_chunks: tuple[torch.Tensor, ...],
        req_id: Optional[str],
    ) -> None:
        self.group_start = group_start
        self.group_end = group_end
        self.memory_objs = memory_objs
        self.slot_mapping_chunks = slot_mapping_chunks
        self.req_id = req_id
        self.waiter = _DeferredLayerwiseBatchWaiter()


class LayerwiseRequestBatch:
    """Collect one scheduler batch before launching a shared grouped D2H."""

    def __init__(
        self,
        group_size: int,
        flush_fn: Callable[
            ["LayerwiseRequestBatch", list[_LayerwiseRequestBatchEntry]],
            None,
        ],
    ) -> None:
        self.group_size = group_size
        self._flush_fn = flush_fn
        self._lock = threading.Lock()
        self._flush_lock = threading.Lock()
        self._pending: list[_LayerwiseRequestBatchEntry] = []
        self._error: Optional[BaseException] = None
        self.layout_signature: Optional[tuple[Any, ...]] = None
        self.cpu_staging: Optional[torch.Tensor] = None
        self.combined_slot_mapping: Optional[torch.Tensor] = None
        self.slot_mapping_ready_event: Optional[torch.cuda.Event] = None
        self.gpu_buffers: dict[int, torch.Tensor] = {}

    def register(self, entry: _LayerwiseRequestBatchEntry) -> None:
        with self._lock:
            if self._error is not None:
                raise RuntimeError(
                    "layerwise request batch is aborted"
                ) from self._error
            self._pending.append(entry)

    def flush(self) -> int:
        with self._lock:
            entries = self._pending
            self._pending = []
            error = self._error
        if error is not None:
            raise RuntimeError("layerwise request batch is aborted") from error
        if not entries:
            return 0
        try:
            with self._flush_lock:
                self._flush_fn(self, entries)
        except BaseException as exc:
            for entry in entries:
                entry.waiter.fail(exc)
            raise
        return len(entries)

    def abort(self, error: BaseException) -> None:
        with self._lock:
            if self._error is None:
                self._error = error
            entries = self._pending
            self._pending = []
        for entry in entries:
            entry.waiter.fail(error)


class _LayerwiseBatchViewAllocator:
    """Release one shared allocator object after all request views are done."""

    def __init__(self, owner: TensorMemoryObj, view_count: int) -> None:
        self._owner: Optional[TensorMemoryObj] = owner
        self._remaining = view_count
        self._lock = threading.Lock()

    def free(self, memory_obj: MemoryObj, allocator_type: Optional[str] = None) -> None:
        del allocator_type
        owner: Optional[TensorMemoryObj] = None
        with self._lock:
            if not memory_obj.is_valid():
                return
            memory_obj.invalidate()
            self._remaining -= 1
            if self._remaining == 0:
                owner = self._owner
                self._owner = None
        if owner is not None:
            owner.ref_count_down()


class _CudaEventLayerwiseOffloadHandle(LayerwiseOffloadHandle):
    """Minimal event-backed handle for the layerwise single-chunk hot path."""

    def __init__(
        self,
        layer_id: int,
        ready_event: torch.cuda.Event,
        device: torch.device,
    ) -> None:
        self.layer_id = layer_id
        self._ready_event = ready_event
        self._device = device

    def wait(self) -> None:
        with torch.cuda.device(self._device):
            self._ready_event.synchronize()


class _GPUBufferReservation:
    """Return pooled GPU buffers only after their CUDA work is safe."""

    def __init__(self, streams: List[torch.cuda.Stream]) -> None:
        self._streams = streams
        self._memory_objs: List[MemoryObj] = []
        self._released = False
        self._lock = threading.Lock()

    def add(self, memory_obj: MemoryObj) -> None:
        with self._lock:
            if self._released:
                raise RuntimeError("cannot add a GPU buffer to a released reservation")
            self._memory_objs.append(memory_obj)

    def release(self, *, synchronize: bool) -> None:
        with self._lock:
            if self._released:
                return
            if synchronize:
                for stream in self._streams:
                    stream.synchronize()
            memory_objs = self._memory_objs
            self._memory_objs = []
            self._released = True

        for memory_obj in memory_objs:
            memory_obj.ref_count_down()


class GPUConnectorInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        # FIXME (Yihua): We shouldn't put start and end here since
        # it's not the responsibility of the GPUConnector to know
        # the token-sequence-related information.
        """Store the data in the memory object into a GPU buffer.
        Sub-classes should define the format of the kwargs.

        :param MemoryObj memory_obj: The memory object to be copied into GPU.
        :param int start: The starting index of the data in the corresponding
            token sequence.
        :param int end: The ending index of the data in the corresponding
            token sequence.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        # FIXME (Yihua): We shouldn't put start and end here since
        # it's not the responsibility of the GPUConnector to know
        # the token-sequence-related information.
        """Load the data from a GPU buffer into the memory object.
        Sub-classes should define the format of the kwargs.

        :param MemoryObj memory_obj: The memory object to store the data from
            GPU.
        :param int start: The starting index of the data in the corresponding
            token sequence.
        :param int end: The ending index of the data in the corresponding
            token sequence.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]], List[MemoryObj]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """
        Batched load the data from a GPU memory into the memory objects.
        Sub-classes should define the format of the kwargs.

        :param Union[List[List[MemoryObj]], List[MemoryObj]] memory_obj:
            The memory objects to store the data from GPU.
        :param List[int] starts: The starting indices of the data in the corresponding
            token sequence.
        :param List[int] ends: The ending indices of the data in the corresponding
            token sequence.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def batched_to_gpu(
        self,
        memory_objs: Union[
            List[List[MemoryObj]], List[MemoryObj], List[int], None
        ] = None,
        starts: Optional[List[int]] = None,
        ends: Optional[List[int]] = None,
        **kwargs,
    ):
        """
        Batched store the data from the memory objects to GPU kv cache.
        Sub-classes should define the format of the kwargs.

        For non-layerwise connectors:
        :param Union[List[List[MemoryObj]], List[MemoryObj]] memory_obj:
            The memory objects to store the data to GPU.
        :param List[int] starts: The starting indices of the data in the corresponding
            token sequence.
        :param List[int] ends: The ending indices of the data in the corresponding
            token sequence.

        For layerwise connectors (generator pattern):
        :param List[int] memory_objs: Actually the starts list
        (positional compatibility)
        :param List[int] starts: Actually the ends list
        (positional compatibility)
        Note: Layerwise connectors receive memory objects
        via generator.send()
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_shape(self, num_tokens: int) -> torch.Size:
        """Get the shape of the data given the number of tokens."""
        raise NotImplementedError

    def initialize_kvcaches_ptr(self, **kwargs):
        """Initialize the kvcaches pointers if not already initialized."""
        if "kvcaches" in kwargs:
            self.kvcaches = kwargs["kvcaches"]
            # Ensure contiguity on every call.  HND tensors from vLLM have a
            # non-contiguous logical view (NHD) that must be permuted back to
            # the physical (HND) shape for correct kernel indexing.
            # permute_kv_caches_to_contiguous is a no-op when already contiguous.
            self.kvcaches = permute_kv_caches_to_contiguous(self.kvcaches)


class VLLMPagedMemGPUConnectorV2(GPUConnectorInterface):
    """
    The GPU KV cache should be a nested tuple of K and V tensors.
    More specifically, we have:
    - GPUTensor = Tuple[KVLayer, ...]
    - KVLayer = Tuple[Tensor, Tensor]
    - Tensor: [num_blocks, block_size, num_heads, head_size]

    It will produce / consume memory object with KV_2LTD format
    """

    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = False,
        **kwargs,
    ):
        """
        If use_gpu is true, it will create a gpu intermediate buffer. In this
        case, it requires the following kwargs:
        - chunk_size: The MAX size of the chunk to be copied to GPU.
        - dtype: The data type of the intermediate buffer.
        """
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.kv_cache_pointers = torch.empty(
            num_layers, dtype=torch.int64, device="cpu"
        )
        # Not sure we need a dict here. Maybe a single GPU connector always
        # works with a single device?
        self.kv_cache_pointers_on_gpu: dict[int, torch.Tensor] = {}

        self.kvcaches: Optional[List[torch.Tensor]] = None

        self.gpu_buffer: Optional[torch.Tensor] = None
        self.use_mla = "use_mla" in kwargs and kwargs["use_mla"]
        self.layout_hints: LayoutHints = (
            kwargs.get(  # type: ignore[assignment]
                "layout_hints"
            )
            or {}
        )
        if use_gpu:
            assert "chunk_size" in kwargs, (
                "chunk_size should be provided to create a GPU buffer."
            )
            assert "dtype" in kwargs, "dtype should be provided to create a GPU buffer."
            assert "device" in kwargs, (
                "device should be provided to create a GPU buffer."
            )
            shape = self.get_shape(kwargs["chunk_size"])
            self.gpu_buffer = torch.empty(
                shape, dtype=kwargs["dtype"], device=kwargs["device"]
            )

        self.store_stream = torch.cuda.Stream()
        self.load_stream = torch.cuda.Stream()
        self._store_lock = threading.Lock()

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheMetadata,
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
        layout_hints: Optional[LayoutHints] = None,
    ) -> "VLLMPagedMemGPUConnectorV2":
        """Create a connector from LMCacheMetadata.

        Args:
            metadata: The LMCache engine metadata containing model configuration.
            use_gpu: Whether to use GPU intermediate buffer.
            device: The device to use for the connector.
            layout_hints: Optional hints about KV cache layout from the
                serving engine.

        Returns:
            A new instance of VLLMPagedMemGPUConnectorV2.
        """
        # Extract parameters from metadata
        # kv_shape: (num_layer, 2 or 1, chunk_size, num_kv_head, head_size)
        num_layers = metadata.kv_shape[0]
        chunk_size = metadata.kv_shape[2]
        num_kv_head = metadata.kv_shape[3]
        head_size = metadata.kv_shape[4]
        hidden_dim_size = num_kv_head * head_size

        return cls(
            hidden_dim_size=hidden_dim_size,
            num_layers=num_layers,
            use_gpu=use_gpu,
            chunk_size=chunk_size,
            dtype=metadata.kv_dtype,
            device=device,
            use_mla=metadata.use_mla,
            layout_hints=layout_hints,
        )

    def _initialize_pointers(self, kv_caches: List[torch.Tensor]) -> torch.Tensor:
        self.device = kv_caches[0].device
        assert self.device.type == "cuda", "The device should be CUDA."
        idx = self.device.index
        if idx in self.kv_cache_pointers_on_gpu:
            return self.kv_cache_pointers_on_gpu[idx]

        # contiguous before pointer capture or format discovery
        kv_caches = ensure_contiguous_kv_caches(
            kv_caches, kv_layout=self.layout_hints.get("kv_layout")
        )

        self.kv_cache_pointers.numpy()[:] = [t.data_ptr() for t in kv_caches]
        self.kv_cache_pointers_on_gpu[idx] = torch.empty(
            self.num_layers, dtype=torch.int64, device=self.device
        )
        self.kv_cache_pointers_on_gpu[idx].copy_(self.kv_cache_pointers)

        self.gpu_kv_format = discover_gpu_kv_format(
            kv_caches, EngineType.VLLM, layout_hints=self.layout_hints
        )
        self.num_blocks = get_num_blocks(kv_caches, self.gpu_kv_format)
        self.block_size = get_block_size(kv_caches, self.gpu_kv_format)
        self.page_buffer_size = self.num_blocks * self.block_size
        self.head_size = get_head_size(kv_caches, self.gpu_kv_format)

        return self.kv_cache_pointers_on_gpu[idx]

    @_lmcache_nvtx_annotate
    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Note:
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)


        :raises ValueError: If 'kvcaches' is not provided in kwargs.
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        self.initialize_kvcaches_ptr(**kwargs)

        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if self.use_mla:
            if memory_obj.metadata.fmt != MemoryFormat.KV_MLA_FMT:
                raise ValueError(
                    "The memory object should be in KV_MLA_FMT format in"
                    " order to be processed by VLLMPagedMemGPUConnector"
                )
        else:
            if memory_obj.metadata.fmt != MemoryFormat.KV_2LTD:
                raise ValueError(
                    "The memory object should be in KV_2LTD format in"
                    " order to be processed by VLLMPagedMemGPUConnector"
                )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        kv_cache_pointers = self._initialize_pointers(self.kvcaches)

        # avoid read/write stream race condition for shared block
        # this will only be potentially non-zero for the first
        # block lmcache is transferring back
        vllm_cached = kwargs.get("vllm_cached_tokens", 0)
        skip_prefix_n_tokens = min(end - start, max(0, vllm_cached - start))

        lmc_ops.multi_layer_kv_transfer(
            memory_obj.tensor,
            kv_cache_pointers,
            slot_mapping[start:end],
            self.device,
            self.page_buffer_size,
            lmc_ops.TransferDirection.H2D,
            self.gpu_kv_format,
            block_size=self.block_size,
            head_size=self.head_size,
            skip_prefix_n_tokens=skip_prefix_n_tokens,
        )

    @_lmcache_nvtx_annotate
    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        # This connector owns one shared staging tensor. Keep the complete
        # gather/copy/synchronize sequence exclusive if callers overlap.
        with self._store_lock:
            return self._from_gpu(memory_obj, start, end, **kwargs)

    def _from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Will set the memory_obj.metadata.fmt to MemoryFormat.KV_2LTD.

        Note:
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)

        :raises ValueError: If 'kvcaches' is not provided in kwargs,
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        kv_cache_pointers = self._initialize_pointers(self.kvcaches)

        with torch.cuda.stream(self.store_stream):
            if self.gpu_buffer is None or end - start != self.gpu_buffer.shape[2]:
                lmc_ops.multi_layer_kv_transfer(
                    memory_obj.tensor,
                    kv_cache_pointers,
                    slot_mapping[start:end],
                    self.kvcaches[0].device,
                    self.page_buffer_size,
                    lmc_ops.TransferDirection.D2H,
                    self.gpu_kv_format,
                    block_size=self.block_size,
                    head_size=self.head_size,
                )
            else:
                # kvcaches -> gpu_buffer -> memobj
                assert self.gpu_buffer.device == self.kvcaches[0].device
                tmp_gpu_buffer = self.gpu_buffer[:, :, : end - start, :]
                lmc_ops.multi_layer_kv_transfer(
                    tmp_gpu_buffer,
                    kv_cache_pointers,
                    slot_mapping[start:end],
                    self.kvcaches[0].device,
                    self.page_buffer_size,
                    lmc_ops.TransferDirection.D2H,
                    self.gpu_kv_format,
                    block_size=self.block_size,
                    head_size=self.head_size,
                )
                memory_obj.tensor.copy_(tmp_gpu_buffer, non_blocking=True)

        if not memory_obj.tensor.is_cuda:
            # Force a synchronize if the target buffer is NOT CUDA device
            # NOTE: for better performance, we may not want to sync for every
            # memory object
            self.store_stream.synchronize()

        if self.use_mla:
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

    # TODO(Jiayi): need to optimize to enable real batching
    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        with torch.cuda.stream(self.load_stream):
            for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
                self.to_gpu(memory_obj, start, end, **kwargs)
        self.load_stream.synchronize()

    # TODO(Jiayi): need to optimize to enable real batching
    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.from_gpu(memory_obj, start, end, **kwargs)

    def get_shape(self, num_tokens: int) -> torch.Size:
        kv_size = 1 if self.use_mla else 2
        return torch.Size([kv_size, self.num_layers, num_tokens, self.hidden_dim_size])


class VLLMPagedMemGPUConnectorV3(GPUConnectorInterface):
    def __init__(
        self,
        metadata: LMCacheMetadata,
        device: torch.device,
        use_gpu: bool = False,
        layout_hints: Optional[LayoutHints] = None,
    ):
        assert device.type == "cuda", "The device should be CUDA."
        self.metadata = metadata
        self.device = device
        self.use_mla = metadata.use_mla
        self.chunk_size = metadata.chunk_size
        self.use_gpu = use_gpu
        self.layout_hints: LayoutHints = layout_hints or {}
        self.kvcaches: Optional[List[torch.Tensor]] = None

        self.init = False
        self.group_kv_cache_pointers_on_gpu: Optional[list[torch.Tensor]] = None
        self.group_tmp_buffer: Optional[list[torch.Tensor]] = None

        self.store_stream = torch.cuda.Stream()
        self.load_stream = torch.cuda.Stream()

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheMetadata,
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
        layout_hints: Optional[LayoutHints] = None,
    ) -> "VLLMPagedMemGPUConnectorV3":
        assert device is not None
        return cls(metadata, device, use_gpu, layout_hints=layout_hints)

    def _initialize_kv_cache_pointers(self):
        if self.init:
            return
        assert self.metadata.kv_layer_groups_manager.kv_layer_groups

        # permute to contiguous before capturing pointers or doing format discovery
        self.kvcaches = ensure_contiguous_kv_caches(
            self.kvcaches, kv_layout=self.layout_hints.get("kv_layout")
        )

        if self.use_gpu:
            # init tmp buffer
            tmp_buf_shapes = self.metadata.get_shapes(self.chunk_size)
            tmp_buf_dtypes = self.metadata.get_dtypes()
            assert len(tmp_buf_shapes) == len(tmp_buf_dtypes)
            self.group_tmp_buffer = [
                torch.empty(tmp_buf_shape, dtype=tmp_buf_dtype, device=self.device)
                for tmp_buf_shape, tmp_buf_dtype in zip(
                    tmp_buf_shapes, tmp_buf_dtypes, strict=True
                )
            ]
        self.group_kv_cache_pointers_on_gpu = []
        for group in self.metadata.kv_layer_groups_manager.kv_layer_groups:
            # init kv cache pointers
            num_layers = group.num_layers
            kv_cache_pointers = torch.empty(num_layers, dtype=torch.int64, device="cpu")
            kv_cache_pointers.numpy()[:] = [
                t.data_ptr()
                for i, t in enumerate(self.kvcaches)
                if i in group.layer_indices
            ]
            kv_cache_pointers_on_gpu = torch.empty(
                num_layers, dtype=torch.int64, device=self.device
            )
            kv_cache_pointers_on_gpu.copy_(kv_cache_pointers)
            self.group_kv_cache_pointers_on_gpu.append(kv_cache_pointers_on_gpu)

        self.gpu_kv_format = discover_gpu_kv_format(
            self.kvcaches, EngineType.VLLM, layout_hints=self.layout_hints
        )
        self.num_blocks = get_num_blocks(self.kvcaches, self.gpu_kv_format)
        self.block_size = get_block_size(self.kvcaches, self.gpu_kv_format)
        self.page_buffer_size = self.num_blocks * self.block_size
        self.head_size = get_head_size(self.kvcaches, self.gpu_kv_format)

        self.init = True
        logger.info("init kv cache pointers success in VLLMPagedMemGPUConnectorV3")

    @_lmcache_nvtx_annotate
    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        assert memory_obj.raw_tensor is not None
        assert "slot_mapping" in kwargs
        if self.use_mla:
            assert memory_obj.metadata.fmt == MemoryFormat.KV_MLA_FMT
        else:
            assert memory_obj.metadata.fmt == MemoryFormat.KV_2LTD

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None
        assert self.kvcaches[0].device == self.device
        self._initialize_kv_cache_pointers()
        assert self.group_kv_cache_pointers_on_gpu is not None

        # avoid read/write stream race condition for shared block
        # this will only be potentially non-zero for the first
        # block lmcache is transferring back
        vllm_cached = kwargs.get("vllm_cached_tokens", 0)
        skip_prefix_n_tokens = min(end - start, max(0, vllm_cached - start))

        for i, kv_cache_pointer in enumerate(self.group_kv_cache_pointers_on_gpu):
            memory_obj_tensor = memory_obj.get_tensor(i)
            assert memory_obj_tensor is not None
            lmc_ops.multi_layer_kv_transfer(
                memory_obj_tensor,
                kv_cache_pointer,
                slot_mapping[start:end],
                self.device,
                self.page_buffer_size,
                lmc_ops.TransferDirection.H2D,
                self.gpu_kv_format,
                block_size=self.block_size,
                head_size=self.head_size,
                skip_prefix_n_tokens=skip_prefix_n_tokens,
            )

    @_lmcache_nvtx_annotate
    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        assert memory_obj.raw_tensor is not None
        assert "slot_mapping" in kwargs

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None
        assert self.kvcaches[0].device == self.device
        self._initialize_kv_cache_pointers()
        assert self.group_kv_cache_pointers_on_gpu is not None
        with torch.cuda.stream(self.store_stream):
            if not self.use_gpu or end - start != self.chunk_size:
                for i, kv_cache_pointer in enumerate(
                    self.group_kv_cache_pointers_on_gpu
                ):
                    memory_obj_tensor = memory_obj.get_tensor(i)
                    assert memory_obj_tensor is not None
                    lmc_ops.multi_layer_kv_transfer(
                        memory_obj_tensor,
                        kv_cache_pointer,
                        slot_mapping[start:end],
                        self.device,
                        self.page_buffer_size,
                        lmc_ops.TransferDirection.D2H,
                        self.gpu_kv_format,
                        block_size=self.block_size,
                        head_size=self.head_size,
                    )
            else:
                # kvcaches -> gpu_buffer -> memobj
                assert self.group_tmp_buffer is not None
                for i, kv_cache_pointer in enumerate(
                    self.group_kv_cache_pointers_on_gpu
                ):
                    tmp_gpu_buffer = self.group_tmp_buffer[i][:, :, : end - start, :]
                    lmc_ops.multi_layer_kv_transfer(
                        tmp_gpu_buffer,
                        kv_cache_pointer,
                        slot_mapping[start:end],
                        self.device,
                        self.page_buffer_size,
                        lmc_ops.TransferDirection.D2H,
                        self.gpu_kv_format,
                        block_size=self.block_size,
                        head_size=self.head_size,
                    )
                    memory_obj_tensor = memory_obj.get_tensor(i)
                    assert memory_obj_tensor is not None
                    memory_obj_tensor.copy_(tmp_gpu_buffer, non_blocking=True)

        if not memory_obj.raw_tensor.is_cuda:
            # Force a synchronize if the target buffer is NOT CUDA device
            # NOTE: for better performance, we may not want to sync for every
            # memory object
            self.store_stream.synchronize()

        if self.use_mla:
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        with torch.cuda.stream(self.load_stream):
            for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
                self.to_gpu(memory_obj, start, end, **kwargs)
        self.load_stream.synchronize()

    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.from_gpu(memory_obj, start, end, **kwargs)

    def get_shape(self, num_tokens: int) -> torch.Size:
        raise NotImplementedError


class VLLMBufferLayerwiseGPUConnector(GPUConnectorInterface):
    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = False,
        use_double_buffer: bool = True,
        **kwargs,
    ):
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers

        self.kvcaches: Optional[List[torch.Tensor]] = None
        self.layout_hints: LayoutHints = (
            kwargs.get(  # type: ignore[assignment]
                "layout_hints"
            )
            or {}
        )

        # TODO(Jiayi): remove this hardcode
        self.cache_positions = True

        self.fused_rotary_emb = None

        assert use_gpu, "use_gpu must be true in VLLMBufferLayerwiseGPUConnector"
        assert "dtype" in kwargs, "dtype should be provided to create a GPU buffer."
        assert "device" in kwargs, "device should be provided to create a GPU buffer."

        self.dtype = kwargs["dtype"]
        self.device = kwargs["device"]

        self.load_stream = torch.cuda.Stream()
        self.store_stream = torch.cuda.Stream()

        self.buffer_mapping: dict[int, MemoryObj] = {}

        # track gap positions between blended chunks
        self.current_gap_positions = None

        self.use_gpu = use_gpu
        self.gpu_buffer_allocator = None
        self.element_size = torch.tensor([], dtype=self.dtype).element_size()

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheMetadata,
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
        layout_hints: Optional[LayoutHints] = None,
    ) -> "VLLMBufferLayerwiseGPUConnector":
        """Create a connector from LMCacheMetadata.

        Args:
            metadata: The LMCache engine metadata containing model configuration.
            use_gpu: Whether to use GPU intermediate buffer.
            device: The device to use for the connector.
            layout_hints: Optional hints about KV cache layout from the
                serving engine.

        Returns:
            A new instance of VLLMBufferLayerwiseGPUConnector.
        """
        # Extract parameters from metadata
        # kv_shape: (num_layer, 2 or 1, chunk_size, num_kv_head, head_size)
        num_layers = metadata.kv_shape[0]
        num_kv_head = metadata.kv_shape[3]
        head_size = metadata.kv_shape[4]
        hidden_dim_size = num_kv_head * head_size

        return cls(
            hidden_dim_size=hidden_dim_size,
            num_layers=num_layers,
            use_gpu=use_gpu,
            dtype=metadata.kv_dtype,
            device=device,
            layout_hints=layout_hints,
        )

    def _lazy_initialize_buffer(self, kv_caches):
        """
        Lazily initialize the GPU buffer allocator if it is not initialized yet.
        Currently, we use the `kv_caches` (kv cache pointer) to determine
        the gpu buffer size in gpu connector.
        Also, the first request might be a bit slower due to buffer creation.
        """
        if self.use_gpu and self.gpu_buffer_allocator is None:
            logger.info("Lazily initializing GPU buffer.")
            # NOTE (Jiayi): We use the first layer to determine the gpu buffer size.
            # NOTE (Jiayi): Using the exact number of tokens in the first layer
            # is okay since fragmentation shouldn't exist in the `gpu_buffer_allocator`
            # in layerwise mode.

            kv_caches = ensure_contiguous_kv_caches(
                kv_caches, kv_layout=self.layout_hints.get("kv_layout")
            )
            self.kvcaches = kv_caches
            self.gpu_kv_format = discover_gpu_kv_format(
                kv_caches, EngineType.VLLM, layout_hints=self.layout_hints
            )
            assert_is_vllm_flash_attn_or_flash_infer(self.gpu_kv_format)
            self.tokens_per_layer = get_tokens_per_layer(kv_caches, self.gpu_kv_format)
            self.elements_per_layer = get_elements_per_layer(
                kv_caches, self.gpu_kv_format
            )
            logger.info(
                f"Lazily initializing GPU buffer (max tokens={self.tokens_per_layer})."
            )
            gpu_buffer_size = self.elements_per_layer * self.element_size
            self.gpu_buffer_allocator = GPUMemoryAllocator(
                gpu_buffer_size, device=self.device
            )

    def get_kv(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the KV cache for the given layer ID.
        This function is used to get the KV cache from the GPU buffer.
        """
        if layer_id not in self.buffer_mapping:
            raise ValueError(f"Layer {layer_id} is not loaded into GPU buffer.")

        gpu_buffer = self.buffer_mapping[layer_id].tensor
        assert gpu_buffer is not None
        return gpu_buffer[0], gpu_buffer[1]

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """ """

        raise NotImplementedError

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """ """

        raise NotImplementedError

    @_lmcache_nvtx_annotate
    def batched_to_gpu(self, starts: List[int], ends: List[int], **kwargs):
        """
        This function is a generator that moves the KV cache from the memory
        objects to buffer GPU memory. In each iteration i, it (1) loads the KV
        cache of layer i from CPU -> GPU buffer, (2) recovers the positional
        encoding of the layer i-1's KV cache in the GPU buffer, and (3)
        moves the KV cache of layer i-2 from GPU buffer to paged GPU memory.
        In total, this the generator will yield num_layers + 2 times.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.
        """

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if self.fused_rotary_emb is None and self.cache_positions:
            # TODO(Jiayi): Make this more elegant
            # First Party
            from lmcache.integration.vllm.utils import ENGINE_NAME

            self.lmc_model = LMCBlenderBuilder.get(ENGINE_NAME).layerwise_model
            self.fused_rotary_emb = self.lmc_model.fused_rotary_emb

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        self._lazy_initialize_buffer(self.kvcaches)

        num_all_tokens = ends[-1] - starts[0]
        slot_mapping_full = slot_mapping[starts[0] : ends[-1]]

        # compute gap positions
        gap_mask = torch.ones(
            num_all_tokens, dtype=torch.bool, device=slot_mapping_full.device
        )
        buf_offset = starts[0]

        for start, end in zip(starts, ends, strict=False):
            gap_mask[start - buf_offset : end - buf_offset] = False

        self.current_gap_positions = torch.where(gap_mask)[0]

        buf_offset = starts[0]
        if self.cache_positions:
            new_positions_full = torch.arange(
                starts[0], ends[-1], dtype=torch.int64, device=self.kvcaches[0].device
            )

        buffer_shape = self.get_shape(num_all_tokens)
        assert self.gpu_buffer_allocator is not None
        compute_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
            buffer_shape, self.dtype, MemoryFormat.KV_2TD
        )
        load_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
            buffer_shape, self.dtype, MemoryFormat.KV_2TD
        )
        assert compute_gpu_buffer_obj is not None, (
            "Failed to allocate GPU buffer in GPUConnector"
        )
        assert load_gpu_buffer_obj is not None, (
            "Failed to allocate GPU buffer in GPUConnector"
        )
        assert compute_gpu_buffer_obj.tensor is not None
        assert load_gpu_buffer_obj.tensor is not None

        # current_stream = torch.cuda.current_stream()

        if self.cache_positions:
            old_positions_full = torch.zeros(
                (num_all_tokens,), dtype=torch.int64, device=self.kvcaches[0].device
            )
        for layer_id in range(self.num_layers + 2):
            if layer_id > 1:
                lmc_ops.single_layer_kv_transfer(
                    self.buffer_mapping[layer_id - 2].tensor,
                    self.kvcaches[layer_id - 2],
                    slot_mapping_full,
                    lmc_ops.TransferDirection.H2D,
                    self.gpu_kv_format,
                    token_major=False,  # shape is [2, num_tokens, hidden_dim]
                )
                del self.buffer_mapping[layer_id - 2]

                logger.debug(f"Finished loading layer {layer_id - 2} into paged memory")

            if layer_id > 0 and layer_id <= self.num_layers:
                # NOTE: wait until both compute and load streams are done
                torch.cuda.synchronize()

                # ping-pong the buffers
                compute_gpu_buffer_obj, load_gpu_buffer_obj = (
                    load_gpu_buffer_obj,
                    compute_gpu_buffer_obj,
                )

                if self.cache_positions:
                    assert compute_gpu_buffer_obj.tensor is not None

                    compute_gpu_buffer_obj.tensor[0] = self.fused_rotary_emb(
                        old_positions_full,
                        new_positions_full,
                        compute_gpu_buffer_obj.tensor[0],
                    )

                # gap zeroing after RoPE
                if self.current_gap_positions.numel():
                    compute_gpu_buffer_obj.tensor[:, self.current_gap_positions] = 0.0

                self.buffer_mapping[layer_id - 1] = compute_gpu_buffer_obj

                logger.debug(f"Finished loading layer {layer_id - 1} into buffer")

            if layer_id < self.num_layers:
                memory_objs_layer = yield

                # memobj -> gpu_buffer
                with torch.cuda.stream(self.load_stream):
                    for start, end, memory_obj in zip(
                        starts, ends, memory_objs_layer, strict=False
                    ):
                        assert memory_obj.metadata.fmt == MemoryFormat.KV_2TD
                        assert load_gpu_buffer_obj.tensor is not None
                        load_gpu_buffer_obj.tensor[0][
                            start - buf_offset : end - buf_offset
                        ].copy_(memory_obj.tensor[0], non_blocking=True)

                        load_gpu_buffer_obj.tensor[1][
                            start - buf_offset : end - buf_offset
                        ].copy_(memory_obj.tensor[1], non_blocking=True)

                        if self.cache_positions and layer_id == 0:
                            old_positions_full[
                                start - buf_offset : end - buf_offset
                            ] = memory_obj.metadata.cached_positions

            elif layer_id == self.num_layers:
                yield

        # free the buffer memory
        load_gpu_buffer_obj.ref_count_down()
        compute_gpu_buffer_obj.ref_count_down()

        assert len(self.buffer_mapping) == 0, (
            "There are still layers in the buffer mapping after "
            "releasing the GPU buffers."
        )

        yield

    # TODO(Jiayi): Reduce repetitive operations in `batched_to_gpu`
    # and `batched_from_gpu`.
    @_lmcache_nvtx_annotate
    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]], List[MemoryObj]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """
        This function is a generator that moves the KV cache from the paged GPU
        memory to the memory objects. The first iteration will prepare some
        related metadata and initiate the transfer in the first layer. In each
        of the following iterations, it will first wait until the storing of
        previous layer finishes, and then initiate string the KV cache of the
        current layer one. The storing process of the KV cache is paged GPU
        memory -> GPU buffer -> memory objects. The last iteration simply waits
        for the last layer to finish.
        In total, this the generator will yield num_layers + 1 times.

        :param memory_objs: The memory objects to store the KV cache. The first
            dimension is the number of layers, and the second dimension is the
            number of memory objects (i.e., number of chunks) for each layer.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.

        :raises ValueError: If 'kvcaches' is not provided in kwargs.

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        self._lazy_initialize_buffer(self.kvcaches)

        buf_start = 0
        slot_mapping_chunks = []
        buf_starts_ends = []
        old_positions_chunks = []
        for start, end in zip(starts, ends, strict=False):
            buf_end = buf_start + end - start
            buf_starts_ends.append((buf_start, buf_end))
            slot_mapping_chunks.append(slot_mapping[start:end])
            buf_start = buf_end
            if self.cache_positions:
                old_positions_chunks.append(
                    torch.arange(
                        start, end, device=self.kvcaches[0].device, dtype=torch.int64
                    )
                )

        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)
        buffer_shape = self.get_shape(num_tokens)
        assert self.gpu_buffer_allocator is not None
        tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
            buffer_shape, self.dtype, MemoryFormat.KV_2TD
        )
        assert tmp_gpu_buffer_obj is not None, (
            "Failed to allocate GPU buffer in GPUConnector"
        )
        assert tmp_gpu_buffer_obj.tensor is not None

        current_stream = torch.cuda.current_stream()

        for layer_id in range(self.num_layers):
            memory_objs_layer = memory_objs[layer_id]
            # kvcaches -> gpu_buffer -> memobj
            with torch.cuda.stream(self.store_stream):
                self.store_stream.wait_stream(current_stream)
                lmc_ops.single_layer_kv_transfer(
                    tmp_gpu_buffer_obj.tensor,
                    self.kvcaches[layer_id],
                    slot_mapping_full,
                    lmc_ops.TransferDirection.D2H,
                    self.gpu_kv_format,
                    token_major=False,  # shape is [2, num_tokens, hidden_dim]
                )
                for (buf_start, buf_end), memory_obj, old_positions in zip(
                    buf_starts_ends,
                    memory_objs_layer,
                    old_positions_chunks,
                    strict=False,
                ):
                    assert memory_obj.tensor is not None
                    memory_obj.tensor[0].copy_(
                        tmp_gpu_buffer_obj.tensor[0][buf_start:buf_end],
                        non_blocking=True,
                    )
                    memory_obj.tensor[1].copy_(
                        tmp_gpu_buffer_obj.tensor[1][buf_start:buf_end],
                        non_blocking=True,
                    )
                    if self.cache_positions:
                        memory_obj.metadata.cached_positions = old_positions

            yield
            self.store_stream.synchronize()
            logger.debug(f"Finished offloading layer {layer_id}")

        # free the buffer memory
        tmp_gpu_buffer_obj.ref_count_down()
        yield

    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size([2, num_tokens, self.hidden_dim_size])


class VLLMPagedMemLayerwiseGPUConnector(GPUConnectorInterface):
    """ """

    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = False,
        **kwargs,
    ):
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.use_gpu = use_gpu
        self.layout_hints: LayoutHints = (
            kwargs.get(  # type: ignore[assignment]
                "layout_hints"
            )
            or {}
        )

        self.gpu_buffer_allocator = None

        assert "chunk_size" in kwargs, (
            "chunk_size should be provided to create a GPU buffer."
        )
        assert "dtype" in kwargs, "dtype should be provided to create a GPU buffer."
        assert "device" in kwargs, "device should be provided to create a GPU buffer."

        self.dtype = kwargs["dtype"]
        self.device = kwargs["device"]

        self.kvcaches: Optional[List[torch.Tensor]] = None
        self.kv_cache_pointers = torch.empty(
            num_layers, dtype=torch.int64, device="cpu"
        )
        self.kv_cache_pointers_on_gpu: Optional[torch.Tensor] = None

        # All sizes are in bytes
        self.element_size = torch.tensor([], dtype=self.dtype).element_size()

        self.load_stream = torch.cuda.Stream()
        self.layerwise_store_stream_count = int(
            os.getenv("LMCACHE_LAYERWISE_STORE_STREAMS", "2")
        )
        if self.layerwise_store_stream_count < 1:
            raise ValueError("LMCACHE_LAYERWISE_STORE_STREAMS must be positive")
        self.store_streams = [
            torch.cuda.Stream(device=self.device)
            for _ in range(self.layerwise_store_stream_count)
        ]
        self.chunk_copy_stream_count = int(
            os.getenv("LMCACHE_CHUNK_COPY_STREAMS", "64")
        )
        if self.chunk_copy_stream_count < 1:
            raise ValueError("LMCACHE_CHUNK_COPY_STREAMS must be positive")
        if self.chunk_copy_stream_count < self.layerwise_store_stream_count:
            raise ValueError(
                "LMCACHE_CHUNK_COPY_STREAMS must be at least "
                "LMCACHE_LAYERWISE_STORE_STREAMS"
            )
        self.chunk_copy_streams_per_store = (
            self.chunk_copy_stream_count // self.layerwise_store_stream_count
        )
        self.chunk_copy_streams = [
            torch.cuda.Stream(device=self.device)
            for _ in range(self.chunk_copy_stream_count)
        ]
        self.use_cpu_staging = os.getenv(
            "LMCACHE_USE_CPU_STAGING", "false"
        ).lower() in ("1", "true", "yes")

        self.use_mla = "use_mla" in kwargs and kwargs["use_mla"]

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheMetadata,
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
        layout_hints: Optional[LayoutHints] = None,
    ) -> "VLLMPagedMemLayerwiseGPUConnector":
        """Create a connector from LMCacheMetadata.

        Args:
            metadata: The LMCache engine metadata containing model configuration.
            use_gpu: Whether to use GPU intermediate buffer.
            device: The device to use for the connector.
            layout_hints: Optional hints about KV cache layout from the
                serving engine.

        Returns:
            A new instance of VLLMPagedMemLayerwiseGPUConnector.
        """
        # Extract parameters from metadata
        # kv_shape: (num_layer, 2 or 1, chunk_size, num_kv_head, head_size)
        num_layers = metadata.kv_shape[0]
        chunk_size = metadata.kv_shape[2]
        num_kv_head = metadata.kv_shape[3]
        head_size = metadata.kv_shape[4]
        hidden_dim_size = num_kv_head * head_size

        return cls(
            hidden_dim_size=hidden_dim_size,
            num_layers=num_layers,
            use_gpu=use_gpu,
            chunk_size=chunk_size,
            dtype=metadata.kv_dtype,
            device=device,
            use_mla=metadata.use_mla,
            layout_hints=layout_hints,
        )

    def initialize_gpu_buffer(self, kv_caches: List[torch.Tensor]) -> None:
        """Reserve the reusable GPU staging pool during engine initialization."""
        if not self.use_gpu or self.gpu_buffer_allocator is not None:
            return

        kv_caches = ensure_contiguous_kv_caches(
            kv_caches, kv_layout=self.layout_hints.get("kv_layout")
        )
        self.kvcaches = kv_caches
        self.gpu_kv_format = discover_gpu_kv_format(
            kv_caches, EngineType.VLLM, layout_hints=self.layout_hints
        )
        assert_is_vllm_flash_attn_or_flash_infer(self.gpu_kv_format)
        self.tokens_per_layer = get_tokens_per_layer(kv_caches, self.gpu_kv_format)
        self.elements_per_layer = get_elements_per_layer(
            kv_caches, self.gpu_kv_format
        )
        self.num_blocks = get_num_blocks(kv_caches, self.gpu_kv_format)
        self.block_size = get_block_size(kv_caches, self.gpu_kv_format)
        self.page_buffer_size = self.num_blocks * self.block_size
        self.head_size = get_head_size(kv_caches, self.gpu_kv_format)
        self.kv_cache_pointers.numpy()[:] = [tensor.data_ptr() for tensor in kv_caches]
        self.kv_cache_pointers_on_gpu = torch.empty(
            self.num_layers, dtype=torch.int64, device=self.device
        )
        self.kv_cache_pointers_on_gpu.copy_(self.kv_cache_pointers)

        gpu_buffer_size = (
            self.elements_per_layer
            * self.element_size
            * self.layerwise_store_stream_count
        )
        logger.info(
            "Initializing reusable GPU staging pool during KV cache registration "
            "(max_tokens=%d, store_streams=%d, size_bytes=%d).",
            self.tokens_per_layer,
            self.layerwise_store_stream_count,
            gpu_buffer_size,
        )
        self.gpu_buffer_allocator = GPUMemoryAllocator(
            gpu_buffer_size, device=self.device
        )

    def _require_gpu_buffer(self) -> None:
        if self.use_gpu and self.gpu_buffer_allocator is None:
            raise RuntimeError(
                "GPU staging pool was not initialized during KV cache registration"
            )

    def _select_layerwise_store_stream(
        self, layer_id: int
    ) -> tuple[int, torch.cuda.Stream]:
        stream_id = (layer_id * 2654435761) % self.layerwise_store_stream_count
        return stream_id, self.store_streams[stream_id]

    def layerwise_store_group_size(self, num_tokens: int) -> int:
        bytes_per_layer = self.get_shape(num_tokens).numel() * self.element_size
        return min(
            self.num_layers,
            max(
                _PD_LAYERWISE_MIN_STORE_GROUP_LAYERS,
                _PD_LAYERWISE_TARGET_STORE_GROUP_BYTES
                // max(bytes_per_layer, 1),
            ),
        )

    def create_layerwise_request_batch(
        self,
        total_tokens: int,
    ) -> LayerwiseRequestBatch:
        """Create a layer-major batch spanning all requests in one forward."""

        bytes_per_layer = self.get_shape(total_tokens).numel() * self.element_size
        group_size = min(
            self.num_layers,
            max(
                1,
                _PD_LAYERWISE_TARGET_STORE_GROUP_BYTES
                // max(bytes_per_layer, 1),
            ),
        )
        return LayerwiseRequestBatch(
            group_size,
            self._flush_layerwise_request_batch,
        )

    def _replace_with_cpu_staging_views(
        self,
        memory_objs: List[MemoryObj],
        staging_tensor: torch.Tensor,
        staging_starts: List[int],
        staging_ends: List[int],
    ) -> Tuple[float, float, float]:
        if not memory_objs:
            return 0.0, 0.0, 0.0

        view_start = time.perf_counter()
        total_tokens = staging_ends[-1]
        bytes_per_token = (
            staging_tensor.numel() * staging_tensor.element_size() // total_tokens
        )
        split_sizes = [
            (end - start) * bytes_per_token
            for start, end in zip(staging_starts, staging_ends, strict=True)
        ]
        raw_views = list(
            torch.split(
                staging_tensor.view(torch.uint8).flatten(),
                split_sizes,
            )
        )
        view_creation_time = time.perf_counter() - view_start

        free_start = time.perf_counter()
        parent_allocator = memory_objs[0].parent()
        if parent_allocator is None or any(
            memory_obj.parent() is not parent_allocator
            for memory_obj in memory_objs
        ):
            raise ValueError(
                "CPU staging destinations must share one parent allocator"
            )
        # TensorMemoryAllocator.batched_free sorts its input in place. Keep the
        # request order intact because raw_views follows that logical order.
        parent_allocator.batched_free(list(memory_objs))
        old_buffer_free_time = time.perf_counter() - free_start

        rebind_start = time.perf_counter()
        fmt = MemoryFormat.KV_MLA_FMT if self.use_mla else memory_objs[0].meta.fmt
        for memory_obj, raw_view in zip(memory_objs, raw_views, strict=True):
            if not isinstance(memory_obj, TensorMemoryObj):
                raise TypeError("CPU staging requires TensorMemoryObj destinations")
            memory_obj.raw_data = raw_view
            memory_obj.meta.address = raw_view.data_ptr()
            memory_obj.meta.phy_size = raw_view.numel()
            memory_obj.meta.ref_count = 1
            memory_obj.meta.pin_count = 0
            memory_obj.meta.fmt = fmt
            memory_obj.parent_allocator = None
            memory_obj.valid = True
        object_rebind_time = time.perf_counter() - rebind_start
        return view_creation_time, object_rebind_time, old_buffer_free_time

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """ """

        raise NotImplementedError

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """ """

        raise NotImplementedError

    @_lmcache_nvtx_annotate
    def batched_to_gpu(self, starts: List[int], ends: List[int], **kwargs):
        """
        This function is a generator that moves the KV cache from the memory
        objects to paged GPU memory. The first iteration will prepare some
        related metadata. In each of the following iterations, it will first
        wait until the loading of the previous layer finish, and then load
        one layer of KV cache from the memory objects -> GPU buffer ->
        paged GPU memory. The last iteration simply waits for the last layer
        to finish.
        In total, this the generator will yield num_layers + 2 times.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """

        reservation = _GPUBufferReservation(
            [self.load_stream] if self.use_gpu else []
        )
        generator = self._batched_to_gpu_impl(
            starts,
            ends,
            _gpu_buffer_reservation=reservation,
            **kwargs,
        )
        try:
            yield from generator
        finally:
            # On cancellation the generator may still have H2D work in flight.
            reservation.release(synchronize=True)

    def _batched_to_gpu_impl(self, starts: List[int], ends: List[int], **kwargs):
        reservation: _GPUBufferReservation = kwargs.pop(
            "_gpu_buffer_reservation"
        )
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        sync: bool = kwargs["sync"]

        self._require_gpu_buffer()

        current_stream = torch.cuda.current_stream()
        # Layerwise retrieval keeps the mapping on CPU until this point so its
        # H2D copy and consumers belong to load_stream, without importing
        # unrelated queued vLLM work through wait_stream().
        slot_mapping_host = slot_mapping if slot_mapping.device.type == "cpu" else None
        mapping_stream = (
            torch.cuda.stream(self.load_stream)
            if slot_mapping_host is not None
            else nullcontext()
        )
        with mapping_stream:
            if slot_mapping_host is not None:
                slot_mapping = slot_mapping_host.to(self.device, non_blocking=True)
            slot_mapping_full = torch.cat(
                [
                    slot_mapping[start:end]
                    for start, end in zip(starts, ends, strict=False)
                ],
                dim=0,
            )

        if slot_mapping_host is None:
            # GPU mappings may have been produced on the current vLLM stream.
            # Preserve the dependency for callers that still pass one directly.
            self.load_stream.wait_stream(current_stream)
        slot_mapping_full.record_stream(self.load_stream)

        num_tokens = len(slot_mapping_full)

        tmp_gpu_buffer_obj: Optional[MemoryObj] = None
        if self.use_gpu:
            buffer_shape = self.get_shape(num_tokens)
            assert self.gpu_buffer_allocator is not None
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, MemoryFormat.KV_T2D
            )
            if tmp_gpu_buffer_obj is None:
                raise RuntimeError("Failed to allocate GPU buffer in GPUConnector")
            reservation.add(tmp_gpu_buffer_obj)
            assert tmp_gpu_buffer_obj.tensor is not None

        offset = starts[0]
        for layer_id in range(self.num_layers):
            memory_objs_layer = yield
            if sync:
                current_stream.wait_stream(self.load_stream)
            if layer_id > 0:
                logger.debug(f"Finished loading layer {layer_id - 1}")

            # memobj -> gpu_buffer -> kvcaches
            with torch.cuda.stream(self.load_stream):
                for start, end, memory_obj in zip(
                    starts, ends, memory_objs_layer, strict=False
                ):
                    # Validate memory format
                    if self.use_mla:
                        assert memory_obj.metadata.fmt == MemoryFormat.KV_MLA_FMT, (
                            f"Expected memory format {MemoryFormat.KV_MLA_FMT}, "
                            f"got {memory_obj.metadata.fmt}"
                        )
                    else:
                        assert memory_obj.metadata.fmt == MemoryFormat.KV_T2D, (
                            f"Expected memory format {MemoryFormat.KV_T2D}, "
                            f"got {memory_obj.metadata.fmt}"
                        )
                    if self.use_gpu:
                        assert tmp_gpu_buffer_obj is not None
                        assert tmp_gpu_buffer_obj.tensor is not None
                        tmp_gpu_buffer_obj.tensor[start - offset : end - offset].copy_(
                            memory_obj.tensor, non_blocking=True
                        )
                    else:
                        lmc_ops.single_layer_kv_transfer(
                            memory_obj.tensor,
                            self.kvcaches[layer_id],
                            slot_mapping[start:end],
                            lmc_ops.TransferDirection.H2D,
                            self.gpu_kv_format,
                            token_major=True,
                        )

                if self.use_gpu:
                    assert tmp_gpu_buffer_obj is not None
                    assert tmp_gpu_buffer_obj.tensor is not None
                    lmc_ops.single_layer_kv_transfer(
                        tmp_gpu_buffer_obj.tensor,
                        self.kvcaches[layer_id],
                        slot_mapping_full,
                        lmc_ops.TransferDirection.H2D,
                        self.gpu_kv_format,
                        token_major=True,
                    )
        yield

        # synchronize the last layer
        if sync:
            current_stream.wait_stream(self.load_stream)

        # free the buffer memory
        reservation.release(synchronize=True)

        logger.debug(f"Finished loading all {self.num_layers} layers.")
        yield

    @_lmcache_nvtx_annotate
    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        reservation = _GPUBufferReservation(
            [*self.store_streams, *self.chunk_copy_streams]
            if self.use_gpu
            else []
        )
        generator = self._batched_from_gpu_impl(
            memory_objs,
            starts,
            ends,
            _gpu_buffer_reservation=reservation,
            **kwargs,
        )
        try:
            yield from generator
        finally:
            # A cancelled save may have already submitted gather/D2H work.
            reservation.release(synchronize=True)

    def _batched_from_gpu_impl(
        self,
        memory_objs: Union[List[List[MemoryObj]]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """
        This function is a generator that moves the KV cache from the paged GPU
        memory to the memory objects. The first iteration will prepare some
        related metadata and initiate the transfer in the first layer. In each
        of the following iterations, it will first wait until the storing of
        previous layer finishes, and then initiate string the KV cache of the
        current layer one. The storing process of the KV cache is paged GPU
        memory -> GPU buffer -> memory objects. The last iteration simply waits
        for the last layer to finish.
        In total, this the generator will yield num_layers + 1 times.

        :param memory_objs: The memory objects to store the KV cache. The first
            dimension is the number of layers, and the second dimension is the
            number of memory objects (i.e., number of chunks) for each layer.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """

        reservation: _GPUBufferReservation = kwargs.pop(
            "_gpu_buffer_reservation"
        )
        setup_start = time.perf_counter()
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        sync: bool = kwargs["sync"]
        skip_d2h = os.getenv("PD_BACKEND_SKIP_D2H", "false").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

        layerwise_request_batch = kwargs.get("layerwise_request_batch")
        self._require_gpu_buffer()

        slot_mapping_chunks = tuple(
            slot_mapping[start:end]
            for start, end in zip(starts, ends, strict=False)
        )
        num_tokens = sum(len(chunk) for chunk in slot_mapping_chunks)
        if len(slot_mapping_chunks) == 1:
            slot_mapping_full: Optional[torch.Tensor] = slot_mapping_chunks[0]
        elif isinstance(layerwise_request_batch, LayerwiseRequestBatch):
            # The shared store stream concatenates all requests once at flush.
            slot_mapping_full = None
        else:
            slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)
        fuse_layerwise_offload = bool(kwargs.get("fuse_layerwise_offload", False))

        if (
            fuse_layerwise_offload
            and self.use_gpu
            and not self.use_cpu_staging
            and not skip_d2h
            and len(starts) == 1
            and len(ends) == 1
            and len(memory_objs) == self.num_layers
            and all(len(layer_objs) == 1 for layer_objs in memory_objs)
        ):
            assert slot_mapping_full is not None
            fused_destination = self._fused_layerwise_destination(memory_objs)
            if fused_destination is not None:
                yield from self._batched_from_gpu_fused_layers(
                    memory_objs=memory_objs,
                    slot_mapping=slot_mapping_full,
                    destination=fused_destination,
                    reservation=reservation,
                    req_id=kwargs.get("req_id"),
                )
                return

        staging_starts = []
        staging_ends = []
        staging_cursor = 0
        for start, end in zip(starts, ends, strict=False):
            staging_starts.append(staging_cursor)
            staging_cursor += end - start
            staging_ends.append(staging_cursor)

        buffer_shape = self.get_shape(num_tokens)
        request_batch_compatible = (
            isinstance(layerwise_request_batch, LayerwiseRequestBatch)
            and len(memory_objs) == self.num_layers
            and all(layer_objs for layer_objs in memory_objs)
        )
        single_chunk_layout = (
            len(starts) == 1
            and len(ends) == 1
            and len(memory_objs) == self.num_layers
            and all(len(layer_objs) == 1 for layer_objs in memory_objs)
        )
        single_chunk_grouped_d2h = (
            single_chunk_layout
            and self._fused_layerwise_destination(memory_objs) is not None
        )
        grouped_layerwise_d2h = (
            bool(kwargs.get("batch_layerwise_puts", False))
            and bool(kwargs.get("contiguous_pinned_d2h", False))
            and (request_batch_compatible or single_chunk_grouped_d2h)
        )
        if slot_mapping_full is None and not grouped_layerwise_d2h:
            slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)
        group_size = (
            layerwise_request_batch.group_size
            if isinstance(layerwise_request_batch, LayerwiseRequestBatch)
            else self.layerwise_store_group_size(num_tokens)
        )

        tmp_gpu_buffer_tensors: list[torch.Tensor] = []
        uses_shared_gpu_buffers = (
            grouped_layerwise_d2h and layerwise_request_batch is not None
        )
        # A shared request batch owns its GPU tile buffers.
        if self.use_gpu and not skip_d2h and not uses_shared_gpu_buffers:
            if grouped_layerwise_d2h:
                active_streams = min(
                    self.layerwise_store_stream_count,
                    (self.num_layers + group_size - 1) // group_size,
                )
                allocation_shape = torch.Size([group_size, *buffer_shape])
                try:
                    for stream_id in range(active_streams):
                        tensor = torch.empty(
                            allocation_shape,
                            dtype=self.dtype,
                            device=self.device,
                        )
                        tensor.record_stream(self.store_streams[stream_id])
                        tmp_gpu_buffer_tensors.append(tensor)
                except torch.OutOfMemoryError as exc:
                    raise RuntimeError(
                        "Failed to allocate request-local grouped D2H buffer"
                    ) from exc
            else:
                assert self.gpu_buffer_allocator is not None
                for _ in range(self.layerwise_store_stream_count):
                    tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                        buffer_shape, self.dtype, MemoryFormat.KV_T2D
                    )
                    if tmp_gpu_buffer_obj is None:
                        raise RuntimeError(
                            "Failed to allocate GPU buffer in GPUConnector"
                        )
                    reservation.add(tmp_gpu_buffer_obj)
                    tmp_gpu_buffer_tensor = tmp_gpu_buffer_obj.tensor
                    assert tmp_gpu_buffer_tensor is not None
                    tmp_gpu_buffer_tensors.append(tmp_gpu_buffer_tensor)

        destination_is_pinned: Optional[bool] = None
        destination_is_contiguous: Optional[bool] = None
        source_is_contiguous: Optional[bool] = None
        use_cpu_staging = (
            self.use_cpu_staging and self.use_gpu and sync and not skip_d2h
        )
        connector_timeline_enabled = (
            os.environ.get("PD_BACKEND_LAYER_TIMING") == "1"
            or os.environ.get("PD_BACKEND_CONNECTOR_TIMELINE") == "1"
        )
        if (
            (not connector_timeline_enabled or grouped_layerwise_d2h)
            and self.use_gpu
            and not use_cpu_staging
            and not skip_d2h
            and (grouped_layerwise_d2h or single_chunk_layout)
        ):
            if grouped_layerwise_d2h:
                yield from self._batched_from_gpu_grouped_layers(
                    memory_objs=memory_objs,
                    slot_mapping=slot_mapping_full,
                    slot_mapping_chunks=slot_mapping_chunks,
                    tmp_gpu_buffer_tensors=tmp_gpu_buffer_tensors,
                    reservation=reservation,
                    dependency_claim=kwargs.get("layerwise_dependency_claim"),
                    group_size=group_size,
                    request_batch=layerwise_request_batch,
                    req_id=kwargs.get("req_id"),
                )
                return
            assert slot_mapping_full is not None
            yield from self._batched_from_gpu_single_chunk_fast(
                memory_objs=memory_objs,
                slot_mapping=slot_mapping_full,
                tmp_gpu_buffer_tensors=tmp_gpu_buffer_tensors,
                reservation=reservation,
                dependency_claim=kwargs.get("layerwise_dependency_claim"),
                contiguous_pinned_d2h=bool(
                    kwargs.get("contiguous_pinned_d2h", False)
                ),
            )
            return

        cpu_staging_tensor: Optional[torch.Tensor] = None
        cpu_staging_allocation_time = 0.0
        cpu_view_creation_time = 0.0
        cpu_object_rebind_time = 0.0
        cpu_old_buffer_free_time = 0.0
        cpu_view_count = 0

        offset = starts[0]
        current_stream = torch.cuda.current_stream()
        setup_time = time.perf_counter() - setup_start
        sync_wait_time = 0.0
        wait_stream_call_time = 0.0
        gather_call_time = 0.0
        chunk_copy_loop_time = 0.0
        tensor_view_time = 0.0
        copy_wait_event_call_time = 0.0
        copy_call_times: list[float] = []
        copy_stream_context_other_time = 0.0
        copy_stream_join_call_time = 0.0
        first_eight_copy_call_time = 0.0
        first_eight_copy_count = 0
        last_eight_copy_call_time = 0.0
        last_eight_copy_count = 0
        staging_d2h_submit_time = 0.0
        stream_dependency_gpu_time_ms = 0.0
        gather_gpu_time_ms = 0.0
        d2h_gpu_time_ms = 0.0
        # CUDA timing events are diagnostic instrumentation, not part of the
        # synchronization contract. Creating and querying them for every layer
        # adds measurable host overhead even when timeline logging is disabled.
        gpu_timing = sync and connector_timeline_enabled
        num_chunks_per_layer = len(memory_objs[0]) if memory_objs else 0
        single_chunk_same_stream = (
            self.use_gpu
            and not use_cpu_staging
            and not skip_d2h
            and num_chunks_per_layer == 1
        )
        effective_copy_stream_count = min(
            self.chunk_copy_streams_per_store, num_chunks_per_layer
        )
        active_copy_stream_count = (
            effective_copy_stream_count
            if self.use_gpu
            and not use_cpu_staging
            and not skip_d2h
            and not single_chunk_same_stream
            else 0
        )
        offload_handles: list[LayerwiseOffloadHandle] = []
        timing_lock = threading.Lock()

        def select_layer_copy_streams(store_stream_id: int) -> list[torch.cuda.Stream]:
            if active_copy_stream_count == 0:
                return []
            stream_start = store_stream_id * self.chunk_copy_streams_per_store
            return [
                self.chunk_copy_streams[stream_start + offset]
                for offset in range(active_copy_stream_count)
            ]

        def log_connector_advance(
            put_layer: int,
            store_stream_id: int,
            advance_start: float,
            advance_end: float,
            sync_wait: float,
            stream_dependency_ms: float,
            gather_gpu_ms: float,
            d2h_gpu_ms: float,
            next_enqueue_layer: int,
            enqueue_wall: float,
            wait_stream_call: float,
            gather_call: float,
            chunk_copy_loop: float,
            tensor_view: float,
            copy_wait_event_call: float,
            copy_call: float,
            copy_stream_context_other: float,
            copy_stream_join_call: float,
            copy_count: int,
            staging_allocation: float,
            staging_submit: float,
            staging_view: float,
            staging_rebind: float,
            staging_free: float,
        ) -> None:
            if not connector_timeline_enabled:
                return
            logger.info(
                "[pd-demo-connector-advance] req_id=%s put_layer=%d "
                "store_stream_id=%d next_enqueue_layer=%d "
                "wall_ms=%.4f sync_wait_ms=%.4f "
                "stream_dependency_gpu_ms=%.4f gather_gpu_ms=%.4f "
                "d2h_gpu_ms=%.4f next_enqueue_wall_ms=%.4f "
                "wait_stream_call_ms=%.4f gather_call_ms=%.4f "
                "chunk_copy_loop_ms=%.4f tensor_view_ms=%.4f "
                "copy_wait_event_call_ms=%.4f copy_call_ms=%.4f "
                "copy_stream_context_other_ms=%.4f copy_stream_join_call_ms=%.4f "
                "copy_count=%d staging_allocation_ms=%.4f "
                "staging_d2h_submit_ms=%.4f staging_view_creation_ms=%.4f "
                "staging_object_rebind_ms=%.4f staging_old_buffer_free_ms=%.4f "
                "cpu_staging=%s copy_streams=%d sync=%s",
                kwargs.get("req_id"),
                put_layer,
                store_stream_id,
                next_enqueue_layer,
                (advance_end - advance_start) * 1000,
                sync_wait * 1000,
                stream_dependency_ms,
                gather_gpu_ms,
                d2h_gpu_ms,
                enqueue_wall * 1000,
                wait_stream_call * 1000,
                gather_call * 1000,
                chunk_copy_loop * 1000,
                tensor_view * 1000,
                copy_wait_event_call * 1000,
                copy_call * 1000,
                copy_stream_context_other * 1000,
                copy_stream_join_call * 1000,
                copy_count,
                staging_allocation * 1000,
                staging_submit * 1000,
                staging_view * 1000,
                staging_rebind * 1000,
                staging_free * 1000,
                use_cpu_staging,
                active_copy_stream_count,
                sync,
            )

        for layer_id in range(self.num_layers):
            enqueue_start = time.perf_counter()
            store_stream_id, selected_store_stream = (
                self._select_layerwise_store_stream(layer_id)
            )
            selected_copy_streams = select_layer_copy_streams(store_stream_id)
            tmp_gpu_buffer_tensor = (
                tmp_gpu_buffer_tensors[store_stream_id]
                if self.use_gpu and not skip_d2h
                else None
            )
            layer_start_event = (
                torch.cuda.Event(enable_timing=True) if gpu_timing else None
            )
            gather_start_event = (
                torch.cuda.Event(enable_timing=True) if gpu_timing else None
            )
            gather_end_event = (
                torch.cuda.Event(enable_timing=True) if gpu_timing else None
            )
            d2h_end_event = (
                torch.cuda.Event(enable_timing=True) if gpu_timing else None
            )
            ready_event = torch.cuda.Event(enable_timing=gpu_timing)
            copy_ready_event = (
                torch.cuda.Event() if active_copy_stream_count else None
            )
            layer_cpu_staging_allocation_time = 0.0
            layer_wait_stream_call_time = 0.0
            layer_gather_call_time = 0.0
            layer_chunk_copy_loop_time = 0.0
            layer_tensor_view_time = 0.0
            layer_copy_wait_event_call_time = 0.0
            layer_copy_call_time = 0.0
            layer_copy_count = 0
            layer_copy_stream_context_other_time = 0.0
            layer_copy_stream_join_call_time = 0.0
            layer_staging_d2h_submit_time = 0.0
            memory_objs_layer = memory_objs[layer_id]
            if use_cpu_staging:
                allocation_start = time.perf_counter()
                cpu_staging_tensor = torch.empty(
                    buffer_shape,
                    dtype=self.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                allocation_elapsed = time.perf_counter() - allocation_start
                cpu_staging_allocation_time += allocation_elapsed
                layer_cpu_staging_allocation_time += allocation_elapsed
                if destination_is_pinned is None:
                    destination_is_pinned = cpu_staging_tensor.is_pinned()
                    destination_is_contiguous = cpu_staging_tensor.is_contiguous()
                    source_is_contiguous = cpu_staging_tensor.is_contiguous()

            # kvcaches -> gpu_buffer -> memobj
            with torch.cuda.stream(selected_store_stream):
                if layer_start_event is not None:
                    layer_start_event.record(selected_store_stream)
                wait_stream_start = time.perf_counter()
                selected_store_stream.wait_stream(current_stream)
                wait_stream_elapsed = time.perf_counter() - wait_stream_start
                wait_stream_call_time += wait_stream_elapsed
                layer_wait_stream_call_time += wait_stream_elapsed
                if gather_start_event is not None:
                    gather_start_event.record(selected_store_stream)
                if self.use_gpu and not skip_d2h:
                    assert tmp_gpu_buffer_tensor is not None
                    gather_call_start = time.perf_counter()
                    lmc_ops.single_layer_kv_transfer(
                        tmp_gpu_buffer_tensor,
                        self.kvcaches[layer_id],
                        slot_mapping_full,
                        lmc_ops.TransferDirection.D2H,
                        self.gpu_kv_format,
                        token_major=True,
                    )
                    gather_call_elapsed = time.perf_counter() - gather_call_start
                    gather_call_time += gather_call_elapsed
                    layer_gather_call_time += gather_call_elapsed
                if gather_end_event is not None:
                    gather_end_event.record(selected_store_stream)
                if copy_ready_event is not None:
                    copy_ready_event.record(selected_store_stream)
                chunk_copy_loop_start = time.perf_counter()
                if skip_d2h:
                    pass
                elif use_cpu_staging:
                    assert tmp_gpu_buffer_tensor is not None
                    assert cpu_staging_tensor is not None
                    staging_submit_start = time.perf_counter()
                    cpu_staging_tensor.copy_(
                        tmp_gpu_buffer_tensor,
                        non_blocking=True,
                    )
                    staging_submit_elapsed = (
                        time.perf_counter() - staging_submit_start
                    )
                    staging_d2h_submit_time += staging_submit_elapsed
                    layer_staging_d2h_submit_time += staging_submit_elapsed
                elif self.use_gpu:
                    assert tmp_gpu_buffer_tensor is not None
                    for chunk_id, (start, end, memory_obj) in enumerate(
                        zip(starts, ends, memory_objs_layer, strict=False)
                    ):
                        tensor_view_start = time.perf_counter()
                        destination = memory_obj.tensor
                        assert destination is not None
                        source = tmp_gpu_buffer_tensor[start - offset : end - offset]
                        tensor_view_elapsed = time.perf_counter() - tensor_view_start
                        tensor_view_time += tensor_view_elapsed
                        layer_tensor_view_time += tensor_view_elapsed
                        if destination_is_pinned is None:
                            destination_is_pinned = destination.is_pinned()
                            destination_is_contiguous = destination.is_contiguous()
                            source_is_contiguous = source.is_contiguous()
                        if single_chunk_same_stream:
                            copy_call_start = time.perf_counter()
                            destination.copy_(source, non_blocking=True)
                            copy_call_time = time.perf_counter() - copy_call_start
                            copy_call_times.append(copy_call_time)
                            layer_copy_call_time += copy_call_time
                            layer_copy_count += 1
                            first_eight_copy_call_time += copy_call_time
                            first_eight_copy_count += 1
                            last_eight_copy_call_time += copy_call_time
                            last_eight_copy_count += 1
                            if self.use_mla:
                                memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT
                            continue

                        assert copy_ready_event is not None
                        copy_stream = selected_copy_streams[
                            chunk_id % active_copy_stream_count
                        ]
                        stream_context_start = time.perf_counter()
                        with torch.cuda.stream(copy_stream):
                            wait_event_start = time.perf_counter()
                            copy_stream.wait_event(copy_ready_event)
                            wait_event_time = time.perf_counter() - wait_event_start
                            copy_wait_event_call_time += wait_event_time
                            layer_copy_wait_event_call_time += wait_event_time
                            copy_call_start = time.perf_counter()
                            destination.copy_(source, non_blocking=True)
                            copy_call_time = time.perf_counter() - copy_call_start
                            copy_call_times.append(copy_call_time)
                            layer_copy_call_time += copy_call_time
                            layer_copy_count += 1
                            if chunk_id < 8:
                                first_eight_copy_call_time += copy_call_time
                                first_eight_copy_count += 1
                            if chunk_id >= len(memory_objs_layer) - 8:
                                last_eight_copy_call_time += copy_call_time
                                last_eight_copy_count += 1
                        stream_context_time = (
                            time.perf_counter() - stream_context_start
                        )
                        copy_stream_context_other_time += max(
                            0.0,
                            stream_context_time - wait_event_time - copy_call_time,
                        )
                        layer_copy_stream_context_other_time += max(
                            0.0,
                            stream_context_time - wait_event_time - copy_call_time,
                        )
                        if self.use_mla:
                            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT
                else:
                    for start, end, memory_obj in zip(
                        starts, ends, memory_objs_layer, strict=False
                    ):
                        tensor_view_start = time.perf_counter()
                        destination = memory_obj.tensor
                        assert destination is not None
                        tensor_view_elapsed = time.perf_counter() - tensor_view_start
                        tensor_view_time += tensor_view_elapsed
                        layer_tensor_view_time += tensor_view_elapsed
                        lmc_ops.single_layer_kv_transfer(
                            destination,
                            self.kvcaches[layer_id],
                            slot_mapping[start:end],
                            lmc_ops.TransferDirection.D2H,
                            self.gpu_kv_format,
                            token_major=True,
                        )
                        if self.use_mla:
                            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT
                chunk_copy_loop_elapsed = time.perf_counter() - chunk_copy_loop_start
                chunk_copy_loop_time += chunk_copy_loop_elapsed
                layer_chunk_copy_loop_time += chunk_copy_loop_elapsed
                if active_copy_stream_count:
                    copy_stream_join_start = time.perf_counter()
                    for copy_stream in selected_copy_streams:
                        selected_store_stream.wait_stream(copy_stream)
                    copy_stream_join_elapsed = (
                        time.perf_counter() - copy_stream_join_start
                    )
                    copy_stream_join_call_time += copy_stream_join_elapsed
                    layer_copy_stream_join_call_time += copy_stream_join_elapsed
                if d2h_end_event is not None:
                    d2h_end_event.record(selected_store_stream)
                ready_event.record(selected_store_stream)

            enqueue_end = time.perf_counter()

            def make_wait_fn(
                *,
                put_layer: int = layer_id,
                layer_store_stream_id: int = store_stream_id,
                layer_ready_event: torch.cuda.Event = ready_event,
                start_event: Optional[torch.cuda.Event] = layer_start_event,
                gather_start: Optional[torch.cuda.Event] = gather_start_event,
                gather_end: Optional[torch.cuda.Event] = gather_end_event,
                d2h_end: Optional[torch.cuda.Event] = d2h_end_event,
                layer_memory_objs: list[MemoryObj] = memory_objs_layer,
                staging_tensor: Optional[torch.Tensor] = cpu_staging_tensor,
                enqueue_wall: float = enqueue_end - enqueue_start,
                wait_stream_call: float = layer_wait_stream_call_time,
                gather_call: float = layer_gather_call_time,
                chunk_copy_loop: float = layer_chunk_copy_loop_time,
                tensor_view: float = layer_tensor_view_time,
                copy_wait_event_call: float = layer_copy_wait_event_call_time,
                copy_call: float = layer_copy_call_time,
                copy_stream_context_other: float = (
                    layer_copy_stream_context_other_time
                ),
                copy_stream_join_call: float = layer_copy_stream_join_call_time,
                copy_count: int = layer_copy_count,
                staging_allocation: float = layer_cpu_staging_allocation_time,
                staging_submit: float = layer_staging_d2h_submit_time,
            ) -> Callable[[], None]:
                waited = False

                def wait_ready() -> None:
                    nonlocal waited
                    nonlocal sync_wait_time
                    nonlocal stream_dependency_gpu_time_ms
                    nonlocal gather_gpu_time_ms
                    nonlocal d2h_gpu_time_ms
                    nonlocal cpu_view_count
                    nonlocal cpu_view_creation_time
                    nonlocal cpu_object_rebind_time
                    nonlocal cpu_old_buffer_free_time
                    if waited:
                        return

                    ready_wait_start = time.perf_counter()
                    with torch.cuda.device(self.device):
                        layer_ready_event.synchronize()
                    layer_ready_wait_time = time.perf_counter() - ready_wait_start
                    layer_stream_dependency_gpu_time_ms = 0.0
                    layer_gather_gpu_time_ms = 0.0
                    layer_d2h_gpu_time_ms = 0.0
                    if gpu_timing:
                        assert start_event is not None
                        assert gather_start is not None
                        assert gather_end is not None
                        assert d2h_end is not None
                        layer_stream_dependency_gpu_time_ms = (
                            start_event.elapsed_time(gather_start)
                        )
                        layer_gather_gpu_time_ms = gather_start.elapsed_time(gather_end)
                        layer_d2h_gpu_time_ms = gather_end.elapsed_time(d2h_end)

                    layer_staging_view_time = 0.0
                    layer_staging_rebind_time = 0.0
                    layer_staging_free_time = 0.0
                    if use_cpu_staging:
                        assert staging_tensor is not None
                        (
                            view_creation_time,
                            object_rebind_time,
                            old_buffer_free_time,
                        ) = self._replace_with_cpu_staging_views(
                            layer_memory_objs,
                            staging_tensor,
                            staging_starts,
                            staging_ends,
                        )
                        layer_staging_view_time = view_creation_time
                        layer_staging_rebind_time = object_rebind_time
                        layer_staging_free_time = old_buffer_free_time

                    ready_wait_end = time.perf_counter()
                    with timing_lock:
                        sync_wait_time += layer_ready_wait_time
                        stream_dependency_gpu_time_ms += (
                            layer_stream_dependency_gpu_time_ms
                        )
                        gather_gpu_time_ms += layer_gather_gpu_time_ms
                        d2h_gpu_time_ms += layer_d2h_gpu_time_ms
                        if use_cpu_staging:
                            cpu_view_count += len(layer_memory_objs)
                        cpu_view_creation_time += layer_staging_view_time
                        cpu_object_rebind_time += layer_staging_rebind_time
                        cpu_old_buffer_free_time += layer_staging_free_time
                    log_connector_advance(
                        put_layer=put_layer,
                        store_stream_id=layer_store_stream_id,
                        advance_start=ready_wait_start,
                        advance_end=ready_wait_end,
                        sync_wait=layer_ready_wait_time,
                        stream_dependency_ms=layer_stream_dependency_gpu_time_ms,
                        gather_gpu_ms=layer_gather_gpu_time_ms,
                        d2h_gpu_ms=layer_d2h_gpu_time_ms,
                        next_enqueue_layer=put_layer,
                        enqueue_wall=enqueue_wall,
                        wait_stream_call=wait_stream_call,
                        gather_call=gather_call,
                        chunk_copy_loop=chunk_copy_loop,
                        tensor_view=tensor_view,
                        copy_wait_event_call=copy_wait_event_call,
                        copy_call=copy_call,
                        copy_stream_context_other=copy_stream_context_other,
                        copy_stream_join_call=copy_stream_join_call,
                        copy_count=copy_count,
                        staging_allocation=staging_allocation,
                        staging_submit=staging_submit,
                        staging_view=layer_staging_view_time,
                        staging_rebind=layer_staging_rebind_time,
                        staging_free=layer_staging_free_time,
                    )
                    logger.debug(f"Finished offloading layer {put_layer}")
                    waited = True

                return wait_ready

            offload_handle = LayerwiseOffloadHandle(layer_id, make_wait_fn())
            offload_handles.append(offload_handle)
            yield offload_handle

        for offload_handle in offload_handles:
            offload_handle.wait()

        logger.info(
            "[req_id=%s] Layerwise offload breakdown: "
            "layers=%d, chunks_per_layer=%d, store_streams=%d, "
            "copy_streams=%d, cpu_staging=%s, "
            "skip_d2h=%s, "
            "setup_time=%.4f ms, "
            "wait_stream_call_time=%.4f ms, gather_call_time=%.4f ms, "
            "chunk_copy_loop_time=%.4f ms, sync_wait_time=%.4f ms, "
            "stream_dependency_gpu_time=%.4f ms, gather_gpu_time=%.4f ms, "
            "d2h_gpu_time=%.4f ms, gpu_timing=%s",
            kwargs.get("req_id"),
            self.num_layers,
            num_chunks_per_layer,
            self.layerwise_store_stream_count,
            active_copy_stream_count,
            use_cpu_staging,
            skip_d2h,
            setup_time * 1000,
            wait_stream_call_time * 1000,
            gather_call_time * 1000,
            chunk_copy_loop_time * 1000,
            sync_wait_time * 1000,
            stream_dependency_gpu_time_ms,
            gather_gpu_time_ms,
            d2h_gpu_time_ms,
            gpu_timing,
        )

        if use_cpu_staging:
            logger.info(
                "[req_id=%s] Layerwise CPU staging breakdown: "
                "layers=%d, views=%d, staging_allocation_time=%.4f ms, "
                "staging_d2h_submit_time=%.4f ms, "
                "view_creation_time=%.4f ms, object_rebind_time=%.4f ms, "
                "old_buffer_free_time=%.4f ms, "
                "dst_pinned=%s, "
                "dst_contiguous=%s, src_contiguous=%s",
                kwargs.get("req_id"),
                self.num_layers,
                cpu_view_count,
                cpu_staging_allocation_time * 1000,
                staging_d2h_submit_time * 1000,
                cpu_view_creation_time * 1000,
                cpu_object_rebind_time * 1000,
                cpu_old_buffer_free_time * 1000,
                destination_is_pinned,
                destination_is_contiguous,
                source_is_contiguous,
            )

        sorted_copy_call_times = sorted(copy_call_times)
        copy_call_p50 = (
            statistics.median(sorted_copy_call_times)
            if sorted_copy_call_times
            else 0.0
        )
        copy_call_p95 = (
            sorted_copy_call_times[
                int(0.95 * (len(sorted_copy_call_times) - 1))
            ]
            if sorted_copy_call_times
            else 0.0
        )
        (logger.info if not use_cpu_staging else logger.debug)(
            "[req_id=%s] Layerwise chunk copy breakdown: "
            "copies=%d, tensor_view_time=%.4f ms, "
            "wait_event_call_time=%.4f ms, copy_call_time=%.4f ms, "
            "stream_context_other_time=%.4f ms, join_call_time=%.4f ms, "
            "copy_call_p50=%.4f ms, copy_call_p95=%.4f ms, "
            "copy_call_max=%.4f ms, first_eight_avg=%.4f ms, "
            "last_eight_avg=%.4f ms, dst_pinned=%s, "
            "dst_contiguous=%s, src_contiguous=%s",
            kwargs.get("req_id"),
            len(copy_call_times),
            tensor_view_time * 1000,
            copy_wait_event_call_time * 1000,
            sum(copy_call_times) * 1000,
            copy_stream_context_other_time * 1000,
            copy_stream_join_call_time * 1000,
            copy_call_p50 * 1000,
            copy_call_p95 * 1000,
            max(copy_call_times, default=0.0) * 1000,
            first_eight_copy_call_time
            / max(first_eight_copy_count, 1)
            * 1000,
            last_eight_copy_call_time
            / max(last_eight_copy_count, 1)
            * 1000,
            destination_is_pinned,
            destination_is_contiguous,
            source_is_contiguous,
        )

        # free the buffer memory
        reservation.release(synchronize=False)

        yield

    def _flush_layerwise_request_batch(
        self,
        request_batch: LayerwiseRequestBatch,
        entries: list[_LayerwiseRequestBatchEntry],
    ) -> None:
        """Launch one gather and D2H for a layer tile across requests."""

        flush_start = time.perf_counter()
        timeline_enabled = os.environ.get("PD_BACKEND_STORE_TIMELINE") == "1"
        submit_timing_enabled = (
            timeline_enabled
            or os.environ.get("PD_BACKEND_AGGREGATE_TIMING") == "1"
        )
        if not entries:
            return
        group_start = entries[0].group_start
        group_end = entries[0].group_end
        if any(
            entry.group_start != group_start or entry.group_end != group_end
            for entry in entries
        ):
            raise RuntimeError(
                "layerwise request batch contains mismatched layer groups"
            )
        if any(
            sum(chunk.numel() for chunk in entry.slot_mapping_chunks) == 0
            for entry in entries
        ):
            raise RuntimeError("layerwise request batch contains an empty slot mapping")

        assert self.kvcaches is not None
        assert self.kv_cache_pointers_on_gpu is not None
        token_counts = [
            sum(int(chunk.numel()) for chunk in entry.slot_mapping_chunks)
            for entry in entries
        ]
        total_tokens = sum(token_counts)
        group_layers = group_end - group_start
        chunk_sizes_by_request: list[tuple[int, ...]] = []
        bytes_per_token = self.get_shape(1).numel() * self.element_size
        for entry, token_count in zip(entries, token_counts, strict=True):
            first_layer_objs = entry.memory_objs[0]
            chunk_sizes = tuple(
                memory_obj.get_size() for memory_obj in first_layer_objs
            )
            if not chunk_sizes or sum(chunk_sizes) != token_count * bytes_per_token:
                raise RuntimeError(
                    "layerwise request batch has inconsistent chunk sizes"
                )
            if any(
                len(layer_objs) != len(chunk_sizes)
                or tuple(memory_obj.get_size() for memory_obj in layer_objs)
                != chunk_sizes
                for layer_objs in entry.memory_objs
            ):
                raise RuntimeError(
                    "layerwise request batch chunk layout differs across layers"
                )
            chunk_sizes_by_request.append(chunk_sizes)
        layout_signature = tuple(
            (entry.req_id, id(entry.memory_objs), chunk_sizes)
            for entry, chunk_sizes in zip(
                entries,
                chunk_sizes_by_request,
                strict=True,
            )
        )
        if (
            request_batch.layout_signature is not None
            and request_batch.layout_signature != layout_signature
        ):
            raise RuntimeError("layerwise request batch membership changed")
        store_stream_id, store_stream = self._select_layerwise_store_stream(
            group_start // request_batch.group_size
        )
        cat_start_event = None
        cat_done_event = None
        cat_submit_start = time.perf_counter()
        if request_batch.combined_slot_mapping is None:
            with torch.cuda.stream(store_stream):
                if timeline_enabled:
                    cat_start_event = torch.cuda.Event(enable_timing=True)
                    cat_done_event = torch.cuda.Event(enable_timing=True)
                    cat_start_event.record(store_stream)
                combined_slot_mapping = torch.cat(
                    [
                        chunk
                        for entry in entries
                        for chunk in entry.slot_mapping_chunks
                    ],
                    dim=0,
                )
                if cat_done_event is not None:
                    cat_done_event.record(store_stream)
                slot_mapping_ready_event = torch.cuda.Event()
                slot_mapping_ready_event.record(store_stream)
            request_batch.combined_slot_mapping = combined_slot_mapping
            request_batch.slot_mapping_ready_event = slot_mapping_ready_event
        else:
            combined_slot_mapping = request_batch.combined_slot_mapping
            slot_mapping_ready_event = request_batch.slot_mapping_ready_event
            assert slot_mapping_ready_event is not None
            store_stream.wait_event(slot_mapping_ready_event)
        cat_submit_ms = (time.perf_counter() - cat_submit_start) * 1000

        gpu_alloc_start = time.perf_counter()
        buffer_shape = self.get_shape(total_tokens)
        gpu_buffer_owner = request_batch.gpu_buffers.get(store_stream_id)
        if gpu_buffer_owner is None:
            allocation_shape = torch.Size(
                [request_batch.group_size, *buffer_shape]
            )
            with torch.cuda.stream(store_stream):
                gpu_buffer_owner = torch.empty(
                    allocation_shape,
                    dtype=self.dtype,
                    device=self.device,
                )
            request_batch.gpu_buffers[store_stream_id] = gpu_buffer_owner
        gpu_buffer = gpu_buffer_owner[:group_layers]
        gpu_alloc_ms = (time.perf_counter() - gpu_alloc_start) * 1000

        expected_sizes_per_layer = [
            chunk_size
            for chunk_sizes in chunk_sizes_by_request
            for chunk_size in chunk_sizes
        ]
        memory_validation_start = time.perf_counter()
        old_free_ms = 0.0
        owner_alloc_ms = 0.0
        view_rebind_ms = 0.0
        if request_batch.cpu_staging is None:
            all_memory_objs = [
                memory_obj
                for layer_id in range(self.num_layers)
                for entry in entries
                for memory_obj in entry.memory_objs[layer_id]
            ]
            parent_allocator = all_memory_objs[0].parent()
            if not isinstance(parent_allocator, TensorMemoryAllocator) or any(
                not isinstance(memory_obj, TensorMemoryObj)
                or memory_obj.parent() is not parent_allocator
                or memory_obj.raw_data.device.type != "cpu"
                or memory_obj.get_size()
                != expected_sizes_per_layer[
                    index % len(expected_sizes_per_layer)
                ]
                for index, memory_obj in enumerate(all_memory_objs)
            ):
                raise RuntimeError(
                    "layerwise request batching requires allocator-owned CPU "
                    "TensorMemoryObjs"
                )
            memory_format = all_memory_objs[0].metadata.fmt
            memory_validation_ms = (
                time.perf_counter() - memory_validation_start
            ) * 1000

            # Request-major allocations cannot provide a contiguous layer-major
            # tile without duplicate capacity. Return the entire batch first,
            # then replace it with one equally-sized layer-major region.
            old_free_start = time.perf_counter()
            parent_allocator.batched_free(list(all_memory_objs))
            old_free_ms = (time.perf_counter() - old_free_start) * 1000

            owner_alloc_start = time.perf_counter()
            full_batch_shape = torch.Size([self.num_layers, *buffer_shape])
            owner = parent_allocator.allocate(
                full_batch_shape,
                self.dtype,
                memory_format,
            )
            if owner is None:
                raise RuntimeError("failed to allocate layerwise request batch buffer")
            full_cpu_staging = owner.tensor
            assert full_cpu_staging is not None
            owner_alloc_ms = (time.perf_counter() - owner_alloc_start) * 1000

            view_rebind_start = time.perf_counter()
            view_allocator = _LayerwiseBatchViewAllocator(
                owner,
                len(all_memory_objs),
            )
            raw_views = torch.split(
                full_cpu_staging.view(torch.uint8).flatten(),
                expected_sizes_per_layer * self.num_layers,
            )
            for memory_obj, raw_view in zip(
                all_memory_objs,
                raw_views,
                strict=True,
            ):
                assert isinstance(memory_obj, TensorMemoryObj)
                memory_obj.raw_data = raw_view
                memory_obj.meta.address = raw_view.data_ptr()
                memory_obj.meta.phy_size = raw_view.numel()
                memory_obj.meta.ref_count = 1
                memory_obj.meta.pin_count = 0
                memory_obj.parent_allocator = view_allocator
                memory_obj.valid = True
                if self.use_mla:
                    memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT
            view_rebind_ms = (time.perf_counter() - view_rebind_start) * 1000
            request_batch.layout_signature = layout_signature
            request_batch.cpu_staging = full_cpu_staging
        else:
            memory_validation_ms = (
                time.perf_counter() - memory_validation_start
            ) * 1000
        full_cpu_staging = request_batch.cpu_staging
        assert full_cpu_staging is not None
        cpu_staging = full_cpu_staging[group_start:group_end]

        ready_event = torch.cuda.Event(enable_timing=timeline_enabled)
        store_start_event = None
        gather_done_event = None
        if timeline_enabled:
            store_start_event = torch.cuda.Event(enable_timing=True)
            gather_done_event = torch.cuda.Event(enable_timing=True)
        gpu_buffer.record_stream(store_stream)
        cuda_enqueue_start = time.perf_counter()
        gather_submit_ms = 0.0
        d2h_submit_ms = 0.0
        with torch.cuda.stream(store_stream):
            # The adapter synchronizes the exact layer-group ready event before
            # entering this function. Waiting on the default stream here would
            # also capture later model work and destroy layerwise overlap.
            if store_start_event is not None:
                store_start_event.record(store_stream)
            kernel_buffer = gpu_buffer.unsqueeze(2) if self.use_mla else gpu_buffer
            gather_submit_start = time.perf_counter()
            lmc_ops.multi_layer_kv_transfer_token_major(
                kernel_buffer,
                self.kv_cache_pointers_on_gpu[group_start:group_end],
                combined_slot_mapping,
                self.device,
                self.page_buffer_size,
                lmc_ops.TransferDirection.D2H,
                self.gpu_kv_format,
                block_size=self.block_size,
                head_size=self.head_size,
            )
            gather_submit_ms = (
                time.perf_counter() - gather_submit_start
            ) * 1000
            if gather_done_event is not None:
                gather_done_event.record(store_stream)
            d2h_submit_start = time.perf_counter()
            lmc_ops.lmcache_memcpy_async(
                cpu_staging.data_ptr(),
                gpu_buffer.data_ptr(),
                cpu_staging.numel() * cpu_staging.element_size(),
                lmc_ops.TransferDirection.D2H,
                0,
                1 << 30,
            )
            d2h_submit_ms = (time.perf_counter() - d2h_submit_start) * 1000
            ready_event.record(store_stream)
        cuda_enqueue_ms = (time.perf_counter() - cuda_enqueue_start) * 1000
        flush_submit_ms = (time.perf_counter() - flush_start) * 1000

        if submit_timing_enabled:
            logger.info(
                "[pd-demo-request-batch] first_layer=%d layers=%d requests=%d "
                "tokens=%d bytes=%d req_ids=%s",
                group_start,
                group_layers,
                len(entries),
                total_tokens,
                cpu_staging.numel() * cpu_staging.element_size(),
                ",".join(str(entry.req_id or "-") for entry in entries),
            )
            logger.info(
                "[pd-request-batch-timing] phase=submit first_layer=%d "
                "layers=%d requests=%d tokens=%d total_ms=%.4f "
                "cat_submit_ms=%.4f gpu_alloc_ms=%.4f validate_ms=%.4f "
                "owner_alloc_ms=%.4f old_free_ms=%.4f view_rebind_ms=%.4f "
                "cuda_enqueue_ms=%.4f gather_submit_ms=%.4f "
                "d2h_submit_ms=%.4f",
                group_start,
                group_layers,
                len(entries),
                total_tokens,
                flush_submit_ms,
                cat_submit_ms,
                gpu_alloc_ms,
                memory_validation_ms,
                owner_alloc_ms,
                old_free_ms,
                view_rebind_ms,
                cuda_enqueue_ms,
                gather_submit_ms,
                d2h_submit_ms,
            )

        wait_lock = threading.Lock()
        waited = False

        def wait_ready() -> None:
            nonlocal waited
            with wait_lock:
                if waited:
                    return
                with torch.cuda.device(self.device):
                    ready_event.synchronize()
                if timeline_enabled:
                    cat_gpu_ms = (
                        cat_start_event.elapsed_time(cat_done_event)
                        if cat_start_event is not None
                        and cat_done_event is not None
                        else 0.0
                    )
                    gather_gpu_ms = (
                        store_start_event.elapsed_time(gather_done_event)
                        if store_start_event is not None
                        and gather_done_event is not None
                        else 0.0
                    )
                    tile_gpu_ms = (
                        store_start_event.elapsed_time(ready_event)
                        if store_start_event is not None
                        else 0.0
                    )
                    logger.info(
                        "[pd-request-batch-timing] phase=complete first_layer=%d "
                        "layers=%d requests=%d tokens=%d "
                        "cat_gpu_ms=%.4f gather_gpu_ms=%.4f d2h_gpu_ms=%.4f "
                        "tile_gpu_ms=%.4f",
                        group_start,
                        group_layers,
                        len(entries),
                        total_tokens,
                        cat_gpu_ms,
                        gather_gpu_ms,
                        max(0.0, tile_gpu_ms - gather_gpu_ms),
                        tile_gpu_ms,
                    )
                waited = True

        for entry in entries:
            entry.waiter.resolve(wait_ready)

    def _batched_from_gpu_grouped_layers(
        self,
        *,
        memory_objs: List[List[MemoryObj]],
        slot_mapping: Optional[torch.Tensor],
        slot_mapping_chunks: tuple[torch.Tensor, ...],
        tmp_gpu_buffer_tensors: list[torch.Tensor],
        reservation: _GPUBufferReservation,
        dependency_claim: Optional[Callable[[int, int], bool]],
        group_size: int,
        request_batch: Optional[LayerwiseRequestBatch],
        req_id: Optional[str],
    ):
        """Offload small layer groups while preserving forward overlap."""

        assert self.kvcaches is not None
        assert self.kv_cache_pointers_on_gpu is not None
        current_stream = torch.cuda.current_stream()
        completion_handles: list[LayerwiseOffloadHandle] = []

        def group_waiter(ready_event: torch.cuda.Event) -> Callable[[], None]:
            wait_lock = threading.Lock()
            waited = False

            def wait_ready() -> None:
                nonlocal waited
                with wait_lock:
                    if waited:
                        return
                    with torch.cuda.device(self.device):
                        ready_event.synchronize()
                    waited = True

            return wait_ready

        for group_start in range(0, self.num_layers, group_size):
            group_end = min(
                group_start + group_size,
                self.num_layers,
            )
            group_layers = group_end - group_start
            if request_batch is not None:
                entry = _LayerwiseRequestBatchEntry(
                    group_start=group_start,
                    group_end=group_end,
                    memory_objs=memory_objs,
                    slot_mapping_chunks=slot_mapping_chunks,
                    req_id=req_id,
                )
                request_batch.register(entry)
                for layer_id in range(group_start, group_end):
                    offload_handle = LayerwiseOffloadHandle(
                        layer_id,
                        entry.waiter.wait,
                        request_batch,
                    )
                    yield offload_handle
                continue

            assert slot_mapping is not None
            store_stream_id, store_stream = self._select_layerwise_store_stream(
                group_start // group_size
            )
            gpu_buffer = tmp_gpu_buffer_tensors[store_stream_id][:group_layers]
            # Each group needs an exact completion event: its CPU destination may
            # be sent while later groups are still gathering on another stream.
            ready_event = torch.cuda.Event()
            wait_ready = group_waiter(ready_event)

            for layer_id in range(group_start, group_end):
                if layer_id == group_end - 1:
                    first_memory_obj = memory_objs[group_start][0]
                    if not isinstance(first_memory_obj, TensorMemoryObj):
                        raise TypeError(
                            "Grouped layerwise D2H requires TensorMemoryObj"
                        )
                    expected_ptr = first_memory_obj.raw_data.data_ptr()
                    total_bytes = 0
                    for grouped_layer_id in range(group_start, group_end):
                        memory_obj = memory_objs[grouped_layer_id][0]
                        if (
                            not isinstance(memory_obj, TensorMemoryObj)
                            or memory_obj.raw_data.device.type != "cpu"
                            or memory_obj.raw_data.data_ptr() != expected_ptr
                        ):
                            raise RuntimeError(
                                "Grouped layerwise D2H destinations are not contiguous"
                            )
                        size = memory_obj.get_size()
                        expected_ptr += size
                        total_bytes += size

                    torch.cuda.set_stream(store_stream)
                    try:
                        should_wait = (
                            group_start == 0
                            or dependency_claim is None
                            or dependency_claim(
                                layer_id,
                                int(current_stream.cuda_stream),
                            )
                        )
                        if should_wait:
                            store_stream.wait_stream(current_stream)
                        kernel_buffer = (
                            gpu_buffer.unsqueeze(2) if self.use_mla else gpu_buffer
                        )
                        lmc_ops.multi_layer_kv_transfer_token_major(
                            kernel_buffer,
                            self.kv_cache_pointers_on_gpu[group_start:group_end],
                            slot_mapping,
                            self.device,
                            self.page_buffer_size,
                            lmc_ops.TransferDirection.D2H,
                            self.gpu_kv_format,
                            block_size=self.block_size,
                            head_size=self.head_size,
                        )
                        lmc_ops.lmcache_memcpy_async(
                            first_memory_obj.raw_data.data_ptr(),
                            gpu_buffer.data_ptr(),
                            total_bytes,
                            lmc_ops.TransferDirection.D2H,
                            0,
                            1 << 30,
                        )
                        ready_event.record(store_stream)
                    finally:
                        torch.cuda.set_stream(current_stream)

                if self.use_mla:
                    memory_objs[layer_id][0].metadata.fmt = MemoryFormat.KV_MLA_FMT
                offload_handle = LayerwiseOffloadHandle(layer_id, wait_ready)
                if layer_id == group_start:
                    completion_handles.append(offload_handle)
                yield offload_handle

        # Shared request batches are already synchronized by LayerwiseStoreBatch;
        # only standalone grouped transfers reach this final wait.
        for offload_handle in completion_handles:
            offload_handle.wait()

        reservation.release(synchronize=False)
        yield

    def _batched_from_gpu_single_chunk_fast(
        self,
        *,
        memory_objs: List[List[MemoryObj]],
        slot_mapping: torch.Tensor,
        tmp_gpu_buffer_tensors: list[torch.Tensor],
        reservation: _GPUBufferReservation,
        dependency_claim: Optional[Callable[[int, int], bool]],
        contiguous_pinned_d2h: bool,
    ):
        """Enqueue one gather and D2H per layer."""

        assert self.kvcaches is not None
        current_stream = torch.cuda.current_stream()
        offload_handles: list[LayerwiseOffloadHandle] = []

        for layer_id in range(self.num_layers):
            store_stream_id, store_stream = self._select_layerwise_store_stream(
                layer_id
            )
            gpu_buffer = tmp_gpu_buffer_tensors[store_stream_id]
            memory_obj = memory_objs[layer_id][0]
            raw_destination = (
                memory_obj.raw_data
                if contiguous_pinned_d2h
                and isinstance(memory_obj, TensorMemoryObj)
                and isinstance(memory_obj.parent(), TensorMemoryAllocator)
                and memory_obj.raw_data.device.type == "cpu"
                and memory_obj.raw_data.is_contiguous()
                else None
            )
            destination = None if raw_destination is not None else memory_obj.tensor
            assert raw_destination is not None or destination is not None
            ready_event = torch.cuda.Event()
            torch.cuda.set_stream(store_stream)
            try:
                # Layer 0 also includes a request-local asynchronous slot-mapping
                # copy, so every request must wait there. On later layers, all
                # requests in one scheduler step consume the same completed model
                # layer on the same store stream; only the first live request needs
                # to establish that stream dependency.
                should_wait = (
                    layer_id == 0
                    or dependency_claim is None
                    or dependency_claim(layer_id, int(current_stream.cuda_stream))
                )
                if should_wait:
                    store_stream.wait_stream(current_stream)
                lmc_ops.single_layer_kv_transfer(
                    gpu_buffer,
                    self.kvcaches[layer_id],
                    slot_mapping,
                    lmc_ops.TransferDirection.D2H,
                    self.gpu_kv_format,
                    token_major=True,
                )
                if raw_destination is not None:
                    # The PD TCP allocator owns one contiguous pinned allocation.
                    # Submit the D2H directly by address to avoid constructing a
                    # typed tensor view and dispatching aten.copy_ for every layer.
                    lmc_ops.lmcache_memcpy_async(
                        raw_destination.data_ptr(),
                        gpu_buffer.data_ptr(),
                        memory_obj.get_size(),
                        lmc_ops.TransferDirection.D2H,
                        0,
                        1 << 30,
                    )
                else:
                    assert destination is not None
                    destination.copy_(gpu_buffer, non_blocking=True)
                ready_event.record(store_stream)
            finally:
                torch.cuda.set_stream(current_stream)

            if self.use_mla:
                memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT
            offload_handle = _CudaEventLayerwiseOffloadHandle(
                layer_id,
                ready_event,
                self.device,
            )
            offload_handles.append(offload_handle)
            yield offload_handle

        for offload_handle in offload_handles:
            offload_handle.wait()

        reservation.release(synchronize=False)
        yield

    def _fused_layerwise_destination(
        self,
        memory_objs: List[List[MemoryObj]],
    ) -> Optional[torch.Tensor]:
        """Return one pinned view when all per-layer objects are contiguous."""

        flat_memory_objs = [layer_objs[0] for layer_objs in memory_objs]
        if not flat_memory_objs or any(
            not isinstance(memory_obj, TensorMemoryObj)
            or memory_obj.raw_data.device.type != "cpu"
            or memory_obj.raw_data.dtype != torch.uint8
            or not memory_obj.raw_data.is_contiguous()
            or memory_obj.raw_data.numel() != memory_obj.get_size()
            for memory_obj in flat_memory_objs
        ):
            return None

        parent = flat_memory_objs[0].parent()
        if parent is None or any(
            memory_obj.parent() is not parent for memory_obj in flat_memory_objs
        ):
            return None

        expected_ptr = flat_memory_objs[0].raw_data.data_ptr()
        total_bytes = 0
        for memory_obj in flat_memory_objs:
            raw_data = memory_obj.raw_data
            if raw_data.data_ptr() != expected_ptr:
                return None
            size = memory_obj.get_size()
            expected_ptr += size
            total_bytes += size

        first_raw_data = flat_memory_objs[0].raw_data
        return first_raw_data.as_strided((total_bytes,), (1,))

    def _batched_from_gpu_fused_layers(
        self,
        *,
        memory_objs: List[List[MemoryObj]],
        slot_mapping: torch.Tensor,
        destination: torch.Tensor,
        reservation: _GPUBufferReservation,
        req_id: Optional[str],
    ):
        """Gather all layers into one GPU buffer and issue one contiguous D2H."""

        assert self.kvcaches is not None
        assert self.gpu_buffer_allocator is not None
        num_tokens = len(slot_mapping)
        if self.use_mla:
            fused_shape = torch.Size(
                [self.num_layers, num_tokens, self.hidden_dim_size]
            )
        else:
            fused_shape = torch.Size(
                [self.num_layers, num_tokens, 2, self.hidden_dim_size]
            )

        gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
            fused_shape,
            self.dtype,
            MemoryFormat.KV_T2D,
        )
        if gpu_buffer_obj is None:
            raise RuntimeError(
                "Failed to allocate fused layerwise GPU buffer in GPUConnector"
            )
        reservation.add(gpu_buffer_obj)
        gpu_buffer = gpu_buffer_obj.tensor
        assert gpu_buffer is not None

        current_stream = torch.cuda.current_stream()
        layer_ids_by_stream: list[list[int]] = [
            [] for _ in range(self.layerwise_store_stream_count)
        ]
        for layer_id in range(self.num_layers):
            store_stream_id, _ = self._select_layerwise_store_stream(layer_id)
            layer_ids_by_stream[store_stream_id].append(layer_id)
        gather_done_events: list[torch.cuda.Event] = []
        for store_stream_id, store_stream in enumerate(self.store_streams):
            layer_ids = layer_ids_by_stream[store_stream_id]
            if not layer_ids:
                continue
            with torch.cuda.stream(store_stream):
                store_stream.wait_stream(current_stream)
                for layer_id in layer_ids:
                    lmc_ops.single_layer_kv_transfer(
                        gpu_buffer[layer_id],
                        self.kvcaches[layer_id],
                        slot_mapping,
                        lmc_ops.TransferDirection.D2H,
                        self.gpu_kv_format,
                        token_major=True,
                    )
                gather_done_event = torch.cuda.Event()
                gather_done_event.record(store_stream)
                gather_done_events.append(gather_done_event)

        copy_stream_id = hash(str(req_id or "")) % self.chunk_copy_stream_count
        copy_stream = self.chunk_copy_streams[copy_stream_id]
        ready_event = torch.cuda.Event()
        with torch.cuda.stream(copy_stream):
            for gather_done_event in gather_done_events:
                copy_stream.wait_event(gather_done_event)
            destination_tensor = destination.view(self.dtype).view(fused_shape)
            destination_tensor.copy_(gpu_buffer, non_blocking=True)
            ready_event.record(copy_stream)

        if self.use_mla:
            for layer_objs in memory_objs:
                layer_objs[0].metadata.fmt = MemoryFormat.KV_MLA_FMT

        for layer_id in range(self.num_layers):
            yield _CudaEventLayerwiseOffloadHandle(
                layer_id,
                ready_event,
                self.device,
            )

        with torch.cuda.device(self.device):
            ready_event.synchronize()
        reservation.release(synchronize=False)
        yield

    def get_shape(self, num_tokens: int) -> torch.Size:
        if self.use_mla:
            # MLA format: [num_tokens, hidden_dim_size]
            return torch.Size([num_tokens, self.hidden_dim_size])
        else:
            # Standard format: [num_tokens, 2, hidden_dim_size]
            return torch.Size([num_tokens, 2, self.hidden_dim_size])


class SGLangGPUConnector(GPUConnectorInterface):
    """
    The GPU KV cache should be a list of tensors, one for each layer,
    with separate key and value pointers.
    More specifically, we have:
    - kvcaches: Tuple[List[Tensor], List[Tensor]]
      - The first element is a list of key tensors, one per layer.
      - The second element is a list of value tensors, one per layer.
    - Each tensor: [page_buffer_size, head_num, head_size]

    The connector manages the transfer of KV cache data between CPU and GPU
    memory for SGLang using pointer arrays for efficient access.
    It will produce/consume memory objects with KV_2LTD format.
    """

    def __init__(
        self, hidden_dim_size: int, num_layers: int, use_gpu: bool = False, **kwargs
    ):
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers

        self.kv_cache_pointers_on_gpu: dict[int, torch.Tensor] = {}
        self.page_buffer_size = 0

        self.gpu_buffer: Optional[torch.Tensor] = None
        self.use_mla = "use_mla" in kwargs and kwargs["use_mla"]

        self.num_kv_cache = num_layers if self.use_mla else num_layers * 2
        self.kv_cache_pointers = torch.empty(
            self.num_kv_cache, dtype=torch.int64, device="cpu"
        )

        if use_gpu:
            assert "chunk_size" in kwargs, (
                "chunk_size should be provided to create a GPU buffer."
            )
            assert "device" in kwargs, (
                "device should be provided to create a GPU buffer."
            )
            shape = self.get_shape(kwargs["chunk_size"])
            self.gpu_buffer = torch.empty(
                shape, dtype=kwargs["dtype"], device=kwargs["device"]
            )
            logger.info(f"GPU buffer: {self.gpu_buffer.shape}")

    def _initialize_pointers(self, kv_caches: List[torch.Tensor]) -> torch.Tensor:
        # Discover format first to handle flattening correctly
        self.gpu_kv_format = discover_gpu_kv_format(kv_caches, EngineType.SGLANG)

        # For TWO_X_NL_X_NBBS_NH_HS format, kv_caches is [[k_list], [v_list]]
        # We need to flatten it to [k0, k1, ..., v0, v1, ...]
        if self.gpu_kv_format == lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS:
            flat_kv_caches = kv_caches[0] + kv_caches[1]  # [k_list] + [v_list]
            device = flat_kv_caches[0].device
        else:
            flat_kv_caches = kv_caches
            device = flat_kv_caches[0].device

        assert len(flat_kv_caches) == self.num_kv_cache, (
            f"Expected {self.num_kv_cache} KV caches, got {len(flat_kv_caches)}"
        )

        self.kv_cache_pointers.numpy()[:] = [t.data_ptr() for t in flat_kv_caches]
        assert device.type == "cuda", "The device should be CUDA."
        idx = device.index
        if idx not in self.kv_cache_pointers_on_gpu:
            self.kv_cache_pointers_on_gpu[idx] = torch.empty(
                self.num_kv_cache, dtype=torch.int64, device=device
            )
        self.kv_cache_pointers_on_gpu[idx].copy_(self.kv_cache_pointers)

        # sglang MLA kv_caches[0].shape: [num_pages * page_size, 1, head_size]
        # sglang MHA kv_caches: [[k_list], [v_list]]
        # each with shape [num_pages * page_size, num_heads, head_size]
        self.page_buffer_size = get_page_buffer_size(kv_caches, self.gpu_kv_format)
        return self.kv_cache_pointers_on_gpu[idx]

    @_lmcache_nvtx_annotate
    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Note:
          1. This function expects the 'slot_mapping' is a "partial slot mapping"
             where its length is the same as the uncached token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)


        :raises ValueError: If 'kvcaches' is not provided in kwargs.
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        if self.use_mla:
            if memory_obj.metadata.fmt != MemoryFormat.KV_MLA_FMT:
                raise ValueError(
                    "The memory object should be in KV_MLA_FMT format in"
                    f" order to be processed by {self.__class__.__name__}"
                )
        else:
            if memory_obj.metadata.fmt != MemoryFormat.KV_2LTD:
                raise ValueError(
                    "The memory object should be in KV_2LTD format in"
                    f" order to be processed by {self.__class__.__name__}"
                )

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        offset = kwargs.get("offset", 0)

        kvcaches: List[torch.Tensor] = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        kv_cache_pointers = self._initialize_pointers(kvcaches)
        lmc_ops.multi_layer_kv_transfer_unilateral(
            memory_obj.tensor,
            kv_cache_pointers,
            slot_mapping[start - offset : end - offset],
            kvcaches[0][0].device,
            self.page_buffer_size,
            lmc_ops.TransferDirection.H2D,
            self.gpu_kv_format,
        )

    @_lmcache_nvtx_annotate
    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Will set the memory_obj.metadata.fmt to MemoryFormat.KV_2LTD.

        Note:
          1. This function expects the 'slot_mapping' is a "partial slot mapping"
             where its length is the same as the uncached token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)

        :raises ValueError: If 'kvcaches' is not provided in kwargs,
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        kvcaches: List[torch.Tensor] = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        kv_cache_pointers = self._initialize_pointers(kvcaches)

        if self.gpu_buffer is None or end - start != self.gpu_buffer.shape[2]:
            lmc_ops.multi_layer_kv_transfer_unilateral(
                memory_obj.tensor,
                kv_cache_pointers,
                slot_mapping[start:end],
                kvcaches[0][0].device,
                self.page_buffer_size,
                lmc_ops.TransferDirection.D2H,
                self.gpu_kv_format,
            )
        else:
            # kvcaches -> gpu_buffer -> memobj
            assert self.gpu_buffer.device == kvcaches[0][0].device
            tmp_gpu_buffer = self.gpu_buffer[:, :, : end - start, :]
            lmc_ops.multi_layer_kv_transfer_unilateral(
                tmp_gpu_buffer,
                kv_cache_pointers,
                slot_mapping[start:end],
                kvcaches[0][0].device,
                self.page_buffer_size,
                lmc_ops.TransferDirection.D2H,
                self.gpu_kv_format,
            )
            memory_obj.tensor.copy_(tmp_gpu_buffer, non_blocking=True)

        if not memory_obj.tensor.is_cuda:
            # Force a synchronize if the target buffer is NOT CUDA device
            # NOTE: for better performance, we may not want to sync for every
            # memory object
            torch.cuda.synchronize()

        if self.use_mla:
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size([2, self.num_layers, num_tokens, self.hidden_dim_size])

    # TODO(Jiayi): need to optimize to enable real batching
    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.to_gpu(memory_obj, start, end, **kwargs)

    # TODO(Yuwei): need to optimize to enable real batching
    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.from_gpu(memory_obj, start, end, **kwargs)


# TODO: support MLA
class SGLangLayerwiseGPUConnector(GPUConnectorInterface):
    """
    The GPU KV cache should be a list of tensors, one for each layer,
    with separate key and value pointers.
    More specifically, we have:
    - kvcaches: Tuple[List[Tensor], List[Tensor]]
      - The first element is a list of key tensors, one per layer.
      - The second element is a list of value tensors, one per layer.
    - Each tensor: [page_buffer_size, head_num, head_size]

    The connector manages the transfer of KV cache data between CPU and GPU
    memory for SGLang using pointer arrays for efficient access.
    It will produce/consume memory objects with KV_2LTD format.
    """

    def __init__(
        self, hidden_dim_size: int, num_layers: int, use_gpu: bool = False, **kwargs
    ):
        assert "dtype" in kwargs, "dtype should be provided to create a GPU buffer."
        self.dtype = kwargs["dtype"]
        assert "device" in kwargs, "device should be provided to create a GPU buffer."
        self.device = kwargs["device"]

        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers

        self.kv_cache_pointers_on_gpu: dict[int, torch.Tensor] = {}
        self.page_buffer_size = 0

        self.gpu_buffer: Optional[torch.Tensor] = None
        self.use_mla = "use_mla" in kwargs and kwargs["use_mla"]

        self.num_kv_cache = num_layers if self.use_mla else num_layers * 2
        self.element_size = torch.tensor([], dtype=self.dtype).element_size()
        self.kv_cache_pointers = torch.empty(
            self.num_kv_cache, dtype=torch.int64, device="cpu"
        )
        self.use_gpu = use_gpu
        self.gpu_buffer_allocator: Optional[GPUMemoryAllocator] = None

    def _lazy_initialize_buffer(self, kv_caches):
        """
        Lazily initialize the GPU buffer allocator if it is not initialized yet.
        Currently, we use the `kv_caches` (kv cache pointer) to determine
        the gpu buffer size in gpu connector.
        Also, the first request might be a bit slower due to buffer creation.
        """
        if self.use_gpu and self.gpu_buffer_allocator is None:
            self.gpu_kv_format = discover_gpu_kv_format(kv_caches, EngineType.SGLANG)
            self.tokens_per_layer = get_tokens_per_layer(kv_caches, self.gpu_kv_format)
            self.elements_per_layer = get_elements_per_layer(
                kv_caches, self.gpu_kv_format
            )
            logger.info(
                f"Lazily initializing GPU buffer (max tokens={self.tokens_per_layer})."
            )
            gpu_buffer_size = self.elements_per_layer * self.element_size
            logger.info(
                f"Lazily initializing GPU buffer (gpu buffer size={gpu_buffer_size})."
            )
            self.gpu_buffer_allocator = GPUMemoryAllocator(
                gpu_buffer_size, device=self.device
            )

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        raise NotImplementedError

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        raise NotImplementedError

    @_lmcache_nvtx_annotate
    def batched_to_gpu(self, starts: List[int], ends: List[int], **kwargs):
        """
        This function is a generator that moves the KV cache from the memory
        objects to paged GPU memory. The first iteration will prepare some
        related metadata. In each of the following iterations, it will first
        wait until the loading of the previous layer finish, and then load
        one layer of KV cache from the memory objects -> GPU buffer ->
        paged GPU memory. The last iteration simply waits for the last layer
        to finish.
        In total, this the generator will yield num_layers + 2 times.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        self._lazy_initialize_buffer(self.kvcaches)

        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])

        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)

        if self.use_gpu:
            buffer_shape = self.get_shape(num_tokens)

            assert self.gpu_buffer_allocator is not None, (
                "GPU buffer allocator should be initialized"
            )
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, MemoryFormat.KV_T2D
            )
            assert tmp_gpu_buffer_obj is not None, (
                "Failed to allocate GPU buffer in GPUConnector"
            )
            assert tmp_gpu_buffer_obj.tensor is not None

        offset = starts[0]

        for layer_id in range(self.num_layers):
            memory_objs_layer = yield
            if layer_id > 0:
                logger.debug(f"Finished loading layer {layer_id - 1}")

            # memobj -> gpu_buffer -> kvcaches
            for start, end, memory_obj in zip(
                starts, ends, memory_objs_layer, strict=False
            ):
                assert memory_obj.metadata.fmt == MemoryFormat.KV_T2D
                if self.use_gpu:
                    tmp_gpu_buffer_obj.tensor[start - offset : end - offset].copy_(
                        memory_obj.tensor, non_blocking=True
                    )
                else:
                    lmc_ops.single_layer_kv_transfer_sgl(
                        memory_obj.tensor,
                        self.kvcaches[0][layer_id],
                        self.kvcaches[1][layer_id],
                        slot_mapping[start:end],
                        lmc_ops.TransferDirection.H2D,
                        token_major=True,
                    )

            if self.use_gpu:
                t, h, d = self.kvcaches[0][layer_id].shape

                lmc_ops.single_layer_kv_transfer_sgl(
                    tmp_gpu_buffer_obj.tensor,
                    self.kvcaches[0][layer_id].view(t, 1, h, d),
                    self.kvcaches[1][layer_id].view(t, 1, h, d),
                    slot_mapping_full,
                    lmc_ops.TransferDirection.H2D,
                    token_major=True,
                )

        # free the buffer memory
        if self.use_gpu:
            tmp_gpu_buffer_obj.ref_count_down()

        logger.debug(f"Finished loading layer {layer_id}")
        yield

    @_lmcache_nvtx_annotate
    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """
        This function is a generator that moves the KV cache from the paged GPU
        memory to the memory objects. The first iteration will prepare some
        related metadata and initiate the transfer in the first layer. In each
        of the following iterations, it will first wait until the storing of
        previous layer finishes, and then initiate string the KV cache of the
        current layer one. The storing process of the KV cache is paged GPU
        memory -> GPU buffer -> memory objects. The last iteration simply waits
        for the last layer to finish.
        In total, this the generator will yield num_layers + 1 times.

        :param memory_objs: The memory objects to store the KV cache. The first
            dimension is the number of layers, and the second dimension is the
            number of memory objects (i.e., number of chunks) for each layer.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        self._lazy_initialize_buffer(self.kvcaches)

        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])

        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)

        if self.use_gpu:
            buffer_shape = self.get_shape(num_tokens)

            assert self.gpu_buffer_allocator is not None, (
                "GPU buffer allocator should be initialized"
            )
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, MemoryFormat.KV_T2D
            )
            assert tmp_gpu_buffer_obj is not None, (
                "Failed to allocate GPU buffer in GPUConnector"
            )
            assert tmp_gpu_buffer_obj.tensor is not None

        for layer_id in range(self.num_layers):
            memory_objs_layer = memory_objs[layer_id]
            # kvcaches -> gpu_buffer -> memobj
            if self.use_gpu:
                t, h, d = self.kvcaches[0][layer_id].shape
                lmc_ops.single_layer_kv_transfer_sgl(
                    tmp_gpu_buffer_obj.tensor,
                    self.kvcaches[0][layer_id].view(t, 1, h, d),
                    self.kvcaches[1][layer_id].view(t, 1, h, d),
                    slot_mapping_full,
                    lmc_ops.TransferDirection.D2H,
                    token_major=True,
                )

            start_idx = 0

            for start, end, memory_obj in zip(
                starts, ends, memory_objs_layer, strict=False
            ):
                assert memory_obj.tensor is not None
                if self.use_gpu:
                    chunk_len = memory_obj.tensor.shape[0]
                    memory_obj.tensor.copy_(
                        tmp_gpu_buffer_obj.tensor[start_idx : start_idx + chunk_len],
                        non_blocking=True,
                    )
                    start_idx += chunk_len
                else:
                    lmc_ops.single_layer_kv_transfer_sgl(
                        memory_obj.tensor,
                        self.kvcaches[0][layer_id],
                        self.kvcaches[1][layer_id],
                        slot_mapping[start:end],
                        lmc_ops.TransferDirection.D2H,
                        token_major=True,
                    )

            yield
            logger.debug(f"Finished offloading layer {layer_id}")

        # free the buffer memory
        if self.use_gpu:
            tmp_gpu_buffer_obj.ref_count_down()
        yield

    def get_shape(self, num_tokens: int) -> torch.Size:
        # TODO: support MLA
        return torch.Size([num_tokens, 2, self.hidden_dim_size])
