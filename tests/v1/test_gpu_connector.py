# SPDX-License-Identifier: Apache-2.0
# Standard
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch
import random
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.gpu_connectors import (
    _GPUBufferReservation,
    SGLangGPUConnector,
    VLLMBufferLayerwiseGPUConnector,
    VLLMPagedMemGPUConnectorV2,
    VLLMPagedMemGPUConnectorV3,
    VLLMPagedMemLayerwiseGPUConnector,
)
from lmcache.v1.gpu_connector.utils import get_dtype
from lmcache.v1.memory_management import (
    GPUMemoryAllocator,
    MemoryFormat,
    PagedTensorMemoryAllocator,
    PinMemoryAllocator,
    TensorMemoryAllocator,
)
from lmcache.v1.metadata import LMCacheMetadata

if torch.cuda.is_available():
    try:
        # First Party
        import lmcache.c_ops as lmc_ops
    except ImportError:
        lmc_ops = None
else:
    lmc_ops = None

# Mock c_ops when not available
if lmc_ops is None:

    class MockGPUKVFormat:
        NL_X_TWO_NB_BS_NH_HS = 0
        NL_X_NB_TWO_BS_NH_HS = 1
        NL_X_NB_BS_HS = 2

    class MockCOps:
        GPUKVFormat = MockGPUKVFormat

    lmc_ops = MockCOps()

# Local
from .utils import (
    check_paged_kv_cache_equal,
    check_paged_kv_cache_equal_with_mla,
    check_sglang_paged_kv_cache_equal,
    generate_kv_cache_paged_list_tensors,
    generate_sglang_kv_cache_paged_list_tensors,
    recover_gpu_connector_states,
)


def test_gpu_buffer_reservation_synchronizes_before_reuse():
    operations = []

    class FakeStream:
        def __init__(self, name):
            self.name = name

        def synchronize(self):
            operations.append(f"sync:{self.name}")

    class FakeMemoryObj:
        def ref_count_down(self):
            operations.append("release")

    reservation = _GPUBufferReservation([FakeStream("store"), FakeStream("copy")])
    reservation.add(FakeMemoryObj())
    reservation.release(synchronize=True)
    reservation.release(synchronize=True)

    assert operations == ["sync:store", "sync:copy", "release"]


def test_layerwise_store_generator_close_drains_before_reuse():
    operations = []

    class FakeStream:
        def __init__(self, name):
            self.name = name

        def synchronize(self):
            operations.append(f"sync:{self.name}")

    class FakeMemoryObj:
        def ref_count_down(self):
            operations.append("release")

    connector = object.__new__(VLLMPagedMemLayerwiseGPUConnector)
    connector.use_gpu = True
    connector.store_streams = [FakeStream("store")]
    connector.chunk_copy_streams = [FakeStream("copy")]

    def fake_batched_from_gpu_impl(memory_objs, starts, ends, **kwargs):
        reservation = kwargs.pop("_gpu_buffer_reservation")
        reservation.add(FakeMemoryObj())
        yield "submitted"

    connector._batched_from_gpu_impl = fake_batched_from_gpu_impl
    generator = connector.batched_from_gpu([], [], [])
    assert next(generator) == "submitted"
    generator.close()

    assert operations == ["sync:store", "sync:copy", "release"]


def test_layerwise_request_batch_defers_slot_mapping_concat():
    connector = object.__new__(VLLMPagedMemLayerwiseGPUConnector)
    connector.num_layers = 2
    connector.device = "cuda"
    connector.kvcaches = [object(), object()]
    connector.kv_cache_pointers_on_gpu = object()
    request_batch = SimpleNamespace(register=Mock())
    chunks = (torch.tensor([0]), torch.tensor([1]))

    with patch.object(torch.cuda, "current_stream", return_value=object()):
        generator = connector._batched_from_gpu_grouped_layers(
            memory_objs=[[object()], [object()]],
            slot_mapping=None,
            slot_mapping_chunks=chunks,
            tmp_gpu_buffer_tensors=[],
            reservation=Mock(),
            dependency_claim=None,
            group_size=2,
            request_batch=request_batch,
            req_id="req",
        )
        next(generator)
        generator.close()

    entry = request_batch.register.call_args.args[0]
    assert entry.slot_mapping_chunks is chunks


def test_layerwise_gpu_buffer_initializes_eagerly_once():
    connector = object.__new__(VLLMPagedMemLayerwiseGPUConnector)
    connector.use_gpu = True
    connector.gpu_buffer_allocator = None
    connector.kv_cache_pointers = torch.empty(2, dtype=torch.int64)
    connector.kv_cache_pointers_on_gpu = None
    connector.num_layers = 2
    connector.layout_hints = {}
    connector.element_size = 2
    connector.layerwise_store_stream_count = 2
    connector.device = "cuda"

    kv_caches = [Mock(), Mock()]
    kv_caches[0].data_ptr.return_value = 100
    kv_caches[1].data_ptr.return_value = 200
    pointer_tensor = Mock()
    allocator = Mock()

    with (
        patch(
            "lmcache.v1.gpu_connector.gpu_connectors.ensure_contiguous_kv_caches",
            side_effect=lambda value, **_: value,
        ),
        patch(
            "lmcache.v1.gpu_connector.gpu_connectors.discover_gpu_kv_format",
            return_value=object(),
        ),
        patch(
            "lmcache.v1.gpu_connector.gpu_connectors."
            "assert_is_vllm_flash_attn_or_flash_infer"
        ),
        patch(
            "lmcache.v1.gpu_connector.gpu_connectors.get_tokens_per_layer",
            return_value=8,
        ),
        patch(
            "lmcache.v1.gpu_connector.gpu_connectors.get_elements_per_layer",
            return_value=16,
        ),
        patch(
            "lmcache.v1.gpu_connector.gpu_connectors.get_num_blocks",
            return_value=4,
        ),
        patch(
            "lmcache.v1.gpu_connector.gpu_connectors.get_block_size",
            return_value=2,
        ),
        patch(
            "lmcache.v1.gpu_connector.gpu_connectors.get_head_size",
            return_value=1,
        ),
        patch(
            "lmcache.v1.gpu_connector.gpu_connectors.GPUMemoryAllocator",
            return_value=allocator,
        ) as allocator_cls,
        patch.object(torch, "empty", return_value=pointer_tensor),
    ):
        connector.initialize_gpu_buffer(kv_caches)
        connector.initialize_gpu_buffer(kv_caches)

    allocator_cls.assert_called_once_with(64, device="cuda")
    pointer_tensor.copy_.assert_called_once_with(connector.kv_cache_pointers)
    assert connector.gpu_buffer_allocator is allocator


def test_layerwise_gpu_transfer_requires_eager_buffer_initialization():
    connector = object.__new__(VLLMPagedMemLayerwiseGPUConnector)
    connector.use_gpu = True
    connector.gpu_buffer_allocator = None

    with pytest.raises(RuntimeError, match="during KV cache registration"):
        connector._require_gpu_buffer()


def test_fused_layerwise_destination_requires_contiguous_exact_allocations():
    allocator = TensorMemoryAllocator(torch.empty(3 * 4096, dtype=torch.uint8))
    memory_objs = allocator.batched_allocate(
        torch.Size([16, 2, 64]),
        torch.float16,
        batch_size=3,
        fmt=MemoryFormat.KV_T2D,
    )
    assert memory_objs is not None
    connector = object.__new__(VLLMPagedMemLayerwiseGPUConnector)

    destination = connector._fused_layerwise_destination(
        [[memory_obj] for memory_obj in memory_objs]
    )

    assert destination is not None
    assert destination.numel() == 3 * 4096
    assert destination.data_ptr() == memory_objs[0].raw_data.data_ptr()
    for memory_obj in memory_objs:
        memory_obj.ref_count_down()
    assert allocator.memcheck()


def test_fused_layerwise_destination_rejects_noncontiguous_order():
    allocator = TensorMemoryAllocator(torch.empty(3 * 4096, dtype=torch.uint8))
    memory_objs = allocator.batched_allocate(
        torch.Size([16, 2, 64]),
        torch.float16,
        batch_size=3,
        fmt=MemoryFormat.KV_T2D,
    )
    assert memory_objs is not None
    connector = object.__new__(VLLMPagedMemLayerwiseGPUConnector)

    destination = connector._fused_layerwise_destination(
        [[memory_objs[0]], [memory_objs[2]], [memory_objs[1]]]
    )

    assert destination is None
    for memory_obj in memory_objs:
        memory_obj.ref_count_down()
    assert allocator.memcheck()


@pytest.fixture(autouse=True, scope="module")
def patch_pin_allocator():
    def fake_pin_init(self, size: int, use_paging: bool = False, **kwargs):
        """
        :param int size: The size of the pinned memory in bytes.
        """

        # self.buffer = torch.empty(size, dtype=torch.uint8)
        # ptr = self.buffer.data_ptr()
        # err = torch.cuda.cudart().cudaHostRegister(ptr, size, 0)
        # assert err == 0, (
        #     f"cudaHostRegister failed: {torch.cuda.cudart().cudaGetErrorString(err)}"
        # )
        self._unregistered = False
        self.buffer = torch.empty(size, dtype=torch.uint8, pin_memory=True)

        if use_paging:
            assert "shapes" in kwargs, (
                "shapes must be specified for paged memory allocator"
            )
            assert "dtypes" in kwargs, (
                "dtypes must be specified for paged memory allocator"
            )
            assert "fmt" in kwargs, "fmt must be specified for paged memory allocator"
            self.allocator = PagedTensorMemoryAllocator(
                tensor=self.buffer,
                shapes=kwargs["shapes"],
                dtypes=kwargs["dtypes"],
                fmt=kwargs["fmt"],
            )
        else:
            self.allocator = TensorMemoryAllocator(self.buffer)

        self.host_mem_lock = threading.Lock() if not use_paging else nullcontext()

    def fake_pin_close(self):
        if not self._unregistered:
            torch.cuda.synchronize()
            # torch.cuda.cudart().cudaHostUnregister(self.buffer.data_ptr())
            self._unregistered = True

    with (
        patch(
            "lmcache.v1.memory_management.PinMemoryAllocator.__init__", fake_pin_init
        ),
        patch("lmcache.v1.memory_management.PinMemoryAllocator.close", fake_pin_close),
    ):
        yield


@pytest.mark.parametrize("use_gpu", [True, False])
@pytest.mark.parametrize(
    "gpu_kv_format",
    [
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,  # vllm non-MLA flash attention
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS,  # vllm non-MLA flash infer
        lmc_ops.GPUKVFormat.NL_X_NB_BS_HS,
    ],  # vllm MLA
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TODO: Add non-CUDA implementation to VLLMPagedMemGPUConnectorV2",
)
def test_vllm_paged_connector_v2_with_gpu_and_mla(use_gpu, gpu_kv_format):
    use_mla = gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_BS_HS
    num_blocks = 100
    block_size = 16
    num_layers = 32
    num_heads = 1 if use_mla else 8
    head_size = 128
    device = "cuda"
    hidden_dim = num_heads * head_size

    num_tokens = 800
    chunk_size = 256

    allocator = PinMemoryAllocator(1024 * 1024 * 1024)

    gpu_kv_src = generate_kv_cache_paged_list_tensors(
        num_blocks=num_blocks,
        device=device,
        block_size=block_size,
        gpu_kv_format=gpu_kv_format,
    )
    gpu_kv_dst = generate_kv_cache_paged_list_tensors(
        num_blocks=num_blocks,
        device=device,
        block_size=block_size,
        gpu_kv_format=gpu_kv_format,
    )
    dtype = get_dtype(gpu_kv_src, gpu_kv_format)

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device, dtype=torch.int64)

    # Check the gpu_kv is not the same before copying
    with pytest.raises(AssertionError):
        if use_mla:
            check_paged_kv_cache_equal_with_mla(
                gpu_kv_src, gpu_kv_dst, slot_mapping, head_size
            )
        else:
            check_paged_kv_cache_equal(
                gpu_kv_src,
                gpu_kv_dst,
                slot_mapping,
                num_heads,
                head_size,
                gpu_kv_format,
            )

    connector = VLLMPagedMemGPUConnectorV2(
        hidden_dim,
        num_layers,
        use_gpu=use_gpu,
        chunk_size=chunk_size,
        dtype=dtype,
        device=device,
        use_mla=use_mla,
    )
    connector2 = VLLMPagedMemGPUConnectorV2(
        hidden_dim,
        num_layers,
        use_gpu=use_gpu,
        chunk_size=chunk_size,
        dtype=dtype,
        device=device,
        use_mla=use_mla,
    )
    assert connector.use_mla == use_mla
    assert connector2.use_mla == use_mla
    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        shape = connector.get_shape(end - start)
        memory_obj = allocator.allocate(shape, dtype)
        connector.from_gpu(
            memory_obj,
            start,
            end,
            kvcaches=gpu_kv_src,
            slot_mapping=slot_mapping,
            offset=0,
        )
        recover_gpu_connector_states(connector)
        if use_mla:
            assert memory_obj.metadata.fmt == MemoryFormat.KV_MLA_FMT
        else:
            assert memory_obj.metadata.fmt == MemoryFormat.KV_2LTD
        connector2.to_gpu(
            memory_obj,
            start,
            end,
            kvcaches=gpu_kv_dst,
            slot_mapping=slot_mapping,
            offset=0,
        )
        allocator.free(memory_obj)
        assert allocator.memcheck()

    if use_mla:
        check_paged_kv_cache_equal_with_mla(
            gpu_kv_src, gpu_kv_dst, slot_mapping, head_size
        )
    else:
        check_paged_kv_cache_equal(
            gpu_kv_src, gpu_kv_dst, slot_mapping, num_heads, head_size, gpu_kv_format
        )
    allocator.close()


@pytest.mark.parametrize("use_gpu", [True, False])
@pytest.mark.parametrize("num_groups", [1, 2, 3])
@pytest.mark.parametrize(
    "gpu_kv_format",
    [
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,  # vllm non-MLA flash attention
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS,  # vllm non-MLA flash infer
        lmc_ops.GPUKVFormat.NL_X_NB_BS_HS,
    ],  # vllm MLA
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TODO: Add non-CUDA implementation to VLLMPagedMemGPUConnectorV3",
)
def test_vllm_paged_connector_v3_with_gpu_and_mla(use_gpu, num_groups, gpu_kv_format):
    use_mla = gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_BS_HS
    head_sizes = [64, 66, 66]
    dtypes = [torch.uint8, torch.bfloat16, torch.uint8]
    num_blocks = 100
    block_size = 16
    num_heads = 1 if use_mla else 8
    device = "cuda"
    num_tokens = 800
    chunk_size = 256

    allocator = PinMemoryAllocator(1024 * 1024 * 1024)

    # generate kv cache tensors
    src_kv_groups: list[list] = []
    dst_kv_groups: list[list] = []
    src_kv_caches: dict[str, torch.Tensor] = {}
    dst_kv_caches: dict[str, torch.Tensor] = {}
    for i in range(num_groups):
        for groups, kv_caches in [
            (src_kv_groups, src_kv_caches),
            (dst_kv_groups, dst_kv_caches),
        ]:
            kv_group = generate_kv_cache_paged_list_tensors(
                num_blocks=num_blocks,
                device=device,
                block_size=block_size,
                dtype=dtypes[i],
                num_layers=8,
                head_size=head_sizes[i],
                gpu_kv_format=gpu_kv_format,
            )
            groups.append(kv_group)
            for j, layer_tensor in enumerate(kv_group):
                kv_caches[f"{i}-{j}"] = layer_tensor

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device, dtype=torch.int64)

    # Check the kv group is not the same before copying
    with pytest.raises(AssertionError):
        for i in range(num_groups):
            if use_mla:
                check_paged_kv_cache_equal_with_mla(
                    src_kv_groups[i], dst_kv_groups[i], slot_mapping, head_sizes[i]
                )
            else:
                check_paged_kv_cache_equal(
                    src_kv_groups[i],
                    dst_kv_groups[i],
                    slot_mapping,
                    num_heads,
                    head_sizes[i],
                    gpu_kv_format,
                )

    # create metadata and init kv layer groups
    metadata = _create_metadata(use_mla, src_kv_caches)
    metadata2 = _create_metadata(use_mla, dst_kv_caches)

    # connector will copy with src_kv_groups
    connector = VLLMPagedMemGPUConnectorV3(
        metadata=metadata,
        use_gpu=use_gpu,
        device=slot_mapping.device,
    )
    # connector2 will copy with dst_kv_groups
    connector2 = VLLMPagedMemGPUConnectorV3(
        metadata=metadata2,
        use_gpu=use_gpu,
        device=slot_mapping.device,
    )
    assert connector.use_mla == use_mla
    assert connector2.use_mla == use_mla

    # copy from src_kv_groups to memory_obj,
    # and then copy from memory_obj to dst_kv_groups
    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        memory_obj = allocator.allocate(
            metadata.get_shapes(end - start), metadata.get_dtypes()
        )
        connector.from_gpu(
            memory_obj,
            start,
            end,
            kvcaches=list(src_kv_caches.values()),
            slot_mapping=slot_mapping,
            offset=0,
        )
        if use_mla:
            assert memory_obj.metadata.fmt == MemoryFormat.KV_MLA_FMT
        else:
            assert memory_obj.metadata.fmt == MemoryFormat.KV_2LTD
        connector2.to_gpu(
            memory_obj,
            start,
            end,
            kvcaches=list(dst_kv_caches.values()),
            slot_mapping=slot_mapping,
            offset=0,
        )
        allocator.free(memory_obj)
        assert allocator.memcheck()

    # Check the kv group is same after copying
    for i in range(num_groups):
        if use_mla:
            check_paged_kv_cache_equal_with_mla(
                src_kv_groups[i], dst_kv_groups[i], slot_mapping, head_sizes[i]
            )
        else:
            check_paged_kv_cache_equal(
                src_kv_groups[i],
                dst_kv_groups[i],
                slot_mapping,
                num_heads,
                head_sizes[i],
                gpu_kv_format,
            )
    allocator.close()


@pytest.mark.parametrize("use_gpu", [True])
@pytest.mark.parametrize(
    "gpu_kv_format",
    [
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,  # vllm non-MLA flash attention
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS,  # vllm non-MLA flash infer
    ],
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TODO: Add non-CUDA implementation to VLLMPagedMemLayerwiseGPUConnector",
)
def test_layerwise_vllm_paged_connector_with_gpu(use_gpu, gpu_kv_format):
    num_blocks = 100
    block_size = 16
    num_layers = 32
    num_heads = 8
    head_size = 128
    device = "cuda"
    hidden_dim = num_heads * head_size

    num_tokens = 800
    chunk_size = 256

    allocator = PinMemoryAllocator(1024 * 1024 * 1024)

    gpu_kv_src = generate_kv_cache_paged_list_tensors(
        num_blocks=num_blocks,
        device=device,
        block_size=block_size,
        gpu_kv_format=gpu_kv_format,
    )
    gpu_kv_dst = generate_kv_cache_paged_list_tensors(
        num_blocks=num_blocks,
        device=device,
        block_size=block_size,
        gpu_kv_format=gpu_kv_format,
    )
    dtype = get_dtype(gpu_kv_src, gpu_kv_format)

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device, dtype=torch.int64)

    # Check the gpu_kv is not the same before copying
    with pytest.raises(AssertionError):
        check_paged_kv_cache_equal(
            gpu_kv_src, gpu_kv_dst, slot_mapping, num_heads, head_size, gpu_kv_format
        )

    connector = VLLMPagedMemLayerwiseGPUConnector(
        hidden_dim,
        num_layers,
        use_gpu=use_gpu,
        chunk_size=chunk_size,
        dtype=dtype,
        device=device,
    )
    connector.initialize_gpu_buffer(gpu_kv_src)

    # from gpu to cpu
    starts = []
    ends = []
    memory_objs = []

    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        shape_single_layer = connector.get_shape(end - start)
        memory_objs_multi_layer = []

        for layer_id in range(num_layers):
            mem_obj_single_layer = allocator.allocate(
                shape_single_layer, dtype, fmt=MemoryFormat.KV_T2D
            )
            memory_objs_multi_layer.append(mem_obj_single_layer)

        starts.append(start)
        ends.append(end)
        memory_objs.append(memory_objs_multi_layer)

    memory_objs = [list(row) for row in zip(*memory_objs, strict=False)]

    mem_obj_generator = connector.batched_from_gpu(
        memory_objs,
        starts,
        ends,
        kvcaches=gpu_kv_src,
        slot_mapping=slot_mapping,
        sync=True,
    )

    for layer_id in range(num_layers + 1):
        next(mem_obj_generator)

    # from cpu to gpu
    mem_obj_consumer = connector.batched_to_gpu(
        starts,
        ends,
        kvcaches=gpu_kv_dst,
        slot_mapping=slot_mapping,
        sync=True,
    )
    next(mem_obj_consumer)
    for layer_id in range(num_layers):
        mem_obj_consumer.send(memory_objs[layer_id])
    next(mem_obj_consumer)

    # free all mem objs
    for mem_obj_multi_layer in memory_objs:
        for mem_obj in mem_obj_multi_layer:
            mem_obj.ref_count_down()

    assert allocator.memcheck()

    assert connector.gpu_buffer_allocator.memcheck()

    check_paged_kv_cache_equal(
        gpu_kv_src, gpu_kv_dst, slot_mapping, num_heads, head_size, gpu_kv_format
    )

    allocator.close()


@pytest.mark.parametrize("use_gpu", [True])
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TODO: Add non-CUDA implementation to VLLMPagedMemLayerwiseGPUConnector",
)
def test_batched_layerwise_vllm_paged_connector_with_gpu(use_gpu):
    num_blocks = 100
    block_size = 16
    num_layers = 32
    num_heads = 8
    head_size = 128
    device = "cuda"
    hidden_dim = num_heads * head_size

    num_tokens_1 = 800
    num_tokens_2 = 500
    num_tokens_total = num_tokens_1 + num_tokens_2
    chunk_size = 256

    allocator = PinMemoryAllocator(1024 * 1024 * 1024)

    gpu_kv_src = generate_kv_cache_paged_list_tensors(num_blocks, device, block_size)
    gpu_kv_dst = generate_kv_cache_paged_list_tensors(num_blocks, device, block_size)
    dtype = gpu_kv_src[0][0].dtype

    slot_mapping_total = random.sample(
        range(0, num_blocks * block_size), num_tokens_total
    )
    slot_mapping_total = torch.tensor(
        slot_mapping_total, device=device, dtype=torch.int64
    )

    # Check the gpu_kv is not the same before copying
    with pytest.raises(AssertionError):
        check_paged_kv_cache_equal(gpu_kv_src, gpu_kv_dst, slot_mapping_total)

    connector = VLLMPagedMemLayerwiseGPUConnector(
        hidden_dim,
        num_layers,
        use_gpu=use_gpu,
        chunk_size=chunk_size,
        dtype=dtype,
        device=device,
    )
    connector.initialize_gpu_buffer(gpu_kv_src)

    # from gpu to cpu
    starts_1 = []
    ends_1 = []
    memory_objs_1 = []

    for start in range(0, num_tokens_1, chunk_size):
        end = min(start + chunk_size, num_tokens_1)
        shape_single_layer = connector.get_shape(end - start)
        memory_objs_multi_layer = []

        for layer_id in range(num_layers):
            mem_obj_single_layer = allocator.allocate(
                shape_single_layer, dtype, fmt=MemoryFormat.KV_T2D
            )
            memory_objs_multi_layer.append(mem_obj_single_layer)

        starts_1.append(start)
        ends_1.append(end)
        memory_objs_1.append(memory_objs_multi_layer)

    memory_objs_1 = [list(row) for row in zip(*memory_objs_1, strict=False)]

    starts_2 = []
    ends_2 = []
    memory_objs_2 = []
    for start in range(num_tokens_1, num_tokens_total, chunk_size):
        end = min(start + chunk_size, num_tokens_total)
        shape_single_layer = connector.get_shape(end - start)
        memory_objs_multi_layer = []

        for layer_id in range(num_layers):
            mem_obj_single_layer = allocator.allocate(
                shape_single_layer, dtype, fmt=MemoryFormat.KV_T2D
            )
            memory_objs_multi_layer.append(mem_obj_single_layer)

        starts_2.append(start)
        ends_2.append(end)
        memory_objs_2.append(memory_objs_multi_layer)

    memory_objs_2 = [list(row) for row in zip(*memory_objs_2, strict=False)]

    mem_obj_generator_1 = connector.batched_from_gpu(
        memory_objs_1,
        starts_1,
        ends_1,
        kvcaches=gpu_kv_src,
        slot_mapping=slot_mapping_total,
        sync=True,
    )

    mem_obj_generator_1 = connector.batched_from_gpu(
        memory_objs_1,
        starts_1,
        ends_1,
        kvcaches=gpu_kv_src,
        slot_mapping=slot_mapping_total,
        sync=True,
    )

    mem_obj_generator_2 = connector.batched_from_gpu(
        memory_objs_2,
        starts_2,
        ends_2,
        kvcaches=gpu_kv_src,
        slot_mapping=slot_mapping_total,
        sync=False,
    )

    for layer_id in range(num_layers + 1):
        next(mem_obj_generator_1)
        next(mem_obj_generator_2)

    # from cpu to gpu
    mem_obj_consumer_1 = connector.batched_to_gpu(
        starts_1,
        ends_1,
        kvcaches=gpu_kv_dst,
        slot_mapping=slot_mapping_total,
        sync=False,
    )
    mem_obj_consumer_2 = connector.batched_to_gpu(
        starts_2,
        ends_2,
        kvcaches=gpu_kv_dst,
        slot_mapping=slot_mapping_total,
        sync=True,
    )

    next(mem_obj_consumer_1)
    next(mem_obj_consumer_2)
    for layer_id in range(num_layers):
        mem_obj_consumer_1.send(memory_objs_1[layer_id])
        mem_obj_consumer_2.send(memory_objs_2[layer_id])
    next(mem_obj_consumer_1)
    next(mem_obj_consumer_2)

    # free all mem objs
    for mem_obj_multi_layer in memory_objs_1:
        for mem_obj in mem_obj_multi_layer:
            mem_obj.ref_count_down()

    for mem_obj_multi_layer in memory_objs_2:
        for mem_obj in mem_obj_multi_layer:
            mem_obj.ref_count_down()

    assert allocator.memcheck()

    assert connector.gpu_buffer_allocator.memcheck()

    check_paged_kv_cache_equal(
        gpu_kv_src,
        gpu_kv_dst,
        slot_mapping_total,
        num_heads,
        head_size,
    )

    allocator.close()


@pytest.mark.skip(reason="This test is skipped due to vllm dependency")
@pytest.mark.parametrize("use_gpu", [True])
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TODO: Add non-CUDA implementation to VLLMBufferLayerwiseGPUConnector",
)
def test_layerwise_vllm_buffer_connector_with_gpu(use_gpu):
    num_blocks = 100
    block_size = 16
    num_layers = 32
    num_heads = 8
    head_size = 128
    device = "cuda"
    hidden_dim = num_heads * head_size

    num_tokens = 800
    chunk_size = 256

    allocator = PinMemoryAllocator(1024 * 1024 * 1024)

    gpu_kv_src = generate_kv_cache_paged_list_tensors(num_blocks, device, block_size)
    gpu_kv_dst = generate_kv_cache_paged_list_tensors(num_blocks, device, block_size)
    dtype = gpu_kv_src[0][0].dtype

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device, dtype=torch.int64)

    # Check the gpu_kv is not the same before copying
    with pytest.raises(AssertionError):
        check_paged_kv_cache_equal(
            gpu_kv_src, gpu_kv_dst, slot_mapping, num_heads, head_size
        )

    connector = VLLMBufferLayerwiseGPUConnector(
        hidden_dim,
        num_layers,
        use_gpu=use_gpu,
        dtype=dtype,
        device=device,
    )

    # from gpu to cpu
    starts = []
    ends = []
    memory_objs = []

    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        shape_single_layer = connector.get_shape(end - start)
        memory_objs_multi_layer = []

        for layer_id in range(num_layers):
            mem_obj_single_layer = allocator.allocate(
                shape_single_layer, dtype, fmt=MemoryFormat.KV_2TD
            )
            memory_objs_multi_layer.append(mem_obj_single_layer)

        starts.append(start)
        ends.append(end)
        memory_objs.append(memory_objs_multi_layer)

    memory_objs = [list(row) for row in zip(*memory_objs, strict=False)]

    mem_obj_generator = connector.batched_from_gpu(
        memory_objs,
        starts,
        ends,
        kvcaches=gpu_kv_src,
        slot_mapping=slot_mapping,
    )

    for layer_id in range(num_layers + 1):
        next(mem_obj_generator)

    # from cpu to gpu
    mem_obj_consumer = connector.batched_to_gpu(
        starts,
        ends,
        kvcaches=gpu_kv_dst,
        slot_mapping=slot_mapping,
    )
    next(mem_obj_consumer)
    for layer_id in range(num_layers):
        mem_obj_consumer.send(memory_objs[layer_id])
    next(mem_obj_consumer)

    # free all mem objs
    for mem_obj_multi_layer in memory_objs:
        for mem_obj in mem_obj_multi_layer:
            mem_obj.ref_count_down()

    assert allocator.memcheck()

    assert connector.gpu_buffer_allocator.memcheck()

    check_paged_kv_cache_equal(
        gpu_kv_src, gpu_kv_dst, slot_mapping, num_heads, head_size
    )

    allocator.close()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TODO: Add non-CUDA implementation to VLLMPagedMemGPUConnectorV2",
)
def test_vllm_paged_connector_v2_to_gpu_bench(benchmark):
    """
    VLLMPagedMemGPUConnectorV2.to_gpu() micro-benchmark.

    This test is to measure the performance of
    VLLMPagedMemGPUConnectorV2.to_gpu() when both KV caches and
    memobject are on GPU.

    """
    num_blocks = 100
    block_size = 16
    num_layers = 32
    num_heads = 8
    head_size = 128
    device = "cuda"
    hidden_dim = num_heads * head_size

    chunk_size = 256

    allocator = GPUMemoryAllocator(1024 * 1024 * 1024)

    gpu_kv_src = generate_kv_cache_paged_list_tensors(num_blocks, device, block_size)
    gpu_kv_dst = generate_kv_cache_paged_list_tensors(num_blocks, device, block_size)

    slot_mapping = random.sample(range(0, num_blocks * block_size), chunk_size)
    slot_mapping = torch.tensor(slot_mapping, device=device, dtype=torch.int64)

    connector = VLLMPagedMemGPUConnectorV2(hidden_dim, num_layers)
    shape = connector.get_shape(chunk_size)
    memory_obj = allocator.allocate(shape, gpu_kv_src[0][0].dtype)
    connector.from_gpu(
        memory_obj,
        0,
        chunk_size,
        kvcaches=gpu_kv_src,
        slot_mapping=slot_mapping,
        offset=0,
    )
    recover_gpu_connector_states(connector)
    assert memory_obj.metadata.fmt == MemoryFormat.KV_2LTD
    benchmark.pedantic(
        connector.to_gpu,
        args=(memory_obj, 0, chunk_size),
        kwargs={
            "kvcaches": gpu_kv_dst,
            "slot_mapping": slot_mapping,
            "offset": 0,
        },
        rounds=100,
        iterations=1000,
        warmup_rounds=10,
    )
    allocator.free(memory_obj)
    assert allocator.memcheck()

    allocator.close()


@pytest.mark.parametrize("use_gpu", [True, False])
@pytest.mark.parametrize("use_mla", [True, False])
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TODO: Add non-CUDA implementation to SGLangGPUConnector",
)
def test_sglang_connector_with_gpu_and_mla(use_gpu, use_mla):
    num_blocks = 100
    block_size = 16
    num_layers = 32
    num_heads = 1 if use_mla else 8
    head_size = 128
    device = "cuda"
    dtype = torch.bfloat16
    hidden_dim = num_heads * head_size

    num_tokens = num_blocks * block_size // 2
    chunk_size = 256

    allocator = PinMemoryAllocator(1024 * 1024 * 1024)

    gpu_kv_src = generate_sglang_kv_cache_paged_list_tensors(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_size=head_size,
        use_mla=use_mla,
        device=device,
        dtype=dtype,
    )
    gpu_kv_dst = generate_sglang_kv_cache_paged_list_tensors(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_size=head_size,
        use_mla=use_mla,
        device=device,
        dtype=dtype,
    )

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device, dtype=torch.int64)

    # Check the gpu_kv is not the same before copying
    with pytest.raises(AssertionError):
        if use_mla:
            check_paged_kv_cache_equal_with_mla(
                gpu_kv_src, gpu_kv_dst, slot_mapping, head_size
            )
        else:
            check_sglang_paged_kv_cache_equal(
                gpu_kv_src, gpu_kv_dst, slot_mapping, num_heads, head_size
            )

    connector = SGLangGPUConnector(
        hidden_dim,
        num_layers,
        use_gpu=use_gpu,
        chunk_size=chunk_size,
        dtype=dtype,
        device=device,
        use_mla=use_mla,
    )
    connector2 = SGLangGPUConnector(
        hidden_dim,
        num_layers,
        use_gpu=use_gpu,
        chunk_size=chunk_size,
        dtype=dtype,
        device=device,
        use_mla=use_mla,
    )
    assert connector.use_mla == use_mla
    assert connector2.use_mla == use_mla
    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        shape = connector.get_shape(end - start)
        memory_obj = allocator.allocate(shape, gpu_kv_src[0][0].dtype)
        connector.from_gpu(
            memory_obj,
            start,
            end,
            kvcaches=gpu_kv_src,
            slot_mapping=slot_mapping,
            offset=0,
        )
        if use_mla:
            assert memory_obj.metadata.fmt == MemoryFormat.KV_MLA_FMT
        else:
            assert memory_obj.metadata.fmt == MemoryFormat.KV_2LTD
        connector2.to_gpu(
            memory_obj,
            start,
            end,
            kvcaches=gpu_kv_dst,
            slot_mapping=slot_mapping,
            offset=0,
        )
        allocator.free(memory_obj)
        assert allocator.memcheck()

    if use_mla:
        check_paged_kv_cache_equal_with_mla(
            gpu_kv_src, gpu_kv_dst, slot_mapping, head_size
        )
    else:
        check_sglang_paged_kv_cache_equal(
            gpu_kv_src, gpu_kv_dst, slot_mapping, num_heads, head_size
        )

    allocator.close()


def _create_metadata(use_mla, kv_caches):
    num_heads = 1 if use_mla else 8
    metadata = LMCacheMetadata(
        model_name="test",
        world_size=8,
        local_world_size=8,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(32, 2, 256, num_heads, 128),
        use_mla=use_mla,
    )
    metadata.kv_layer_groups_manager.build_kv_layer_groups(kv_caches)
    return metadata
