# SPDX-License-Identifier: Apache-2.0
"""
Shared Disk Backend for disaggregated prefill-decode architecture.

This backend enables KV cache transfer between prefill and decode engines
through a shared storage system (local disk, NFS, GCS, S3, etc.).

Key features:
- Layer-wise pipelined KV transfer
- Pluggable storage connector interface
- Support for both sender (prefiller) and receiver (decoder) roles
- Async I/O for overlapping with computation
"""

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Union
import asyncio
import os
import threading
import time

import msgspec
import torch
import zmq

from lmcache.logging import init_logger
from lmcache.utils import (
    CacheEngineKey,
    STR_DTYPE_TO_TORCH_DTYPE,
    TORCH_DTYPE_TO_STR_DTYPE,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    PagedCpuGpuMemoryAllocator,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.rpc_utils import get_zmq_context, get_zmq_socket
from lmcache.v1.storage_backend.abstract_backend import AllocatorBackendInterface
from lmcache.v1.storage_backend.connector.shared_storage_connector import (
    SharedStorageConfig,
    SharedStorageConnector,
    create_shared_storage_connector,
)
from lmcache.v1.transfer_channel.transfer_utils import get_correct_device

logger = init_logger(__name__)


# Message types for coordination between sender and receiver
class SharedDiskMsgBase(msgspec.Struct, tag=True):
    """Base class for shared disk messages."""
    pass


class ChunkReadyNotif(SharedDiskMsgBase):
    """Notification that a chunk is ready to be read."""
    req_id: str
    keys: list[str]  # Keys that are ready
    fmt: int
    shape: list[int]
    dtype: str
    last_chunk_toks: int


class ChunkAck(SharedDiskMsgBase):
    """Acknowledgment that chunks have been received."""
    req_id: str
    received_keys: list[str]


class ProxyNotif(SharedDiskMsgBase):
    """Notification to proxy that transfer is complete."""
    req_id: str


SharedDiskMsg = Union[ChunkReadyNotif, ChunkAck, ProxyNotif]


@dataclass
class SharedDiskConfig:
    """Configuration for SharedDiskBackend."""
    
    role: str  # "sender" or "receiver"
    
    # Storage connector configuration
    storage_path: str
    storage_type: str = "local_disk"  # "local_disk", "gcs", "s3"
    max_storage_size: int = 0  # bytes, 0 = unlimited
    
    # Communication configuration
    peer_host: Optional[str] = None
    peer_port: Optional[int] = None
    proxy_host: Optional[str] = None
    proxy_port: Optional[int] = None
    
    # Buffer configuration
    buffer_size: int = 1073741824  # 1GB default
    buffer_device: str = "cuda"
    
    # Extra options
    use_odirect: bool = False
    
    @staticmethod
    def from_cache_engine_config(
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        tp_rank: int,
    ) -> "SharedDiskConfig":
        """Create SharedDiskConfig from LMCacheEngineConfig."""
        
        # Get role from pd_role or extra_config
        role = config.pd_role
        assert role in ["sender", "receiver"], f"Invalid role: {role}"
        
        extra = config.extra_config or {}
        
        # Get storage path from config or extra_config
        storage_path = getattr(config, "shared_disk_path", None)
        if storage_path is None:
            storage_path = extra.get(
                "shared_disk_path",
                "/tmp/lmcache_shared_disk"
            )
        
        storage_type = getattr(config, "shared_disk_type", None)
        if storage_type is None:
            storage_type = extra.get("shared_disk_type", "local_disk")
        
        # Get buffer configuration
        buffer_size = config.pd_buffer_size or 1073741824
        buffer_device = get_correct_device(
            config.pd_buffer_device or "cuda",
            metadata.worker_id
        )
        
        # Get peer/proxy configuration based on role
        peer_host = config.pd_peer_host
        peer_port = None
        if config.pd_peer_init_port is not None:
            peer_port = config.pd_peer_init_port[tp_rank]
        
        proxy_host = config.pd_proxy_host
        proxy_port = config.pd_proxy_port
        
        return SharedDiskConfig(
            role=role,
            storage_path=storage_path,
            storage_type=storage_type,
            max_storage_size=int(config.max_local_disk_size * 1024**3),
            peer_host=peer_host,
            peer_port=peer_port,
            proxy_host=proxy_host,
            proxy_port=proxy_port,
            buffer_size=buffer_size,
            buffer_device=buffer_device,
            use_odirect=extra.get("use_odirect", False),
        )


class SharedDiskBackend(AllocatorBackendInterface):
    """
    Storage backend for disaggregated prefill-decode using shared disk.
    
    This backend enables KV cache transfer between prefill and decode engines
    through a shared storage system. The sender (prefiller) writes KV cache
    to disk layer-by-layer, and the receiver (decoder) reads it layer-by-layer,
    enabling pipelined transfer that overlaps with computation.
    """
    
    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
    ):
        self.running = True
        self.tp_rank = metadata.worker_id
        
        # Parse configuration
        self.sd_config = SharedDiskConfig.from_cache_engine_config(
            config, metadata, self.tp_rank
        )
        
        self.corrected_device = self.sd_config.buffer_device
        
        # Data storage for receiver
        self.data: dict[CacheEngineKey, MemoryObj] = {}
        self.data_lock = threading.Lock()
        
        # Initialize memory allocator
        self.memory_allocator = self.initialize_allocator(config, metadata)
        assert isinstance(self.memory_allocator, PagedCpuGpuMemoryAllocator)
        
        # Initialize storage connector
        self.connector = self._create_connector()
        
        # ZMQ context for coordination
        self.zmq_context = get_zmq_context(use_asyncio=False)
        self.running_threads: list[threading.Thread] = []
        self.side_channels: list[zmq.Socket] = []
        
        # Initialize role-specific components
        if self.sd_config.role == "sender":
            self._init_sender()
        elif self.sd_config.role == "receiver":
            self._init_receiver()
        else:
            raise ValueError(f"Invalid role: {self.sd_config.role}")
        
        self.full_chunk_size = config.chunk_size
        
        logger.info(
            f"SharedDiskBackend initialized: role={self.sd_config.role}, "
            f"storage_path={self.sd_config.storage_path}, "
            f"device={self.corrected_device}"
        )
    
    def __str__(self):
        return "SharedDiskBackend"
    
    def _create_connector(self) -> SharedStorageConnector:
        """Create the appropriate storage connector based on configuration."""
        
        connector_config = SharedStorageConfig(
            base_path=self.sd_config.storage_path,
            max_size=self.sd_config.max_storage_size,
            use_async=True,
            extra_options={"use_odirect": self.sd_config.use_odirect},
        )
        
        return create_shared_storage_connector(
            self.sd_config.storage_type,
            connector_config,
        )
    
    def initialize_allocator(
        self, config: LMCacheEngineConfig, metadata: LMCacheMetadata
    ) -> PagedCpuGpuMemoryAllocator:
        """Initialize the memory allocator for this backend."""
        
        if self.corrected_device != "cpu":
            logger.info(f"Setting cuda device to {self.corrected_device}")
            torch.cuda.set_device(self.corrected_device)
        
        paged_mem_allocator = PagedCpuGpuMemoryAllocator()
        
        init_func = (
            paged_mem_allocator.init_cpu_memory_allocator
            if self.corrected_device == "cpu"
            else paged_mem_allocator.init_gpu_memory_allocator
        )
        
        init_func(
            self.sd_config.buffer_size,
            [torch.Size(metadata.kv_shape)],
            [metadata.kv_dtype],
            MemoryFormat.KV_2LTD,
        )
        
        return paged_mem_allocator
    
    def get_memory_allocator(self) -> PagedCpuGpuMemoryAllocator:
        return self.memory_allocator
    
    def get_allocator_backend(self):
        return self
    
    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[MemoryObj]:
        if fmt is None:
            fmt = MemoryFormat.KV_2LTD
        alloc_type = "cpu" if self.corrected_device == "cpu" else "gpu"
        return self.memory_allocator.allocate(
            shapes, dtypes, fmt=fmt, allocator_type=alloc_type
        )
    
    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[list[MemoryObj]]:
        if fmt is None:
            fmt = MemoryFormat.KV_2LTD
        alloc_type = "cpu" if self.corrected_device == "cpu" else "gpu"
        return self.memory_allocator.batched_allocate(
            shapes, dtypes, batch_size, fmt, allocator_type=alloc_type
        )
    
    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        assert isinstance(key, CacheEngineKey)
        with self.data_lock:
            if mem_obj := self.data.get(key, None):
                if pin:
                    mem_obj.ref_count_up()
                return True
            return False
    
    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        return False
    
    # =========================================================================
    # Sender (Prefiller) Methods
    # =========================================================================
    
    def _init_sender(self):
        """Initialize sender-specific components."""
        # Connect to proxy for completion notification
        if self.sd_config.proxy_host and self.sd_config.proxy_port:
            proxy_url = f"{self.sd_config.proxy_host}:{self.sd_config.proxy_port}"
            self.proxy_side_channel = get_zmq_socket(
                self.zmq_context,
                proxy_url,
                "tcp",
                zmq.PUSH,
                "connect",
            )
        else:
            self.proxy_side_channel = None
            logger.warning("No proxy configured for sender")
        
        # Track peer connections
        self.initialized_peers: set[str] = set()
        self.notif_sockets: dict[str, zmq.Socket] = {}
    
    def _ensure_peer_connection(
        self,
        receiver_id: str,
        receiver_host: str,
        receiver_port: int,
    ) -> None:
        """Establish connection to receiver for notifications."""
        if receiver_id in self.initialized_peers:
            return
        
        notif_url = f"{receiver_host}:{receiver_port}"
        notif_socket = get_zmq_socket(
            self.zmq_context,
            notif_url,
            "tcp",
            zmq.PUSH,
            "connect",
        )
        self.notif_sockets[receiver_id] = notif_socket
        self.initialized_peers.add(receiver_id)
        
        logger.info(f"Connected to receiver {receiver_id} at {notif_url}")
    
    def _write_chunk_to_storage(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
    ) -> bool:
        """Write a single chunk to storage."""
        key_str = key.to_string()
        buffer = memory_obj.byte_array
        
        success = self.connector.write(key_str, buffer)
        if success:
            logger.debug(f"Wrote chunk {key_str} to storage")
        else:
            logger.error(f"Failed to write chunk {key_str}")
        
        return success
    
    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """
        Submit batched put tasks to write KV caches to shared storage.
        
        For sender role: Write chunks to disk and notify receiver.
        """
        # Increment ref counts
        for mem_obj in memory_objs:
            mem_obj.ref_count_up()
        
        # Get receiver info from transfer_spec
        receiver_init_port = transfer_spec.receiver_init_port[self.tp_rank]
        receiver_id = transfer_spec.receiver_host + str(receiver_init_port)
        receiver_host = transfer_spec.receiver_host
        
        self._ensure_peer_connection(
            receiver_id=receiver_id,
            receiver_host=receiver_host,
            receiver_port=receiver_init_port,
        )
        
        # Write each chunk to storage
        written_keys = []
        for key, mem_obj in zip(keys, memory_objs, strict=False):
            if self._write_chunk_to_storage(key, mem_obj):
                written_keys.append(key.to_string())
        
        # Send notification to receiver
        if written_keys:
            fmt = memory_objs[0].meta.fmt
            shape = memory_objs[0].meta.shape
            dtype = TORCH_DTYPE_TO_STR_DTYPE[memory_objs[0].meta.dtype]
            token_dim = fmt.token_dim()
            last_chunk_toks = memory_objs[-1].meta.shape[token_dim]
            
            notif = ChunkReadyNotif(
                req_id=transfer_spec.req_id,
                keys=written_keys,
                fmt=fmt.value,
                shape=list(shape),
                dtype=dtype,
                last_chunk_toks=last_chunk_toks,
            )
            
            notif_socket = self.notif_sockets[receiver_id]
            notif_socket.send(msgspec.msgpack.encode(notif))
            logger.debug(f"Sent notification for {len(written_keys)} chunks")
        
        # Release ref counts
        for mem_obj in memory_objs:
            mem_obj.ref_count_down()
        
        # Notify proxy if this is the last prefill
        if transfer_spec.is_last_prefill:
            if self.proxy_side_channel is not None:
                proxy_notif = ProxyNotif(req_id=transfer_spec.req_id)
                self.proxy_side_channel.send(msgspec.msgpack.encode(proxy_notif))
                logger.debug(f"Notified proxy for request {transfer_spec.req_id}")
        
        # Call completion callbacks
        if on_complete_callback is not None:
            for key in keys:
                try:
                    on_complete_callback(key)
                except Exception as e:
                    logger.warning(f"on_complete_callback failed for {key}: {e}")
    
    # =========================================================================
    # Receiver (Decoder) Methods
    # =========================================================================
    
    def _init_receiver(self):
        """Initialize receiver-specific components."""
        # Set up notification listener
        if self.sd_config.peer_port:
            listen_url = f"*:{self.sd_config.peer_port}"
            self.notif_listen_socket = get_zmq_socket(
                self.zmq_context,
                listen_url,
                "tcp",
                zmq.PULL,
                "bind",
            )
            self.side_channels.append(self.notif_listen_socket)
            
            # Start notification listener thread
            self.notif_thread = threading.Thread(
                target=self._notif_listener_loop,
                daemon=True,
            )
            self.notif_thread.start()
            self.running_threads.append(self.notif_thread)
            
            logger.info(f"Receiver listening on port {self.sd_config.peer_port}")
        else:
            logger.warning("No listen port configured for receiver")
    
    def _notif_listener_loop(self):
        """Listen for chunk ready notifications from sender."""
        while self.running:
            try:
                msg_bytes = self.notif_listen_socket.recv(flags=zmq.NOBLOCK)
                msg = msgspec.msgpack.decode(msg_bytes, type=SharedDiskMsg)
                
                if isinstance(msg, ChunkReadyNotif):
                    self._handle_chunk_ready(msg)
                else:
                    logger.warning(f"Unexpected message type: {type(msg)}")
                    
            except zmq.Again:
                # No message available, sleep briefly
                time.sleep(0.001)
            except Exception as e:
                if self.running:
                    logger.error(f"Error in notification listener: {e}")
                    time.sleep(0.01)
    
    def _handle_chunk_ready(self, notif: ChunkReadyNotif):
        """Handle chunk ready notification by loading from storage."""
        fmt = MemoryFormat(notif.fmt)
        dtype = STR_DTYPE_TO_TORCH_DTYPE[notif.dtype]
        shape = list(notif.shape)
        
        for idx, key_str in enumerate(notif.keys):
            key = CacheEngineKey.from_string(key_str)
            
            # Skip if already loaded
            if self.contains(key):
                continue
            
            # Adjust shape for last chunk
            if idx == len(notif.keys) - 1:
                token_dim = fmt.token_dim()
                shape[token_dim] = notif.last_chunk_toks
            
            # Allocate memory
            mem_obj = self.allocate(torch.Size(shape), dtype, fmt)
            
            # Retry allocation with backoff
            wait_time = 0.01
            while mem_obj is None:
                logger.warning("Memory allocation failed, retrying...")
                time.sleep(wait_time)
                wait_time = min(wait_time * 2, 1.0)
                mem_obj = self.allocate(torch.Size(shape), dtype, fmt)
            
            # Load from storage
            buffer = mem_obj.byte_array
            if self.connector.read_into(key_str, buffer):
                self.put(key, mem_obj)
                logger.debug(f"Loaded chunk {key_str} from storage")
            else:
                logger.error(f"Failed to load chunk {key_str}")
                mem_obj.ref_count_down()
    
    def put(self, key: CacheEngineKey, mem_obj: MemoryObj):
        """Store a memory object in the local cache."""
        with self.data_lock:
            self.data[key] = mem_obj
    
    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Get a memory object from the local cache."""
        with self.data_lock:
            mem_obj = self.data.get(key, None)
            if mem_obj is None:
                # Try loading from storage
                key_str = key.to_string()
                if self.connector.exists(key_str):
                    # We don't have shape/dtype info here, so can't load
                    logger.warning(
                        f"Key {key} exists in storage but not in local cache"
                    )
            return mem_obj
    
    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        """Remove a key from local cache and optionally from storage."""
        with self.data_lock:
            if mem_obj := self.data.get(key, None):
                if mem_obj.get_ref_count() == 1:
                    del self.data[key]
                    # Optionally delete from storage
                    key_str = key.to_string()
                    self.connector.delete(key_str)
                return True
            return False
    
    def pin(self, key: CacheEngineKey) -> bool:
        return True
    
    def unpin(self, key: CacheEngineKey) -> bool:
        return True
    
    def close(self) -> None:
        """Close the backend and release resources."""
        self.running = False
        
        # Wait for threads
        for thread in self.running_threads:
            thread.join(timeout=5.0)
        
        # Close sockets
        for socket in self.side_channels:
            socket.close()
        
        # Close connector
        self.connector.close()
        
        # Terminate ZMQ context
        self.zmq_context.term()
        
        logger.info("SharedDiskBackend closed")
