# SPDX-License-Identifier: Apache-2.0
"""
Shared storage connector interface for disaggregated KV cache transfer.

This module defines the abstract interface for shared storage connectors used
by SharedDiskBackend. Implementations include LocalDiskConnector for local/NFS
storage, and can be extended for GCS, S3, etc.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import os
import threading
import time

from lmcache.logging import init_logger

logger = init_logger(__name__)


@dataclass
class SharedStorageConfig:
    """Configuration for shared storage connectors."""
    
    # Base path/prefix for storage (e.g., "/tmp/lmcache" or "gs://bucket/prefix")
    base_path: str
    
    # Maximum storage size in bytes (0 = unlimited)
    max_size: int = 0
    
    # Whether to use async I/O where supported
    use_async: bool = True
    
    # Additional connector-specific options
    extra_options: Optional[dict] = None


class SharedStorageConnector(ABC):
    """
    Abstract interface for shared storage connectors.
    
    These connectors handle the actual I/O operations for reading and writing
    KV cache data in a shared storage system. They abstract the underlying 
    storage system (local disk, NFS, GCS, S3, etc.) from the SharedDiskBackend.
    """
    
    def __init__(self, config: SharedStorageConfig):
        self.config = config
        self._lock = threading.Lock()
    
    @abstractmethod
    def write(self, key: str, data: bytes) -> bool:
        """
        Write data to storage.
        
        Args:
            key: Unique identifier for the data (will be converted to path/key)
            data: Raw bytes to write
            
        Returns:
            True if write succeeded, False otherwise
        """
        raise NotImplementedError
    
    @abstractmethod
    def read(self, key: str) -> Optional[bytes]:
        """
        Read data from storage.
        
        Args:
            key: Unique identifier for the data
            
        Returns:
            Raw bytes if found, None otherwise
        """
        raise NotImplementedError
    
    @abstractmethod
    def read_into(self, key: str, buffer: bytearray) -> bool:
        """
        Read data from storage into a pre-allocated buffer.
        
        This is more efficient than read() when the buffer is already allocated.
        
        Args:
            key: Unique identifier for the data
            buffer: Pre-allocated buffer to read into
            
        Returns:
            True if read succeeded, False otherwise
        """
        raise NotImplementedError
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if data exists in storage.
        
        Args:
            key: Unique identifier for the data
            
        Returns:
            True if exists, False otherwise
        """
        raise NotImplementedError
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete data from storage.
        
        Args:
            key: Unique identifier for the data
            
        Returns:
            True if deleted (or didn't exist), False on error
        """
        raise NotImplementedError
    
    @abstractmethod
    def list_keys(self, prefix: str = "") -> list[str]:
        """
        List all keys with the given prefix.
        
        Args:
            prefix: Key prefix to filter by
            
        Returns:
            List of matching keys
        """
        raise NotImplementedError
    
    def key_to_path(self, key: str) -> str:
        """
        Convert a cache key to a storage path.
        
        Args:
            key: Cache key string
            
        Returns:
            Full path/URI for storage
        """
        # Default implementation: sanitize key and join with base path
        safe_key = key.replace("/", "-").replace(":", "_")
        return f"{self.config.base_path}/{safe_key}"
    
    @abstractmethod
    def close(self) -> None:
        """
        Close the connector and release resources.
        """
        raise NotImplementedError
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.config.base_path})"


class LocalSharedDiskConnector(SharedStorageConnector):
    """
    Local disk storage connector for shared storage.
    
    Implements storage operations on the local filesystem (or NFS mount) with 
    support for:
    - Regular file I/O
    - O_DIRECT for bypassing OS page cache (when aligned)
    - Async-compatible operations
    """
    
    def __init__(self, config: SharedStorageConfig):
        super().__init__(config)
        
        # Ensure base directory exists
        if not os.path.exists(config.base_path):
            os.makedirs(config.base_path, exist_ok=True)
            logger.info(f"Created shared storage directory: {config.base_path}")
        
        # Get filesystem block size for O_DIRECT alignment
        stat = os.statvfs(config.base_path)
        self.block_size = stat.f_bsize
        
        # Check if O_DIRECT should be used
        self.use_odirect = False
        if config.extra_options:
            self.use_odirect = config.extra_options.get("use_odirect", False)
        
        logger.info(
            f"LocalSharedDiskConnector initialized: path={config.base_path}, "
            f"block_size={self.block_size}, use_odirect={self.use_odirect}"
        )
    
    def key_to_path(self, key: str) -> str:
        """Convert cache key to filesystem path."""
        safe_key = key.replace("/", "-").replace(":", "_")
        return os.path.join(self.config.base_path, f"{safe_key}.bin")
    
    def write(self, key: str, data: bytes) -> bool:
        """Write data to local disk."""
        path = self.key_to_path(key)
        size = len(data)
        
        try:
            start_time = time.time()
            
            # Check if we can use O_DIRECT (requires alignment)
            if self.use_odirect and size % self.block_size == 0:
                fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_DIRECT, 0o644)
                try:
                    os.write(fd, data)
                finally:
                    os.close(fd)
            else:
                with open(path, "wb") as f:
                    f.write(data)
            
            elapsed = time.time() - start_time
            bandwidth = size / elapsed / 1e6 if elapsed > 0 else 0
            logger.debug(
                f"Shared disk write: key={key}, size={size} bytes, "
                f"bandwidth={bandwidth:.2f} MB/s"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to write key {key}: {e}")
            return False
    
    def read(self, key: str) -> Optional[bytes]:
        """Read data from local disk."""
        path = self.key_to_path(key)
        
        if not os.path.exists(path):
            return None
        
        try:
            start_time = time.time()
            
            with open(path, "rb") as f:
                data = f.read()
            
            elapsed = time.time() - start_time
            size = len(data)
            bandwidth = size / elapsed / 1e6 if elapsed > 0 else 0
            logger.debug(
                f"Shared disk read: key={key}, size={size} bytes, "
                f"bandwidth={bandwidth:.2f} MB/s"
            )
            return data
            
        except Exception as e:
            logger.error(f"Failed to read key {key}: {e}")
            return None
    
    def read_into(self, key: str, buffer: bytearray) -> bool:
        """Read data from local disk into a pre-allocated buffer."""
        path = self.key_to_path(key)
        
        if not os.path.exists(path):
            logger.warning(f"File not found: {path}")
            return False
        
        try:
            start_time = time.time()
            size = len(buffer)
            
            # Check if we can use O_DIRECT
            if self.use_odirect and size % self.block_size == 0:
                fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
                try:
                    with os.fdopen(fd, "rb", buffering=0) as fdo:
                        fdo.readinto(buffer)
                except Exception:
                    os.close(fd)
                    raise
            else:
                with open(path, "rb") as f:
                    f.readinto(buffer)
            
            elapsed = time.time() - start_time
            bandwidth = size / elapsed / 1e6 if elapsed > 0 else 0
            logger.debug(
                f"Shared disk read_into: key={key}, size={size} bytes, "
                f"bandwidth={bandwidth:.2f} MB/s"
            )
            return True
            
        except FileNotFoundError:
            logger.warning(f"File not found: {path}")
            return False
        except Exception as e:
            logger.error(f"Failed to read key {key} into buffer: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if data exists on disk."""
        path = self.key_to_path(key)
        return os.path.exists(path)
    
    def delete(self, key: str) -> bool:
        """Delete data from disk."""
        path = self.key_to_path(key)
        
        try:
            if os.path.exists(path):
                os.remove(path)
            return True
        except Exception as e:
            logger.error(f"Failed to delete key {key}: {e}")
            return False
    
    def list_keys(self, prefix: str = "") -> list[str]:
        """List all keys in the storage directory."""
        keys = []
        
        try:
            for filename in os.listdir(self.config.base_path):
                if filename.endswith(".bin"):
                    # Remove .bin extension and convert back to key format
                    key = filename[:-4].replace("-", "/").replace("_", ":")
                    if key.startswith(prefix):
                        keys.append(key)
        except Exception as e:
            logger.error(f"Failed to list keys: {e}")
        
        return keys
    
    def close(self) -> None:
        """Close the connector (no-op for local disk)."""
        pass


def create_shared_storage_connector(
    storage_type: str,
    config: SharedStorageConfig,
) -> SharedStorageConnector:
    """
    Factory function to create the appropriate shared storage connector.
    
    Args:
        storage_type: Type of storage ("local_disk", "gcs", "s3")
        config: Storage configuration
        
    Returns:
        An instance of SharedStorageConnector
        
    Raises:
        ValueError: If storage_type is unknown
        ImportError: If required dependencies are not available
    """
    if storage_type == "local_disk":
        return LocalSharedDiskConnector(config)
    elif storage_type == "gcs":
        # Future: Import and return GCS connector
        raise ImportError(
            "GCS connector not yet implemented. "
            "Please use local_disk for now."
        )
    elif storage_type == "s3":
        # Future: Import and return S3 connector
        raise ImportError(
            "S3 connector for shared disk not yet implemented. "
            "Please use local_disk for now."
        )
    else:
        raise ValueError(f"Unknown storage type: {storage_type}")
