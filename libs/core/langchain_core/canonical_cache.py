"""Canonical prompt cache for KV cache optimization.

This module provides a cache system that stores canonical prompts and their
associated KV cache entries, enabling efficient reuse across semantically
similar prompts.

Key features:
- Maps canonical prompts to KV cache entries
- LRU-based cache eviction policy
- Cache hit rate tracking and metrics
- Thread-safe operations
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from typing import Any

from pydantic import BaseModel, Field


class KVCacheEntry(BaseModel):
    """Represents a KV cache entry."""

    canonical_prompt: str = Field(description="The canonical prompt for this cache entry.")
    token_ids: list[int] = Field(description="Token IDs for the canonical prompt.")
    cache_key: str = Field(description="Unique key for this cache entry.")
    created_at: float = Field(default_factory=time.time, description="Creation timestamp.")
    last_accessed: float = Field(default_factory=time.time, description="Last access timestamp.")
    access_count: int = Field(default=0, description="Number of times this entry was accessed.")
    size_bytes: int = Field(default=0, description="Size of the cache entry in bytes.")

    def touch(self) -> None:
        """Update the last accessed timestamp."""
        self.last_accessed = time.time()
        self.access_count += 1


class CanonicalPromptCache(BaseModel):
    """Cache for canonical prompts and their KV cache entries.

    This cache manages KV cache entries keyed by canonical prompts,
    implementing LRU eviction policy and tracking cache metrics.

    Example:
        ```python
        from langchain_core.canonical_cache import CanonicalPromptCache

        cache = CanonicalPromptCache(
            max_size=1000,
            max_memory_bytes=1024 * 1024 * 1024,
        )

        cache.add(
            canonical_prompt="You are a helpful assistant.",
            token_ids=[123, 456, 789],
        )

        entry = cache.get("You are a helpful assistant.")
        if entry:
            print(f"Cache hit! Access count: {entry.access_count}")
        ```
    """

    max_size: int = Field(
        default=1000,
        ge=1,
        description="Maximum number of entries in the cache.",
    )
    max_memory_bytes: int = Field(
        default=1024 * 1024 * 1024,
        ge=1,
        description="Maximum memory usage in bytes.",
    )
    enable_lru: bool = Field(
        default=True,
        description="Whether to enable LRU eviction policy.",
    )
    enable_metrics: bool = Field(
        default=True,
        description="Whether to track cache metrics.",
    )
    eviction_batch_size: int = Field(
        default=10,
        ge=1,
        description="Number of entries to evict when cache is full.",
    )
    cleanup_interval: float = Field(
        default=3600.0,
        ge=0.0,
        description="Interval in seconds for periodic cleanup of expired entries (0 to disable).",
    )
    enable_compression: bool = Field(
        default=False,
        description="Whether to compress large cache entries to save memory.",
    )
    compression_threshold: int = Field(
        default=1024,
        ge=1,
        description="Minimum size in bytes before compression is applied.",
    )
    max_token_ids_per_entry: int = Field(
        default=10000,
        ge=1,
        description="Maximum number of token IDs allowed per cache entry.",
    )
    enable_async_cleanup: bool = Field(
        default=False,
        description="Whether to enable asynchronous cleanup of expired entries.",
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._cache: OrderedDict[str, KVCacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._metrics = CacheMetrics()
        self._last_cleanup_time = time.time()
        self._async_cleanup_task = None

        # Start async cleanup if enabled
        if self.enable_async_cleanup and self.cleanup_interval > 0:
            self._start_async_cleanup()

    def add(
        self,
        canonical_prompt: str,
        token_ids: list[int],
        size_bytes: int = 0,
    ) -> KVCacheEntry:
        """Add a KV cache entry for a canonical prompt.

        Args:
            canonical_prompt: The canonical prompt.
            token_ids: Token IDs for the prompt.
            size_bytes: Size of the cache entry in bytes.

        Returns:
            The created KVCacheEntry.
        """
        # Validate inputs
        if canonical_prompt is None:
            raise ValueError("canonical_prompt cannot be None")
        if token_ids is None:
            raise ValueError("token_ids cannot be None")

        # Check token ID limit
        if len(token_ids) > self.max_token_ids_per_entry:
            raise ValueError(f"token_ids length {len(token_ids)} exceeds maximum {self.max_token_ids_per_entry}")

        # Compress token IDs if enabled
        compressed_token_ids = self._compress_token_ids(token_ids)

        cache_key = self._generate_cache_key(canonical_prompt)

        with self._lock:
            # Perform periodic cleanup if needed
            if self.cleanup_interval > 0:
                current_time = time.time()
                if current_time - self._last_cleanup_time > self.cleanup_interval:
                    self._perform_cleanup()
                    self._last_cleanup_time = current_time

            if cache_key in self._cache:
                entry = self._cache[cache_key]
                entry.touch()
                if self.enable_metrics:
                    self._metrics.update_hit()
                return entry

            entry = KVCacheEntry(
                canonical_prompt=canonical_prompt,
                token_ids=compressed_token_ids,
                cache_key=cache_key,
                size_bytes=size_bytes,
            )

            self._cache[cache_key] = entry

            if self.enable_lru:
                self._cache.move_to_end(cache_key)

            if self.enable_metrics:
                self._metrics.update_add(entry.size_bytes)

            self._evict_if_needed()

            return entry

    def get(self, canonical_prompt: str) -> KVCacheEntry | None:
        """Get a KV cache entry for a canonical prompt.

        Args:
            canonical_prompt: The canonical prompt.

        Returns:
            The KVCacheEntry if found, None otherwise.
        """
        cache_key = self._generate_cache_key(canonical_prompt)

        with self._lock:
            if cache_key not in self._cache:
                if self.enable_metrics:
                    self._metrics.update_miss()
                return None

            entry = self._cache[cache_key]
            entry.touch()

            if self.enable_lru:
                self._cache.move_to_end(cache_key)

            if self.enable_metrics:
                self._metrics.update_hit()

            # Decompress token IDs if they were compressed
            if self.enable_compression and isinstance(entry.token_ids, bytes):
                entry.token_ids = self._decompress_token_ids(entry.token_ids)

            return entry

    def remove(self, canonical_prompt: str) -> bool:
        """Remove a KV cache entry.

        Args:
            canonical_prompt: The canonical prompt.

        Returns:
            True if the entry was removed, False if not found.
        """
        cache_key = self._generate_cache_key(canonical_prompt)

        with self._lock:
            if cache_key not in self._cache:
                return False

            entry = self._cache.pop(cache_key)
            if self.enable_metrics:
                self._metrics.update_remove(entry.size_bytes)

            return True

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            if self.enable_metrics:
                self._metrics.reset()

    def contains(self, canonical_prompt: str) -> bool:
        """Check if a canonical prompt is in the cache.

        Args:
            canonical_prompt: The canonical prompt.

        Returns:
            True if the prompt is cached, False otherwise.
        """
        cache_key = self._generate_cache_key(canonical_prompt)
        return cache_key in self._cache

    def get_size(self) -> int:
        """Get the current number of entries in the cache.

        Returns:
            Number of entries.
        """
        return len(self._cache)

    def get_memory_usage(self) -> int:
        """Get the current memory usage in bytes.

        Returns:
            Memory usage in bytes.
        """
        return sum(entry.size_bytes for entry in self._cache.values())

    def get_metrics(self) -> CacheMetrics:
        """Get cache metrics.

        Returns:
            CacheMetrics object with current metrics.
        """
        return self._metrics

    def reset_metrics(self) -> None:
        """Reset cache metrics."""
        self._metrics.reset()

    def _generate_cache_key(self, canonical_prompt: str) -> str:
        """Generate a cache key from a canonical prompt.

        Args:
            canonical_prompt: The canonical prompt.

        Returns:
            A hash-based cache key.
        """
        if canonical_prompt is None:
            raise ValueError("canonical_prompt cannot be None")
        return hashlib.sha256(canonical_prompt.encode()).hexdigest()

    def _evict_if_needed(self) -> None:
        """Evict entries if cache exceeds limits."""
        while self._eviction_needed():
            self._evict_lru()

    def _eviction_needed(self) -> bool:
        """Check if eviction is needed.

        Returns:
            True if eviction is needed.
        """
        if len(self._cache) > self.max_size:
            return True
        if self.get_memory_usage() > self.max_memory_bytes:
            return True
        return False

    def _evict_lru(self) -> None:
        """Evict the least recently used entries in batch."""
        if not self._cache:
            return

        # Evict multiple entries based on eviction_batch_size
        for _ in range(min(self.eviction_batch_size, len(self._cache))):
            if not self._eviction_needed():
                break

            cache_key, entry = self._cache.popitem(last=False)
            if self.enable_metrics:
                self._metrics.update_evict(entry.size_bytes)


class CacheMetrics(BaseModel):
    """Metrics for cache performance tracking."""

    hits: int = Field(default=0, description="Number of cache hits.")
    misses: int = Field(default=0, description="Number of cache misses.")
    evictions: int = Field(default=0, description="Number of cache evictions.")
    total_added: int = Field(default=0, description="Total entries added.")
    total_removed: int = Field(default=0, description="Total entries removed.")
    current_memory_bytes: int = Field(default=0, description="Current memory usage in bytes.")

    @property
    def total_requests(self) -> int:
        """Total number of requests (hits + misses)."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate as a fraction (0-1)."""
        total_requests = self.hits + self.misses
        if total_requests == 0:
            return 0.0
        return self.hits / total_requests

    def update_hit(self) -> None:
        """Record a cache hit."""
        self.hits += 1

    def update_miss(self) -> None:
        """Record a cache miss."""
        self.misses += 1

    def update_add(self, size_bytes: int) -> None:
        """Record an entry addition."""
        self.total_added += 1
        self.current_memory_bytes += size_bytes

    def update_remove(self, size_bytes: int) -> None:
        """Record an entry removal."""
        self.total_removed += 1
        self.current_memory_bytes = max(0, self.current_memory_bytes - size_bytes)

    def update_evict(self, size_bytes: int) -> None:
        """Record an entry eviction."""
        self.evictions += 1
        self.current_memory_bytes = max(0, self.current_memory_bytes - size_bytes)

    def get_hit_rate(self) -> float:
        """Calculate cache hit rate.

        Returns:
            Hit rate as a percentage (0-100).
        """
        total_requests = self.hits + self.misses
        if total_requests == 0:
            return 0.0
        return (self.hits / total_requests) * 100

    def reset(self) -> None:
        """Reset all metrics."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.total_added = 0
        self.total_removed = 0
        self.current_memory_bytes = 0


# Additional methods for CanonicalPromptCache configuration

def _start_async_cleanup(self) -> None:
    """Start asynchronous cleanup task."""
    import threading

    def cleanup_worker():
        while self.enable_async_cleanup:
            time.sleep(self.cleanup_interval)
            if self.enable_async_cleanup:  # Check again after sleep
                self._perform_cleanup()

    self._async_cleanup_task = threading.Thread(target=cleanup_worker, daemon=True)
    self._async_cleanup_task.start()

def _perform_cleanup(self) -> None:
    """Perform periodic cleanup of expired entries."""
    with self._lock:
        current_time = time.time()
        expired_keys = []

        for cache_key, entry in self._cache.items():
            # Remove entries that haven't been accessed for a long time
            if current_time - entry.last_accessed > self.cleanup_interval * 2:
                expired_keys.append(cache_key)

        for cache_key in expired_keys:
            if cache_key in self._cache:
                entry = self._cache.pop(cache_key)
                if self.enable_metrics:
                    self._metrics.update_remove(entry.size_bytes)

def _compress_token_ids(self, token_ids: list[int]) -> bytes:
    """Compress token IDs if compression is enabled."""
    if not self.enable_compression:
        return token_ids

    # Simple compression for large token lists
    if len(token_ids) * 4 > self.compression_threshold:  # 4 bytes per int
        import zlib
        token_bytes = bytes(token_ids)
        compressed = zlib.compress(token_bytes)
        return compressed

    return token_ids

def _decompress_token_ids(self, compressed_data: bytes) -> list[int]:
    """Decompress token IDs if they were compressed."""
    if isinstance(compressed_data, list):
        return compressed_data

    if self.enable_compression and isinstance(compressed_data, bytes):
        import zlib
        try:
            decompressed = zlib.decompress(compressed_data)
            return list(decompressed)
        except:
            return []

    return compressed_data

def stop_async_cleanup(self) -> None:
    """Stop asynchronous cleanup task."""
    self.enable_async_cleanup = False
    if self._async_cleanup_task and self._async_cleanup_task.is_alive():
        self._async_cleanup_task.join(timeout=1.0)

def get_config_summary(self) -> dict[str, Any]:
    """Get a summary of current configuration."""
    return {
        "max_size": self.max_size,
        "max_memory_bytes": self.max_memory_bytes,
        "enable_lru": self.enable_lru,
        "enable_metrics": self.enable_metrics,
        "eviction_batch_size": self.eviction_batch_size,
        "cleanup_interval": self.cleanup_interval,
        "enable_compression": self.enable_compression,
        "compression_threshold": self.compression_threshold,
        "max_token_ids_per_entry": self.max_token_ids_per_entry,
        "enable_async_cleanup": self.enable_async_cleanup,
        "current_entries": len(self._cache),
        "current_memory_usage": self.get_memory_usage(),
    }
