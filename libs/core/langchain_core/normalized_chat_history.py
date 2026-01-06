"""Normalized chat history with KV cache optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.canonical_cache import CanonicalPromptCache
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.prompt_normalization import PromptNormalizer
from pydantic import ConfigDict

if TYPE_CHECKING:
    from collections.abc import Sequence


class NormalizedChatHistory(InMemoryChatMessageHistory):
    """Chat history with prompt normalization and KV cache optimization.

    This class extends InMemoryChatMessageHistory to add:
    - Prompt normalization for semantic similarity matching
    - KV cache management for canonical prompts
    - Cache hit rate tracking and metrics

    Example:
        ```python
        from langchain_core.embeddings import OpenAIEmbeddings
        from langchain_core.normalized_chat_history import NormalizedChatHistory
        from langchain_core.prompt_normalization import PromptNormalizer
        from langchain_core.canonical_cache import CanonicalPromptCache

        normalizer = PromptNormalizer(
            embeddings=OpenAIEmbeddings(),
            similarity_threshold=0.95,
        )
        cache = CanonicalPromptCache(max_size=1000)

        history = NormalizedChatHistory(
            normalizer=normalizer,
            cache=cache,
        )

        # Add messages - they will be normalized automatically
        history.add_messages([HumanMessage("Hello, how are you?")])
        ```
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    normalizer: PromptNormalizer | None = None
    """Prompt normalizer for semantic similarity matching."""

    cache: CanonicalPromptCache | None = None
    """KV cache for storing canonical prompt entries."""

    enable_normalization: bool = True
    """Whether to enable prompt normalization."""

    _last_normalization_result: dict[str, object] | None = None

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Add messages to the history with optional normalization.

        Args:
            messages: A sequence of BaseMessage objects to store.
        """
        for message in messages:
            self.add_message(message)

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the history with optional normalization.

        Args:
            message: The message to add.
        """
        if self.enable_normalization and self.normalizer is not None:
            normalized_message = self._normalize_message(message)
            super().add_message(normalized_message)
        else:
            super().add_message(message)

    def _normalize_message(self, message: BaseMessage) -> BaseMessage:
        """Normalize a message content using the normalizer.

        Args:
            message: The message to normalize.

        Returns:
            The normalized message.
        """
        if not message.content:
            return message

        result = self.normalizer.normalize(str(message.content))

        self._last_normalization_result = {
            "original": message.content,
            "canonical": result.canonical,
            "normalized": result.normalized,
            "similarity": result.similarity,
            "method": result.method,
        }

        if result.normalized and result.canonical != message.content:
            return message.__class__(content=result.canonical)

        return message

    def check_cache_hit(self, prompt: str) -> bool:
        """Check if a prompt has a cache hit.

        Args:
            prompt: The prompt to check.

        Returns:
            True if the prompt has a cache hit, False otherwise.
        """
        if self.cache is None:
            return False

        if self.normalizer is not None:
            result = self.normalizer.normalize(prompt)
            canonical_prompt = result.canonical
        else:
            canonical_prompt = prompt

        # Use get() instead of contains() to properly track metrics
        entry = self.cache.get(canonical_prompt)
        return entry is not None

    def get_cache_entry(self, prompt: str):
        """Get the cache entry for a prompt.

        Args:
            prompt: The prompt to get the cache entry for.

        Returns:
            The KV cache entry if found, None otherwise.
        """
        if self.cache is None:
            return None

        if self.normalizer is not None:
            result = self.normalizer.normalize(prompt)
            canonical_prompt = result.canonical
        else:
            canonical_prompt = prompt

        return self.cache.get(canonical_prompt)

    def add_cache_entry(
        self,
        canonical_prompt: str,
        token_ids: list[int],
        size_bytes: int = 0,
    ):
        """Add a cache entry for a canonical prompt.

        Args:
            canonical_prompt: The canonical prompt.
            token_ids: The token IDs for the prompt.
            size_bytes: The size of the cache entry in bytes.
        """
        if self.cache is not None:
            self.cache.add(canonical_prompt, token_ids, size_bytes)

    def get_normalization_metrics(self):
        """Get normalization metrics.

        Returns:
            NormalizationMetrics object with current metrics, or None if no normalizer.
        """
        if self.normalizer is None:
            return None

        return self.normalizer.get_metrics()

    def get_cache_metrics(self):
        """Get cache metrics.

        Returns:
            CacheMetrics object with current metrics, or None if no cache.
        """
        if self.cache is None:
            return None

        return self.cache.get_metrics()

    def get_metrics(self) -> dict[str, object]:
        """Get combined metrics for normalization and cache.

        Returns:
            A dictionary containing combined metrics.
        """
        metrics = {}

        if self.normalizer is not None:
            norm_metrics = self.normalizer.get_metrics()
            metrics["normalization"] = {
                "total_normalizations": norm_metrics.total_normalizations,
                "exact_match_hits": norm_metrics.exact_match_hits,
                "semantic_match_hits": norm_metrics.semantic_match_hits,
            }

        if self.cache is not None:
            cache_metrics = self.cache.get_metrics()
            metrics["cache"] = {
                "hits": cache_metrics.hits,
                "misses": cache_metrics.misses,
                "evictions": cache_metrics.evictions,
                "total_added": cache_metrics.total_added,
                "total_removed": cache_metrics.total_removed,
                "current_memory_bytes": cache_metrics.current_memory_bytes,
                "total_requests": cache_metrics.total_requests,
            }

        return metrics

    def get_last_normalization_result(self) -> dict[str, object] | None:
        """Get the result of the last normalization operation.

        Returns:
            A dictionary containing the last normalization result,
            or None if no normalization has been performed.
        """
        return self._last_normalization_result

    def clear_normalization_history(self) -> None:
        """Clear the normalization history."""
        if self.normalizer is not None:
            self.normalizer.clear_history()

    def clear_cache(self) -> None:
        """Clear the KV cache."""
        if self.cache is not None:
            self.cache.clear()

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        if self.normalizer is not None:
            self.normalizer.reset_metrics()

        if self.cache is not None:
            self.cache.reset_metrics()
