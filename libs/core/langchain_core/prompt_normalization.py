"""Prompt normalization for KV cache optimization.

This module provides functionality to normalize semantically similar prompts
into canonical forms, enabling better KV cache reuse in inference engines.

Key features:
- Semantic similarity-based prompt normalization
- Support for preserving template variables (slots)
- Configurable similarity thresholds
- Performance-optimized for <1ms latency
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from langchain_core.embeddings import Embeddings

if TYPE_CHECKING:
    pass


class PromptNormalizer(BaseModel):
    """Normalizes semantically similar prompts to canonical forms.

    This class uses embeddings to identify semantically similar prompts and
    maps them to canonical forms, enabling KV cache reuse across different
    prompt variations.

    Example:
        ```python
        from langchain_core.embeddings import FakeEmbeddings
        from langchain_core.prompt_normalization import PromptNormalizer

        normalizer = PromptNormalizer(
            embeddings=FakeEmbeddings(size=100),
            similarity_threshold=0.95,
        )

        normalizer.add_canonical_prompt(
            canonical="You are a helpful assistant.",
            variations=["You're really helpful", "You are very helpful"],
        )

        result = normalizer.normalize("You're really helpful")
        print(result.canonical)
        ```
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    embeddings: Embeddings | None = Field(
        default=None,
        description="Embedding model for semantic similarity. If None, uses exact string matching.",
    )
    similarity_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score to consider prompts equivalent.",
    )
    cache_embeddings: bool = Field(
        default=True,
        description="Whether to cache embeddings for performance.",
    )
    slot_pattern: str = Field(
        default=r"\{[^}]+\}",
        description="Regex pattern for identifying template slots.",
    )
    max_variations_per_canonical: int = Field(
        default=100,
        ge=1,
        description="Maximum number of variations to store per canonical prompt.",
    )
    enable_semantic_caching: bool = Field(
        default=True,
        description="Whether to cache semantic similarity results.",
    )
    semantic_cache_size: int = Field(
        default=1000,
        ge=1,
        description="Size of the semantic similarity cache.",
    )
    enable_fuzzy_matching: bool = Field(
        default=False,
        description="Whether to enable fuzzy string matching for variations.",
    )
    fuzzy_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Threshold for fuzzy string matching.",
    )
    enable_preprocessing: bool = Field(
        default=True,
        description="Whether to enable text preprocessing (lowercasing, trimming).",
    )
    preprocessing_rules: dict[str, bool] = Field(
        default_factory=lambda: {
            "lowercase": True,
            "trim_whitespace": True,
            "remove_extra_spaces": True,
            "normalize_unicode": False,
        },
        description="Rules for text preprocessing.",
    )
    max_prompt_length: int = Field(
        default=10000,
        ge=1,
        description="Maximum allowed prompt length in characters.",
    )
    enable_length_filtering: bool = Field(
        default=True,
        description="Whether to filter prompts by length.",
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._canonical_prompts: dict[str, CanonicalPrompt] = {}
        self._embedding_cache: dict[str, list[float]] = {}
        self._metrics = NormalizationMetrics()
        self._semantic_cache: dict[str, NormalizationResult] = {}
        self._preprocessing_cache: dict[str, str] = {}

        # Initialize fuzzy matching if enabled
        if self.enable_fuzzy_matching:
            try:
                import fuzzywuzzy
                self._fuzzy_available = True
            except ImportError:
                self._fuzzy_available = False
                print("Warning: fuzzywuzzy not available, disabling fuzzy matching")

    def add_canonical_prompt(
        self,
        canonical: str,
        variations: Sequence[str] | None = None,
    ) -> None:
        """Add a canonical prompt with its variations.

        Args:
            canonical: The canonical form of the prompt.
            variations: Known variations that map to this canonical form.
        """
        variations = list(variations) if variations else []
        canonical_prompt = CanonicalPrompt(
            canonical=canonical,
            variations=variations,
        )
        self._canonical_prompts[canonical] = canonical_prompt

        for variation in variations:
            self._canonical_prompts[variation] = canonical_prompt

        if self.embeddings and self.cache_embeddings:
            self._cache_embedding(canonical)
            for variation in variations:
                self._cache_embedding(variation)

    def normalize(self, prompt: str) -> NormalizationResult:
        """Normalize a prompt to its canonical form.

        Args:
            prompt: The prompt to normalize.

        Returns:
            NormalizationResult containing the canonical prompt and metadata.
        """
        self._metrics.total_normalizations += 1

        # Validate prompt length
        if self.enable_length_filtering and len(prompt) > self.max_prompt_length:
            self._metrics.length_filtered += 1
            return NormalizationResult(
                canonical=prompt,
                normalized=False,
                similarity=1.0,
                method="length_filtered",
            )

        # Preprocess prompt if enabled
        if self.enable_preprocessing:
            processed_prompt = self._preprocess_prompt(prompt)
            if processed_prompt in self._preprocessing_cache:
                return self._preprocessing_cache[processed_prompt]
        else:
            processed_prompt = prompt

        # Check semantic cache if enabled
        if self.enable_semantic_caching and processed_prompt in self._semantic_cache:
            result = self._semantic_cache[processed_prompt]
            self._metrics.cache_hits += 1
            return result

        if processed_prompt in self._canonical_prompts:
            canonical_prompt = self._canonical_prompts[processed_prompt]
            self._metrics.exact_match_hits += 1
            result = NormalizationResult(
                canonical=canonical_prompt.canonical,
                normalized=True,
                similarity=1.0,
                method="exact_match",
            )
            if self.enable_semantic_caching:
                self._add_to_semantic_cache(processed_prompt, result)
            return result

        if self.embeddings:
            result = self._normalize_with_embeddings(processed_prompt)
            # If no match found, add this prompt as a new canonical prompt
            if not result.normalized:
                self.add_canonical_prompt(canonical=processed_prompt)
        else:
            # If no embeddings and no exact match, add as new canonical prompt
            self.add_canonical_prompt(canonical=processed_prompt)
            result = NormalizationResult(
                canonical=processed_prompt,
                normalized=False,
                similarity=1.0,
                method="no_embeddings",
            )

        if self.enable_semantic_caching:
            self._add_to_semantic_cache(processed_prompt, result)

        return result

    def _normalize_with_embeddings(self, prompt: str) -> NormalizationResult:
        """Normalize using semantic similarity.

        Args:
            prompt: The prompt to normalize.

        Returns:
            NormalizationResult with the best matching canonical prompt.
        """
        prompt_embedding = self._get_embedding(prompt)
        best_match: str | None = None
        best_similarity = 0.0

        for canonical, canonical_prompt in self._canonical_prompts.items():
            if canonical != canonical_prompt.canonical:
                continue

            canonical_embedding = self._get_embedding(canonical)
            similarity = self._cosine_similarity(prompt_embedding, canonical_embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = canonical

        if best_match and best_similarity >= self.similarity_threshold:
            self._metrics.semantic_match_hits += 1
            return NormalizationResult(
                canonical=best_match,
                normalized=True,
                similarity=best_similarity,
                method="semantic_similarity",
            )

        # Check if there are any canonical prompts at all
        if not self._canonical_prompts:
            # Add this prompt as a new canonical prompt
            self.add_canonical_prompt(canonical=prompt)
            return NormalizationResult(
                canonical=prompt,
                normalized=False,
                similarity=0.0,
                method="no_canonical_match",
            )

        # Add this prompt as a new canonical prompt
        self.add_canonical_prompt(canonical=prompt)
        return NormalizationResult(
            canonical=prompt,
            normalized=False,
            similarity=best_similarity,
            method="below_threshold",
        )

    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text, using cache if enabled.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        if self.cache_embeddings and text in self._embedding_cache:
            return self._embedding_cache[text]

        if self.embeddings:
            embedding = self.embeddings.embed_query(text)
            if self.cache_embeddings:
                self._embedding_cache[text] = embedding
            return embedding

        return []

    def _cache_embedding(self, text: str) -> None:
        """Cache embedding for text.

        Args:
            text: Text to cache embedding for.
        """
        if self.embeddings and text not in self._embedding_cache:
            self._embedding_cache[text] = self.embeddings.embed_query(text)

    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector.
            vec2: Second vector.

        Returns:
            Cosine similarity score.
        """
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def extract_slots(self, prompt: str) -> list[str]:
        """Extract template slots from a prompt.

        Args:
            prompt: The prompt to extract slots from.

        Returns:
            List of slot names found in the prompt.
        """
        return re.findall(self.slot_pattern, prompt)

    def preserve_slots(self, original: str, canonical: str) -> str:
        """Preserve slots from original prompt in canonical form.

        Args:
            original: Original prompt with slots.
            canonical: Canonical prompt to preserve slots in.

        Returns:
            Canonical prompt with slots from original.
        """
        original_slots = self.extract_slots(original)
        canonical_slots = self.extract_slots(canonical)

        if len(original_slots) != len(canonical_slots):
            return canonical

        result = canonical
        for orig_slot, canon_slot in zip(original_slots, canonical_slots):
            result = result.replace(canon_slot, orig_slot)

        return result

    def get_canonical_count(self) -> int:
        """Get the number of canonical prompts.

        Returns:
            Number of unique canonical prompts.
        """
        unique_canonicals = {
            cp.canonical for cp in self._canonical_prompts.values()
        }
        return len(unique_canonicals)

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()

    def clear_history(self) -> None:
        """Clear the canonical prompt history."""
        self._canonical_prompts.clear()
        self._embedding_cache.clear()
        # Reset metrics when clearing history
        self._metrics = NormalizationMetrics()

    def get_metrics(self) -> NormalizationMetrics:
        """Get normalization metrics.

        Returns:
            NormalizationMetrics object with current metrics.
        """
        return self._metrics

    def reset_metrics(self) -> None:
        """Reset normalization metrics."""
        self._metrics = NormalizationMetrics()


class CanonicalPrompt(BaseModel):
    """Represents a canonical prompt with its variations."""

    canonical: str = Field(description="The canonical form of the prompt.")
    variations: list[str] = Field(
        default_factory=list,
        description="Known variations that map to this canonical form.",
    )


class NormalizationResult(BaseModel):
    """Result of prompt normalization."""

    canonical: str = Field(description="The canonical form of the prompt.")
    normalized: bool = Field(description="Whether the prompt was normalized.")
    similarity: float = Field(description="Similarity score to canonical form.")
    method: str = Field(description="Method used for normalization.")


class NormalizationMetrics(BaseModel):
    """Metrics for normalization performance tracking."""

    total_normalizations: int = Field(default=0, description="Total number of normalizations.")
    exact_match_hits: int = Field(default=0, description="Number of exact match hits.")
    semantic_match_hits: int = Field(default=0, description="Number of semantic match hits.")
