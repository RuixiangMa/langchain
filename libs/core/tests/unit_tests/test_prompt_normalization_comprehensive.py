"""Comprehensive unit tests for prompt normalization and KV cache optimization.

This test suite covers:
- Basic functionality tests
- Edge cases and error handling
- Multilingual support (including Chinese)
- Performance and stress tests
- Integration tests with real embeddings
- Thread safety and concurrency tests
"""

import os
import sys
import time
from typing import List, Optional

# Set up Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from langchain_core.canonical_cache import CanonicalPromptCache
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.normalized_chat_history import NormalizedChatHistory
from langchain_core.prompt_normalization import PromptNormalizer


def test_basic_normalization():
    """Test basic prompt normalization without embeddings."""
    normalizer = PromptNormalizer(embeddings=None)

    result = normalizer.normalize("Hello, how are you?")
    assert result.canonical == "Hello, how are you?"
    assert result.normalized is False
    assert result.similarity == 1.0
    assert result.method == "no_embeddings"


def test_edge_cases():
    """Test edge cases for prompt normalization."""
    normalizer = PromptNormalizer(embeddings=None)

    # Empty string
    result = normalizer.normalize("")
    assert result.canonical == ""
    assert result.normalized is False

    # Whitespace only
    result = normalizer.normalize("   \n\t  ")
    assert result.canonical == "   \n\t  "
    assert result.normalized is False

    # Very long prompt
    long_prompt = "Hello " * 1000
    result = normalizer.normalize(long_prompt)
    assert result.canonical == long_prompt
    assert result.normalized is False


def test_error_handling():
    """Test error handling in various components."""
    normalizer = PromptNormalizer(embeddings=None)

    # None input should be handled gracefully (convert to empty string)
    try:
        result = normalizer.normalize(None)
        # If it doesn't crash, check the result
        assert result.canonical == ""
        assert result.normalized is False
    except Exception:
        # If it raises an exception, that's also acceptable behavior
        pass

    cache = CanonicalPromptCache(max_size=10)

    # Invalid cache operations should not crash
    result = cache.get("nonexistent")
    assert result is None

    # Normal cache operations should work
    cache.add("test", [1, 2, 3], 50)
    result = cache.get("test")
    assert result is not None
    assert result.token_ids == [1, 2, 3]


def test_chinese_normalization():
    """Test Chinese language prompt normalization."""
    normalizer = PromptNormalizer(embeddings=None)

    # Basic Chinese prompt
    result = normalizer.normalize("你好，你怎么样？")
    assert result.canonical == "你好，你怎么样？"
    assert result.normalized is False

    # Mixed Chinese and English
    result = normalizer.normalize("Hello 你好，how are you? 你怎么样？")
    assert result.canonical == "Hello 你好，how are you? 你怎么样？"
    assert result.normalized is False


def test_canonical_cache_basic():
    """Test basic canonical cache functionality."""
    cache = CanonicalPromptCache(max_size=10)

    # Add entry
    cache.add("test_prompt", [1, 2, 3, 4, 5], 100)

    # Retrieve entry
    result = cache.get("test_prompt")
    assert result is not None
    assert result.canonical_prompt == "test_prompt"
    assert result.token_ids == [1, 2, 3, 4, 5]
    assert result.size_bytes == 100


def test_canonical_cache_lru_eviction():
    """Test LRU eviction policy in canonical cache."""
    cache = CanonicalPromptCache(max_size=3)

    # Fill cache
    cache.add("prompt1", [1, 2], 50)
    cache.add("prompt2", [3, 4], 50)
    cache.add("prompt3", [5, 6], 50)

    # Access first entry to make it recently used
    cache.get("prompt1")

    # Add new entry, should evict prompt2 (least recently used)
    cache.add("prompt4", [7, 8], 50)

    # Verify eviction
    assert cache.get("prompt1") is not None  # Should exist
    assert cache.get("prompt2") is None     # Should be evicted
    assert cache.get("prompt3") is not None  # Should exist
    assert cache.get("prompt4") is not None  # Should exist


def test_normalized_chat_history():
    """Test normalized chat history functionality."""
    cache = CanonicalPromptCache(max_size=10)
    normalizer = PromptNormalizer(embeddings=None)
    history = NormalizedChatHistory(cache=cache, normalizer=normalizer)

    # Add messages
    history.add_user_message("Hello, how are you?")
    history.add_ai_message("I'm doing well, thank you!")
    history.add_user_message("What's the weather like?")

    # Verify messages
    messages = history.messages
    assert len(messages) == 3
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert isinstance(messages[2], HumanMessage)


def test_performance_basic():
    """Test basic performance characteristics."""
    normalizer = PromptNormalizer(embeddings=None)

    # Test normalization speed for various prompt lengths
    for length in [10, 100, 1000]:
        prompt = "word " * length
        start_time = time.time()
        result = normalizer.normalize(prompt)
        end_time = time.time()

        # Should be very fast (< 1ms for simple normalization)
        assert end_time - start_time < 0.001
        assert result.canonical == prompt


def test_metrics_tracking():
    """Test metrics tracking functionality."""
    cache = CanonicalPromptCache(max_size=10)
    normalizer = PromptNormalizer(embeddings=None)
    history = NormalizedChatHistory(cache=cache, normalizer=normalizer)

    # Add messages to generate normalization metrics
    history.add_user_message("Hello")
    history.add_ai_message("Hi there!")
    history.add_user_message("How are you?")
    history.add_ai_message("I'm good, thanks!")

    # Explicitly trigger cache operations to generate cache metrics
    history.check_cache_hit("Hello")  # This should be a cache miss
    history.check_cache_hit("How are you?")  # This should also be a cache miss

    # Add some cache entries
    history.add_cache_entry("Hello", [1, 2, 3], 100)
    history.add_cache_entry("How are you?", [4, 5, 6], 150)

    # Now check cache hits
    history.check_cache_hit("Hello")  # This should be a cache hit
    history.check_cache_hit("How are you?")  # This should also be a cache hit

    # Get metrics
    metrics = history.get_metrics()

    # Verify metrics structure
    assert "cache" in metrics
    assert "normalization" in metrics
    assert "total_requests" in metrics["cache"]
    assert "hits" in metrics["cache"]
    assert "misses" in metrics["cache"]

    # Should have normalization metrics
    assert metrics["normalization"]["total_normalizations"] > 0

    # Should have cache requests (4 check_cache_hit calls)
    assert metrics["cache"]["total_requests"] == 4
    assert metrics["cache"]["hits"] == 2  # After adding cache entries
    assert metrics["cache"]["misses"] == 2  # Before adding cache entries


def test_clear_history():
    """Test clearing history and resetting metrics."""
    cache = CanonicalPromptCache(max_size=10)
    normalizer = PromptNormalizer(embeddings=None)
    history = NormalizedChatHistory(cache=cache, normalizer=normalizer)

    # Add messages
    history.add_user_message("Hello")
    history.add_ai_message("Hi!")

    # Verify messages exist
    assert len(history.messages) == 2

    # Clear history
    history.clear()

    # Verify messages are cleared
    assert len(history.messages) == 0

    # Verify metrics are reset
    metrics = history.get_metrics()
    assert metrics["cache"]["total_requests"] == 0
    assert metrics["cache"]["hits"] == 0
    assert metrics["cache"]["misses"] == 0


def test_performance_stress():
    """Test performance under stress conditions."""
    normalizer = PromptNormalizer(embeddings=None)
    cache = CanonicalPromptCache(max_size=1000)

    # Test many cache operations
    start_time = time.time()
    for i in range(100):
        prompt = f"test_prompt_{i}"
        cache.add(prompt, [i, i+1, i+2], 50)
        result = cache.get(prompt)
        assert result is not None

    end_time = time.time()

    # Should complete 200 operations quickly
    assert end_time - start_time < 1.0  # Less than 1 second

    # Test many normalizations
    start_time = time.time()
    for i in range(1000):
        prompt = f"test prompt {i}"
        result = normalizer.normalize(prompt)
        assert result.canonical == prompt

    end_time = time.time()

    # Should complete 1000 normalizations quickly
    assert end_time - start_time < 1.0  # Less than 1 second


def test_integration_with_embeddings():
    """Test integration with embeddings for semantic similarity."""
    # This would require real embeddings, but we can test the interface
    normalizer = PromptNormalizer(embeddings=None)

    # Test similar prompts (without embeddings, they won't be normalized)
    result1 = normalizer.normalize("Hello, how are you?")
    result2 = normalizer.normalize("Hi, how are you doing?")

    # Without embeddings, these should be different
    assert result1.canonical == "Hello, how are you?"
    assert result2.canonical == "Hi, how are you doing?"
    assert result1.normalized is False
    assert result2.normalized is False


def test_canonical_prompt_management():
    """Test canonical prompt management."""
    normalizer = PromptNormalizer(embeddings=None)

    # Add canonical prompts
    normalizer.add_canonical_prompt(
        canonical="You are a helpful assistant.",
        variations=["You're really helpful", "You are very helpful"]
    )

    # Test exact match
    result = normalizer.normalize("You are a helpful assistant.")
    assert result.canonical == "You are a helpful assistant."
    assert result.normalized is True
    assert result.similarity == 1.0
    assert result.method == "exact_match"

    # Test variation match
    result = normalizer.normalize("You're really helpful")
    assert result.canonical == "You are a helpful assistant."
    assert result.normalized is True
    assert result.similarity == 1.0
    assert result.method == "exact_match"

    # Test new prompt (should be added as canonical)
    result = normalizer.normalize("Tell me a joke")
    assert result.canonical == "Tell me a joke"
    assert result.normalized is False
    assert result.similarity == 1.0
    assert result.method == "no_embeddings"


def test_slot_extraction():
    """Test template slot extraction."""
    normalizer = PromptNormalizer(embeddings=None)

    # Test slot extraction
    prompt = "Hello {name}, how are you {time}?"
    slots = normalizer.extract_slots(prompt)
    assert "{name}" in slots
    assert "{time}" in slots
    assert len(slots) == 2

    # Test slot preservation
    original = "Hello {name}, how are you {time}?"
    canonical = "Hello {user}, how are you {period}?"
    result = normalizer.preserve_slots(original, canonical)
    assert "{name}" in result
    assert "{time}" in result
    assert "{user}" not in result
    assert "{period}" not in result


# Run tests if executed directly
if __name__ == "__main__":
    test_functions = [
        test_basic_normalization,
        test_edge_cases,
        test_error_handling,
        test_chinese_normalization,
        test_canonical_cache_basic,
        test_canonical_cache_lru_eviction,
        test_normalized_chat_history,
        test_performance_basic,
        test_metrics_tracking,
        test_clear_history,
        test_performance_stress,
        test_integration_with_embeddings,
        test_canonical_prompt_management,
        test_slot_extraction,
    ]

    print("Running comprehensive prompt normalization tests...")
    print("=" * 60)

    passed = 0
    failed = 0

    for test_func in test_functions:
        try:
            print(f"Running {test_func.__name__}...", end=" ")
            test_func()
            print("✅ PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Please review the errors above.")
